#![doc = include_str!("../README.md")]
use rusqlite::session::{ConflictAction, Session, invert_strm};
use rusqlite::{
    Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior, params,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fmt,
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

const ROW_BYTES: usize = 8192;
const ROW_TOTAL: usize = 131072;
const CAPTURE_BYTES: usize = 262144;
const INPUT_BYTES: usize = 65536;
const OUTPUT_BYTES: usize = 262144;
const SCHEMA: &str = "
PRAGMA application_id=1448563249;
PRAGMA user_version=1;
CREATE TABLE meta(id INTEGER PRIMARY KEY CHECK(id=1),project TEXT NOT NULL,sequence INTEGER NOT NULL,schema_version INTEGER NOT NULL);
CREATE TABLE tasks(id INTEGER PRIMARY KEY,title TEXT NOT NULL,owner TEXT,payload BLOB);
CREATE TABLE versions(id INTEGER PRIMARY KEY,revision INTEGER NOT NULL CHECK(revision>0) REFERENCES requests(revision) DEFERRABLE INITIALLY DEFERRED,live INTEGER NOT NULL CHECK(live IN (0,1)));
CREATE TABLE requests(id TEXT PRIMARY KEY NOT NULL,actor TEXT NOT NULL,command BLOB NOT NULL,
 revision INTEGER NOT NULL UNIQUE,receipt TEXT NOT NULL,capture BLOB NOT NULL,undo_of TEXT REFERENCES requests(id));
CREATE TABLE changes(request TEXT NOT NULL REFERENCES requests(id),id INTEGER NOT NULL,
 PRIMARY KEY(request,id)) WITHOUT ROWID;
CREATE INDEX changes_by_id ON changes(id,request);
";

#[derive(Debug)]
pub enum Error {
    Invalid,
    Conflict,
    ReusedKey,
    Cancelled,
    Busy,
    Limit,
    Schema,
    UnknownOutcome,
    Database(rusqlite::Error),
    Io(io::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "writer prototype: {self:?}")
    }
}
impl std::error::Error for Error {}
impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Self::Database(e)
    }
}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Serialize)]
pub enum Mutation {
    Create {
        id: i64,
        title: String,
        payload: Option<Vec<u8>>,
    },
    Claim {
        id: i64,
        expected: i64,
    },
    Delete {
        id: i64,
        expected: i64,
    },
}
impl Mutation {
    fn id(&self) -> i64 {
        match self {
            Self::Create { id, .. } | Self::Claim { id, .. } | Self::Delete { id, .. } => *id,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub enum Command {
    Batch(Vec<Mutation>),
    Undo { request: String },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Changed {
    pub id: i64,
    pub live: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    pub request: String,
    pub revision: i64,
    pub changes: Vec<Changed>,
    pub undo_of: Option<String>,
}
#[derive(Debug, PartialEq, Eq)]
pub struct Task {
    pub title: String,
    pub owner: Option<String>,
    pub payload: Option<Vec<u8>>,
    pub revision: i64,
}
#[derive(Debug, PartialEq, Eq)]
pub struct History {
    pub request: String,
    pub actor: String,
    pub revision: i64,
    pub undo_of: Option<String>,
}

struct Limited {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}
impl Limited {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            exceeded: false,
        }
    }
    fn finish(self, result: rusqlite::Result<()>) -> Result<Vec<u8>> {
        if self.exceeded {
            return Err(Error::Limit);
        }
        result?;
        Ok(self.bytes)
    }
}
impl Write for Limited {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other("byte limit"));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Mutated,
    Captured,
    Ledger,
    BeforeCommit,
    Committed,
}
#[cfg(test)]
type Hook = Box<dyn Fn(Stage, &Connection, &AtomicBool) -> Result<()> + Send>;

/// Owns a separate connection to a database created explicitly for this prototype.
pub struct Writer {
    db: Option<Connection>,
    project: String,
    actor: String,
    #[cfg(test)]
    hook: Option<Hook>,
    #[cfg(test)]
    capture_limit: usize,
}
fn identity(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 128 {
        return Err(Error::Invalid);
    }
    Ok(())
}
fn cancelled(flag: &AtomicBool) -> Result<()> {
    if flag.load(Ordering::SeqCst) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
impl Writer {
    /// Creates a new file exclusively. A failed setup is never treated as a ready fixture.
    pub fn create(path: &Path, project: &str, actor: &str) -> Result<Self> {
        identity(project)?;
        identity(actor)?;
        drop(OpenOptions::new().write(true).create_new(true).open(path)?);
        let mut db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
        db.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;",
        )?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(SCHEMA)?;
        let version: i64 = tx.query_row("PRAGMA schema_version", [], |r| r.get(0))?;
        tx.execute(
            "INSERT INTO meta VALUES(1,?,0,?)",
            params![project, version],
        )?;
        tx.commit()?;
        drop(db);
        Self::open(path, project, actor)
    }
    pub fn open(path: &Path, project: &str, actor: &str) -> Result<Self> {
        identity(project)?;
        identity(actor)?;
        let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
        db.busy_timeout(Duration::ZERO)?;
        db.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")?;
        let writer = Self {
            db: Some(db),
            project: project.into(),
            actor: actor.into(),
            #[cfg(test)]
            hook: None,
            #[cfg(test)]
            capture_limit: CAPTURE_BYTES,
        };
        writer.check_schema(writer.connection()?)?;
        Ok(writer)
    }
    fn connection(&self) -> Result<&Connection> {
        self.db.as_ref().ok_or(Error::UnknownOutcome)
    }
    fn check_schema(&self, db: &Connection) -> Result<()> {
        let application: i64 = db.query_row("PRAGMA application_id", [], |r| r.get(0))?;
        let version: i64 = db.query_row("PRAGMA schema_version", [], |r| r.get(0))?;
        let user: i64 = db.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if application != 1448563249 || user != 1 {
            return Err(Error::Schema);
        }
        let valid: bool = db.query_row(
            "SELECT project=? AND schema_version=? FROM meta WHERE id=1",
            params![self.project, version],
            |r| r.get(0),
        )?;
        if !valid {
            return Err(Error::Schema);
        }
        Ok(())
    }
    fn checkpoint(&self, stage: Stage, db: &Connection, cancel: &AtomicBool) -> Result<()> {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(stage, db, cancel)?;
        }
        #[cfg(not(test))]
        let _ = (stage, db);
        if stage != Stage::Committed {
            cancelled(cancel)?;
        }
        Ok(())
    }
    fn begin<'a>(&self, db: &'a Connection, cancel: &AtomicBool) -> Result<Transaction<'a>> {
        let deadline = Instant::now() + Duration::from_millis(250);
        loop {
            cancelled(cancel)?;
            match Transaction::new_unchecked(db, TransactionBehavior::Immediate) {
                Ok(tx) => return Ok(tx),
                Err(e) if e.sqlite_error_code() == Some(rusqlite::ErrorCode::DatabaseBusy) => {
                    if Instant::now() >= deadline {
                        return Err(Error::Busy);
                    }
                    thread::sleep(Duration::from_millis(2));
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
    fn row_bytes(db: &Connection, ids: &[i64]) -> Result<()> {
        let mut total = 0usize;
        for id in ids {
            let bytes: i64=db.query_row("SELECT coalesce((SELECT length(CAST(title AS BLOB))+coalesce(length(CAST(owner AS BLOB)),0)+coalesce(length(payload),0) FROM tasks WHERE id=?),0)",[id],|r|r.get(0))?;
            let size = usize::try_from(bytes).map_err(|_| Error::Limit)?;
            total = total.checked_add(size).ok_or(Error::Limit)?;
            if size > ROW_BYTES || total > ROW_TOTAL {
                return Err(Error::Limit);
            }
        }
        Ok(())
    }
    fn revision(db: &Connection, id: i64) -> Result<Option<(i64, bool)>> {
        Ok(db
            .query_row("SELECT revision,live FROM versions WHERE id=?", [id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .optional()?)
    }
    /// Returns only committed receipts. Unknown commit/rollback outcomes poison this handle.
    pub fn apply(
        &mut self,
        request: &str,
        command: &Command,
        cancel: &AtomicBool,
    ) -> Result<Receipt> {
        let result = self.apply_inner(request, command, cancel);
        if matches!(result, Err(Error::UnknownOutcome)) {
            self.db.take();
        }
        result
    }
    fn apply_inner(
        &self,
        request: &str,
        command: &Command,
        cancel: &AtomicBool,
    ) -> Result<Receipt> {
        identity(request)?;
        if let Command::Batch(ops) = command {
            if ops.is_empty() || ops.len() > 64 {
                return Err(Error::Invalid);
            }
            let mut seen = HashSet::new();
            for op in ops {
                if !seen.insert(op.id()) {
                    return Err(Error::Invalid);
                }
                match op {
                    Mutation::Create { title, payload, .. }
                        if title.len() + payload.as_ref().map_or(0, Vec::len) > ROW_BYTES =>
                    {
                        return Err(Error::Limit);
                    }
                    Mutation::Claim { expected, .. } | Mutation::Delete { expected, .. }
                        if *expected <= 0 =>
                    {
                        return Err(Error::Invalid);
                    }
                    _ => {}
                }
            }
        } else if let Command::Undo { request: target } = command {
            identity(target)?;
        }
        let mut canonical = Limited::new(INPUT_BYTES);
        serde_json::to_writer(&mut canonical, &(&self.project, &self.actor, command))
            .map_err(|_| Error::Limit)?;
        let db = self.connection()?;
        let tx = self.begin(db, cancel)?;
        let result = (|| {
            self.check_schema(&tx)?;
            let old: Option<(i64, i64)> = tx
                .query_row(
                    "SELECT length(command),length(receipt) FROM requests WHERE id=?",
                    [request],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            if let Some((bytes, receipt_bytes)) = old {
                if !(0..=INPUT_BYTES as i64).contains(&bytes)
                    || !(0..=OUTPUT_BYTES as i64).contains(&receipt_bytes)
                {
                    return Err(Error::Limit);
                }
                let (prior, receipt): (Vec<u8>, String) = tx.query_row(
                    "SELECT command,receipt FROM requests WHERE id=?",
                    [request],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?;
                if prior != canonical.bytes {
                    return Err(Error::ReusedKey);
                }
                return serde_json::from_str(&receipt).map_err(|_| Error::Schema);
            }
            self.perform(&tx, request, command, &canonical.bytes, cancel)
        })();
        match result {
            Ok(receipt) => {
                if let Err(e) = self.checkpoint(Stage::BeforeCommit, &tx, cancel) {
                    tx.rollback().map_err(|_| Error::UnknownOutcome)?;
                    return Err(e);
                }
                tx.commit().map_err(|_| Error::UnknownOutcome)?;
                self.checkpoint(Stage::Committed, db, cancel)
                    .map_err(|_| Error::UnknownOutcome)?;
                Ok(receipt)
            }
            Err(e) => {
                tx.rollback().map_err(|_| Error::UnknownOutcome)?;
                Err(e)
            }
        }
    }
    fn perform(
        &self,
        tx: &Transaction<'_>,
        request: &str,
        command: &Command,
        canonical: &[u8],
        cancel: &AtomicBool,
    ) -> Result<Receipt> {
        let mut ids = Vec::new();
        let mut inverse = None;
        let mut undo_of = None;
        match command {
            Command::Batch(ops) => {
                for op in ops {
                    let version = Self::revision(tx, op.id())?;
                    let valid = match op {
                        Mutation::Create { .. } => version.is_none(),
                        Mutation::Claim { expected, .. } | Mutation::Delete { expected, .. } => {
                            version == Some((*expected, true))
                        }
                    };
                    if !valid {
                        return Err(Error::Conflict);
                    }
                    ids.push(op.id());
                }
            }
            Command::Undo { request: target } => {
                let stored: Option<(i64, i64)> = tx
                    .query_row(
                        "SELECT revision,length(capture) FROM requests WHERE id=?",
                        [target],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()?;
                let (revision, size) = stored.ok_or(Error::Conflict)?;
                if !(0..=CAPTURE_BYTES as i64).contains(&size) {
                    return Err(Error::Limit);
                }
                let mut stmt =
                    tx.prepare("SELECT id FROM changes WHERE request=? ORDER BY id LIMIT 65")?;
                ids = stmt
                    .query_map([target], |r| r.get(0))?
                    .collect::<rusqlite::Result<Vec<i64>>>()?;
                if ids.is_empty() || ids.len() > 64 {
                    return Err(Error::Schema);
                }
                for id in &ids {
                    if Self::revision(tx, *id)?.map(|v| v.0) != Some(revision) {
                        return Err(Error::Conflict);
                    }
                }
                let blob: Vec<u8> =
                    tx.query_row("SELECT capture FROM requests WHERE id=?", [target], |r| {
                        r.get(0)
                    })?;
                let mut output = Limited::new(CAPTURE_BYTES);
                let result = invert_strm(&mut blob.as_slice(), &mut output);
                inverse = Some(output.finish(result)?);
                undo_of = Some(target.clone());
            }
        }
        Self::row_bytes(tx, &ids)?;
        let sequence: i64 =
            tx.query_row("SELECT sequence FROM meta WHERE id=1", [], |r| r.get(0))?;
        let revision = sequence
            .checked_add(1)
            .filter(|v| *v > 0)
            .ok_or(Error::Limit)?;
        let mut capture = Session::new(tx)?;
        capture.attach(Some("tasks"))?;
        match command {
            Command::Batch(ops) => {
                for op in ops {
                    cancelled(cancel)?;
                    match op {
                        Mutation::Create { id, title, payload } => {
                            tx.execute(
                                "INSERT INTO tasks VALUES(?,?,NULL,?)",
                                params![id, title, payload],
                            )?;
                        }
                        Mutation::Claim { id, .. } => {
                            if tx.execute(
                                "UPDATE tasks SET owner=? WHERE id=? AND owner IS NULL",
                                params![self.actor, id],
                            )? != 1
                            {
                                return Err(Error::Conflict);
                            }
                        }
                        Mutation::Delete { id, .. } => {
                            if tx.execute("DELETE FROM tasks WHERE id=?", [id])? != 1 {
                                return Err(Error::Conflict);
                            }
                        }
                    }
                }
            }
            Command::Undo { .. } => {
                let blob = inverse.as_ref().ok_or(Error::Schema)?;
                tx.apply_strm(
                    &mut blob.as_slice(),
                    Some(|name: &str| name == "tasks"),
                    |_, _| ConflictAction::SQLITE_CHANGESET_ABORT,
                )?;
            }
        }
        self.checkpoint(Stage::Mutated, tx, cancel)?;
        Self::row_bytes(tx, &ids)?;
        let limit = CAPTURE_BYTES;
        #[cfg(test)]
        let limit = limit.min(self.capture_limit);
        let mut bytes = Limited::new(limit);
        let result = capture.changeset_strm(&mut bytes);
        let bytes = bytes.finish(result)?;
        if bytes.is_empty() {
            return Err(Error::Conflict);
        }
        self.checkpoint(Stage::Captured, tx, cancel)?;
        let mut changed = Vec::new();
        for id in ids {
            let live: bool =
                tx.query_row("SELECT EXISTS(SELECT 1 FROM tasks WHERE id=?)", [id], |r| {
                    r.get(0)
                })?;
            tx.execute("INSERT INTO versions VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,live=excluded.live",params![id,revision,live])?;
            changed.push(Changed { id, live });
        }
        changed.sort_by_key(|r| r.id);
        let receipt = Receipt {
            request: request.into(),
            revision,
            changes: changed,
            undo_of,
        };
        let json = serde_json::to_string(&receipt).map_err(|_| Error::Schema)?;
        if json.len() > OUTPUT_BYTES {
            return Err(Error::Limit);
        }
        tx.execute("UPDATE meta SET sequence=? WHERE id=1", [revision])?;
        tx.execute(
            "INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
            params![
                request,
                self.actor,
                canonical,
                revision,
                json,
                bytes,
                receipt.undo_of
            ],
        )?;
        for change in &receipt.changes {
            tx.execute(
                "INSERT INTO changes VALUES(?,?)",
                params![request, change.id],
            )?;
        }
        self.checkpoint(Stage::Ledger, tx, cancel)?;
        Ok(receipt)
    }
    pub fn task(&self, id: i64) -> Result<Option<Task>> {
        let tx = self.connection()?.unchecked_transaction()?;
        self.check_schema(&tx)?;
        Self::row_bytes(&tx, &[id])?;
        let task=tx.query_row("SELECT title,owner,payload,revision FROM tasks JOIN versions USING(id) WHERE id=? AND live=1",[id],|r|Ok(Task{title:r.get(0)?,owner:r.get(1)?,payload:r.get(2)?,revision:r.get(3)?})).optional()?;
        tx.rollback()?;
        Ok(task)
    }
    pub fn history(&self, id: i64, limit: usize) -> Result<Vec<History>> {
        if !(1..=100).contains(&limit) {
            return Err(Error::Invalid);
        }
        let tx = self.connection()?.unchecked_transaction()?;
        self.check_schema(&tx)?;
        let limit = i64::try_from(limit).map_err(|_| Error::Invalid)?;
        let history = {
            let mut stmt=tx.prepare("SELECT r.id,r.actor,r.revision,r.undo_of FROM changes c JOIN requests r ON r.id=c.request WHERE c.id=? ORDER BY r.revision DESC LIMIT ?")?;
            stmt.query_map(params![id, limit], |r| {
                Ok(History {
                    request: r.get(0)?,
                    actor: r.get(1)?,
                    revision: r.get(2)?,
                    undo_of: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?
        };
        tx.rollback()?;
        Ok(history)
    }
}

#[cfg(test)]
mod tests;
