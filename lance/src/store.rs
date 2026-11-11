//! `LanceStore`: rows keyed by rowid in a Lance dataset, following SQLite
//! transactions. Shared by the `lance0` virtual table and the vec0 storage
//! provider, so the transaction rules live in one place:
//!
//! - Writes wait in a pending map (rowid -> row, or deletion) until `sync`
//!   (SQLite's xSync), which writes them to Lance and records the version
//!   Lance reached in a SQLite shadow table, inside the same transaction.
//! - Reads open Lance at the recorded version and merge the pending map, so a
//!   connection sees its own writes and never sees an orphan Lance version
//!   left by a failed commit or a crash. The next writer restores Lance to the
//!   recorded version first.
//! - Savepoints (including the implicit one per statement) roll the pending
//!   map back through an undo log.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_int;
use std::sync::{Arc, OnceLock};

use lance::Dataset;
use lance::dataset::{MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched, WriteMode,
                     WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_index::vector::{hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, sq::builder::SQBuildParams};
use lance_linalg::distance::DistanceType;
use lance::deps::arrow_array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
    LargeBinaryArray, LargeStringArray, RecordBatch, RecordBatchIterator, StringArray,
};
use lance::deps::arrow_schema::{DataType, Field, Schema};
use rusqlite::types::Value;
use rusqlite::{Connection, Error, Result, ffi};
use tokio::runtime::Runtime;

pub fn runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    // ponytail: 2 workers is enough for one embedded writer; make it
    // configurable if readers start to contend.
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime")
    })
}

pub fn block<F: std::future::Future>(f: F) -> F::Output {
    runtime().block_on(f)
}

pub(crate) fn id_literal(id: i64) -> String {
    // Lance parses the magnitude of i64::MIN as a float before applying '-'.
    if id == i64::MIN { "CAST('-9223372036854775808' AS BIGINT)".into() }
    else { id.to_string() }
}

/// Rows appended since the rowid index last covered them, before a commit
/// updates it. Until then Lance scans those rows and uses the index for the
/// rest. ponytail: fixed threshold; make it a table option if a workload
/// needs fresher indexing.
const INDEX_LAG_ROWS: i64 = 1024;
const ID_INDEX: &str = "id_idx";
const VECTOR_INDEX_MIN_ROWS: usize = 16384;
const VECTOR_INDEX: &str = "vector_idx";
/// Fragment count that triggers compaction during a commit. Every small
/// transaction appends its own fragment; without compaction a table written
/// by many small transactions accumulates files and deleted-row tombstones.
/// ponytail: fixed threshold; make it a table option if a workload needs
/// control over the write amplification this trades away.
const COMPACT_MIN_FRAGMENTS: usize = 16;

pub fn lance_err(e: impl std::fmt::Display) -> Error {
    Error::ModuleError(format!("lance: {e}"))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Integer,
    Real,
    Text,
    Blob,
    /// float32[dim], stored as a FixedSizeList so Lance can search it. As a
    /// SQLite value it is a little-endian float32 BLOB.
    Vector(usize),
}

pub type Row = Vec<Value>;

#[derive(Clone)]
struct Pending {
    row: Option<Row>, // Some = the row as this transaction sees it; None = deleted
    replaces: bool,   // hide the committed row (a delete or update happened)
    /// Some(columns): a committed row where only these columns changed, so
    /// the commit patches just them. None: the whole row was written.
    changed: Option<Vec<usize>>,
    /// False for a patch whose `row` holds only the patched values: reads
    /// merge the committed row in, so writing a patch needs no read.
    complete: bool,
}

pub struct LanceStore {
    db: *mut ffi::sqlite3,
    database: String, // SQLite schema containing the virtual table and its metadata
    shadow: String, // unquoted name of the version table
    uri: String,
    pub cols: Vec<(String, Kind)>,
    /// Only lance0 searches vectors in Lance; vec0 maintains its own indexes.
    pub ann_column: Option<usize>,
    schema: Arc<Schema>,
    pending: HashMap<i64, Pending>,
    undo: Vec<(i64, Option<Pending>)>,
    marks: Vec<(c_int, usize, bool)>,
    in_write: bool,
    /// Maintenance requested in this transaction; runs at commit (xSync).
    optimize: bool,
    /// The dataset opened at the committed version, reused until it changes:
    /// opening reads the manifest from storage on every call otherwise.
    opened: RefCell<Option<(u64, Dataset)>>,
}

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn qualified(database: &str, table: &str) -> String {
    format!("{}.{}", quote(database), quote(table))
}

impl LanceStore {
    /// Open (or, with `create`, create) the store. `shadow` is the SQLite
    /// table that records the committed Lance version.
    pub fn open(db: *mut ffi::sqlite3, database: &str, shadow: &str, uri: &str, cols: Vec<(String, Kind)>,
                create: bool) -> Result<Self> {
        let mut fields = vec![Field::new("id", DataType::Int64, false)];
        for (name, kind) in &cols {
            let t = match kind {
                Kind::Integer => DataType::Int64,
                Kind::Real => DataType::Float64,
                Kind::Text => DataType::Utf8,
                Kind::Blob => DataType::LargeBinary,
                Kind::Vector(d) => DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)), *d as i32),
            };
            fields.push(Field::new(name, t, true));
        }
        let store = LanceStore {
            db,
            database: database.to_owned(),
            shadow: shadow.to_owned(),
            uri: uri.to_owned(),
            cols,
            ann_column: None,
            schema: Arc::new(Schema::new(fields)),
            pending: HashMap::new(),
            undo: Vec::new(),
            marks: Vec::new(),
            in_write: false,
            optimize: false,
            opened: RefCell::new(None),
        };
        store.init(create)?;
        Ok(store)
    }

    fn conn(&self) -> Result<Connection> {
        // Borrows the host connection; it is not closed on drop.
        unsafe { Connection::from_handle(self.db) }
    }

    fn init(&self, create: bool) -> Result<()> {
        let conn = self.conn()?;
        let shadow = qualified(&self.database, &self.shadow);
        conn.execute(&format!("create table if not exists {shadow}(k primary key, v)"), [])?;
        let recorded: Option<i64> = conn
            .query_row(&format!("select v from {shadow} where k = 'version'"), [], |r| r.get(0))
            .ok();
        if recorded.is_some() {
            return Ok(());
        }
        let schema = self.schema.clone();
        let uri = self.uri.clone();
        let version = block(async move {
            if let Ok(ds) = Dataset::open(&uri).await {
                if create {
                    return Err(lance::Error::invalid_input(format!(
                        "a Lance dataset already exists at {uri}"
                    )));
                }
                return Ok(ds.version().version);
            }
            let reader = RecordBatchIterator::new(vec![Ok(RecordBatch::new_empty(schema.clone()))], schema);
            let params = WriteParams { mode: WriteMode::Create, ..Default::default() };
            Ok(Dataset::write(reader, uri.as_str(), Some(params)).await?.version().version)
        })
        .map_err(lance_err)?;
        conn.execute(&format!("insert into {shadow} values ('version', ?1)"), [version as i64])?;
        // The location, so later opens (e.g. after a rename) find the same data.
        conn.execute(&format!("insert or replace into {shadow} values ('uri', ?1)"), [&self.uri])?;
        Ok(())
    }

    /// The dataset location recorded by an earlier `open`, if any.
    pub fn recorded_uri(db: *mut ffi::sqlite3, database: &str, shadow: &str) -> Option<String> {
        let conn = unsafe { Connection::from_handle(db) }.ok()?;
        conn.query_row(&format!("select v from {} where k = 'uri'", qualified(database, shadow)), [], |r| r.get(0)).ok()
    }

    fn committed_version(&self) -> Result<u64> {
        let v: i64 = self.conn()?.query_row(
            &format!("select v from {} where k = 'version'", qualified(&self.database, &self.shadow)),
            [],
            |r| r.get(0),
        )?;
        Ok(v as u64)
    }

    /// The dataset at the version SQLite committed.
    pub fn reader(&self) -> Result<Dataset> {
        let v = self.committed_version()?;
        if let Some((ver, ds)) = self.opened.borrow().as_ref() {
            if *ver == v {
                return Ok(ds.clone());
            }
        }
        let ds = block(async { Dataset::open(&self.uri).await?.checkout_version(v).await }).map_err(lance_err)?;
        *self.opened.borrow_mut() = Some((v, ds.clone()));
        Ok(ds)
    }

    fn begin_write(&mut self) -> Result<()> {
        if self.in_write {
            return Ok(());
        }
        // Restore over any orphan version from a crash or a failed commit, so
        // this transaction's Lance writes build on what SQLite committed.
        let committed = self.committed_version()?;
        block(async {
            let ds = Dataset::open(&self.uri).await?;
            if ds.version().version != committed {
                ds.checkout_version(committed).await?.restore().await?;
            }
            Ok::<_, lance::Error>(())
        })
        .map_err(lance_err)?;
        self.in_write = true;
        Ok(())
    }

    // ---------------------------------------------------------------- writes

    fn set(&mut self, id: i64, row: Option<Row>, replaces: bool, changed: Option<Vec<usize>>) {
        let prior = self.pending.get(&id);
        let replaces = replaces || prior.is_some_and(|p| p.replaces);
        // Column patches accumulate; any whole-row write makes the row whole.
        let changed = match (prior, changed) {
            (_, None) => None,
            (None, Some(c)) => Some(c),
            (Some(p), Some(mut c)) => p.changed.as_ref().map(|old| {
                c.extend(old.iter().copied());
                c.sort_unstable();
                c.dedup();
                c
            }),
        };
        let prev = self.pending.insert(id, Pending { row, replaces, changed, complete: true });
        if !self.marks.is_empty() {
            self.undo.push((id, prev));
        }
    }

    /// A new row (a rowid not in the store).
    pub fn insert(&mut self, id: i64, row: Row) -> Result<()> {
        self.begin_write()?;
        self.set(id, Some(row), false, None);
        Ok(())
    }

    /// Logical identity, without materializing vector/blob payloads.
    pub fn contains(&self, id: i64) -> Result<bool> {
        if let Some(pending) = self.pending.get(&id) {
            return Ok(pending.row.is_some());
        }
        let ds = self.reader()?;
        let mut scan = ds.scan();
        scan.project(&["id"]).map_err(lance_err)?;
        scan.filter(&format!("id = {}", id_literal(id))).map_err(lance_err)?;
        scan.limit(Some(1), None).map_err(lance_err)?;
        Ok(block(scan.try_into_batch()).map_err(lance_err)?.num_rows() != 0)
    }

    /// Replace the whole row (or create it).
    pub fn replace(&mut self, id: i64, row: Row) -> Result<()> {
        self.begin_write()?;
        self.set(id, Some(row), true, None);
        Ok(())
    }

    /// Change some columns of an existing row. The commit writes only these
    /// columns, so updating metadata never rewrites a row's vector or blob.
    pub fn update_columns(&mut self, id: i64, updates: Vec<(usize, Value)>) -> Result<()> {
        self.begin_write()?;
        let mut cols: Vec<usize> = updates.iter().map(|u| u.0).collect();
        let entry = match self.pending.get(&id) {
            Some(Pending { row: None, .. }) => return Err(lance_err(format!("no row {id}"))),
            Some(p) => {
                let mut p = p.clone();
                let row = p.row.as_mut().unwrap();
                for (c, v) in updates {
                    row[c] = v;
                }
                if let Some(old) = &p.changed {
                    cols.extend(old.iter().copied());
                    cols.sort_unstable();
                    cols.dedup();
                    p.changed = Some(cols);
                } // else: a fresh insert or whole-row write stays whole
                p
            }
            None => {
                // The committed row is not read here: only the patch is kept.
                let mut row = vec![Value::Null; self.cols.len()];
                for (c, v) in updates {
                    row[c] = v;
                }
                Pending { row: Some(row), replaces: true, changed: Some(cols), complete: false }
            }
        };
        let prev = self.pending.insert(id, entry);
        if !self.marks.is_empty() {
            self.undo.push((id, prev));
        }
        Ok(())
    }

    /// Table maintenance, run when the transaction commits:
    /// - `optimize`: merge small data files and maintain rowid and configured ANN indexes.
    /// Returns false for commands this store does not know.
    pub fn command(&mut self, cmd: &str) -> Result<bool> {
        if cmd == "optimize" {
            self.begin_write()?;
            self.optimize = true;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn delete(&mut self, id: i64) -> Result<()> {
        self.begin_write()?;
        self.set(id, None, true, None);
        Ok(())
    }

    // --------------------------------------------------------- transactions

    pub fn savepoint(&mut self, id: c_int) {
        self.marks.retain(|m| m.0 < id);
        // First enlistment announces only the innermost already-open savepoint.
        let first = self.marks.last().map_or(0, |m| m.0 + 1);
        for nested in first..=id {
            self.marks.push((nested, self.undo.len(), self.optimize));
        }
    }

    pub fn rollback_to(&mut self, id: c_int) {
        let Some(&(_, len, optimize)) = self.marks.iter().find(|m| m.0 == id) else { return };
        self.optimize = optimize;
        while self.undo.len() > len {
            let (rowid, prev) = self.undo.pop().unwrap();
            match prev {
                Some(p) => { self.pending.insert(rowid, p); }
                None => { self.pending.remove(&rowid); }
            }
        }
        self.marks.retain(|m| m.0 <= id);
    }

    pub fn release(&mut self, id: c_int) {
        self.marks.retain(|m| m.0 < id);
        if self.marks.is_empty() {
            self.undo.clear();
        }
    }

    /// SQLite's xSync: write pending changes to Lance and record the version.
    pub fn sync(&mut self) -> Result<()> {
        if !self.in_write {
            return Ok(());
        }
        // Group the transaction's changes by how Lance writes them.
        let mut deletes = Vec::new();
        let mut fresh: Vec<(i64, &Row)> = Vec::new();
        let mut whole: Vec<(i64, &Row)> = Vec::new();
        let mut patches: HashMap<Vec<usize>, Vec<(i64, &Row)>> = HashMap::new();
        let mut ids: Vec<&i64> = self.pending.keys().collect();
        ids.sort();
        for id in ids {
            let p = &self.pending[id];
            match (&p.row, p.replaces, &p.changed) {
                (None, _, _) => deletes.push(*id),
                (Some(r), false, _) => fresh.push((*id, r)),
                (Some(r), true, None) => whole.push((*id, r)),
                (Some(r), true, Some(c)) => patches.entry(c.clone()).or_default().push((*id, r)),
            }
        }
        let appended = fresh.len() + whole.len();
        let fresh = if fresh.is_empty() { None } else { Some(self.batch(&fresh)?) };
        let whole = if whole.is_empty() { None } else { Some(self.batch(&whole)?) };
        let mut patch_batches = Vec::new();
        for (cols, rows) in patches {
            let full = self.batch(&rows)?;
            let keep: Vec<usize> = std::iter::once(0).chain(cols.iter().map(|c| c + 1)).collect();
            patch_batches.push(full.project(&keep).map_err(lance_err)?);
        }

        let uri = self.uri.clone();
        let schema = self.schema.clone();
        let conn = self.conn()?;
        let shadow = qualified(&self.database, &self.shadow);
        let lag: i64 = conn
            .query_row(&format!("select v from {shadow} where k = 'unindexed'"), [], |r| r.get(0))
            .unwrap_or(0)
            + appended as i64;
        let optimize = self.optimize;
        let ann_column = self.ann_column.map(|col| self.cols[col].0.clone());
        let reindex = lag >= INDEX_LAG_ROWS || (optimize && lag > 0);
        let latest = block(async move {
            let mut ds = Dataset::open(&uri).await?;
            for chunk in deletes.chunks(1000) {
                let ids: Vec<String> = chunk.iter().map(|d| id_literal(*d)).collect();
                ds.delete(&format!("id IN ({})", ids.join(","))).await?;
            }
            // Replaced rows: one upsert instead of a delete plus an append.
            if let Some(batch) = whole {
                let s = batch.schema();
                let (out, _) = MergeInsertBuilder::try_new(Arc::new(ds), vec!["id".to_owned()])?
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched(WhenNotMatched::InsertAll)
                    .try_build()?
                    .execute_reader(RecordBatchIterator::new(vec![Ok(batch)], s))
                    .await?;
                ds = (*out).clone();
            }
            // Column patches: write id + the changed columns only.
            for batch in patch_batches {
                let s = batch.schema();
                let (out, _) = MergeInsertBuilder::try_new(Arc::new(ds), vec!["id".to_owned()])?
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched(WhenNotMatched::DoNothing)
                    .write_mode(MergeInsertWriteMode::RewriteColumns)
                    .try_build()?
                    .execute_reader(RecordBatchIterator::new(vec![Ok(batch)], s))
                    .await?;
                ds = (*out).clone();
            }
            if let Some(batch) = fresh {
                let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
                let params = WriteParams { mode: WriteMode::Append, ..Default::default() };
                Dataset::write(reader, uri.as_str(), Some(params)).await?;
                ds = Dataset::open(&uri).await?;
            }
            if reindex {
                // A BTREE index on id turns rowid lookups into index probes.
                let exists = ds.load_indices().await?.iter().any(|i| i.name == ID_INDEX);
                if exists {
                    let opts = OptimizeOptions::merge(1).index_names(vec![ID_INDEX.to_owned()]);
                    ds.optimize_indices(&opts).await?;
                } else {
                    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
                    ds.create_index(&["id"], IndexType::BTree, Some(ID_INDEX.to_owned()), &params, true).await?;
                }
            }
            if optimize || ds.get_fragments().len() >= COMPACT_MIN_FRAGMENTS {
                // Merge small fragments and materialize deletions. Indices
                // are remapped eagerly (the default), so the rowid index
                // stays valid; the version recorded below is the compacted
                // one, keeping Lance and SQLite in step.
                let opts = lance::dataset::optimize::CompactionOptions::default();
                lance::dataset::optimize::compact_files(&mut ds, opts, None).await?;
            }
            if optimize && let Some(column) = ann_column {
                if ds.load_indices().await?.iter().any(|i| i.name == VECTOR_INDEX) {
                    let opts = OptimizeOptions::merge(1).index_names(vec![VECTOR_INDEX.to_owned()]);
                    ds.optimize_indices(&opts).await?;
                } else {
                    let rows = ds.count_rows(None).await?;
                    if rows >= VECTOR_INDEX_MIN_ROWS {
                        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
                            DistanceType::L2,
                            IvfBuildParams::new((rows / 256).clamp(1, 256)),
                            HnswBuildParams::default(), SQBuildParams::default(),
                        );
                        ds.create_index(&[&column], IndexType::IvfHnswSq,
                                        Some(VECTOR_INDEX.to_owned()), &params, true).await?;
                    }
                }
            }
            Ok::<_, lance::Error>(ds.version().version)
        })
        .map_err(lance_err)?;
        // Several Lance operations make several versions; SQLite commits the last.
        conn.execute(&format!("update {shadow} set v = ?1 where k = 'version'"), [latest as i64])?;
        conn.execute(&format!("insert or replace into {shadow} values ('unindexed', ?1)"),
                     [if reindex { 0 } else { lag }])?;
        Ok(())
    }

    pub fn end_transaction(&mut self) {
        self.optimize = false;
        self.pending.clear();
        self.undo.clear();
        self.marks.clear();
        self.in_write = false;
    }

    // ----------------------------------------------------------------- reads

    /// This transaction's rows (patched rows merged with their committed
    /// values, fetched in one batch), by rowid.
    pub fn pending_rows(&self) -> Result<Vec<(i64, Row)>> {
        self.pending_rows_for(None)
    }

    /// Like `pending_rows`, limited to one rowid when `only` is set, so a
    /// rowid lookup completes just that row instead of every patch.
    pub fn pending_rows_for(&self, only: Option<i64>) -> Result<Vec<(i64, Row)>> {
        let incomplete: Vec<i64> = self.pending.iter()
            .filter(|(id, p)| !p.complete && only.is_none_or(|o| o == **id))
            .map(|(id, _)| *id).collect();
        let mut bases = self.committed_rows(&incomplete)?;
        let mut rows: Vec<(i64, Row)> = Vec::with_capacity(self.pending.len());
        for (id, p) in &self.pending {
            if only.is_some_and(|o| o != *id) {
                continue;
            }
            let Some(r) = &p.row else { continue };
            if p.complete {
                rows.push((*id, r.clone()));
            } else if let Some(mut base) = bases.remove(id) {
                for c in p.changed.iter().flatten() {
                    base[*c] = r[*c].clone();
                }
                rows.push((*id, base));
            }
        }
        rows.sort_by_key(|r| r.0);
        Ok(rows)
    }

    /// Whether this transaction wrote, patched or deleted this rowid.
    pub fn has_pending(&self, id: i64) -> bool {
        self.pending.contains_key(&id)
    }

    /// Whether this transaction hides the committed row with this rowid.
    pub fn hides(&self, id: i64) -> bool {
        self.pending.get(&id).is_some_and(|p| p.replaces)
    }

    pub fn hidden_count(&self) -> usize {
        self.pending.values().filter(|p| p.replaces).count()
    }

    /// The row as this transaction sees it: Some(Some(row)), Some(None) when
    /// it was deleted, None when the pending map does not decide.
    pub fn pending_get(&self, id: i64) -> Result<Option<Option<Row>>> {
        let Some(p) = self.pending.get(&id) else { return Ok(None) };
        if p.complete || p.row.is_none() {
            return Ok(Some(p.row.clone()));
        }
        Ok(Some(self.get_many(&[id])?.remove(&id)))
    }

    /// Committed rows in Lance with these rowids, ignoring the pending map.
    fn committed_rows(&self, ids: &[i64]) -> Result<HashMap<i64, Row>> {
        let mut out = HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        let ds = self.reader()?;
        for chunk in ids.chunks(1000) {
            let list: Vec<String> = chunk.iter().map(|d| id_literal(*d)).collect();
            for (id, row, _) in self.scan(&ds, Some(format!("id IN ({})", list.join(","))), None)? {
                out.insert(id, row);
            }
        }
        Ok(out)
    }

    /// Rows with these rowids, as this transaction sees them.
    pub fn get_many(&self, ids: &[i64]) -> Result<HashMap<i64, Row>> {
        let mut out = HashMap::new();
        let mut from_lance = Vec::new();
        for &id in ids {
            match self.pending.get(&id) {
                Some(Pending { row: Some(r), complete: true, .. }) => { out.insert(id, r.clone()); }
                Some(Pending { row: None, .. }) => {}
                _ => from_lance.push(id), // not pending, or a patch to merge
            }
        }
        for (id, mut row) in self.committed_rows(&from_lance)? {
            match self.pending.get(&id) {
                None => { out.insert(id, row); }
                Some(p) => {
                    let patch = p.row.as_ref().unwrap();
                    for c in p.changed.iter().flatten() {
                        row[*c] = patch[*c].clone();
                    }
                    out.insert(id, row);
                }
            }
        }
        Ok(out)
    }

    /// Committed rows matching `filter`, optionally the `k` nearest to `q` in
    /// vector column `col`. Distances are L2 (Lance reports it squared).
    /// Does not apply the pending map; callers merge it.
    pub fn scan(&self, ds: &Dataset, filter: Option<String>, nearest: Option<(usize, &[f32], usize)>)
                -> Result<Vec<(i64, Row, Option<f64>)>> {
        let mut scan = ds.scan();
        let names: Vec<&str> = std::iter::once("id").chain(self.cols.iter().map(|c| c.0.as_str())).collect();
        scan.project(&names).map_err(lance_err)?;
        if let Some(f) = &filter {
            scan.filter(f).map_err(lance_err)?;
        }
        if let Some((col, q, k)) = nearest {
            scan.nearest(&self.cols[col].0, &Float32Array::from(q.to_vec()), k).map_err(lance_err)?;
            scan.prefilter(true);
            scan.minimum_nprobes(32).ef(k.saturating_mul(4).max(128)).refine(4);
        }
        let batch = block(scan.try_into_batch()).map_err(lance_err)?;
        self.rows_from_batch(&batch)
    }

    /// One column of every committed row, as (rowid, value). Does not apply
    /// the pending map; callers merge it.
    pub fn scan_column(&self, ds: &Dataset, col: usize) -> Result<Vec<(i64, Value)>> {
        let mut scan = ds.scan();
        scan.project(&["id", self.cols[col].0.as_str()]).map_err(lance_err)?;
        let batch = block(scan.try_into_batch()).map_err(lance_err)?;
        let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().ok_or_else(|| lance_err("id type"))?;
        let values = batch.column(1);
        Ok((0..batch.num_rows()).map(|i| (ids.value(i), value_at(values, i, self.cols[col].1))).collect())
    }

    // ----------------------------------------------------------------- Arrow

    fn batch(&self, rows: &[(i64, &Row)]) -> Result<RecordBatch> {
        let mut arrays: Vec<ArrayRef> =
            vec![Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0)))];
        for (c, (name, kind)) in self.cols.iter().enumerate() {
            let vals = rows.iter().map(|r| &r.1[c]);
            let wrong = |what: &str| Error::ModuleError(format!("lance: column {name} expects {what}"));
            let array: ArrayRef = match kind {
                Kind::Integer => Arc::new(vals.map(|v| match v {
                    Value::Null => Ok(None),
                    Value::Integer(i) => Ok(Some(*i)),
                    _ => Err(wrong("an integer")),
                }).collect::<Result<Int64Array>>()?),
                Kind::Real => Arc::new(vals.map(|v| match v {
                    Value::Null => Ok(None),
                    Value::Real(f) => Ok(Some(*f)),
                    Value::Integer(i) => Ok(Some(*i as f64)),
                    _ => Err(wrong("a real")),
                }).collect::<Result<Float64Array>>()?),
                Kind::Text => Arc::new(vals.map(|v| match v {
                    Value::Null => Ok(None),
                    Value::Text(t) => Ok(Some(t.clone())),
                    _ => Err(wrong("text")),
                }).collect::<Result<StringArray>>()?),
                Kind::Blob => Arc::new(vals.map(|v| match v {
                    Value::Null => Ok(None),
                    Value::Blob(b) => Ok(Some(b.clone())),
                    _ => Err(wrong("a blob")),
                }).collect::<Result<LargeBinaryArray>>()?),
                Kind::Vector(d) => {
                    let mut flat = Vec::with_capacity(rows.len() * d);
                    let mut valid = Vec::with_capacity(rows.len());
                    for v in vals {
                        match v {
                            Value::Null => { flat.extend(std::iter::repeat_n(0f32, *d)); valid.push(false); }
                            Value::Blob(b) if b.len() == d * 4 => {
                                flat.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                                valid.push(true);
                            }
                            _ => return Err(wrong(&format!("float32[{d}]"))),
                        }
                    }
                    let item = Arc::new(Field::new("item", DataType::Float32, true));
                    Arc::new(FixedSizeListArray::try_new(item, *d as i32, Arc::new(Float32Array::from(flat)),
                                                         Some(valid.into())).map_err(lance_err)?)
                }
            };
            arrays.push(array);
        }
        RecordBatch::try_new(self.schema.clone(), arrays).map_err(lance_err)
    }

    fn rows_from_batch(&self, b: &RecordBatch) -> Result<Vec<(i64, Row, Option<f64>)>> {
        let col = |name: &str| {
            b.column_by_name(name).ok_or_else(|| lance_err(format!("missing column {name}")))
        };
        let ids = col("id")?.as_any().downcast_ref::<Int64Array>().ok_or_else(|| lance_err("id type"))?.clone();
        let cols: Vec<ArrayRef> = self.cols.iter().map(|c| col(&c.0).cloned()).collect::<Result<_>>()?;
        let dist = b.column_by_name("_distance").and_then(|c| c.as_any().downcast_ref::<Float32Array>().cloned());
        let mut out = Vec::with_capacity(b.num_rows());
        for i in 0..b.num_rows() {
            let mut row = Vec::with_capacity(cols.len());
            for (a, (_, kind)) in cols.iter().zip(&self.cols) {
                row.push(value_at(a, i, *kind));
            }
            out.push((ids.value(i), row, dist.as_ref().map(|d| (d.value(i) as f64).max(0.0).sqrt())));
        }
        Ok(out)
    }

    // ------------------------------------------------------------ lifecycle

    /// DROP TABLE removes only SQLite metadata. Retain datasets because
    /// xDestroy can be rolled back and existing readers may still need them.
    pub fn destroy(&self) -> Result<()> {
        if self.in_write {
            return Err(lance_err("cannot drop with uncommitted Lance writes or maintenance; commit or roll back first"));
        }
        self.conn()?.execute(&format!("drop table if exists {}", qualified(&self.database, &self.shadow)), [])?;
        Ok(())
    }

    pub fn rename(&mut self, new_shadow: &str) -> Result<()> {
        self.conn()?.execute(
            &format!("alter table {} rename to {}", qualified(&self.database, &self.shadow), quote(new_shadow)),
            [],
        )?;
        self.shadow = new_shadow.to_owned();
        Ok(())
    }
}

fn value_at(a: &ArrayRef, i: usize, kind: Kind) -> Value {
    if a.is_null(i) {
        return Value::Null;
    }
    let any = a.as_any();
    match kind {
        Kind::Integer => Value::Integer(any.downcast_ref::<Int64Array>().map_or(0, |x| x.value(i))),
        Kind::Real => Value::Real(any.downcast_ref::<Float64Array>().map_or(0.0, |x| x.value(i))),
        Kind::Text => Value::Text(
            any.downcast_ref::<StringArray>().map(|x| x.value(i).to_owned())
                .or_else(|| any.downcast_ref::<LargeStringArray>().map(|x| x.value(i).to_owned()))
                .unwrap_or_default()),
        Kind::Blob => Value::Blob(
            any.downcast_ref::<LargeBinaryArray>().map(|x| x.value(i).to_vec())
                .or_else(|| any.downcast_ref::<BinaryArray>().map(|x| x.value(i).to_vec()))
                .unwrap_or_default()),
        Kind::Vector(_) => {
            let list = any.downcast_ref::<FixedSizeListArray>().map(|x| x.value(i));
            Value::Blob(list.and_then(|l| l.as_any().downcast_ref::<Float32Array>()
                .map(|f| f.values().iter().flat_map(|x| x.to_le_bytes()).collect()))
                .unwrap_or_default())
        }
    }
}

/// L2 distance between a float32 BLOB value and a query.
pub fn l2_blob(v: &Value, q: &[f32]) -> Option<f64> {
    let Value::Blob(b) = v else { return None };
    if b.len() != q.len() * 4 {
        return None;
    }
    Some(b.chunks_exact(4).zip(q)
        .map(|(c, y)| ((f32::from_le_bytes([c[0], c[1], c[2], c[3]]) - y) as f64).powi(2))
        .sum::<f64>().sqrt())
}
