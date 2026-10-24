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
use lance::dataset::{WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
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

/// Rows appended since the rowid index last covered them, before a commit
/// updates it. Until then Lance scans those rows and uses the index for the
/// rest. ponytail: fixed threshold; make it a table option if a workload
/// needs fresher indexing.
const INDEX_LAG_ROWS: i64 = 1024;
const ID_INDEX: &str = "id_idx";

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
    row: Option<Row>, // Some = written row; None = deleted
    replaces: bool,   // hide the committed row (a delete or update happened)
}

pub struct LanceStore {
    db: *mut ffi::sqlite3,
    shadow: String, // unquoted name of the version table
    uri: String,
    pub cols: Vec<(String, Kind)>,
    schema: Arc<Schema>,
    pending: HashMap<i64, Pending>,
    undo: Vec<(i64, Option<Pending>)>,
    marks: Vec<(c_int, usize)>,
    in_write: bool,
    /// The dataset opened at the committed version, reused until it changes:
    /// opening reads the manifest from storage on every call otherwise.
    opened: RefCell<Option<(u64, Dataset)>>,
}

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

impl LanceStore {
    /// Open (or, with `create`, create) the store. `shadow` is the SQLite
    /// table that records the committed Lance version.
    pub fn open(db: *mut ffi::sqlite3, shadow: &str, uri: &str, cols: Vec<(String, Kind)>,
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
            shadow: shadow.to_owned(),
            uri: uri.to_owned(),
            cols,
            schema: Arc::new(Schema::new(fields)),
            pending: HashMap::new(),
            undo: Vec::new(),
            marks: Vec::new(),
            in_write: false,
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
        let shadow = quote(&self.shadow);
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
    pub fn recorded_uri(db: *mut ffi::sqlite3, shadow: &str) -> Option<String> {
        let conn = unsafe { Connection::from_handle(db) }.ok()?;
        conn.query_row(&format!("select v from {} where k = 'uri'", quote(shadow)), [], |r| r.get(0)).ok()
    }

    fn committed_version(&self) -> Result<u64> {
        let v: i64 = self.conn()?.query_row(
            &format!("select v from {} where k = 'version'", quote(&self.shadow)),
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

    fn set(&mut self, id: i64, row: Option<Row>, replaces: bool) {
        let replaces = replaces || self.pending.get(&id).is_some_and(|p| p.replaces);
        let prev = self.pending.insert(id, Pending { row, replaces });
        if !self.marks.is_empty() {
            self.undo.push((id, prev));
        }
    }

    /// A new row (a rowid not in the store).
    pub fn insert(&mut self, id: i64, row: Row) -> Result<()> {
        self.begin_write()?;
        self.set(id, Some(row), false);
        Ok(())
    }

    /// Replace the row (or create it).
    pub fn replace(&mut self, id: i64, row: Row) -> Result<()> {
        self.begin_write()?;
        self.set(id, Some(row), true);
        Ok(())
    }

    pub fn delete(&mut self, id: i64) -> Result<()> {
        self.begin_write()?;
        self.set(id, None, true);
        Ok(())
    }

    // --------------------------------------------------------- transactions

    pub fn savepoint(&mut self, id: c_int) {
        self.marks.retain(|m| m.0 < id);
        self.marks.push((id, self.undo.len()));
    }

    pub fn rollback_to(&mut self, id: c_int) {
        let Some(&(_, len)) = self.marks.iter().find(|m| m.0 == id) else { return };
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
        let deletes: Vec<i64> = self.pending.iter().filter(|(_, p)| p.replaces).map(|(id, _)| *id).collect();
        let rows = self.pending_rows();
        let batch = if rows.is_empty() { None } else { Some(self.batch(&rows)?) };
        let uri = self.uri.clone();
        let schema = self.schema.clone();
        let conn = self.conn()?;
        let shadow = quote(&self.shadow);
        let lag: i64 = conn
            .query_row(&format!("select v from {shadow} where k = 'unindexed'"), [], |r| r.get(0))
            .unwrap_or(0)
            + rows.len() as i64;
        let reindex = lag >= INDEX_LAG_ROWS;
        let latest = block(async move {
            if !deletes.is_empty() {
                let mut ds = Dataset::open(&uri).await?;
                for chunk in deletes.chunks(1000) {
                    let ids: Vec<String> = chunk.iter().map(|d| d.to_string()).collect();
                    ds.delete(&format!("id IN ({})", ids.join(","))).await?;
                }
            }
            if let Some(batch) = batch {
                let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
                let params = WriteParams { mode: WriteMode::Append, ..Default::default() };
                Dataset::write(reader, uri.as_str(), Some(params)).await?;
            }
            let mut ds = Dataset::open(&uri).await?;
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
        self.pending.clear();
        self.undo.clear();
        self.marks.clear();
        self.in_write = false;
    }

    // ----------------------------------------------------------------- reads

    /// This transaction's rows, by rowid.
    pub fn pending_rows(&self) -> Vec<(i64, &Row)> {
        let mut rows: Vec<(i64, &Row)> =
            self.pending.iter().filter_map(|(id, p)| p.row.as_ref().map(|r| (*id, r))).collect();
        rows.sort_by_key(|r| r.0);
        rows
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
    pub fn pending_get(&self, id: i64) -> Option<Option<&Row>> {
        self.pending.get(&id).map(|p| p.row.as_ref())
    }

    /// Committed rows (as this transaction sees them) with these rowids.
    pub fn get_many(&self, ids: &[i64]) -> Result<HashMap<i64, Row>> {
        let mut out = HashMap::new();
        let mut from_lance = Vec::new();
        for &id in ids {
            match self.pending_get(id) {
                Some(Some(row)) => { out.insert(id, row.clone()); }
                Some(None) => {}
                None => from_lance.push(id),
            }
        }
        if from_lance.is_empty() {
            return Ok(out);
        }
        let ds = self.reader()?;
        for chunk in from_lance.chunks(1000) {
            let ids: Vec<String> = chunk.iter().map(|d| d.to_string()).collect();
            for (id, row, _) in self.scan(&ds, Some(format!("id IN ({})", ids.join(","))), None)? {
                if !self.hides(id) {
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

    /// DROP TABLE: remove the version table and, for a local path, the data.
    /// ponytail: remote (s3://...) datasets are left in place; deleting
    /// objects under a URI is a decision for the owner of the bucket.
    pub fn destroy(&self) -> Result<()> {
        self.conn()?.execute(&format!("drop table if exists {}", quote(&self.shadow)), [])?;
        if !self.uri.contains("://") {
            match std::fs::remove_dir_all(&self.uri) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(lance_err(e)),
            }
        }
        Ok(())
    }

    pub fn rename(&mut self, new_shadow: &str) -> Result<()> {
        self.conn()?.execute(
            &format!("alter table {} rename to {}", quote(&self.shadow), quote(new_shadow)),
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
