//! `lance0`: a virtual table over a Lance dataset (vectors + blobs), with the
//! transaction rules of `LanceStore`.
//!
//! ```sql
//! create virtual table emb using lance0('/path/emb.lance', 4);
//! insert into emb(rowid, label, vector, payload) values (1, 'cat', :f32_blob, :bytes);
//! select rowid, distance from emb where query = :f32_blob and k = 5;
//! select rowid from emb where query = :q and k = 5 and label = 'cat';  -- prefilter in Lance
//! ```
//!
//! Vectors use sqlite-vec's formats: a float32 BLOB or JSON text. Distances
//! are L2, as vec0 reports them (not squared).

use std::borrow::Cow;
use std::ffi::{CStr, CString, c_int};
use std::marker::PhantomData;
use std::sync::OnceLock;

use rusqlite::types::{Value, ValueRef};
use rusqlite::vtab::{
    Context, CreateVTab, Filters, IndexConstraintOp, IndexInfo, Inserts, Module, TransactionVTab,
    UpdateVTab, Updates, VTab, VTabConnection, VTabCursor, VTabKind,
};
use rusqlite::{Connection, Error, Result, ffi};

use crate::store::{Kind, LanceStore, Row, l2_blob};

const COL_LABEL: c_int = 0;
const COL_VECTOR: c_int = 1;
const COL_PAYLOAD: c_int = 2;
const COL_DISTANCE: c_int = 3;
const COL_QUERY: c_int = 4;
const COL_K: c_int = 5;
const COL_ROWID: c_int = -1;
// LanceStore column order
const S_LABEL: usize = 0;
const S_VECTOR: usize = 1;

#[repr(C)]
struct LanceTab {
    /// Base class. Must be first.
    base: ffi::sqlite3_vtab,
    dim: usize,
    store: LanceStore,
}

/// A vector in sqlite-vec's formats: little-endian float32 BLOB or JSON array.
pub fn vector_arg(v: ValueRef<'_>, dim: usize) -> Result<Vec<f32>> {
    let wrong = || Error::ModuleError(format!("lance0: vector must be float32[{dim}]"));
    let out: Vec<f32> = match v {
        ValueRef::Blob(b) => {
            if b.len() != dim * 4 {
                return Err(wrong());
            }
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        ValueRef::Text(t) => {
            let t = std::str::from_utf8(t)?.trim();
            let inner = t.strip_prefix('[').and_then(|t| t.strip_suffix(']'))
                .ok_or_else(|| Error::ModuleError("lance0: JSON vector must be an array".into()))?;
            if inner.trim().is_empty() {
                Vec::new()
            } else {
                inner.split(',').map(|x| x.trim().parse::<f32>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|_| Error::ModuleError("lance0: JSON vector must hold numbers".into()))?
            }
        }
        _ => return Err(Error::ModuleError("lance0: vector must be a BLOB or JSON text".into())),
    };
    if out.len() != dim {
        return Err(wrong());
    }
    Ok(out)
}

fn f32_blob(v: &[f32]) -> Value {
    Value::Blob(v.iter().flat_map(|x| x.to_le_bytes()).collect())
}

impl LanceTab {
    fn connect_create(db: &mut VTabConnection, table_name: &[u8], args: &[&[u8]])
                      -> Result<(Cow<'static, CStr>, Self)> {
        // Our arguments are the last two: uri, dim.
        if args.len() < 2 {
            return Err(Error::ModuleError(
                "usage: create virtual table t using lance0('/path/to.lance', <dim>)".into()));
        }
        let arg = |i: usize| -> Result<String> {
            Ok(std::str::from_utf8(args[i])?.trim().trim_matches(|c| c == '\'' || c == '"').to_owned())
        };
        let uri = arg(args.len() - 2)?;
        let dim: usize = arg(args.len() - 1)?.parse().ok().filter(|d| *d > 0)
            .ok_or_else(|| Error::ModuleError("lance0: dim must be a positive integer".into()))?;
        let name = std::str::from_utf8(table_name)?;
        let cols = vec![
            ("label".to_owned(), Kind::Text),
            ("vector".to_owned(), Kind::Vector(dim)),
            ("payload".to_owned(), Kind::Blob),
        ];
        let store = LanceStore::open(unsafe { db.handle() }, &format!("{name}_lance"), &uri, cols, false)?;
        let schema = CString::new(
            "create table x(label text, vector blob, payload blob, distance real, query hidden, k hidden)",
        ).unwrap();
        Ok((Cow::Owned(schema), LanceTab { base: ffi::sqlite3_vtab::default(), dim, store }))
    }

    /// Column c (label, vector, payload) as stored.
    fn value(&self, c: usize, v: ValueRef<'_>) -> Result<Value> {
        Ok(match (c, v) {
            (_, ValueRef::Null) if c != S_VECTOR => Value::Null,
            (0, ValueRef::Text(t)) => Value::Text(std::str::from_utf8(t)?.to_owned()),
            (0, _) => return Err(Error::ModuleError("lance0: label must be text".into())),
            (1, v) => f32_blob(&vector_arg(v, self.dim)?),
            (_, ValueRef::Blob(b)) => Value::Blob(b.to_vec()),
            _ => return Err(Error::ModuleError("lance0: payload must be a blob".into())),
        })
    }

    fn row(&self, args: &[ValueRef<'_>]) -> Result<Row> {
        // args: label, vector, payload (from the 3rd xUpdate argument on)
        (0..3).map(|c| self.value(c, args[c])).collect()
    }
}

unsafe impl<'vtab> VTab<'vtab> for LanceTab {
    type Aux = ();
    type Cursor = LanceCursor<'vtab>;

    fn connect(db: &mut VTabConnection, _aux: Option<&()>, _module_name: &[u8], _database_name: &[u8],
               table_name: &[u8], args: &[&[u8]]) -> Result<(Cow<'static, CStr>, Self)> {
        Self::connect_create(db, table_name, args)
    }

    /// The SQL-to-Lance translation: SQLite hands us the constraints it can
    /// delegate, and we record which ones Lance will apply.
    fn best_index(&self, info: &mut IndexInfo) -> Result<bool> {
        let mut cols = Vec::new();
        let mut usages = Vec::new();
        for (i, c) in info.constraints().enumerate() {
            let pushable = c.is_usable()
                && c.operator() == IndexConstraintOp::SQLITE_INDEX_CONSTRAINT_EQ
                && matches!(c.column(), COL_LABEL | COL_QUERY | COL_K | COL_ROWID);
            if pushable && !cols.contains(&c.column()) {
                cols.push(c.column());
                usages.push((i, cols.len() as c_int));
            }
        }
        for (i, argv) in usages {
            let mut u = info.constraint_usage(i);
            u.set_argv_index(argv);
            u.set_omit(true);
        }
        let knn = cols.contains(&COL_QUERY) && cols.contains(&COL_K);
        let by_distance = info.num_of_order_by() == 1
            && info.order_bys().next().is_some_and(|o| o.column() == COL_DISTANCE && !o.is_order_by_desc());
        info.set_order_by_consumed(knn && by_distance);
        info.set_estimated_cost(if cols.contains(&COL_ROWID) { 1.0 } else if knn { 10.0 } else { 1e6 });
        let idx: Vec<String> = cols.iter().map(|c| c.to_string()).collect();
        info.set_idx_str(&idx.join(","));
        Ok(true)
    }

    fn open(&'vtab mut self) -> Result<LanceCursor<'vtab>> {
        Ok(LanceCursor { base: ffi::sqlite3_vtab_cursor::default(), rows: Vec::new(), i: 0, phantom: PhantomData })
    }
}

impl CreateVTab<'_> for LanceTab {
    const KIND: VTabKind = VTabKind::Default;
}

impl UpdateVTab<'_> for LanceTab {
    fn delete(&mut self, arg: ValueRef<'_>) -> Result<()> {
        self.store.delete(arg.as_i64()?)
    }

    fn insert(&mut self, args: &Inserts<'_>) -> Result<i64> {
        // args: [old rowid (NULL), new rowid, label, vector, payload, distance, query, k]
        let rowid: Option<i64> = args.get(1)?;
        let rowid = rowid.ok_or_else(|| Error::ModuleError("lance0: insert needs an explicit rowid".into()))?;
        let vals: Vec<ValueRef<'_>> = args.iter().skip(2).take(3).collect();
        let row = self.row(&vals)?;
        self.store.insert(rowid, row)?;
        Ok(rowid)
    }

    fn update(&mut self, args: &Updates<'_>) -> Result<()> {
        let old: i64 = args.get(0)?;
        let new: i64 = args.get(1)?;
        // Columns the UPDATE did not set arrive as "no change" (see column());
        // patch only the others, so a label change never rewrites a vector.
        let unchanged: Vec<bool> = (2..5).map(|i| args.no_change(i)).collect();
        if old == new && unchanged.iter().any(|u| *u) {
            let vals: Vec<ValueRef<'_>> = args.iter().skip(2).take(3).collect();
            let mut updates = Vec::new();
            for (c, v) in vals.iter().enumerate() {
                if !unchanged[c] {
                    updates.push((c, self.value(c, *v)?));
                }
            }
            return if updates.is_empty() { Ok(()) } else { self.store.update_columns(new, updates) };
        }
        let vals: Vec<ValueRef<'_>> = args.iter().skip(2).take(3).collect();
        let row = self.row(&vals)?;
        if old != new {
            self.store.delete(old)?;
        }
        self.store.replace(new, row)
    }
}

impl TransactionVTab<'_> for LanceTab {
    fn sync(&mut self) -> Result<()> {
        self.store.sync()
    }

    fn commit(&mut self) -> Result<()> {
        self.store.end_transaction();
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        // Lance may already be ahead; readers ignore it and the next writer
        // restores it.
        self.store.end_transaction();
        Ok(())
    }
}

struct OutRow {
    id: i64,
    row: Row,
    distance: Option<f64>,
}

#[repr(C)]
struct LanceCursor<'vtab> {
    /// Base class. Must be first.
    base: ffi::sqlite3_vtab_cursor,
    rows: Vec<OutRow>,
    i: usize,
    phantom: PhantomData<&'vtab LanceTab>,
}

impl LanceCursor<'_> {
    fn vtab(&self) -> &LanceTab {
        unsafe { &*(self.base.pVtab as *const LanceTab) }
    }
}

unsafe impl VTabCursor for LanceCursor<'_> {
    fn filter(&mut self, _idx_num: c_int, idx_str: Option<&str>, args: &Filters<'_>) -> Result<()> {
        let cols: Vec<c_int> = idx_str.unwrap_or("").split(',').filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap()).collect();
        let tab = self.vtab();
        let mut filters = Vec::new();
        let mut label: Option<Option<String>> = None; // Some(None): `label = NULL`
        let mut rowid: Option<i64> = None;
        let mut query: Option<Vec<f32>> = None;
        let mut k: Option<i64> = None;
        for (i, c) in cols.iter().enumerate() {
            let v = args.iter().nth(i).unwrap();
            match *c {
                COL_LABEL => match v.as_str_or_null()? {
                    Some(s) => {
                        filters.push(format!("label = '{}'", s.replace('\'', "''")));
                        label = Some(Some(s.to_owned()));
                    }
                    None => {
                        filters.push("false".into()); // `= NULL` matches nothing
                        label = Some(None);
                    }
                },
                COL_ROWID => {
                    let id = v.as_i64()?;
                    filters.push(format!("id = {id}"));
                    rowid = Some(id);
                }
                COL_QUERY => query = Some(vector_arg(v, tab.dim)?),
                COL_K => k = Some(v.as_i64()?),
                _ => {}
            }
        }
        let knn = match (&query, k) {
            (Some(_), Some(k)) if k < 1 => return Err(Error::ModuleError("lance0: k must be at least 1".into())),
            (Some(q), Some(k)) => Some((q.as_slice(), k as usize)),
            _ => None,
        };
        let store = &tab.store;
        // Rows this transaction replaced are hidden; ask Lance for enough
        // extra neighbors that hiding them still leaves k.
        let ds = store.reader()?;
        let nearest = knn.map(|(q, k)| (S_VECTOR, q, k + store.hidden_count()));
        let filter = (!filters.is_empty()).then(|| filters.join(" AND "));
        let mut rows: Vec<OutRow> = store.scan(&ds, filter, nearest)?
            .into_iter()
            .filter(|(id, _, _)| !store.hides(*id))
            .map(|(id, row, distance)| OutRow { id, row, distance })
            .collect();
        for (id, row) in store.pending_rows_for(rowid)? {
            let label_ok = label.as_ref().is_none_or(|l| match (&row[S_LABEL], l) {
                (Value::Text(t), Some(l)) => t == l,
                _ => false,
            });
            if rowid.is_some_and(|r| r != id) || !label_ok {
                continue;
            }
            let distance = knn.and_then(|(q, _)| l2_blob(&row[S_VECTOR], q));
            rows.push(OutRow { id, row, distance });
        }
        if let Some((_, k)) = knn {
            rows.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            rows.truncate(k);
        }
        self.rows = rows;
        self.i = 0;
        Ok(())
    }

    fn next(&mut self) -> Result<()> {
        self.i += 1;
        Ok(())
    }

    fn eof(&self) -> bool {
        self.i >= self.rows.len()
    }

    fn column(&self, ctx: &mut Context, i: c_int) -> Result<()> {
        // During an UPDATE, SQLite asks whether it may skip unchanged columns.
        if ctx.no_change() && matches!(i, COL_LABEL | COL_VECTOR | COL_PAYLOAD) {
            return Ok(());
        }
        let r = &self.rows[self.i];
        match i {
            COL_LABEL | COL_VECTOR | COL_PAYLOAD => ctx.set_result(&r.row[i as usize]),
            COL_DISTANCE => ctx.set_result(&r.distance),
            _ => ctx.set_result(&None::<i64>),
        }
    }

    fn rowid(&self) -> Result<i64> {
        Ok(self.rows[self.i].id)
    }
}

unsafe extern "C" fn x_savepoint(v: *mut ffi::sqlite3_vtab, id: c_int) -> c_int {
    unsafe { (*(v as *mut LanceTab)).store.savepoint(id) };
    ffi::SQLITE_OK
}

unsafe extern "C" fn x_release(v: *mut ffi::sqlite3_vtab, id: c_int) -> c_int {
    unsafe { (*(v as *mut LanceTab)).store.release(id) };
    ffi::SQLITE_OK
}

unsafe extern "C" fn x_rollback_to(v: *mut ffi::sqlite3_vtab, id: c_int) -> c_int {
    unsafe { (*(v as *mut LanceTab)).store.rollback_to(id) };
    ffi::SQLITE_OK
}

/// rusqlite's module plus savepoint methods, which rusqlite does not expose.
/// `Module` is `#[repr(transparent)]` over `sqlite3_module`.
struct SyncModule(ffi::sqlite3_module);
unsafe impl Sync for SyncModule {}
unsafe impl Send for SyncModule {}

fn module() -> &'static ffi::sqlite3_module {
    static M: OnceLock<SyncModule> = OnceLock::new();
    &M.get_or_init(|| {
        const BASE: Module<LanceTab> = Module::update_module_with_tx();
        let mut m: ffi::sqlite3_module = unsafe { std::mem::transmute_copy(&BASE) };
        m.iVersion = m.iVersion.max(2);
        m.xSavepoint = Some(x_savepoint);
        m.xRelease = Some(x_release);
        m.xRollbackTo = Some(x_rollback_to);
        SyncModule(m)
    })
    .0
}

pub fn register(db: &Connection) -> Result<()> {
    let rc = unsafe {
        ffi::sqlite3_create_module_v2(db.handle(), c"lance0".as_ptr(), module(), std::ptr::null_mut(), None)
    };
    if rc != ffi::SQLITE_OK {
        return Err(Error::SqliteFailure(ffi::Error::new(rc), None));
    }
    Ok(())
}
