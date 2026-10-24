//! The Lance storage provider for vec0: implements `sqlite3_vec_storage_methods`
//! (see sqlite-vec.h) on top of `LanceStore`, so a vec0 table declared with
//! `storage=lance` keeps its vectors and auxiliary columns in a Lance dataset.
//!
//! Columns are numbered as vec0 numbers them: auxiliary columns first, then
//! vectors. float32 vectors become Lance vector columns; other element types
//! are stored as fixed-size blobs.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use rusqlite::types::Value;
use rusqlite::{Connection, Error, Result, ffi};

use crate::store::{Kind, LanceStore, Row, lance_err};

type ScanCallback = unsafe extern "C" fn(*mut c_void, i64, *const c_void, c_int) -> c_int;

/// Mirrors `sqlite3_vec_storage_methods` in sqlite-vec.h, field for field.
#[repr(C)]
pub struct StorageMethods {
    i_version: c_int,
    x_open: unsafe extern "C" fn(*mut ffi::sqlite3, *const c_char, *const c_char, *const c_char, c_int,
                                 c_int, *const *const c_char, *const c_int,
                                 c_int, *const *const c_char, *const c_int, *const c_int,
                                 *mut *mut Provider, *mut *mut c_char) -> c_int,
    x_close: unsafe extern "C" fn(*mut Provider),
    x_destroy: unsafe extern "C" fn(*mut Provider) -> c_int,
    x_rename: unsafe extern "C" fn(*mut Provider, *const c_char) -> c_int,
    x_put: unsafe extern "C" fn(*mut Provider, i64, *mut *mut ffi::sqlite3_value, *const *const c_void) -> c_int,
    x_set: unsafe extern "C" fn(*mut Provider, i64, c_int, *mut ffi::sqlite3_value) -> c_int,
    x_set_vector: unsafe extern "C" fn(*mut Provider, i64, c_int, *const c_void) -> c_int,
    x_delete: unsafe extern "C" fn(*mut Provider, i64) -> c_int,
    x_column: unsafe extern "C" fn(*mut Provider, i64, c_int, *mut ffi::sqlite3_context) -> c_int,
    x_read: unsafe extern "C" fn(*mut Provider, i64, c_int, *mut *const c_void, *mut c_int) -> c_int,
    x_scan: unsafe extern "C" fn(*mut Provider, c_int, ScanCallback, *mut c_void) -> c_int,
    x_prefetch: unsafe extern "C" fn(*mut Provider, *const i64, c_int) -> c_int,
    x_sync: unsafe extern "C" fn(*mut Provider) -> c_int,
    x_commit: unsafe extern "C" fn(*mut Provider) -> c_int,
    x_rollback: unsafe extern "C" fn(*mut Provider) -> c_int,
    x_savepoint: unsafe extern "C" fn(*mut Provider, c_int) -> c_int,
    x_release: unsafe extern "C" fn(*mut Provider, c_int) -> c_int,
    x_rollback_to: unsafe extern "C" fn(*mut Provider, c_int) -> c_int,
    x_errmsg: unsafe extern "C" fn(*mut Provider) -> *const c_char,
}

const SQLITE_VEC_STORAGE_VERSION: c_int = 1;

pub static METHODS: StorageMethods = StorageMethods {
    i_version: SQLITE_VEC_STORAGE_VERSION,
    x_open, x_close, x_destroy, x_rename, x_put, x_set, x_set_vector, x_delete, x_column, x_read,
    x_scan, x_prefetch, x_sync, x_commit, x_rollback, x_savepoint, x_release, x_rollback_to, x_errmsg,
};

pub struct Provider {
    db: *mut ffi::sqlite3,
    schema: CString,
    store: LanceStore,
    n_aux: usize,
    vec_bytes: Vec<usize>,
    /// Committed rows read from Lance (None: no such row) and whole committed
    /// columns read by xScan, valid while the pager's data version is
    /// `cache_version`. ponytail: whole-column caches are unbounded; add a
    /// budget if columns outgrow memory.
    cache: HashMap<i64, Option<Row>>,
    columns: HashMap<usize, Vec<(i64, Value)>>,
    cache_version: u32,
    read_buf: Value,
    err: CString,
}

impl Provider {
    fn data_version(&self) -> u32 {
        let mut v: u32 = 0;
        let rc = unsafe {
            ffi::sqlite3_file_control(self.db, self.schema.as_ptr(), ffi::SQLITE_FCNTL_DATA_VERSION,
                                      (&mut v as *mut u32).cast::<c_void>())
        };
        if rc == ffi::SQLITE_OK { v } else { u32::MAX }
    }

    fn check_cache(&mut self) {
        let v = self.data_version();
        if v != self.cache_version || v == u32::MAX {
            self.cache.clear();
            self.columns.clear();
            self.cache_version = v;
        }
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
        self.columns.clear();
    }

    /// The row as this connection sees it.
    fn row(&mut self, id: i64) -> Result<Option<Row>> {
        if let Some(p) = self.store.pending_get(id) {
            return Ok(p.cloned());
        }
        self.check_cache();
        if let Some(r) = self.cache.get(&id) {
            return Ok(r.clone());
        }
        let row = self.store.get_many(&[id])?.remove(&id);
        self.cache.insert(id, row.clone());
        Ok(row)
    }

    fn column_index(&self, i: c_int) -> Result<usize> {
        let i = i as usize;
        if i >= self.store.cols.len() {
            return Err(lance_err(format!("column {i} out of range")));
        }
        Ok(i)
    }
}

// ------------------------------------------------------------------ helpers

unsafe fn sqlite_string(msg: &str) -> *mut c_char {
    let bytes = msg.as_bytes();
    let p = unsafe { ffi::sqlite3_malloc((bytes.len() + 1) as c_int) }.cast::<u8>();
    if !p.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
            *p.add(bytes.len()) = 0;
        }
    }
    p.cast()
}

unsafe fn cstr(p: *const c_char) -> Result<String> {
    Ok(unsafe { CStr::from_ptr(p) }.to_str()?.to_owned())
}

unsafe fn value_from(v: *mut ffi::sqlite3_value) -> Value {
    unsafe {
        match ffi::sqlite3_value_type(v) {
            ffi::SQLITE_INTEGER => Value::Integer(ffi::sqlite3_value_int64(v)),
            ffi::SQLITE_FLOAT => Value::Real(ffi::sqlite3_value_double(v)),
            ffi::SQLITE_TEXT => {
                let p = ffi::sqlite3_value_text(v);
                let n = ffi::sqlite3_value_bytes(v) as usize;
                Value::Text(String::from_utf8_lossy(std::slice::from_raw_parts(p, n)).into_owned())
            }
            ffi::SQLITE_BLOB => {
                let n = ffi::sqlite3_value_bytes(v) as usize;
                let p = ffi::sqlite3_value_blob(v).cast::<u8>();
                Value::Blob(if n == 0 { Vec::new() } else { std::slice::from_raw_parts(p, n).to_vec() })
            }
            _ => Value::Null,
        }
    }
}

unsafe fn result(ctx: *mut ffi::sqlite3_context, v: &Value) {
    unsafe {
        match v {
            Value::Null => ffi::sqlite3_result_null(ctx),
            Value::Integer(i) => ffi::sqlite3_result_int64(ctx, *i),
            Value::Real(f) => ffi::sqlite3_result_double(ctx, *f),
            Value::Text(t) => ffi::sqlite3_result_text64(ctx, t.as_ptr().cast(), t.len() as u64,
                                                          ffi::SQLITE_TRANSIENT(), ffi::SQLITE_UTF8 as u8),
            Value::Blob(b) => ffi::sqlite3_result_blob64(ctx, b.as_ptr().cast(), b.len() as u64,
                                                          ffi::SQLITE_TRANSIENT()),
        }
    }
}

fn bytes_of(v: &Value) -> Option<&[u8]> {
    match v {
        Value::Blob(b) => Some(b),
        Value::Text(t) => Some(t.as_bytes()),
        _ => None,
    }
}

/// Run `f` on the provider, mapping errors (and panics, which must not cross
/// into C) to SQLITE_ERROR with a message for xErrmsg.
unsafe fn guard(p: *mut Provider, f: impl FnOnce(&mut Provider) -> Result<()>) -> c_int {
    let p = unsafe { &mut *p };
    let out = catch_unwind(AssertUnwindSafe(|| f(p)));
    let msg = match out {
        Ok(Ok(())) => return ffi::SQLITE_OK,
        Ok(Err(e)) => e.to_string(),
        Err(_) => "internal error (panic) in the Lance storage provider".to_owned(),
    };
    p.err = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    ffi::SQLITE_ERROR
}

// ------------------------------------------------------------------ methods

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn x_open(db: *mut ffi::sqlite3, z_schema: *const c_char, z_table: *const c_char,
                            z_uri: *const c_char, b_create: c_int,
                            n_aux: c_int, az_aux: *const *const c_char, a_aux_type: *const c_int,
                            n_vec: c_int, az_vec: *const *const c_char, a_vec_bytes: *const c_int,
                            a_vec_dims: *const c_int,
                            pp_store: *mut *mut Provider, pz_err: *mut *mut c_char) -> c_int {
    let open = || -> Result<Provider> {
        let schema = unsafe { cstr(z_schema) }?;
        let table = unsafe { cstr(z_table) }?;
        let shadow = format!("{table}_storage");
        let recorded = if b_create == 0 { LanceStore::recorded_uri(db, &shadow) } else { None };
        let uri = if let Some(uri) = recorded {
            uri
        } else if z_uri.is_null() {
            // Next to the database file: <db>.<table>.lance
            let schema_c = CString::new(schema.clone()).unwrap_or_default();
            let f = unsafe { ffi::sqlite3_db_filename(db, schema_c.as_ptr()) };
            let file = if f.is_null() { String::new() } else { unsafe { cstr(f) }? };
            if file.is_empty() {
                return Err(Error::ModuleError(
                    "storage_uri is required for in-memory and temporary databases".into()));
            }
            format!("{file}.{table}.lance")
        } else {
            unsafe { cstr(z_uri) }?
        };
        let mut cols = Vec::new();
        for i in 0..n_aux as usize {
            let name = unsafe { cstr(*az_aux.add(i)) }?;
            let kind = match unsafe { *a_aux_type.add(i) } {
                ffi::SQLITE_INTEGER => Kind::Integer,
                ffi::SQLITE_FLOAT => Kind::Real,
                ffi::SQLITE_TEXT => Kind::Text,
                ffi::SQLITE_BLOB => Kind::Blob,
                t => return Err(lance_err(format!("unsupported column type {t}"))),
            };
            cols.push((name, kind));
        }
        let mut vec_bytes = Vec::new();
        for j in 0..n_vec as usize {
            let name = unsafe { cstr(*az_vec.add(j)) }?;
            let dims = unsafe { *a_vec_dims.add(j) } as usize;
            vec_bytes.push(unsafe { *a_vec_bytes.add(j) } as usize);
            cols.push((name, if dims > 0 { Kind::Vector(dims) } else { Kind::Blob }));
        }
        let store = LanceStore::open(db, &shadow, &uri, cols, b_create != 0)?;
        Ok(Provider { db, schema: CString::new(schema).unwrap_or_default(), store, n_aux: n_aux as usize,
                      vec_bytes, cache: HashMap::new(), columns: HashMap::new(), cache_version: u32::MAX,
                      read_buf: Value::Null, err: CString::default() })
    };
    match catch_unwind(AssertUnwindSafe(open)) {
        Ok(Ok(p)) => {
            unsafe { *pp_store = Box::into_raw(Box::new(p)) };
            ffi::SQLITE_OK
        }
        Ok(Err(e)) => {
            unsafe { *pz_err = sqlite_string(&format!("vec0 storage=lance: {e}")) };
            ffi::SQLITE_ERROR
        }
        Err(_) => {
            unsafe { *pz_err = sqlite_string("vec0 storage=lance: internal error (panic)") };
            ffi::SQLITE_ERROR
        }
    }
}

unsafe extern "C" fn x_close(p: *mut Provider) {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p) });
    }
}

unsafe extern "C" fn x_destroy(p: *mut Provider) -> c_int {
    unsafe { guard(p, |p| p.store.destroy()) }
}

unsafe extern "C" fn x_rename(p: *mut Provider, z_new: *const c_char) -> c_int {
    unsafe { guard(p, |p| { let new = cstr(z_new)?; p.store.rename(&format!("{new}_storage")) }) }
}

unsafe extern "C" fn x_put(p: *mut Provider, rowid: i64, ap_aux: *mut *mut ffi::sqlite3_value,
                           ap_vec: *const *const c_void) -> c_int {
    unsafe {
        guard(p, |p| {
            let mut row: Row = (0..p.n_aux).map(|i| value_from(*ap_aux.add(i))).collect();
            for (j, &n) in p.vec_bytes.iter().enumerate() {
                let v = *ap_vec.add(j);
                row.push(if v.is_null() { Value::Null }
                         else { Value::Blob(std::slice::from_raw_parts(v.cast::<u8>(), n).to_vec()) });
            }
            p.store.insert(rowid, row)
        })
    }
}

fn set_column(p: &mut Provider, rowid: i64, i: usize, v: Value) -> Result<()> {
    let n = p.store.cols.len();
    let mut row = p.row(rowid)?.unwrap_or_else(|| vec![Value::Null; n]);
    row[i] = v;
    p.store.replace(rowid, row)
}

unsafe extern "C" fn x_set(p: *mut Provider, rowid: i64, i_col: c_int, val: *mut ffi::sqlite3_value) -> c_int {
    unsafe {
        guard(p, |p| {
            let i = p.column_index(i_col)?;
            set_column(p, rowid, i, value_from(val))
        })
    }
}

unsafe extern "C" fn x_set_vector(p: *mut Provider, rowid: i64, i_vec: c_int, v: *const c_void) -> c_int {
    unsafe {
        guard(p, |p| {
            let j = i_vec as usize;
            let n = *p.vec_bytes.get(j).ok_or_else(|| lance_err(format!("vector {i_vec} out of range")))?;
            let blob = Value::Blob(std::slice::from_raw_parts(v.cast::<u8>(), n).to_vec());
            let i = p.n_aux + j;
            set_column(p, rowid, i, blob)
        })
    }
}

unsafe extern "C" fn x_delete(p: *mut Provider, rowid: i64) -> c_int {
    unsafe { guard(p, |p| p.store.delete(rowid)) }
}

unsafe extern "C" fn x_column(p: *mut Provider, rowid: i64, i_col: c_int, ctx: *mut ffi::sqlite3_context) -> c_int {
    unsafe {
        guard(p, |p| {
            let i = p.column_index(i_col)?;
            let row = p.row(rowid)?;
            result(ctx, row.as_ref().map_or(&Value::Null, |r| &r[i]));
            Ok(())
        })
    }
}

unsafe extern "C" fn x_read(p: *mut Provider, rowid: i64, i_col: c_int, pp_data: *mut *const c_void,
                            pn_data: *mut c_int) -> c_int {
    unsafe {
        guard(p, |p| {
            let i = p.column_index(i_col)?;
            p.read_buf = p.row(rowid)?.map_or(Value::Null, |mut r| std::mem::replace(&mut r[i], Value::Null));
            match bytes_of(&p.read_buf) {
                Some(b) => { *pp_data = b.as_ptr().cast(); *pn_data = b.len() as c_int; }
                None => { *pp_data = std::ptr::null(); *pn_data = 0; }
            }
            Ok(())
        })
    }
}

unsafe extern "C" fn x_scan(p: *mut Provider, i_col: c_int, cb: ScanCallback, arg: *mut c_void) -> c_int {
    let mut cb_rc = ffi::SQLITE_OK;
    let rc = unsafe {
        guard(p, |p| {
            let i = p.column_index(i_col)?;
            p.check_cache();
            if !p.columns.contains_key(&i) {
                let ds = p.store.reader()?;
                let col = p.store.scan_column(&ds, i)?;
                p.columns.insert(i, col);
            }
            // committed rows this transaction did not touch, then its own rows
            for (id, v) in &p.columns[&i] {
                if p.store.pending_get(*id).is_some() {
                    continue;
                }
                if let Some(b) = bytes_of(v) {
                    cb_rc = cb(arg, *id, b.as_ptr().cast(), b.len() as c_int);
                    if cb_rc != ffi::SQLITE_OK {
                        return Ok(());
                    }
                }
            }
            for (id, row) in p.store.pending_rows() {
                if let Some(b) = bytes_of(&row[i]) {
                    cb_rc = cb(arg, id, b.as_ptr().cast(), b.len() as c_int);
                    if cb_rc != ffi::SQLITE_OK {
                        return Ok(());
                    }
                }
            }
            Ok(())
        })
    };
    if rc != ffi::SQLITE_OK { rc } else { cb_rc }
}

unsafe extern "C" fn x_prefetch(p: *mut Provider, a_rowid: *const i64, n: c_int) -> c_int {
    unsafe {
        guard(p, |p| {
            p.check_cache();
            let ids: Vec<i64> = std::slice::from_raw_parts(a_rowid, n.max(0) as usize).iter().copied()
                .filter(|id| p.store.pending_get(*id).is_none() && !p.cache.contains_key(id))
                .collect();
            if ids.is_empty() {
                return Ok(());
            }
            let mut got = p.store.get_many(&ids)?;
            for id in ids {
                let row = got.remove(&id);
                p.cache.insert(id, row);
            }
            Ok(())
        })
    }
}

unsafe extern "C" fn x_sync(p: *mut Provider) -> c_int {
    unsafe { guard(p, |p| p.store.sync()) }
}

unsafe extern "C" fn x_commit(p: *mut Provider) -> c_int {
    unsafe {
        guard(p, |p| {
            p.store.end_transaction();
            p.clear_cache(); // the committed state just changed
            Ok(())
        })
    }
}

unsafe extern "C" fn x_rollback(p: *mut Provider) -> c_int {
    unsafe { guard(p, |p| { p.store.end_transaction(); Ok(()) }) }
}

unsafe extern "C" fn x_savepoint(p: *mut Provider, i: c_int) -> c_int {
    unsafe { guard(p, |p| { p.store.savepoint(i); Ok(()) }) }
}

unsafe extern "C" fn x_release(p: *mut Provider, i: c_int) -> c_int {
    unsafe { guard(p, |p| { p.store.release(i); Ok(()) }) }
}

unsafe extern "C" fn x_rollback_to(p: *mut Provider, i: c_int) -> c_int {
    unsafe { guard(p, |p| { p.store.rollback_to(i); Ok(()) }) }
}

unsafe extern "C" fn x_errmsg(p: *mut Provider) -> *const c_char {
    if p.is_null() { std::ptr::null() } else { unsafe { (*p).err.as_ptr() } }
}

/// Make storage=lance available to vec0 tables on this connection.
pub fn register(db: &Connection) -> Result<()> {
    let rc = unsafe {
        ffi::sqlite3_set_clientdata(db.handle(), c"sqlite-vec-storage:lance".as_ptr(),
                                    (&METHODS as *const StorageMethods).cast_mut().cast(), None)
    };
    if rc != ffi::SQLITE_OK {
        return Err(Error::SqliteFailure(ffi::Error::new(rc), None));
    }
    Ok(())
}
