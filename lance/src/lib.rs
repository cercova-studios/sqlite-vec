//! The sqlite-vec hybrid extension: the sqlite-vec C core (vec0, vec_*
//! functions) plus Lance, in one loadable library.
//!
//! - `store`: `LanceStore`, rows in a Lance dataset that follow SQLite
//!   transactions. The one place the commit and rollback rules live.
//! - `lance0`: a virtual table over a Lance dataset (vectors + blobs).
//! - `provider`: `storage=lance` for vec0 tables, through the C storage
//!   interface in sqlite-vec.h.

use std::ffi::{c_char, c_int};

use rusqlite::{Connection, Result, ffi};

mod lance0;
mod provider;
mod store;

unsafe extern "C" {
    // The sqlite-vec C core, compiled in by build.rs.
    fn sqlite3_vec_init(
        db: *mut ffi::sqlite3,
        pz_err_msg: *mut *mut c_char,
        p_api: *const ffi::sqlite3_api_routines,
    ) -> c_int;
}

fn init(db: Connection) -> Result<bool> {
    lance0::register(&db)?;
    provider::register(&db)?;
    Ok(false)
}

/// # Safety
/// Called by SQLite when the extension is loaded. Registers sqlite-vec first,
/// so Lance features load only on top of a working vec0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sqlite3_extension_init(
    db: *mut ffi::sqlite3,
    pz_err_msg: *mut *mut c_char,
    p_api: *mut ffi::sqlite3_api_routines,
) -> c_int {
    let rc = unsafe { sqlite3_vec_init(db, pz_err_msg, p_api) };
    if rc != ffi::SQLITE_OK {
        return rc;
    }
    unsafe { Connection::extension_init2(db, pz_err_msg, p_api, init) }
}
