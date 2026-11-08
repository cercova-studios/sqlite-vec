use std::ffi::{c_char, c_int, c_void};

#[link(name = "sqlite_vec0")]
extern "C" {
    /// Register on a live connection from the same SQLite library linked to this crate.
    /// Pass null for `pz_err_msg` and `p_api` when registering statically.
    pub fn sqlite3_vec_init(
        db: *mut c_void,
        pz_err_msg: *mut *mut c_char,
        p_api: *const c_void,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    use rusqlite::{ffi::sqlite3_auto_extension, Connection};

    #[test]
    fn test_rusqlite_registration() {
        let conn = Connection::open_in_memory().unwrap();
        unsafe {
            assert_eq!(
                sqlite3_vec_init(conn.handle().cast(), std::ptr::null_mut(), std::ptr::null()),
                rusqlite::ffi::SQLITE_OK
            );
        }

        let result: String = conn
            .query_row("select vec_version()", [], |x| x.get(0))
            .unwrap();

        assert!(result.starts_with("v"));

        conn.execute_batch(
            "CREATE VIRTUAL TABLE t USING vec0(v float[2] indexed by hnsw(m=4));
             INSERT INTO t(rowid,v) VALUES(1,'[0,0]'),(2,'[1,0]');
             CREATE TABLE e(id,src,dst); INSERT INTO e VALUES(7,1,2);",
        )
        .unwrap();
        let nearest: i64 = conn
            .query_row(
                "SELECT rowid FROM t WHERE v MATCH '[1,0]' AND k=1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(nearest, 2);
        let witness: String = conn
            .query_row(
                "SELECT graph_witness('e','src','dst','id',1,2,1000,65536)",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(witness, "[[1,null],[2,7]]");
    }

    #[test]
    fn test_rusqlite_auto_extension() {
        let rc = unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())))
        };
        // Apple's SQLite rejects process-global registration; per-connection works above.
        if rc == rusqlite::ffi::SQLITE_MISUSE {
            assert!(cfg!(target_vendor = "apple"));
            assert_ne!(
                unsafe {
                    rusqlite::ffi::sqlite3_compileoption_used(
                        b"OMIT_LOAD_EXTENSION\0".as_ptr().cast(),
                    )
                },
                0
            );
        } else {
            assert_eq!(rc, rusqlite::ffi::SQLITE_OK);
            let conn = Connection::open_in_memory().unwrap();
            let version: String = conn
                .query_row("SELECT vec_version()", [], |row| row.get(0))
                .unwrap();
            assert!(version.starts_with('v'));
        }
    }
}
