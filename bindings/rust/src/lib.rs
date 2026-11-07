#[link(name = "sqlite_vec0")]
extern "C" {
    pub fn sqlite3_vec_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    use rusqlite::{ffi::sqlite3_auto_extension, Connection};

    #[test]
    fn test_rusqlite_auto_extension() {
        unsafe {
            assert_eq!(
                sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ()))),
                rusqlite::ffi::SQLITE_OK
            );
        }

        let conn = Connection::open_in_memory().unwrap();

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
}
