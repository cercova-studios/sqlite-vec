use rusqlite::{Connection, Result};
use std::ffi::{c_char, c_int, c_void};

fn open(path: &str, version: &str) -> Result<Connection> {
    let db = Connection::open(path)?;
    // 0.1.9 declares a zero-argument symbol; the C initializer has this ABI.
    let init: unsafe extern "C" fn(*mut c_void, *mut *mut c_char, *const c_void) -> c_int =
        unsafe { std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ()) };
    // SAFETY: db is live and both crates resolve SQLite calls to rusqlite's bundled library.
    assert_eq!(
        unsafe { init(db.handle().cast(), std::ptr::null_mut(), std::ptr::null()) },
        0
    );
    let actual: String = db.query_row("SELECT vec_version()", [], |r| r.get(0))?;
    assert_eq!(actual, version);
    println!("{actual}: SQLite {}", rusqlite::version());
    Ok(db)
}

fn verify(db: &Connection) -> Result<()> {
    let mut stmt = db.prepare(
        "SELECT rowid FROM t WHERE v MATCH '[0,0]' AND k=10
                              AND area=1 AND score>=10.0 AND blocked=0 ORDER BY distance",
    )?;
    let ids = stmt
        .query_map([], |r| r.get::<_, i64>(0))?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(ids, [2, 3]);
    let mut stmt = db.prepare(
        "SELECT rowid, distance FROM t WHERE v MATCH '[0,0]' AND k=10
                              ORDER BY distance",
    )?;
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)))?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(rows, [(1, 0.0), (2, 5.0), (3, 10.0), (4, 13.0)]);
    let text: String = db.query_row("SELECT label FROM t WHERE rowid=2", [], |r| r.get(0))?;
    assert_eq!(text, "two");
    let integrity: String = db.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
    assert_eq!(integrity, "ok");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let db = open(&args[2], &args[3])?;
    match args[1].as_str() {
        "create" => {
            db.execute_batch(
                "CREATE VIRTUAL TABLE t USING vec0(
                v float[2], area integer, score float, blocked boolean, +label text, chunk_size=8);
                INSERT INTO t(rowid,v,area,score,blocked,label) VALUES
                (1,'[0,0]',0,0.0,0,'one'),(2,'[3,4]',1,10.0,0,'two'),
                (3,'[6,8]',1,20.0,0,'three'),(4,'[5,12]',1,30.0,1,'four');",
            )?;
            verify(&db)?;
        }
        "read" => verify(&db)?,
        "write" => {
            verify(&db)?;
            db.execute_batch("BEGIN; SAVEPOINT s;
                UPDATE t SET area=7, score=999.0 WHERE rowid=2;
                DELETE FROM t WHERE rowid=3;
                INSERT INTO t(rowid,v,area,score,blocked,label) VALUES(5,'[9,12]',7,999.0,0,'five');")?;
            let ids: Vec<i64> = db.prepare("SELECT rowid FROM t WHERE v MATCH '[0,0]' AND k=10 AND area=7 ORDER BY distance")?
                .query_map([], |r| r.get(0))?.collect::<Result<_>>()?;
            assert_eq!(ids, [2, 5]);
            db.execute_batch(
                "ROLLBACK TO s; RELEASE s; COMMIT;
                BEGIN; UPDATE t SET score=-100.0 WHERE rowid=2; ROLLBACK;
                BEGIN; UPDATE t SET score=11.0 WHERE rowid=2;
                INSERT INTO t(rowid,v,area,score,blocked,label) VALUES(5,'[9,12]',7,999.0,0,'five');
                DELETE FROM t WHERE rowid=5; COMMIT;",
            )?;
            verify(&db)?;
        }
        "clear-zonemaps" => {
            let removed = db.execute("DELETE FROM t_info WHERE key GLOB 'zonemap_*'", [])?;
            assert!(
                removed > 0,
                "downgrade fixture must exercise persisted bounds"
            );
        }
        "mixed-chunks" => {
            db.execute_batch("BEGIN")?;
            for id in 5..=12 {
                db.execute("INSERT INTO t(rowid,v,area,score,blocked,label) VALUES(?1,?2,9,999.0,0,'new chunk')",
                           rusqlite::params![id, format!("[{},0]", 100 + id)])?;
            }
            db.execute_batch("COMMIT")?;
            let bounds: i64 = db.query_row(
                "SELECT count(*) FROM t_info WHERE key GLOB 'zonemap_*'",
                [],
                |r| r.get(0),
            )?;
            assert_eq!(bounds, 0, "legacy tables retain their unpruned scan path");
        }
        "verify-mixed" => {
            let ids = db.prepare("SELECT rowid FROM t WHERE v MATCH '[0,0]' AND k=10 AND area=9 AND score=999.0 ORDER BY distance")?
                .query_map([], |r| r.get::<_, i64>(0))?.collect::<Result<Vec<_>>>()?;
            assert_eq!(ids, (5..=12).collect::<Vec<_>>());
        }
        "legacy-write" => {
            db.execute("UPDATE t SET area=9, score=999.0 WHERE rowid=1", [])?;
        }
        "verify-legacy-write" => {
            let id: i64 = db.query_row(
                "SELECT rowid FROM t WHERE v MATCH '[0,0]' AND k=1 AND area=9 AND score=999.0",
                [],
                |r| r.get(0),
            )?;
            assert_eq!(id, 1);
        }
        "features" => {
            db.execute_batch(
                "CREATE VIRTUAL TABLE h USING vec0(v float[2] indexed by hnsw(m=4));
                INSERT INTO h(rowid,v) VALUES(1,'[0,0]'),(2,'[3,4]');",
            )?;
            let id: i64 = db.query_row(
                "SELECT rowid FROM h WHERE v MATCH '[3,4]' AND k=1 AND rowid IN (2)",
                [],
                |r| r.get(0),
            )?;
            assert_eq!(id, 2);
        }
        _ => panic!("unknown fixture operation"),
    }
    Ok(())
}
