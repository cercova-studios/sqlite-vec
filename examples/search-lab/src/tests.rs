use super::*;
use rusqlite::trace::{TraceEvent, TraceEventCodes};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture {
    directory: PathBuf,
    database: PathBuf,
    config: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "sqlite-vec-rust-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let database = directory.join("fixture.db");
        let config = directory.join("config.json");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let extension = repo
            .join(if cfg!(target_os = "macos") {
                "dist/vec0.dylib"
            } else {
                "dist/vec0.so"
            })
            .canonicalize()
            .unwrap();
        let db = Connection::open(&database).unwrap();
        // SAFETY: fixed repository build artifact, never a tool request.
        unsafe {
            let _guard = LoadExtensionGuard::new(&db).unwrap();
            db.load_extension(&extension, None::<&str>).unwrap();
        }
        db.execute_batch("create table documents(id integer primary key,document_id text,tenant text,valid_from integer,valid_to integer,invalidated_at integer,source_id text,body text,start_ms integer,end_ms integer);
            create virtual table vectors using vec0(embedding float[2]);create virtual table lexical using fts5(body);
            create table edges(src,dst);create view admitted_edges as select src,dst from edges;
            insert into documents values (1,'doc:1','ours',0,null,null,'source:1','old evidence',0,100);
            insert into documents values (2,'doc:2','ours',0,null,null,'source:2','second evidence',100,200);
            insert into documents values (3,'doc:3','ours',0,null,null,'source:3','third evidence',200,300);
            insert into edges values (1,2);").unwrap();
        std::fs::write(
            &config,
            json!({"database":database,"extension":extension,"corpora":{"demo":{
            "documents":"documents","vectors":"vectors","fts":"lexical","edges":"admitted_edges",
            "tenant":"ours","as_of":100,"dimensions":2}}})
            .to_string(),
        )
        .unwrap();
        Self {
            directory,
            database,
            config,
        }
    }
    fn adapter(&self) -> Adapter {
        Adapter::open(&self.config).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.directory).unwrap();
    }
}
fn request(op: &str, args: Value) -> Value {
    json!({"op":op,"corpus":"demo","args":args})
}

#[test]
fn read_only_defenses_and_metadata() {
    let fixture = Fixture::new();
    let mut a = fixture.adapter();
    for sql in [
        "delete from documents",
        "create temp table x(a)",
        "pragma query_only=off",
        "attach ':memory:' as other",
        "select load_extension('ignored')",
    ] {
        assert!(
            a.db.execute_batch(sql).is_err(),
            "write boundary admitted {sql}"
        );
    }
    let result = a.call(&request("describe", json!({})));
    assert_eq!(result["sqlite_version"], rusqlite::version());
    assert_eq!(result["ok"], true);
    a.db.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert_eq!(
        a.db.query_row("pragma query_only", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    // TEMP would be writable on a read-only file handle without query_only.
    assert!(
        a.db.execute_batch("create temp table forbidden(a)")
            .is_err()
    );
    // The native loader is independently disabled after startup.
    let extension: String =
        a.db.query_row("select 'ignored'", [], |r| r.get(0))
            .unwrap();
    let err = unsafe { a.db.load_extension(&extension, None::<&str>) }.unwrap_err();
    assert!(err.to_string().contains("not authorized"));
}

#[test]
fn read_only_file_flag_survives_removing_other_layers() {
    let fixture = Fixture::new();
    let mut a = fixture.adapter();
    a.db.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    a.db.execute_batch("pragma query_only=off").unwrap();
    assert!(
        a.db.execute_batch("delete from documents")
            .unwrap_err()
            .to_string()
            .contains("readonly")
    );
    // With the other defenses removed, TEMP writes work; main file writes do not.
    a.db.execute_batch("create temp table allowed_in_this_test(a)")
        .unwrap();
}

#[test]
fn failed_operation_releases_transaction() {
    let fixture = Fixture::new();
    let mut a = fixture.adapter();
    a.corpora[0].1.documents = "absent".into();
    assert_eq!(
        a.call(&request("fetch", json!({"ids":[1]})))["error"]["code"],
        "database_error"
    );
    assert!(a.db.is_autocommit());
    a.corpora[0].1.documents = "documents".into();
    assert_eq!(
        a.call(&request("fetch", json!({"ids":[1]})))["rows"][0]["body"],
        "old evidence"
    );
    assert!(a.db.is_autocommit());
}

#[test]
fn graph_cache_reuse_and_external_commit() {
    let fixture = Fixture::new();
    let mut a = fixture.adapter();
    let count = Arc::new(AtomicUsize::new(0));
    let observed = count.clone();
    a.db.trace_v2(
        TraceEventCodes::SQLITE_TRACE_STMT,
        Some(move |event: TraceEvent<'_>| {
            if let TraceEvent::Stmt(_, sql) = event
                && sql == "-- SELECT \"src\", \"dst\" FROM \"admitted_edges\""
            {
                observed.fetch_add(1, Ordering::Relaxed);
            }
        }),
    )
    .unwrap();
    let r = request("traverse", json!({"ids":[1],"depth":2}));
    assert_eq!(
        a.call(&r)["rows"],
        json!([{"id":1,"depth":0},{"id":2,"depth":1}])
    );
    assert_eq!(a.call(&r)["ok"], true);
    assert_eq!(count.load(Ordering::Relaxed), 1);
    Connection::open(&fixture.database)
        .unwrap()
        .execute_batch("insert into edges values (2,3)")
        .unwrap();
    assert_eq!(a.call(&r)["rows"][2], json!({"id":3,"depth":2}));
    assert_eq!(count.load(Ordering::Relaxed), 2);
}

#[test]
fn operation_snapshot_during_external_commit() {
    let fixture = Fixture::new();
    let writer = Connection::open(&fixture.database).unwrap();
    writer.execute_batch("pragma journal_mode=wal").unwrap();
    let mut a = fixture.adapter();
    let committed = Arc::new(AtomicBool::new(false));
    let flag = committed.clone();
    a.db.trace_v2(
        TraceEventCodes::SQLITE_TRACE_STMT,
        Some(move |event: TraceEvent<'_>| {
            if let TraceEvent::Stmt(_, sql) = event
                && sql.starts_with("select id,document_id,body,")
                && !flag.load(Ordering::Relaxed)
            {
                writer
                    .execute_batch("update documents set body='new evidence' where id=1")
                    .unwrap();
                flag.store(true, Ordering::Relaxed);
            }
        }),
    )
    .unwrap();
    let rec = request("fetch", json!({"ids":[1]}));
    assert_eq!(a.call(&rec)["rows"][0]["body"], "old evidence");
    assert!(committed.load(Ordering::Relaxed));
    assert_eq!(a.call(&rec)["rows"][0]["body"], "new evidence");
}

#[test]
fn database_growth_rechecked_before_operation() {
    let fixture = Fixture::new();
    let a = fixture.adapter();
    let writer = Connection::open(&fixture.database).unwrap();
    writer
        .execute_batch(
            "create table growth(payload);insert into growth values (zeroblob(67108864))",
        )
        .unwrap();
    assert_eq!(
        a.call(&request("describe", json!({})))["error"]["code"],
        "workload_limit"
    );
    assert_eq!(
        Adapter::open(&fixture.config).err().unwrap().code,
        "workload_limit"
    );
}

#[test]
fn unicode_byte_accounting_and_vector_validation() {
    assert_eq!(canonical_size(&json!("é")), 8);
    assert_eq!(canonical_size(&json!("😀")), 14);
    let fixture = Fixture::new();
    let a = fixture.adapter();
    let result = a.call(&request(
        "search",
        json!({"text":"😀".repeat(5500),"vector":[0,0]}),
    ));
    assert_eq!(result["error"]["code"], "input_limit");
    for value in [json!(true), json!(1.0), json!(9223372036854775808u64)] {
        assert_eq!(
            a.call(&request("fetch", json!({"ids":[value]})))["error"]["code"],
            "invalid_input"
        );
    }
}

#[test]
fn cli_bad_utf8_final_record_and_empty_input() {
    let fixture = Fixture::new();
    let a = fixture.adapter();
    let mut output = Vec::new();
    assert_eq!(cli(&a, io::Cursor::new(b""), &mut output).unwrap(), 0);
    assert!(output.is_empty());
    let mut input = vec![255, b'\n'];
    input.extend_from_slice(request("describe", json!({})).to_string().as_bytes());
    assert_eq!(cli(&a, io::Cursor::new(input), &mut output).unwrap(), 1);
    let rows = output
        .split(|b| *b == b'\n')
        .filter(|s| !s.is_empty())
        .map(|v| serde_json::from_slice::<Value>(v).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows[0]["error"]["code"], "invalid_json");
    assert_eq!(rows[1]["ok"], true);
}

#[test]
fn bounded_witness_cancellation_releases_transaction_and_progress_hook() {
    let fixture = Fixture::new();
    let writer = Connection::open(&fixture.database).unwrap();
    writer
        .execute_batch("create table evidence_edges(id integer primary key,src,dst)")
        .unwrap();
    writer.execute_batch("BEGIN").unwrap();
    {
        let mut stmt = writer
            .prepare("insert into evidence_edges values (?1,1,2)")
            .unwrap();
        for i in 0..19999 {
            stmt.execute([i]).unwrap();
        }
    }
    writer.execute_batch("COMMIT").unwrap();
    let mut config: Value =
        serde_json::from_slice(&std::fs::read(&fixture.config).unwrap()).unwrap();
    config["corpora"]["demo"]["edges"] = json!("evidence_edges");
    config["corpora"]["demo"]["graph"] =
        json!({"node_type":"Function","edge_type":"CALLS","edge_id":"id"});
    std::fs::write(&fixture.config, config.to_string()).unwrap();
    let mut adapter = fixture.adapter();
    let cancelled = Arc::new(AtomicBool::new(false));
    let started = Arc::new(AtomicBool::new(false));
    let observed = started.clone();
    // Trace runs immediately before the native SELECT enters SQLite execution.
    adapter
        .db
        .trace_v2(
            TraceEventCodes::SQLITE_TRACE_STMT,
            Some(move |event: TraceEvent<'_>| {
                if let TraceEvent::Stmt(_, sql) = event {
                    if sql.contains("graph_witness(") {
                        observed.store(true, Ordering::SeqCst);
                    }
                }
            }),
        )
        .unwrap();
    let interrupt = adapter.interrupt_handle();
    let flag = cancelled.clone();
    let worker = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !started.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "native witness never started");
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(5));
        flag.store(true, Ordering::SeqCst);
        interrupt.interrupt();
    });
    let input = request(
        "witness",
        json!({"version":"g2/1","node_type":"Function","edge_type":"CALLS","direction":"out","source":1,"target":2,"work":1000000000,"bytes":4194304}),
    );
    let result = adapter.call_cancellable(&input, cancelled);
    worker.join().unwrap();
    assert_eq!(result["error"]["code"], "cancelled", "{result}");
    assert!(adapter.db.is_autocommit());
    assert_eq!(
        adapter.call(&request("fetch", json!({"ids":[1]})))["ok"],
        true
    );
    let result = adapter.call_cancellable(&input, Arc::new(AtomicBool::new(true)));
    assert_eq!(result["error"]["code"], "cancelled");
    assert!(adapter.db.is_autocommit());
    assert_eq!(adapter.call(&request("describe", json!({})))["ok"], true);
}

#[test]
fn keyword_membership_preserves_scores_and_transaction_visibility() {
    let fixture = Fixture::new();
    let writer = Connection::open(&fixture.database).unwrap();
    writer.execute_batch("insert into lexical(rowid,body) values (1,'old evidence'),(2,'old evidence'),(3,'third evidence');
        insert into documents values (-9223372036854775808,'min','ours',100,null,null,'min','old evidence',0,1);
        insert into documents values (9223372036854775807,'max','ours',0,100,null,'max','old evidence',0,1);
        insert into lexical(rowid,body) values (-9223372036854775808,'old evidence'),(9223372036854775807,'old evidence');").unwrap();
    let old = "with eligible as materialized (select id from documents where tenant=:tenant and invalidated_at is null and valid_from<=:as_of and (valid_to is null or valid_to>:as_of)) select rowid id,rank from lexical where lexical match :text and rowid in (select id from eligible) order by rank,rowid limit :cap";
    let new = old.replace(
        "rowid in (select id from eligible)",
        "exists (select 1 from eligible where eligible.id = lexical.rowid)",
    );
    let check = || {
        for text in [
            "old",
            "\"old evidence\"",
            "old OR third",
            "old NOT third",
            "zznomatchzz",
        ] {
            for cap in [1, 2, 50] {
                let run = |sql: &str| {
                    writer
                        .prepare(sql)
                        .unwrap()
                        .query_map(
                            named_params! {":tenant":"ours",":as_of":100,":text":text,":cap":cap},
                            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)),
                        )
                        .unwrap()
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .unwrap()
                };
                assert_eq!(run(old), run(&new));
            }
        }
    };
    check();
    writer
        .execute_batch("begin; update documents set invalidated_at=100 where id=1;")
        .unwrap();
    check();
    writer.execute_batch("savepoint s; update documents set tenant='other' where id=2; update lexical set body='changed' where rowid=2").unwrap();
    check();
    writer.execute_batch("rollback to s").unwrap();
    check();
    writer.execute_batch("rollback").unwrap();
    check();
    // Compare the shipped adapter branch against the unchanged IN reference.
    let adapter = fixture.adapter();
    let reference = writer
        .prepare(old)
        .unwrap()
        .query_map(
            named_params! {":tenant":"ours",":as_of":100,":text":"old",":cap":50},
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)),
        )
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    let result = adapter.call(&request(
        "search",
        json!({"mode":"lexical","text":"old","candidates":50,"limit":100}),
    ));
    assert_eq!(result["ok"], true, "{result}");
    let actual = result["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["id"].as_i64().unwrap(),
                r["keyword_bm25"].as_f64().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, reference);
    writer
        .execute_batch("begin; update documents set invalidated_at=100 where id=1; commit")
        .unwrap();
    check();
    let result = adapter.call(&request(
        "search",
        json!({"mode":"lexical","text":"old","candidates":50,"limit":100}),
    ));
    assert!(
        result["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["id"] != 1)
    );
}
