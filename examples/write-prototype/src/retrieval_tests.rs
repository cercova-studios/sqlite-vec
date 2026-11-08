use super::*;
use std::{
    path::PathBuf,
    process::Command as Process,
    sync::{Arc, Barrier, atomic::AtomicU64},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    directory: PathBuf,
    path: PathBuf,
}
fn extension() -> PathBuf {
    std::env::var_os("VEC_WRITER_EXTENSION")
        .expect("set VEC_WRITER_EXTENSION to a verified native vec0 library for indexed tests")
        .into()
}
fn model() -> Model {
    Model {
        id: "fixture/v1-l2".into(),
        dimensions: 2,
    }
}
impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "vec-sync-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("tasks.db");
        // SAFETY: test runner supplies the verified host-selected vec0 library.
        drop(
            unsafe { Writer::create_indexed(&path, "project", "actor", &extension(), &model()) }
                .unwrap(),
        );
        Self { directory, path }
    }
    fn writer(&self, actor: &str) -> Writer {
        // SAFETY: same verified test library on every SQLite connection.
        unsafe { Writer::open_indexed(&self.path, "project", actor, &extension(), &model()) }
            .unwrap()
    }
    fn raw(&self) -> Connection {
        let db = Connection::open(&self.path).unwrap();
        // SAFETY: same verified test library as the writer.
        unsafe { retrieval::load(&db, &extension()) }.unwrap();
        db
    }
    fn seed(&self) -> Writer {
        let mut w = self.writer("actor");
        w.apply(
            "seed",
            &Command::Batch(vec![create(1), create(2)]),
            &AtomicBool::new(false),
        )
        .unwrap();
        w
    }
    fn oracle(&self, title: &str, vector: [f32; 2], revision: i64, requests: i64) {
        let db = self.raw();
        let expected: Vec<u8> = vector.into_iter().flat_map(f32::to_le_bytes).collect();
        let actual:(String,Vec<u8>,String,i64,i64)=db.query_row("SELECT t.title,s.vector,f.body,v.source_revision,i.source_revision FROM tasks t JOIN source_vectors s USING(id) JOIN versions v USING(id) JOIN index_state i USING(id) JOIN lexical f ON f.rowid=t.id WHERE t.id=1",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?))).unwrap();
        assert_eq!(
            actual,
            (title.into(), expected, title.into(), revision, revision)
        );
        let distances:Vec<(i64,f64)>=db.prepare("SELECT rowid,distance FROM vectors WHERE embedding MATCH ? AND k=2 ORDER BY distance").unwrap().query_map([[0u8;8].as_slice()],|r|Ok((r.get(0)?,r.get(1)?))).unwrap().collect::<rusqlite::Result<_>>().unwrap();
        assert_eq!(distances.len(), 2);
        let distance = distances.into_iter().find(|r| r.0 == 1).unwrap().1;
        assert!((distance - (vector[0] as f64).hypot(vector[1] as f64)).abs() < 1e-6);
        assert_eq!(
            db.query_row("SELECT count(*) FROM requests", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            requests
        );
        assert!(
            !db.prepare("PRAGMA foreign_key_check")
                .unwrap()
                .exists([])
                .unwrap()
        );
        assert_eq!(
            db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "ok"
        );
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.directory).unwrap();
    }
}
fn embedding(title: &str, values: [f32; 2]) -> PreparedEmbedding {
    PreparedEmbedding {
        model: model().id,
        source: title.into(),
        values: values.into(),
    }
}
fn create(id: i64) -> Mutation {
    Mutation::CreateSource {
        id,
        title: "alpha".into(),
        payload: Some(vec![0, 255]),
        embedding: embedding("alpha", [3., 4.]),
    }
}
fn update(id: i64, expected: i64, expected_source: i64, title: &str, values: [f32; 2]) -> Mutation {
    Mutation::UpdateSource {
        id,
        expected,
        expected_source,
        title: title.into(),
        embedding: embedding(title, values),
    }
}
fn undo(request: &str) -> Command {
    Command::Undo {
        request: request.into(),
    }
}

#[test]
fn publication_snapshot_vector_only_and_monotonic_undo() {
    let f = Fixture::new();
    let mut w = f.seed();
    let c = AtomicBool::new(false);
    f.oracle("alpha", [3., 4.], 1, 1);
    let reader = f.raw();
    reader.execute_batch("BEGIN").unwrap();
    assert_eq!(
        reader
            .query_row(
                "SELECT count(*) FROM lexical WHERE lexical MATCH 'alpha'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        2
    );
    w.apply(
        "update",
        &Command::Batch(vec![update(1, 1, 1, "beta", [6., 8.])]),
        &c,
    )
    .unwrap();
    assert_eq!(
        reader
            .query_row("SELECT body FROM lexical WHERE rowid=1", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        "alpha"
    );
    let old: Vec<u8> = reader
        .query_row("SELECT embedding FROM vectors WHERE rowid=1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        old,
        [3f32, 4.]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>()
    );
    reader.execute_batch("ROLLBACK").unwrap();
    f.oracle("beta", [6., 8.], 2, 2);
    assert_eq!(
        reader
            .query_row(
                "SELECT count(*) FROM lexical WHERE lexical MATCH 'beta'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
    w.apply(
        "vector",
        &Command::Batch(vec![update(1, 2, 2, "beta", [0., 0.])]),
        &c,
    )
    .unwrap();
    f.oracle("beta", [0., 0.], 3, 3);
    w.apply("undo-vector", &undo("vector"), &c).unwrap();
    f.oracle("beta", [6., 8.], 4, 4);
    let delete = Command::Batch(vec![Mutation::Delete { id: 1, expected: 4 }]);
    let receipt = w.apply("delete", &delete, &c).unwrap();
    assert!(w.source(1).unwrap().is_none());
    let db = f.raw();
    for table in ["source_vectors", "lexical", "vectors", "index_state"] {
        let key = if table == "lexical" || table == "vectors" {
            "rowid"
        } else {
            "id"
        };
        assert_eq!(
            db.query_row(
                &format!("SELECT count(*) FROM {table} WHERE {key}=1"),
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
    }
    w.apply("restore", &undo("delete"), &c).unwrap();
    f.oracle("beta", [6., 8.], 6, 6);
    assert_eq!(w.apply("delete", &delete, &c).unwrap(), receipt);
    f.oracle("beta", [6., 8.], 6, 6);
    w.apply("undo-restore", &undo("restore"), &c).unwrap();
    assert!(w.source(1).unwrap().is_none());
    w.apply("restore-again", &undo("undo-restore"), &c).unwrap();
    f.oracle("beta", [6., 8.], 8, 8);
    drop(w);
    assert_eq!(
        f.writer("actor")
            .source(1)
            .unwrap()
            .unwrap()
            .source_revision,
        8
    );
}

#[test]
fn claims_and_undo_do_not_write_any_index_rows() {
    let f = Fixture::new();
    let mut w = f.seed();
    let c = AtomicBool::new(false);
    let original = w.source(1).unwrap().unwrap();
    w.hook = Some(Box::new(|stage, _, _| {
        assert!(!matches!(stage, Stage::Fts | Stage::Vector));
        Ok(())
    }));
    for (request, command) in [
        (
            "claim",
            Command::Batch(vec![Mutation::Claim { id: 1, expected: 1 }]),
        ),
        ("unclaim", undo("claim")),
    ] {
        let before = w.connection().unwrap().total_changes();
        w.apply(request, &command, &c).unwrap();
        assert_eq!(w.connection().unwrap().total_changes() - before, 5);
        assert_eq!(w.source(1).unwrap().unwrap(), original);
    }
    f.oracle("alpha", [3., 4.], 1, 3);
}

#[test]
fn rollback_and_cancellation_after_each_participant_are_atomic_and_retryable() {
    for stage in [
        Stage::Mutated,
        Stage::Fts,
        Stage::Vector,
        Stage::Captured,
        Stage::Ledger,
        Stage::BeforeCommit,
    ] {
        for cancel in [false, true] {
            let f = Fixture::new();
            let mut w = f.seed();
            let c = AtomicBool::new(false);
            let command = Command::Batch(vec![
                update(1, 1, 1, "beta", [6., 8.]),
                update(2, 1, 1, "beta", [0., 0.]),
            ]);
            w.hook = Some(Box::new(move |at, _, flag| {
                if at == stage {
                    if cancel {
                        flag.store(true, Ordering::SeqCst);
                    } else {
                        return Err(Error::Conflict);
                    }
                }
                Ok(())
            }));
            let result = w.apply("update", &command, &c);
            assert!(if cancel {
                matches!(result, Err(Error::Cancelled))
            } else {
                matches!(result, Err(Error::Conflict))
            });
            f.oracle("alpha", [3., 4.], 1, 1);
            assert_eq!(w.source(2).unwrap().unwrap().title, "alpha");
            w.hook = None;
            c.store(false, Ordering::SeqCst);
            let receipt = w.apply("update", &command, &c).unwrap();
            f.oracle("beta", [6., 8.], 2, 2);
            assert_eq!(w.apply("update", &command, &c).unwrap(), receipt);
        }
    }
}

#[test]
fn native_failure_savepoint_and_poisoned_commit_roll_back_indexes() {
    let f = Fixture::new();
    let mut w = f.seed();
    let c = AtomicBool::new(false);
    let command = Command::Batch(vec![
        update(1, 1, 1, "beta", [6., 8.]),
        update(2, 1, 1, "beta", [0., 0.]),
    ]);
    // A real vec0 insertion failure after FTS succeeds must roll back both participants.
    w.hook = Some(Box::new(|at, db, _| {
        if at == Stage::Fts {
            db.execute(
                "INSERT INTO vectors(rowid,embedding) VALUES(99,zeroblob(3))",
                [],
            )?;
        }
        Ok(())
    }));
    assert!(matches!(
        w.apply("update", &command, &c),
        Err(Error::Database(_))
    ));
    f.oracle("alpha", [3., 4.], 1, 1);
    w.capture_limit = 1;
    w.hook = None;
    assert!(matches!(w.apply("update", &command, &c), Err(Error::Limit)));
    f.oracle("alpha", [3., 4.], 1, 1);
    w.capture_limit = CAPTURE_BYTES;
    w.hook = Some(Box::new(|at, db, _| {
        if at == Stage::Vector {
            db.execute_batch("SAVEPOINT probe;DELETE FROM lexical WHERE rowid=2;DELETE FROM vectors WHERE rowid=2;UPDATE source_vectors SET vector=zeroblob(8) WHERE id=2;ROLLBACK TO probe;RELEASE probe;")?;
        }
        Ok(())
    }));
    w.apply("update", &command, &c).unwrap();
    f.oracle("beta", [6., 8.], 2, 2);
    w.hook = Some(Box::new(|at, db, _| {
        if at == Stage::BeforeCommit {
            db.execute_batch("ROLLBACK")?;
        }
        Ok(())
    }));
    assert!(matches!(
        w.apply("undo", &undo("update"), &c),
        Err(Error::UnknownOutcome)
    ));
    assert!(matches!(w.source(1), Err(Error::UnknownOutcome)));
    f.oracle("beta", [6., 8.], 2, 2);
    f.writer("actor")
        .apply("undo", &undo("update"), &c)
        .unwrap();
    f.oracle("alpha", [3., 4.], 3, 3);
}

#[test]
fn indexed_claim_race_and_exact_retry_have_one_receipt() {
    for same_request in [false, true] {
        let f = Fixture::new();
        drop(f.seed());
        let barrier = Arc::new(Barrier::new(3));
        let threads: Vec<_> = (0..2)
            .map(|i| {
                let mut w = f.writer(if same_request {
                    "same"
                } else if i == 0 {
                    "a"
                } else {
                    "b"
                });
                let b = barrier.clone();
                thread::spawn(move || {
                    b.wait();
                    w.apply(
                        if same_request {
                            "claim"
                        } else if i == 0 {
                            "a"
                        } else {
                            "b"
                        },
                        &Command::Batch(vec![Mutation::Claim { id: 1, expected: 1 }]),
                        &AtomicBool::new(false),
                    )
                })
            })
            .collect();
        barrier.wait();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        if same_request {
            assert_eq!(results[0].as_ref().unwrap(), results[1].as_ref().unwrap());
        } else {
            assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
            assert!(
                results
                    .iter()
                    .filter(|r| r.is_err())
                    .all(|r| matches!(r, Err(Error::Conflict | Error::Busy)))
            );
        }
        f.oracle("alpha", [3., 4.], 1, 2);
    }
}

#[test]
fn later_row_failure_and_strict_undo_conflict_preserve_every_row() {
    let f = Fixture::new();
    let mut w = f.seed();
    let c = AtomicBool::new(false);
    let command = Command::Batch(vec![
        update(1, 1, 1, "beta", [6., 8.]),
        update(2, 1, 1, "beta", [0., 0.]),
    ]);
    let count = Arc::new(AtomicU64::new(0));
    let counter = count.clone();
    w.hook = Some(Box::new(move |stage, _, _| {
        if stage == Stage::Vector && counter.fetch_add(1, Ordering::SeqCst) == 1 {
            return Err(Error::Conflict);
        }
        Ok(())
    }));
    assert!(matches!(
        w.apply("update", &command, &c),
        Err(Error::Conflict)
    ));
    assert_eq!(count.load(Ordering::SeqCst), 2);
    f.oracle("alpha", [3., 4.], 1, 1);
    w.hook = None;
    w.apply("update", &command, &c).unwrap();
    f.raw()
        .execute("UPDATE tasks SET payload=X'01' WHERE id=2", [])
        .unwrap();
    // Source updates do not touch payload, so use an all-row claim inverse to conflict.
    w.apply(
        "claim",
        &Command::Batch(vec![
            Mutation::Claim { id: 1, expected: 2 },
            Mutation::Claim { id: 2, expected: 2 },
        ]),
        &c,
    )
    .unwrap();
    f.raw()
        .execute("UPDATE tasks SET owner='unmanaged' WHERE id=2", [])
        .unwrap();
    assert!(matches!(
        w.apply("undo", &undo("claim"), &c),
        Err(Error::Database(_))
    ));
    assert_eq!(w.task(1).unwrap().unwrap().owner.as_deref(), Some("actor"));
    assert_eq!(
        w.task(2).unwrap().unwrap().owner.as_deref(),
        Some("unmanaged")
    );
    f.oracle("beta", [6., 8.], 2, 3);
}

#[test]
fn canonical_embedding_bytes_count_toward_old_row_set_admission() {
    let f = Fixture::new();
    let path = f.directory.join("large.db");
    let model = Model {
        id: "large/v1".into(),
        dimensions: 4096,
    };
    // SAFETY: same verified test extension; a distinct fresh fixture avoids migration.
    let mut w =
        unsafe { Writer::create_indexed(&path, "project", "actor", &extension(), &model) }.unwrap();
    let c = AtomicBool::new(false);
    for id in 1..=8 {
        w.apply(
            &format!("create{id}"),
            &Command::Batch(vec![Mutation::CreateSource {
                id,
                title: "alpha".into(),
                payload: None,
                embedding: PreparedEmbedding {
                    model: model.id.clone(),
                    source: "alpha".into(),
                    values: vec![0.; 4096],
                },
            }]),
            &c,
        )
        .unwrap();
    }
    assert!(matches!(
        w.apply(
            "too-many-bytes",
            &Command::Batch(
                (1..=8)
                    .map(|id| Mutation::Claim { id, expected: id })
                    .collect()
            ),
            &c
        ),
        Err(Error::Limit)
    ));
    assert_eq!(w.task(1).unwrap().unwrap().owner, None);
    assert_eq!(w.history(1, 100).unwrap().len(), 1);
    w.apply(
        "fits",
        &Command::Batch(
            (1..=7)
                .map(|id| Mutation::Claim { id, expected: id })
                .collect(),
        ),
        &c,
    )
    .unwrap();
    assert_eq!(w.source(1).unwrap().unwrap().source_revision, 1);
}

#[test]
fn stale_inputs_corrupt_derived_and_oversize_canonical_values_are_rejected() {
    let f = Fixture::new();
    let mut w = f.seed();
    let c = AtomicBool::new(false);
    assert!(matches!(
        Writer::open(&f.path, "project", "actor"),
        Err(Error::Schema)
    ));
    let wrong = Model {
        id: "other/v1".into(),
        dimensions: 2,
    };
    // SAFETY: verified test extension; incompatible model must be rejected.
    assert!(matches!(
        unsafe { Writer::open_indexed(&f.path, "project", "actor", &extension(), &wrong) },
        Err(Error::Schema)
    ));
    for mutation in [
        update(1, 1, 2, "beta", [6., 8.]),
        Mutation::UpdateSource {
            id: 1,
            expected: 1,
            expected_source: 1,
            title: "beta".into(),
            embedding: embedding("alpha", [6., 8.]),
        },
    ] {
        assert!(matches!(
            w.apply("invalid", &Command::Batch(vec![mutation]), &c),
            Err(Error::Conflict)
        ));
    }
    for values in [vec![f32::NAN, 0.], vec![0.], vec![1e11, 0.]] {
        let mutation = Mutation::UpdateSource {
            id: 1,
            expected: 1,
            expected_source: 1,
            title: "alpha".into(),
            embedding: PreparedEmbedding {
                model: model().id,
                source: "alpha".into(),
                values,
            },
        };
        assert!(matches!(
            w.apply("invalid", &Command::Batch(vec![mutation]), &c),
            Err(Error::Invalid)
        ));
    }
    for sql in [
        "DELETE FROM lexical WHERE rowid=1",
        "DELETE FROM vectors WHERE rowid=1",
        "DELETE FROM index_state WHERE id=1",
        "UPDATE index_state SET source_revision=99 WHERE id=1",
        "UPDATE source_vectors SET model='wrong' WHERE id=1",
    ] {
        let db = f.raw();
        db.execute_batch("BEGIN").unwrap();
        db.execute_batch(sql).unwrap();
        // Publish corruption temporarily so the independently opened writer must detect it.
        db.execute_batch("COMMIT").unwrap();
        assert!(matches!(w.source(1), Err(Error::Schema)));
        assert!(matches!(
            w.apply(
                "claim",
                &Command::Batch(vec![Mutation::Claim { id: 1, expected: 1 }]),
                &c
            ),
            Err(Error::Schema)
        ));
        match sql {
            "DELETE FROM lexical WHERE rowid=1" => {
                db.execute("INSERT INTO lexical(rowid,body) VALUES(1,'alpha')", [])
                    .unwrap();
            }
            "DELETE FROM vectors WHERE rowid=1" => {
                let v: Vec<u8> = [3f32, 4.].into_iter().flat_map(f32::to_ne_bytes).collect();
                db.execute("INSERT INTO vectors(rowid,embedding) VALUES(1,?)", [v])
                    .unwrap();
            }
            "UPDATE index_state SET source_revision=99 WHERE id=1" => {
                db.execute("UPDATE index_state SET source_revision=1 WHERE id=1", [])
                    .unwrap();
            }
            "DELETE FROM index_state WHERE id=1" => {
                db.execute("INSERT INTO index_state VALUES(1,1,?)", [model().id])
                    .unwrap();
            }
            _ => {
                db.execute("UPDATE source_vectors SET model=? WHERE id=1", [model().id])
                    .unwrap();
            }
        }
    }
    f.oracle("alpha", [3., 4.], 1, 1);
    let db = f.raw();
    db.execute("INSERT INTO index_state VALUES(99,1,?)", [model().id])
        .unwrap();
    assert!(matches!(w.source(99), Err(Error::Schema)));
    assert!(matches!(
        w.apply("orphan", &Command::Batch(vec![create(99)]), &c),
        Err(Error::Schema)
    ));
    db.execute("DELETE FROM index_state WHERE id=99", [])
        .unwrap();
    let canonical: Vec<u8> = [3f32, 4.].into_iter().flat_map(f32::to_le_bytes).collect();
    f.raw()
        .execute(
            "UPDATE source_vectors SET vector=zeroblob(7) WHERE id=1",
            [],
        )
        .unwrap();
    assert!(matches!(w.source(1), Err(Error::Schema)));
    f.raw()
        .execute("UPDATE source_vectors SET vector=? WHERE id=1", [canonical])
        .unwrap();
    f.raw()
        .execute(
            "UPDATE source_vectors SET vector=zeroblob(1048576) WHERE id=1",
            [],
        )
        .unwrap();
    assert!(matches!(w.source(1), Err(Error::Limit)));
    assert!(matches!(
        w.apply(
            "claim",
            &Command::Batch(vec![Mutation::Claim { id: 1, expected: 1 }]),
            &c
        ),
        Err(Error::Limit)
    ));
}

#[test]
#[ignore = "subprocess helper explicitly run by indexed_process_exit_recovery"]
fn indexed_crash_worker() {
    let path = PathBuf::from(std::env::var_os("VEC_SYNC_TEST_DB").unwrap());
    // SAFETY: inherited verified extension environment from the test runner.
    let mut w =
        unsafe { Writer::open_indexed(&path, "project", "actor", &extension(), &model()) }.unwrap();
    let committed = std::env::var("VEC_SYNC_TEST_COMMITTED").unwrap() == "1";
    w.hook = Some(Box::new(move |stage, _, _| {
        if stage
            == if committed {
                Stage::Committed
            } else {
                Stage::Vector
            }
        {
            std::process::exit(42);
        }
        Ok(())
    }));
    w.apply(
        "exit",
        &Command::Batch(vec![update(1, 1, 1, "beta", [6., 8.])]),
        &AtomicBool::new(false),
    )
    .unwrap();
    panic!("exit not reached");
}
#[test]
fn indexed_process_exit_recovery() {
    for committed in [false, true] {
        let f = Fixture::new();
        drop(f.seed());
        let status = Process::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "retrieval_tests::indexed_crash_worker",
                "--ignored",
            ])
            .env("VEC_SYNC_TEST_DB", &f.path)
            .env("VEC_SYNC_TEST_COMMITTED", if committed { "1" } else { "0" })
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(42));
        if committed {
            f.oracle("beta", [6., 8.], 2, 2);
        } else {
            f.oracle("alpha", [3., 4.], 1, 1);
        }
        let mut w = f.writer("actor");
        w.apply(
            "exit",
            &Command::Batch(vec![update(1, 1, 1, "beta", [6., 8.])]),
            &AtomicBool::new(false),
        )
        .unwrap();
        f.oracle("beta", [6., 8.], 2, 2);
    }
}
