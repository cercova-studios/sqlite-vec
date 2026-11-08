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
impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "vec-writer-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("tasks.db");
        drop(Writer::create(&path, "project", "seed").unwrap());
        Self { directory, path }
    }
    fn writer(&self, actor: &str) -> Writer {
        Writer::open(&self.path, "project", actor).unwrap()
    }
    fn raw(&self) -> Connection {
        Connection::open(&self.path).unwrap()
    }
    fn seed(&self, ids: &[i64]) -> Receipt {
        self.writer("seed")
            .apply(
                "seed",
                &Command::Batch(
                    ids.iter()
                        .map(|id| Mutation::Create {
                            id: *id,
                            title: format!("task {id} 🦀"),
                            payload: if *id % 2 == 0 {
                                None
                            } else {
                                Some(vec![0, 255, 2])
                            },
                        })
                        .collect(),
                ),
                &AtomicBool::new(false),
            )
            .unwrap()
    }
    fn counts(&self) -> (i64, i64, i64, i64) {
        assert!(
            !self
                .raw()
                .prepare("PRAGMA foreign_key_check")
                .unwrap()
                .exists([])
                .unwrap()
        );
        self.raw().query_row("SELECT (SELECT sequence FROM meta),(SELECT count(*) FROM requests),(SELECT count(*) FROM changes),(SELECT count(*) FROM tasks)",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.directory).unwrap();
    }
}
fn claim(id: i64, expected: i64) -> Command {
    Command::Batch(vec![Mutation::Claim { id, expected }])
}
fn undo(request: &str) -> Command {
    Command::Undo {
        request: request.into(),
    }
}

#[test]
fn concurrent_claims_have_one_winner() {
    for _ in 0..12 {
        let f = Fixture::new();
        f.seed(&[1]);
        let barrier = Arc::new(Barrier::new(3));
        let threads: Vec<_> = ["a", "b"]
            .into_iter()
            .map(|actor| {
                let mut w = f.writer(actor);
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    w.apply(actor, &claim(1, 1), &AtomicBool::new(false))
                })
            })
            .collect();
        barrier.wait();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert!(
            results
                .iter()
                .filter(|r| r.is_err())
                .all(|r| matches!(r, Err(Error::Conflict | Error::Busy)))
        );
        assert_eq!(f.counts(), (2, 2, 2, 1));
        let task = f.writer("observer").task(1).unwrap().unwrap();
        assert_eq!(task.revision, 2);
        assert!(matches!(task.owner.as_deref(), Some("a" | "b")));
    }
}
#[test]
fn concurrent_retries_and_reopen_return_one_receipt() {
    let f = Fixture::new();
    f.seed(&[1]);
    let barrier = Arc::new(Barrier::new(3));
    let threads: Vec<_> = (0..2)
        .map(|_| {
            let mut w = f.writer("actor");
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                w.apply("same", &claim(1, 1), &AtomicBool::new(false))
                    .unwrap()
            })
        })
        .collect();
    barrier.wait();
    let receipts: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(receipts[0], receipts[1]);
    assert_eq!(f.counts(), (2, 2, 2, 1));
    let mut w = f.writer("actor");
    assert_eq!(
        w.apply("same", &claim(1, 1), &AtomicBool::new(false))
            .unwrap(),
        receipts[0]
    );
    assert!(matches!(
        w.apply("same", &claim(1, 2), &AtomicBool::new(false)),
        Err(Error::ReusedKey)
    ));
    assert!(matches!(
        f.writer("other")
            .apply("same", &claim(1, 1), &AtomicBool::new(false)),
        Err(Error::ReusedKey)
    ));
}
#[test]
fn undo_is_monotonic_conflict_aware_and_attributed() {
    let f = Fixture::new();
    f.seed(&[1, 2]);
    let mut a = f.writer("a");
    let cancel = AtomicBool::new(false);
    a.apply("claim", &claim(1, 1), &cancel).unwrap();
    assert!(matches!(
        a.apply("bad undo", &undo("seed"), &cancel),
        Err(Error::Conflict)
    ));
    assert_eq!(a.task(2).unwrap().unwrap().revision, 1);
    let restored = a.apply("undo claim", &undo("claim"), &cancel).unwrap();
    assert_eq!(restored.revision, 3);
    assert_eq!(a.task(1).unwrap().unwrap().owner, None);
    assert!(matches!(
        a.apply("stale", &claim(1, 1), &cancel),
        Err(Error::Conflict)
    ));
    let mut b = f.writer("b");
    b.apply("new claim", &claim(1, 3), &cancel).unwrap();
    assert!(matches!(
        a.apply("erase later", &undo("claim"), &cancel),
        Err(Error::Conflict)
    ));
    assert_eq!(a.task(1).unwrap().unwrap().owner.as_deref(), Some("b"));
    let history = a.history(1, 100).unwrap();
    assert_eq!(history.len(), 4);
    assert_eq!(
        history[1],
        History {
            request: "undo claim".into(),
            actor: "a".into(),
            revision: 3,
            undo_of: Some("claim".into())
        }
    );
}
#[test]
fn delete_restore_and_undo_of_undo_preserve_blob_null_and_tombstones() {
    let f = Fixture::new();
    f.seed(&[1, 2]);
    let mut w = f.writer("a");
    let c = AtomicBool::new(false);
    let original = w.task(1).unwrap().unwrap();
    w.apply(
        "delete",
        &Command::Batch(vec![Mutation::Delete { id: 1, expected: 1 }]),
        &c,
    )
    .unwrap();
    assert!(w.task(1).unwrap().is_none());
    assert!(matches!(
        w.apply(
            "reuse",
            &Command::Batch(vec![Mutation::Create {
                id: 1,
                title: "reuse".into(),
                payload: None
            }]),
            &c
        ),
        Err(Error::Conflict)
    ));
    w.apply("restore", &undo("delete"), &c).unwrap();
    let restored = w.task(1).unwrap().unwrap();
    assert_eq!(restored.revision, 3);
    assert_eq!(
        (restored.title, restored.owner, restored.payload),
        (original.title, original.owner, original.payload)
    );
    w.apply("undo restore", &undo("restore"), &c).unwrap();
    assert!(w.task(1).unwrap().is_none());
    w.apply("redo restore", &undo("undo restore"), &c).unwrap();
    assert_eq!(w.task(1).unwrap().unwrap().revision, 5);
    assert_eq!(w.task(2).unwrap().unwrap().payload, None);
    let f = Fixture::new();
    f.seed(&[8]);
    let mut w = f.writer("a");
    w.apply("undo create", &undo("seed"), &c).unwrap();
    assert!(w.task(8).unwrap().is_none());
    w.apply("restore create", &undo("undo create"), &c).unwrap();
    assert_eq!(w.task(8).unwrap().unwrap().revision, 3);
}
#[test]
fn failed_batch_rolls_back_earlier_successful_mutation() {
    let f = Fixture::new();
    f.seed(&[1, 2]);
    let mut w = f.writer("a");
    let c = AtomicBool::new(false);
    w.apply("owned", &claim(2, 1), &c).unwrap();
    let before = f.counts();
    let batch = Command::Batch(vec![
        Mutation::Claim { id: 1, expected: 1 },
        Mutation::Claim { id: 2, expected: 2 },
    ]);
    assert!(matches!(
        w.apply("partial", &batch, &c),
        Err(Error::Conflict)
    ));
    assert_eq!(f.counts(), before);
    assert_eq!(w.task(1).unwrap().unwrap().owner, None);
}
#[test]
fn statement_capture_and_history_failures_can_retry_without_lost_state() {
    for stage in [Stage::Mutated, Stage::Captured, Stage::Ledger] {
        let f = Fixture::new();
        f.seed(&[1]);
        let mut w = f.writer("a");
        let before = f.counts();
        w.hook = Some(Box::new(move |at, db, _| {
            if at == stage {
                db.execute("INSERT INTO tasks VALUES(1,'duplicate',NULL,NULL)", [])?;
            }
            Ok(())
        }));
        assert!(matches!(
            w.apply("retry", &claim(1, 1), &AtomicBool::new(false)),
            Err(Error::Database(_))
        ));
        assert_eq!(f.counts(), before);
        assert_eq!(w.task(1).unwrap().unwrap().owner, None);
        w.hook = None;
        assert_eq!(
            w.apply("retry", &claim(1, 1), &AtomicBool::new(false))
                .unwrap()
                .revision,
            2
        );
    }
    let f = Fixture::new();
    f.seed(&[1]);
    let mut w = f.writer("a");
    w.capture_limit = 1;
    assert!(
        w.apply("capture fail", &claim(1, 1), &AtomicBool::new(false))
            .is_err()
    );
    assert_eq!(f.counts(), (1, 1, 1, 1));
    w.capture_limit = CAPTURE_BYTES;
    assert_eq!(
        w.apply("capture fail", &claim(1, 1), &AtomicBool::new(false))
            .unwrap()
            .revision,
        2
    );
}
#[test]
fn cancellation_rolls_back_every_stage_but_not_a_committed_receipt() {
    for stage in [
        Stage::Mutated,
        Stage::Captured,
        Stage::Ledger,
        Stage::BeforeCommit,
        Stage::Committed,
    ] {
        let f = Fixture::new();
        f.seed(&[1]);
        let mut w = f.writer("a");
        let c = AtomicBool::new(false);
        w.hook = Some(Box::new(move |at, _, c| {
            if at == stage {
                c.store(true, Ordering::SeqCst);
            }
            Ok(())
        }));
        let r = w.apply("cancel", &claim(1, 1), &c);
        if stage == Stage::Committed {
            assert_eq!(r.unwrap().revision, 2);
            assert_eq!(f.counts(), (2, 2, 2, 1));
        } else {
            assert!(matches!(r, Err(Error::Cancelled)));
            assert_eq!(f.counts(), (1, 1, 1, 1));
        }
    }
}
#[test]
fn savepoint_rollback_is_absent_from_capture_and_history() {
    let f = Fixture::new();
    f.seed(&[1]);
    let mut w = f.writer("a");
    w.hook = Some(Box::new(|at, db, _| {
        if at == Stage::Mutated {
            db.execute_batch(
                "SAVEPOINT probe;UPDATE tasks SET title='discard';ROLLBACK TO probe;RELEASE probe;",
            )?;
        }
        Ok(())
    }));
    let c = AtomicBool::new(false);
    w.apply("claim", &claim(1, 1), &c).unwrap();
    w.apply("undo", &undo("claim"), &c).unwrap();
    assert_eq!(w.task(1).unwrap().unwrap().title, "task 1 🦀");
    assert_eq!(w.task(1).unwrap().unwrap().owner, None);
}
#[test]
fn failed_commit_or_rollback_poison_connection_and_resolve_by_reopening() {
    for before_commit in [false, true] {
        let f = Fixture::new();
        f.seed(&[1]);
        let mut w = f.writer("a");
        w.hook = Some(Box::new(move |at, db, _| {
            if at
                == if before_commit {
                    Stage::BeforeCommit
                } else {
                    Stage::Ledger
                }
            {
                db.execute_batch("ROLLBACK")?;
                if !before_commit {
                    return Err(Error::Conflict);
                }
            }
            Ok(())
        }));
        assert!(matches!(
            w.apply("ambiguous", &claim(1, 1), &AtomicBool::new(false)),
            Err(Error::UnknownOutcome)
        ));
        assert!(matches!(w.task(1), Err(Error::UnknownOutcome)));
        assert_eq!(f.counts(), (1, 1, 1, 1));
        assert_eq!(
            f.writer("a")
                .apply("ambiguous", &claim(1, 1), &AtomicBool::new(false))
                .unwrap()
                .revision,
            2
        );
    }
}
#[test]
fn admission_overflow_schema_and_existing_file_guards() {
    let f = Fixture::new();
    f.seed(&[1]);
    let mut w = f.writer("a");
    let c = AtomicBool::new(false);
    assert!(Writer::create(&f.path, "project", "a").is_err());
    assert!(matches!(
        Writer::open(&f.path, "wrong project", "a"),
        Err(Error::Schema)
    ));
    for command in [
        Command::Batch(vec![]),
        Command::Batch(vec![Mutation::Claim { id: 1, expected: 1 }; 65]),
        Command::Batch(vec![Mutation::Claim { id: 1, expected: 1 }; 2]),
    ] {
        assert!(matches!(
            w.apply("invalid", &command, &c),
            Err(Error::Invalid)
        ));
    }
    let huge = Command::Batch(vec![Mutation::Create {
        id: 2,
        title: "x".repeat(ROW_BYTES + 1),
        payload: None,
    }]);
    assert!(matches!(w.apply("huge", &huge, &c), Err(Error::Limit)));
    f.raw()
        .execute("UPDATE meta SET sequence=?", [i64::MAX])
        .unwrap();
    assert!(matches!(
        w.apply("overflow", &claim(1, 1), &c),
        Err(Error::Limit)
    ));
    f.raw().execute_batch("CREATE TABLE drift(x)").unwrap();
    assert!(matches!(
        w.apply("drift", &claim(1, 1), &c),
        Err(Error::Schema)
    ));
}
#[test]
fn busy_wait_is_bounded_and_cancellable() {
    let f = Fixture::new();
    f.seed(&[1]);
    let db = f.raw();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    let mut w = f.writer("a");
    let c = Arc::new(AtomicBool::new(false));
    let flag = c.clone();
    let t = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        flag.store(true, Ordering::SeqCst);
    });
    assert!(matches!(
        w.apply("cancel wait", &claim(1, 1), &c),
        Err(Error::Cancelled)
    ));
    t.join().unwrap();
    assert!(matches!(
        w.apply("busy", &claim(1, 1), &AtomicBool::new(false)),
        Err(Error::Busy)
    ));
    assert_eq!(f.counts(), (1, 1, 1, 1));
    db.execute_batch("ROLLBACK").unwrap();
}

#[test]
fn request_and_old_row_budgets_fail_without_mutating() {
    let f = Fixture::new();
    let mut w = f.writer("a");
    let c = AtomicBool::new(false);
    let oversized = Command::Batch(
        (0..3)
            .map(|id| Mutation::Create {
                id,
                title: "x".into(),
                payload: Some(vec![255; 8190]),
            })
            .collect(),
    );
    assert!(matches!(
        w.apply("request bytes", &oversized, &c),
        Err(Error::Limit)
    ));
    assert_eq!(f.counts(), (0, 0, 0, 0));
    for id in 1..=17 {
        w.apply(
            &format!("create {id}"),
            &Command::Batch(vec![Mutation::Create {
                id,
                title: "x".into(),
                payload: Some(vec![0; 8190]),
            }]),
            &c,
        )
        .unwrap();
    }
    let command = Command::Batch(
        (1..=17)
            .map(|id| Mutation::Claim { id, expected: id })
            .collect(),
    );
    assert!(matches!(
        w.apply("old bytes", &command, &c),
        Err(Error::Limit)
    ));
    assert_eq!(f.counts(), (17, 17, 17, 17));
    f.raw()
        .execute(
            "UPDATE tasks SET payload=zeroblob(?) WHERE id=1",
            [ROW_BYTES as i64 + 1],
        )
        .unwrap();
    assert!(matches!(w.task(1), Err(Error::Limit)));
}

#[test]
fn strict_changeset_conflict_rolls_back_all_undo_rows() {
    let f = Fixture::new();
    f.seed(&[1, 2]);
    let mut w = f.writer("a");
    let c = AtomicBool::new(false);
    w.apply(
        "claims",
        &Command::Batch(vec![
            Mutation::Claim { id: 1, expected: 1 },
            Mutation::Claim { id: 2, expected: 1 },
        ]),
        &c,
    )
    .unwrap();
    // Simulate an unmanaged writer; strict capture comparison must still refuse overwrite.
    f.raw()
        .execute("UPDATE tasks SET owner='unmanaged' WHERE id=2", [])
        .unwrap();
    assert!(matches!(
        w.apply("conflict", &undo("claims"), &c),
        Err(Error::Database(_))
    ));
    assert_eq!(w.task(1).unwrap().unwrap().owner.as_deref(), Some("a"));
    assert_eq!(
        w.task(2).unwrap().unwrap().owner.as_deref(),
        Some("unmanaged")
    );
    assert_eq!(f.counts(), (2, 2, 4, 2));
}
#[test]
#[ignore = "subprocess worker executed explicitly by process_exit_recovery"]
fn crash_worker() {
    let path = std::env::var_os("VEC_WRITER_TEST_DB").unwrap();
    let mut w = Writer::open(Path::new(&path), "project", "a").unwrap();
    let committed = std::env::var("VEC_WRITER_TEST_COMMITTED").unwrap() == "1";
    w.hook = Some(Box::new(move |at, _, _| {
        if at
            == if committed {
                Stage::Committed
            } else {
                Stage::BeforeCommit
            }
        {
            std::process::exit(42);
        }
        Ok(())
    }));
    w.apply("exit", &claim(1, 1), &AtomicBool::new(false))
        .unwrap();
    panic!("exit hook not reached");
}
#[test]
fn process_exit_recovery() {
    for committed in [false, true] {
        let f = Fixture::new();
        f.seed(&[1]);
        let status = Process::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::crash_worker", "--ignored"])
            .env("VEC_WRITER_TEST_DB", &f.path)
            .env(
                "VEC_WRITER_TEST_COMMITTED",
                if committed { "1" } else { "0" },
            )
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(42));
        assert_eq!(
            f.counts(),
            if committed {
                (2, 2, 2, 1)
            } else {
                (1, 1, 1, 1)
            }
        );
        assert_eq!(
            f.raw()
                .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "ok"
        );
        let receipt = f
            .writer("a")
            .apply("exit", &claim(1, 1), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(receipt.revision, 2);
        assert_eq!(f.counts(), (2, 2, 2, 1));
    }
}
