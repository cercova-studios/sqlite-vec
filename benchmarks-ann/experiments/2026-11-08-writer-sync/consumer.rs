//! Independent public-API and SQL acceptance consumer, run after implementation handoff.
use rusqlite::{Connection, LoadExtensionGuard, params};
use sqlite_vec_write_prototype::{Command, Error, Model, Mutation, PreparedEmbedding, Writer};
use std::{path::Path, sync::atomic::AtomicBool};

fn embedding(text: &str, values: [f32; 2]) -> PreparedEmbedding {
    PreparedEmbedding {
        model: "fixture/v1-l2".into(),
        source: text.into(),
        values: values.into(),
    }
}
fn nearest(db: &Connection) -> Vec<(i64, f64)> {
    let zero = [0u8; 8];
    db.prepare(
        "SELECT rowid,distance FROM vectors WHERE embedding MATCH ? AND k=2 ORDER BY distance",
    )
    .unwrap()
    .query_map([zero.as_slice()], |r| Ok((r.get(0)?, r.get(1)?)))
    .unwrap()
    .collect::<rusqlite::Result<_>>()
    .unwrap()
}
fn lexical(db: &Connection, query: &str) -> Vec<i64> {
    db.prepare("SELECT rowid FROM lexical WHERE lexical MATCH ? ORDER BY rowid")
        .unwrap()
        .query_map([query], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let path = Path::new(&args[1]);
    let extension = Path::new(&args[2]);
    let model = Model {
        id: "fixture/v1-l2".into(),
        dimensions: 2,
    };
    let cancel = AtomicBool::new(false);
    // SAFETY: caller verifies the host-selected bundle hash before this fresh-fixture test.
    let mut writer =
        unsafe { Writer::create_indexed(path, "consumer", "actor", extension, &model) }.unwrap();
    let raw = Connection::open(path).unwrap();
    // SAFETY: same verified, ABI-compatible SQLite extension as the indexed writer.
    unsafe {
        let _guard = LoadExtensionGuard::new(&raw).unwrap();
        raw.load_extension(extension, None::<&str>).unwrap();
    }
    let first = writer
        .apply(
            "first",
            &Command::Batch(vec![Mutation::CreateSource {
                id: 1,
                title: "alpha only".into(),
                payload: Some(vec![0, 255]),
                embedding: embedding("alpha only", [0., 0.]),
            }]),
            &cancel,
        )
        .unwrap();
    writer
        .apply(
            "second",
            &Command::Batch(vec![Mutation::CreateSource {
                id: 2,
                title: "beta only".into(),
                payload: None,
                embedding: embedding("beta only", [3., 4.]),
            }]),
            &cancel,
        )
        .unwrap();
    let initial_source = writer.source(1).unwrap().unwrap();
    assert_eq!(lexical(&raw, "alpha"), vec![1]);
    assert_eq!(nearest(&raw), vec![(1, 0.), (2, 5.)]);
    let snapshot = raw.unchecked_transaction().unwrap();
    assert_eq!(lexical(&snapshot, "alpha"), vec![1]);
    let update = writer
        .apply(
            "update",
            &Command::Batch(vec![Mutation::UpdateSource {
                id: 1,
                expected: first.revision,
                expected_source: initial_source.source_revision,
                title: "gamma only".into(),
                embedding: embedding("gamma only", [6., 8.]),
            }]),
            &cancel,
        )
        .unwrap();
    assert_eq!(lexical(&snapshot, "alpha"), vec![1]);
    assert_eq!(nearest(&snapshot), vec![(1, 0.), (2, 5.)]);
    snapshot.rollback().unwrap();
    assert!(lexical(&raw, "alpha").is_empty());
    assert_eq!(lexical(&raw, "gamma"), vec![1]);
    assert_eq!(nearest(&raw), vec![(2, 5.), (1, 10.)]);
    let updated_source = writer.source(1).unwrap().unwrap();
    assert!(updated_source.source_revision > initial_source.source_revision);
    let claim = writer
        .apply(
            "claim",
            &Command::Batch(vec![Mutation::Claim {
                id: 1,
                expected: update.revision,
            }]),
            &cancel,
        )
        .unwrap();
    assert_eq!(writer.source(1).unwrap().unwrap(), updated_source);
    let before: i64 = raw
        .query_row("SELECT count(*) FROM requests", [], |r| r.get(0))
        .unwrap();
    assert!(matches!(
        writer.apply(
            "stale-source",
            &Command::Batch(vec![Mutation::UpdateSource {
                id: 1,
                expected: claim.revision,
                expected_source: initial_source.source_revision,
                title: "delta only".into(),
                embedding: embedding("delta only", [1., 1.]),
            }]),
            &cancel
        ),
        Err(Error::Conflict)
    ));
    assert_eq!(
        raw.query_row("SELECT count(*) FROM requests", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        before
    );
    assert_eq!(writer.source(1).unwrap().unwrap(), updated_source);
    let unclaimed = writer
        .apply(
            "unclaim",
            &Command::Undo {
                request: "claim".into(),
            },
            &cancel,
        )
        .unwrap();
    assert_eq!(writer.source(1).unwrap().unwrap(), updated_source);
    let delete = Command::Batch(vec![Mutation::Delete {
        id: 1,
        expected: unclaimed.revision,
    }]);
    let deleted = writer.apply("delete", &delete, &cancel).unwrap();
    assert!(writer.source(1).unwrap().is_none());
    assert!(lexical(&raw, "gamma").is_empty());
    assert_eq!(nearest(&raw), vec![(2, 5.)]);
    let restored = writer
        .apply(
            "restore",
            &Command::Undo {
                request: "delete".into(),
            },
            &cancel,
        )
        .unwrap();
    assert_eq!(writer.apply("delete", &delete, &cancel).unwrap(), deleted);
    let restored_source = writer.source(1).unwrap().unwrap();
    assert!(restored_source.source_revision > updated_source.source_revision);
    assert_eq!(restored_source.values, vec![6., 8.]);
    assert_eq!(lexical(&raw, "gamma"), vec![1]);
    assert_eq!(nearest(&raw), vec![(2, 5.), (1, 10.)]);
    writer
        .apply(
            "vector-only",
            &Command::Batch(vec![Mutation::UpdateSource {
                id: 1,
                expected: restored.revision,
                expected_source: restored_source.source_revision,
                title: "gamma only".into(),
                embedding: embedding("gamma only", [0., 0.]),
            }]),
            &cancel,
        )
        .unwrap();
    assert_eq!(nearest(&raw), vec![(1, 0.), (2, 5.)]);
    writer
        .apply(
            "undo-vector",
            &Command::Undo {
                request: "vector-only".into(),
            },
            &cancel,
        )
        .unwrap();
    assert_eq!(nearest(&raw), vec![(2, 5.), (1, 10.)]);
    let final_source = writer.source(1).unwrap().unwrap();
    let canonical: Vec<u8> = raw
        .query_row("SELECT vector FROM source_vectors WHERE id=1", [], |r| {
            r.get(0)
        })
        .unwrap();
    let expected: Vec<u8> = [6f32, 8.].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(canonical, expected);
    raw.execute(
        "UPDATE source_vectors SET model='wrong-profile' WHERE id=1",
        [],
    )
    .unwrap();
    assert!(matches!(writer.source(1), Err(Error::Schema)));
    raw.execute("UPDATE source_vectors SET model=? WHERE id=1", [&model.id])
        .unwrap();
    raw.execute("DELETE FROM lexical WHERE rowid=1", [])
        .unwrap();
    assert!(matches!(writer.source(1), Err(Error::Schema)));
    raw.execute(
        "INSERT INTO lexical(rowid,body) VALUES(1,?)",
        [&final_source.title],
    )
    .unwrap();
    assert_eq!(writer.source(1).unwrap().unwrap(), final_source);
    raw.execute("INSERT INTO index_state VALUES(99,1,?)", [&model.id])
        .unwrap();
    assert!(matches!(writer.source(99), Err(Error::Schema)));
    raw.execute("DELETE FROM index_state WHERE id=99", [])
        .unwrap();
    raw.execute("DELETE FROM index_state WHERE id=1", [])
        .unwrap();
    assert!(matches!(writer.source(1), Err(Error::Schema)));
    raw.execute(
        "INSERT INTO index_state VALUES(1,?,?)",
        params![final_source.source_revision, model.id],
    )
    .unwrap();
    let mismatches: i64 = raw
        .query_row(
            "SELECT count(*) FROM (
        SELECT t.id FROM tasks t LEFT JOIN versions v USING(id)
        LEFT JOIN source_vectors s USING(id) LEFT JOIN index_state i USING(id)
        WHERE v.id IS NULL OR v.live<>1 OR s.id IS NULL OR i.id IS NULL
          OR i.source_revision<>v.source_revision OR i.model<>s.model
        UNION ALL SELECT i.id FROM index_state i LEFT JOIN tasks t USING(id) WHERE t.id IS NULL
    )",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(mismatches, 0);
    assert!(
        !raw.prepare("PRAGMA foreign_key_check")
            .unwrap()
            .exists([])
            .unwrap()
    );
    assert_eq!(
        raw.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "ok"
    );
    assert_eq!(writer.task(1).unwrap().unwrap().payload, Some(vec![0, 255]));
    drop(writer);
    // SAFETY: the caller retains the same verified host-selected extension.
    let writer =
        unsafe { Writer::open_indexed(path, "consumer", "actor", extension, &model) }.unwrap();
    assert_eq!(writer.source(1).unwrap().unwrap(), final_source);
    let counts:(i64,i64,i64)=raw.query_row("SELECT (SELECT count(*) FROM tasks),(SELECT count(*) FROM lexical),(SELECT count(*) FROM vectors)",params![],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap();
    assert_eq!(counts, (2, 2, 2));
    assert_eq!(lexical(&raw, "beta"), vec![2]);
    println!(
        "Independent consumer: snapshot/update/claim/retry/delete/undo/vector-only/reopen and four corruption controls passed."
    );
}
