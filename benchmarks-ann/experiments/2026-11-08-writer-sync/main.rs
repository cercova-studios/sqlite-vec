use rusqlite::Connection;
use serde_json::{Value, json};
use sqlite_vec_write_prototype::{Command, Error, Mutation, Writer};
use std::{collections::BTreeMap, path::Path, sync::atomic::AtomicBool, time::Instant};

fn timed<T>(
    samples: &mut BTreeMap<&'static str, Vec<u64>>,
    name: &'static str,
    f: impl FnOnce() -> T,
) -> T {
    let start = Instant::now();
    let value = f();
    samples
        .entry(name)
        .or_default()
        .push(start.elapsed().as_nanos().try_into().unwrap());
    value
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let path = Path::new(&args[1]);
    let batch: i64 = args[2].parse().unwrap();
    let payload: usize = args[3].parse().unwrap();
    let rounds: i64 = args[4].parse().unwrap();
    let cancel = AtomicBool::new(false);
    let mut samples = BTreeMap::new();
    let mut writer = timed(&mut samples, "create_fixture", || {
        Writer::create(path, "bench", "actor").unwrap()
    });
    let raw = Connection::open(path).unwrap();
    let mut expected_sequence = 0;
    for round in 0..rounds {
        let ids: Vec<_> = (round * batch + 1..=(round + 1) * batch).collect();
        let create = Command::Batch(
            ids.iter()
                .map(|id| Mutation::Create {
                    id: *id,
                    title: format!("task {id} 🦀"),
                    payload: Some(vec![0xa5; payload]),
                })
                .collect(),
        );
        let created = timed(&mut samples, "create", || {
            writer
                .apply(&format!("new-{round}"), &create, &cancel)
                .unwrap()
        });
        expected_sequence += 1;
        assert_eq!(created.revision, expected_sequence);
        let claim = Command::Batch(
            ids.iter()
                .map(|id| Mutation::Claim {
                    id: *id,
                    expected: created.revision,
                })
                .collect(),
        );
        let request = format!("claim-{round}");
        let claimed = timed(&mut samples, "claim", || {
            writer.apply(&request, &claim, &cancel).unwrap()
        });
        expected_sequence += 1;
        assert_eq!(claimed.revision, expected_sequence);
        assert_eq!(
            timed(&mut samples, "retry", || writer
                .apply(&request, &claim, &cancel)
                .unwrap()),
            claimed
        );
        assert!(matches!(
            timed(&mut samples, "stale_conflict", || writer.apply(
                &format!("stale-{round}"),
                &claim,
                &cancel
            )),
            Err(Error::Conflict)
        ));
        let undo = Command::Undo { request };
        let undone = timed(&mut samples, "undo_claim", || {
            writer
                .apply(&format!("undo-claim-{round}"), &undo, &cancel)
                .unwrap()
        });
        expected_sequence += 1;
        assert_eq!(undone.revision, expected_sequence);
        let delete = Command::Batch(
            ids.iter()
                .map(|id| Mutation::Delete {
                    id: *id,
                    expected: undone.revision,
                })
                .collect(),
        );
        let request = format!("delete-{round}");
        timed(&mut samples, "delete", || {
            writer.apply(&request, &delete, &cancel).unwrap()
        });
        expected_sequence += 1;
        let undo = Command::Undo { request };
        let restored = timed(&mut samples, "undo_delete", || {
            writer
                .apply(&format!("restore-{round}"), &undo, &cancel)
                .unwrap()
        });
        expected_sequence += 1;
        assert_eq!(restored.revision, expected_sequence);
        let history = timed(&mut samples, "history_100", || {
            writer.history(ids[0], 100).unwrap()
        });
        assert_eq!(history.len(), 5);
        // The independent SQL oracle is outside timed public calls.
        let snapshot = raw.unchecked_transaction().unwrap();
        for id in &ids {
            let (title, owner, data, revision, live): (String, Option<String>, Vec<u8>, i64, bool) = snapshot.query_row(
                "SELECT title,owner,payload,revision,live FROM tasks JOIN versions USING(id) WHERE id=?", [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))).unwrap();
            assert_eq!(title, format!("task {id} 🦀"));
            assert!(owner.is_none() && live);
            assert_eq!(data, vec![0xa5; payload]);
            assert_eq!(revision, restored.revision);
        }
        let (sequence, requests, changes): (i64, i64, i64) = snapshot.query_row(
            "SELECT (SELECT sequence FROM meta),(SELECT count(*) FROM requests),(SELECT count(*) FROM changes)", [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).unwrap();
        assert_eq!(
            (sequence, requests, changes),
            (
                expected_sequence,
                expected_sequence,
                expected_sequence * batch
            )
        );
        snapshot.rollback().unwrap();
    }
    let mut history_growth = Vec::new();
    let mut revision = writer.task(1).unwrap().unwrap().revision;
    for round in 0..500 {
        let request = format!("growth-claim-{round}");
        writer
            .apply(
                &request,
                &Command::Batch(vec![Mutation::Claim {
                    id: 1,
                    expected: revision,
                }]),
                &cancel,
            )
            .unwrap();
        revision = writer
            .apply(
                &format!("growth-undo-{round}"),
                &Command::Undo { request },
                &cancel,
            )
            .unwrap()
            .revision;
        if [0, 49, 499].contains(&round) {
            let mut times = Vec::new();
            for _ in 0..30 {
                let start = Instant::now();
                let rows = writer.history(1, 100).unwrap();
                times.push(start.elapsed().as_nanos() as u64);
                assert_eq!(rows.len(), (7 + round as usize * 2).min(100));
                assert_eq!(rows[0].revision, revision);
            }
            history_growth
                .push(json!({"row_history_entries": 7 + round * 2, "limit":100,"ns": times}));
        }
    }
    let (capture_bytes, request_bytes, receipt_bytes, history_rows): (i64,i64,i64,i64) = raw.query_row(
        "SELECT sum(length(capture)),sum(length(command)),sum(length(receipt)),count(*) FROM requests", [],
        |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).unwrap();
    assert_eq!(
        raw.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "ok"
    );
    assert!(
        !raw.prepare("PRAGMA foreign_key_check")
            .unwrap()
            .exists([])
            .unwrap()
    );
    let sqlite: String = raw
        .query_row("SELECT sqlite_version()", [], |r| r.get(0))
        .unwrap();
    let pragmas: Vec<Value> = [
        "journal_mode",
        "synchronous",
        "page_size",
        "wal_autocheckpoint",
    ]
    .iter()
    .map(|name| {
        let value: rusqlite::types::Value = raw
            .query_row(&format!("PRAGMA {name}"), [], |r| r.get(0))
            .unwrap();
        let value = match value {
            rusqlite::types::Value::Text(value) => value,
            rusqlite::types::Value::Integer(value) => value.to_string(),
            other => panic!("unexpected PRAGMA value: {other:?}"),
        };
        json!({"name":name,"value":value})
    })
    .collect();
    let disk = |suffix: &str| {
        std::fs::metadata(format!("{}{suffix}", path.display())).map_or(0, |m| m.len())
    };
    let open_bytes = json!({"db":disk(""),"wal":disk("-wal"),"shm":disk("-shm")});
    drop(raw);
    drop(writer);
    let writer = timed(&mut samples, "reopen", || {
        Writer::open(path, "bench", "actor").unwrap()
    });
    assert_eq!(writer.task(1).unwrap().unwrap().revision, revision);
    println!(
        "{}",
        json!({"batch":batch,"payload_bytes":payload,"rounds":rounds,"sqlite":sqlite,"pragmas":pragmas,
        "samples_ns":samples,"history_growth":history_growth,"capture_bytes":capture_bytes,"canonical_request_bytes":request_bytes,
        "receipt_bytes":receipt_bytes,"history_rows":history_rows,"open_files_bytes":open_bytes,"reopened_db_bytes":disk("") ,"oracle":"passed"})
    );
}
