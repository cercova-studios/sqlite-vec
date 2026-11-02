//! Compare the adapter's keyword predicates using its locked bundled SQLite.
use rusqlite::{Connection, StatementStatus, named_params};
use serde_json::{Value, json};
use std::{fs, time::Instant};

fn sql(exists: bool) -> String {
    let predicate = if exists {
        "exists (select 1 from eligible where eligible.id = \"lexical\".rowid)"
    } else {
        "rowid in (select id from eligible)"
    };
    format!(
        "with eligible as materialized (select id from \"documents\" where tenant=:tenant and invalidated_at is null and valid_from<=:as_of and (valid_to is null or valid_to>:as_of)) select rowid id,rank from \"lexical\" where \"lexical\" match :text and {predicate} order by rank,rowid limit :cap"
    )
}
fn execute(db: &Connection, sql: &str, text: &str) -> (Vec<(i64, f64)>, f64, i32) {
    let start = Instant::now();
    let mut stmt = db.prepare(sql).unwrap();
    let rows = stmt
        .query_map(
            named_params! {":tenant":"ours",":as_of":100,":text":text,":cap":50},
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    (
        rows,
        start.elapsed().as_secs_f64() * 1000.,
        stmt.get_status(StatementStatus::VmStep),
    )
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let input: Value = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch("create table documents(id integer primary key,tenant text,valid_from integer,valid_to integer,invalidated_at integer); create virtual table lexical using fts5(body); begin").unwrap();
    for d in input["documents"].as_array().unwrap() {
        db.execute(
            "insert into documents values (?1,'ours',0,null,null)",
            [d["id"].as_i64().unwrap()],
        )
        .unwrap();
        db.execute(
            "insert into lexical(rowid,body) values (?1,?2)",
            rusqlite::params![d["id"].as_i64().unwrap(), d["body"].as_str().unwrap()],
        )
        .unwrap();
    }
    db.execute_batch("commit").unwrap();
    let mut timings = vec![];
    let mut quality = vec![];
    let mut plans = vec![];
    for scope in ["full", "scoped"] {
        db.execute_batch("update documents set tenant='ours',valid_from=0,valid_to=null,invalidated_at=null; begin").unwrap();
        if scope == "scoped" {
            for d in input["documents"].as_array().unwrap() {
                let b = d["bucket"].as_i64().unwrap();
                db.execute("update documents set tenant=?1,valid_from=?2,valid_to=?3,invalidated_at=?4 where id=?5",rusqlite::params![if b==0 {"other"} else {"ours"},if b==2 {101} else {0},if b==1 {Some(100)} else {None},if b==3 {Some(99)} else {None},d["id"].as_i64().unwrap()]).unwrap();
            }
        }
        db.execute_batch("commit").unwrap();
        for q in input["queries"].as_array().unwrap() {
            let text = q["text"].as_str().unwrap();
            let (candidate, _, _) = execute(&db, &sql(true), text);
            let historical_ids = q["expected"][scope]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r[0].as_i64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                candidate.iter().map(|r| r.0).collect::<Vec<_>>(),
                historical_ids
            );
            let (baseline, _, _) = execute(&db, &sql(false), text);
            assert_eq!(
                baseline, candidate,
                "same-build scores and ordering {} {scope}",
                q["id"]
            );
            quality.push(json!({"scope":scope,"id":q["id"],"rows":candidate}));
            if !input["timed_ids"].as_array().unwrap().contains(&q["id"]) {
                continue;
            }
            for variant in [false, true] {
                assert_eq!(execute(&db, &sql(variant), text).0, candidate);
            }
            for round in 0..3 {
                for variant in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let (rows, ms, steps) = execute(&db, &sql(variant), text);
                    assert_eq!(rows, candidate);
                    timings.push(json!({"scope":scope,"id":q["id"],"round":round,"variant":if variant {"exists"} else {"in"},"ms":ms,"vm_steps":steps}));
                }
            }
        }
        for exists in [false, true] {
            let mut stmt = db
                .prepare(&format!("explain query plan {}", sql(exists)))
                .unwrap();
            let plan=stmt.query_map(named_params!{":tenant":"ours",":as_of":100,":text":"cell OR cancer",":cap":50},|r|r.get::<_,String>(3)).unwrap().collect::<Result<Vec<_>,_>>().unwrap();
            plans.push(json!({"scope":scope,"exists":exists,"plan":plan}));
        }
        eprintln!("{scope}: 300 reference checks and 24 timed pairs complete");
    }
    let mut controls = vec![];
    for count in [0, 1, 10, 100, 1000, 5183] {
        db.execute_batch(
            "update documents set tenant='other',valid_from=0,valid_to=null,invalidated_at=null",
        )
        .unwrap();
        db.execute("update documents set tenant='ours' where id in (select id from documents order by id limit ?1)",[count]).unwrap();
        for text in ["cell OR cancer", "\"T cells\"", "zznomatchzz"] {
            let expected = execute(&db, &sql(false), text).0;
            for round in 0..3 {
                for variant in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let (rows, ms, steps) = execute(&db, &sql(variant), text);
                    assert_eq!(rows, expected);
                    controls.push(json!({"eligible":count,"text":text,"round":round,"variant":if variant {"exists"} else {"in"},"ms":ms,"vm_steps":steps}));
                }
            }
        }
    }
    fs::write(&args[2],serde_json::to_vec(&json!({"sqlite":rusqlite::version(),"data_sha256":input["data_sha256"],"timed_ids":input["timed_ids"],"sql":{"in":sql(false),"exists":sql(true)},"quality":quality,"timings":timings,"controls":controls,"plans":plans})).unwrap()).unwrap();
}
