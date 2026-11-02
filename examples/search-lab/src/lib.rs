use regex::Regex;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use rusqlite::{Connection, LoadExtensionGuard, OpenFlags, OptionalExtension, ToSql, named_params};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    io::{self, BufRead, Write},
    path::Path,
    time::{Duration, Instant},
};

pub const OPERATIONS: [&str; 6] = [
    "describe", "search", "traverse", "fetch", "reach", "witness",
];
pub const INPUT_BYTES: usize = 65536;
pub const OUTPUT_BYTES: usize = 262144;
const DATABASE_BYTES: u64 = 67108864;

pub fn limits() -> Value {
    json!({"input_bytes":INPUT_BYTES,"output_bytes":OUTPUT_BYTES,"candidates":4096,
        "rows":100,"ids":64,"depth":8,"graph_edges":20000,"corpus_rows":10000,
        "dimensions":4096,"database_bytes":DATABASE_BYTES})
}

#[derive(Debug)]
pub struct Rejected {
    pub code: &'static str,
    pub message: String,
}
type Result<T> = std::result::Result<T, Rejected>;
fn reject(code: &'static str, message: impl Into<String>) -> Rejected {
    Rejected {
        code,
        message: message.into(),
    }
}
impl From<rusqlite::Error> for Rejected {
    fn from(e: rusqlite::Error) -> Self {
        let code = match e.sqlite_error_code() {
            Some(rusqlite::ErrorCode::OperationInterrupted) => "cancelled",
            Some(rusqlite::ErrorCode::DiskFull) if e.to_string().contains("graph_witness:") => {
                "work_limit"
            }
            Some(rusqlite::ErrorCode::TooBig) if e.to_string().contains("graph_witness:") => {
                "memory_limit"
            }
            _ => "database_error",
        };
        reject(code, e.to_string())
    }
}
impl From<io::Error> for Rejected {
    fn from(e: io::Error) -> Self {
        reject("startup_error", e.to_string())
    }
}
pub fn error(code: &str, message: impl Into<String>) -> Value {
    json!({"ok":false,"error":{"code":code,"message":message.into()}})
}

// Count non-ASCII strings as escaped UTF-16, including surrogate pairs, so
// changing host language does not loosen Unicode admission bounds.
fn canonical_size(v: &Value) -> usize {
    v.to_string()
        .chars()
        .map(|c| {
            if c.is_ascii() {
                1
            } else if c as u32 <= 0xffff {
                6
            } else {
                12
            }
        })
        .sum()
}
fn object<'a>(value: &'a Value, message: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| reject("invalid_input", message))
}
fn only(a: &Map<String, Value>, fields: &[&str], message: &str) -> Result<()> {
    if a.keys().any(|k| !fields.contains(&k.as_str())) {
        return Err(reject("invalid_input", message));
    }
    Ok(())
}
fn integer(v: &Value, name: &str, min: i64, max: i64) -> Result<i64> {
    v.as_i64()
        .filter(|n| *n >= min && *n <= max)
        .ok_or_else(|| {
            reject(
                "invalid_input",
                format!("{name} must be an integer in [{min}, {max}]"),
            )
        })
}
fn bounded(a: &Map<String, Value>, name: &str, default: i64, min: i64, max: i64) -> Result<i64> {
    integer(a.get(name).unwrap_or(&json!(default)), name, min, max)
}
fn ids(v: &Value) -> Result<Vec<i64>> {
    let values = v
        .as_array()
        .filter(|a| !a.is_empty() && a.len() <= 64)
        .ok_or_else(|| {
            reject(
                "invalid_input",
                "ids must contain 1 to 64 unique signed 64-bit integers",
            )
        })?;
    let keys = values
        .iter()
        .map(|v| integer(v, "id", i64::MIN, i64::MAX))
        .collect::<Result<Vec<_>>>()?;
    if keys.iter().collect::<HashSet<_>>().len() != keys.len() {
        return Err(reject("invalid_input", "ids must be unique"));
    }
    Ok(keys)
}
fn identifier(name: &str) -> Result<String> {
    if name.is_empty()
        || !name
            .bytes()
            .enumerate()
            .all(|(i, b)| b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit()))
    {
        return Err(reject(
            "configuration",
            "mapping names must be simple SQL identifiers",
        ));
    }
    Ok(format!("\"{name}\""))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    database: String,
    extension: String,
    corpora: Map<String, Value>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    documents: String,
    vectors: String,
    fts: String,
    edges: String,
    tenant: String,
    as_of: i64,
    dimensions: usize,
    #[serde(default)]
    graph: Option<GraphBinding>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphBinding {
    node_type: String,
    edge_type: String,
    #[serde(default)]
    edge_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReachQuery {
    version: String,
    node_type: String,
    edge_type: String,
    direction: String,
    anchors: Vec<i64>,
    max_depth: i64,
    limit: i64,
}
impl Corpus {
    fn tables(&self) -> [&str; 4] {
        [&self.documents, &self.vectors, &self.fts, &self.edges]
    }
    fn eligible(&self) -> String {
        format!(
            "select id from \"{}\" where tenant=:tenant and invalidated_at is null and valid_from<=:as_of and (valid_to is null or valid_to>:as_of)",
            self.documents
        )
    }
}

fn authorizer(ctx: AuthContext<'_>) -> Authorization {
    use AuthAction::*;
    match ctx.action {
        Select | Read { .. } | Transaction { .. } | Recursive => Authorization::Allow,
        Function { function_name }
            if !["load_extension", "readfile", "writefile"]
                .iter()
                .any(|n| function_name.eq_ignore_ascii_case(n)) =>
        {
            Authorization::Allow
        }
        Pragma { pragma_name, .. }
            if [
                "table_xinfo",
                "table_info",
                "data_version",
                "page_count",
                "page_size",
            ]
            .contains(&pragma_name) =>
        {
            Authorization::Allow
        }
        _ => Authorization::Deny,
    }
}

pub struct Adapter {
    db: Connection,
    corpora: Vec<(String, Corpus)>,
    extension_version: String,
}
impl Adapter {
    pub fn tools(&self) -> Vec<rmcp::model::Tool> {
        use rmcp::model::{Tool, ToolAnnotations};
        let properties: Value = serde_json::from_str(include_str!("tool-arguments.json"))
            .expect("embedded tool schemas");
        OPERATIONS.iter().map(|op| {
            let schema=json!({"type":"object","properties":{
                "corpus":{"type":"string","enum":self.corpora.iter().map(|(n,_)|n).collect::<Vec<_>>()},
                "args":{"type":"object","properties":properties[op],"required":if ["traverse","fetch"].contains(op) {vec!["ids"]} else if *op=="reach" {vec!["version","node_type","edge_type","direction","anchors","max_depth","limit"]} else if *op=="witness" {vec!["version","node_type","edge_type","direction","source","target","work","bytes"]} else {vec![]},"additionalProperties":false}},
                "required":["corpus"],"additionalProperties":false});
            let mut annotations=ToolAnnotations::default();
            annotations.read_only_hint=Some(true);annotations.open_world_hint=Some(false);
            Tool::new(*op,format!("Read-only {op} in the configured retrieval toolset; use describe for limits."),schema.as_object().unwrap().clone())
                .with_annotations(annotations)
        }).collect()
    }

    pub fn open(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let config: Config = serde_json::from_slice(&std::fs::read(&path)?)
            .map_err(|e| reject("configuration", e.to_string()))?;
        let parent = path.parent().expect("canonical file has parent");
        let database = parent.join(config.database).canonicalize()?;
        let extension = parent.join(config.extension).canonicalize()?;
        if database.metadata()?.len() > DATABASE_BYTES {
            return Err(reject(
                "workload_limit",
                "database exceeds 64 MiB admission limit",
            ));
        }
        if config.corpora.is_empty() || config.corpora.len() > 4 {
            return Err(reject("configuration", "configure 1 to 4 corpora"));
        }
        let mut corpora = Vec::new();
        for (name, value) in config.corpora {
            identifier(&name)?;
            let c: Corpus = serde_json::from_value(value)
                .map_err(|e| reject("configuration", e.to_string()))?;
            for table in c.tables() {
                identifier(table)?;
            }
            if let Some(graph) = &c.graph {
                identifier(&graph.node_type)?;
                identifier(&graph.edge_type)?;
                if let Some(id) = &graph.edge_id {
                    identifier(id)?;
                }
            }
            if c.tenant.is_empty() || c.tenant.chars().count() > 128 {
                return Err(reject(
                    "configuration",
                    "tenant must contain 1 to 128 characters",
                ));
            }
            if !(1..=4096).contains(&c.dimensions) {
                return Err(reject(
                    "configuration",
                    "dimensions must be an integer in [1, 4096]",
                ));
            }
            corpora.push((name, c));
        }
        let mut db = Connection::open_with_flags(
            database,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        db.busy_timeout(Duration::from_secs(1))?;
        // SAFETY: only the trusted host config supplies this executable library.
        // The guard disables loading on every exit, including a load failure.
        unsafe {
            let _guard = LoadExtensionGuard::new(&db)?;
            db.load_extension(extension, None::<&str>)?;
        }
        db.execute_batch("pragma query_only=on")?;
        for (_, c) in &corpora {
            for name in c.tables() {
                db.prepare(&format!("select * from {} limit 0", identifier(name)?))?
                    .query([])?
                    .next()?;
            }
            let ddl: String = db.query_row(
                "select sql from sqlite_schema where name=?",
                [&c.vectors],
                |r| r.get(0),
            )?;
            // shortcut: canonical single-column flat/L2 schema only; expand when
            // other metrics/indexes have their own parity coverage.
            let pattern = format!(
                r"(?i)using\s+vec0\s*\(\s*embedding\s+float\s*\[\s*{}\s*\]\s*\)\s*$",
                c.dimensions
            );
            if !Regex::new(&pattern)
                .expect("constant schema pattern")
                .is_match(&ddl)
            {
                return Err(reject(
                    "configuration",
                    "vectors must be vec0(embedding float[dimensions]) without options",
                ));
            }
        }
        db.authorizer(Some(authorizer))?;
        let extension_version = db.query_row("select vec_version()", [], |r| r.get(0))?;
        Ok(Self {
            db,
            corpora,
            extension_version,
        })
    }

    fn rows(&self, sql: &str, params: &[(&str, &dyn ToSql)]) -> Result<Vec<Value>> {
        let mut statement = self.db.prepare(sql)?;
        let names = statement
            .column_names()
            .map(|v| v.to_string())
            .collect::<Vec<_>>();
        let mut cursor = statement.query(params)?;
        let mut rows = Vec::new();
        while let Some(row) = cursor.next()? {
            let mut item = Map::new();
            for (i, name) in names.iter().enumerate() {
                use rusqlite::types::ValueRef::*;
                let v = match row.get_ref(i)? {
                    Null => Value::Null,
                    Integer(v) => json!(v),
                    Real(v) => {
                        if !v.is_finite() {
                            return Err(reject(
                                "database_error",
                                "stored vectors produced a non-finite L2 distance",
                            ));
                        }
                        json!(v)
                    }
                    Text(v) => json!(
                        std::str::from_utf8(v)
                            .map_err(|_| reject("database_error", "database text is not UTF-8"))?
                    ),
                    Blob(_) => return Err(reject("database_error", "unexpected blob in evidence")),
                };
                item.insert(name.clone(), v);
            }
            rows.push(Value::Object(item));
        }
        Ok(rows)
    }

    pub fn interrupt_handle(&self) -> rusqlite::InterruptHandle {
        self.db.get_interrupt_handle()
    }

    pub fn call_cancellable(
        &mut self,
        request: &Value,
        cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Value {
        if let Err(e) = self.db.progress_handler(
            1,
            Some(move || cancelled.load(std::sync::atomic::Ordering::SeqCst)),
        ) {
            return error("database_error", e.to_string());
        }
        let result = self.call(request);
        if let Err(e) = self.db.progress_handler(0, None::<fn() -> bool>) {
            return error("database_error", e.to_string());
        }
        if !self.db.is_autocommit() {
            if let Err(e) = self.db.execute_batch("ROLLBACK") {
                return error(
                    "database_error",
                    format!("cancellation cleanup failed: {e}"),
                );
            }
        }
        result
    }

    pub fn call(&self, request: &Value) -> Value {
        let start = Instant::now();
        match self.run(request) {
            Ok(mut result) => {
                result["elapsed_ms"] = json!(start.elapsed().as_secs_f64() * 1000.0);
                if canonical_size(&result) > OUTPUT_BYTES {
                    error(
                        "output_limit",
                        "result exceeds 262144 encoded bytes; request fewer rows",
                    )
                } else {
                    result
                }
            }
            Err(e) => error(e.code, e.message),
        }
    }

    fn run(&self, request: &Value) -> Result<Value> {
        if canonical_size(request) > INPUT_BYTES {
            return Err(reject("input_limit", "request exceeds 65536 encoded bytes"));
        }
        let req = object(request, "request accepts only op, corpus and args")?;
        only(
            req,
            &["op", "corpus", "args"],
            "request accepts only op, corpus and args",
        )?;
        let op = req
            .get("op")
            .and_then(Value::as_str)
            .filter(|v| OPERATIONS.contains(v))
            .ok_or_else(|| {
                reject(
                    "unknown_operation",
                    "supported operations: describe, search, traverse, fetch, reach, witness",
                )
            })?;
        let empty = json!({});
        let args = object(req.get("args").unwrap_or(&empty), "args must be an object")?;
        let (name, c) = req
            .get("corpus")
            .and_then(Value::as_str)
            .and_then(|name| self.corpora.iter().find(|(n, _)| n == name))
            .ok_or_else(|| reject("unknown_corpus", "choose a host-configured corpus"))?;
        // An RAII transaction rolls back on every return, including query errors.
        let transaction = self.db.unchecked_transaction()?;
        let pages: i64 = transaction.query_row("pragma page_count", [], |r| r.get(0))?;
        let page_size: i64 = transaction.query_row("pragma page_size", [], |r| r.get(0))?;
        if pages as u64 * page_size as u64 > DATABASE_BYTES {
            return Err(reject(
                "workload_limit",
                "database exceeds 64 MiB logical page limit",
            ));
        }
        for (field, table) in ["documents", "vectors", "fts"].iter().zip(c.tables()) {
            let count: i64 = transaction.query_row(
                &format!("select count(*) from (select 1 from \"{table}\" limit 10001)"),
                [],
                |r| r.get(0),
            )?;
            if count > 10000 {
                return Err(reject(
                    "workload_limit",
                    format!("{field} exceeds 10000 rows"),
                ));
            }
        }
        let mut result = match op {
            "describe" => self.describe(c, args)?,
            "search" => self.search(c, args)?,
            "fetch" => self.fetch(c, args)?,
            "traverse" => self.traverse(c, args)?,
            "reach" => self.reach(c, args)?,
            "witness" => self.witness(c, args)?,
            _ => unreachable!(),
        };
        transaction.rollback()?;
        result["ok"] = json!(true);
        result["corpus"] = json!(name);
        result["scope"] = json!({"tenant":c.tenant,"as_of":c.as_of});
        result["operation"] = json!(op);
        result["termination"] = json!(if result["truncated"].as_bool().unwrap_or(false) {
            "row_limit"
        } else {
            "complete"
        });
        result["consistency"] = json!("one_read_transaction_per_operation");
        Ok(result)
    }

    fn describe(&self, c: &Corpus, a: &Map<String, Value>) -> Result<Value> {
        if !a.is_empty() {
            return Err(reject("invalid_input", "describe takes no arguments"));
        }
        Ok(
            json!({"operations":OPERATIONS,"toolset":"retrieval","sqlite_version":rusqlite::version(),
            "extension_version":self.extension_version,"dimensions":c.dimensions,"metric":"l2",
            "available_corpora":self.corpora.iter().map(|(n,_)|n).collect::<Vec<_>>(),
            "mappings":{"documents":c.documents,"vectors":c.vectors,"fts":c.fts,"edges":c.edges},
            "graph_types":c.graph.as_ref().map(|g|json!({"node_type":g.node_type,"edge_type":g.edge_type})),
            "scope_is_authorization":false,"id_domain":"corpus + signed_int64_document_id","limits":limits(),
            "approximation":false,"score_directions":{"l2":"lower","bm25":"lower","rrf":"higher"},
            "traversal":"outgoing distinct nodes with minimum hops; anchor included if eligible",
            "cancellation":false,"native_work_budget":false,"native_memory_budget":false,
            "witness_limits":{"work_max":1000000000,"bytes_max":67108864,"native_work_budget":true,"native_memory_budget":true,"cancellation":true},
            "continuation":false,"edge_witnesses":c.graph.as_ref().is_some_and(|g|g.edge_id.is_some()),"cross_call_snapshot":false}),
        )
    }

    fn vector(&self, c: &Corpus, a: &Map<String, Value>) -> Result<Vec<u8>> {
        if a.contains_key("vector") == a.contains_key("embedding_id") {
            return Err(reject(
                "invalid_input",
                "supply exactly one of vector or embedding_id",
            ));
        }
        let values: Vec<Value> = if let Some(value) = a.get("embedding_id") {
            let id = integer(value, "embedding_id", i64::MIN, i64::MAX)?;
            let data: Option<Vec<u8>> = self
                .db
                .query_row(
                    &format!(
                        "select embedding from \"{}\" where rowid=:id and rowid in ({})",
                        c.vectors,
                        c.eligible()
                    ),
                    named_params! {":id":id,":tenant":c.tenant,":as_of":c.as_of},
                    |r| r.get(0),
                )
                .optional()?;
            let data = data.ok_or_else(|| {
                reject("missing_embedding", "embedding ID is absent or ineligible")
            })?;
            if data.len() != c.dimensions * 4 {
                return Err(reject(
                    "database_error",
                    "stored vector has wrong dimensions",
                ));
            }
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|b| json!(f32::from_ne_bytes(*b)))
                .collect()
        } else {
            a["vector"]
                .as_array()
                .filter(|v| v.len() == c.dimensions)
                .cloned()
                .ok_or_else(|| {
                    reject(
                        "invalid_vector",
                        "vector length must match configured dimensions",
                    )
                })?
        };
        let mut bytes = Vec::with_capacity(c.dimensions * 4);
        for v in values {
            let number = v
                .as_f64()
                .filter(|n| n.is_finite() && n.abs() <= 1e10)
                .ok_or_else(|| {
                    reject(
                        "invalid_vector",
                        "vector coordinates must be finite numbers with magnitude <= 1e10",
                    )
                })?;
            bytes.extend_from_slice(&(number as f32).to_ne_bytes());
        }
        Ok(bytes)
    }

    fn search(&self, c: &Corpus, a: &Map<String, Value>) -> Result<Value> {
        only(
            a,
            &[
                "mode",
                "text",
                "vector",
                "embedding_id",
                "candidates",
                "limit",
            ],
            "unknown search argument",
        )?;
        let mode = a.get("mode").unwrap_or(&Value::Null);
        let mode = if !a.contains_key("mode") {
            "hybrid"
        } else {
            mode.as_str().unwrap_or("")
        };
        if !["lexical", "dense", "hybrid"].contains(&mode) {
            return Err(reject(
                "invalid_input",
                "mode must be lexical, dense or hybrid",
            ));
        }
        let cap = bounded(a, "candidates", 50, 1, 4096)?;
        let limit = bounded(a, "limit", 10, 1, 100)? as usize;
        let mut branches = Vec::new();
        if mode != "lexical" {
            let vector = self.vector(c, a)?;
            let sql = format!(
                "with eligible as materialized ({}) select rowid id,vec_distance_l2(embedding,:vector) distance from \"{}\" where rowid in (select id from eligible) order by distance,rowid limit :cap",
                c.eligible(),
                c.vectors
            );
            let rows = self.rows(
                &sql,
                named_params! {":tenant":c.tenant,":as_of":c.as_of,":vector":vector,":cap":cap},
            )?;
            if rows.iter().any(|r| r["distance"].is_null()) {
                return Err(reject(
                    "database_error",
                    "stored vectors produced a non-finite L2 distance",
                ));
            }
            branches.push(("vector", rows));
        } else if a.contains_key("vector") || a.contains_key("embedding_id") {
            return Err(reject(
                "invalid_input",
                "lexical mode does not accept a vector",
            ));
        }
        if mode != "dense" {
            let text = a
                .get("text")
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty() && t.chars().count() <= 2048)
                .ok_or_else(|| {
                    reject(
                        "invalid_input",
                        "text must contain 1 to 2048 characters of FTS5 query syntax",
                    )
                })?;
            let sql = format!(
                "with eligible as materialized ({}) select rowid id,rank from \"{}\" where \"{}\" match :text and exists (select 1 from eligible where eligible.id = \"{}\".rowid) order by rank,rowid limit :cap",
                c.eligible(),
                c.fts,
                c.fts,
                c.fts
            );
            branches.push((
                "keyword",
                self.rows(
                    &sql,
                    named_params! {":tenant":c.tenant,":as_of":c.as_of,":text":text,":cap":cap},
                )?,
            ));
        } else if a.contains_key("text") {
            return Err(reject("invalid_input", "dense mode does not accept text"));
        }
        let mut merged = BTreeMap::new();
        let mut info = Map::new();
        for (branch, rows) in &branches {
            info.insert(
                branch.to_string(),
                json!({"returned":rows.len(),"boundary_hit":rows.len()==cap as usize}),
            );
            for (rank, r) in rows.iter().enumerate() {
                let id = r["id"].as_i64().expect("SQLite rowid");
                let row=merged.entry(id).or_insert_with(||json!({"id":id,"vector_rank":null,"keyword_rank":null,"vector_distance":null,"keyword_bm25":null}));
                row[format!("{branch}_rank")] = json!(rank + 1);
                row[if *branch == "vector" {
                    "vector_distance"
                } else {
                    "keyword_bm25"
                }] = r[if *branch == "vector" {
                    "distance"
                } else {
                    "rank"
                }]
                .clone();
            }
        }
        let mut ordered = merged.into_values().collect::<Vec<_>>();
        for row in &mut ordered {
            let score = ["vector_rank", "keyword_rank"]
                .iter()
                .filter_map(|b| row[b].as_u64())
                .map(|r| 1.0 / (60 + r) as f64)
                .sum::<f64>();
            row["score"] = json!(score);
        }
        ordered.sort_by(|a, b| {
            b["score"]
                .as_f64()
                .unwrap()
                .total_cmp(&a["score"].as_f64().unwrap())
                .then_with(|| a["id"].as_i64().cmp(&b["id"].as_i64()))
        });
        let union = ordered.len();
        ordered.truncate(limit);
        for row in &mut ordered {
            let source = self.rows(
                &format!("select source_id from \"{}\" where id=:id", c.documents),
                named_params! {":id":row["id"].as_i64()},
            )?;
            row["source_id"] = source[0]["source_id"].clone();
        }
        Ok(
            json!({"returned":ordered.len(),"rows":ordered,"mode":mode,"requested":limit,"candidate_union":union,
            "candidates_requested":cap,"branches":info,"truncated":union>limit,"complete":union<=limit,
            "approximation":false,"completeness_domain":"bounded_candidate_union","score":"rrf_60_higher_is_better",
            "dense_strategy":"exhaustive_scalar_l2","tie_policy":"score desc,id asc; branch score asc,id asc"}),
        )
    }

    fn fetch(&self, c: &Corpus, a: &Map<String, Value>) -> Result<Value> {
        if a.len() != 1 || !a.contains_key("ids") {
            return Err(reject("invalid_input", "fetch requires ids only"));
        }
        let keys = ids(&a["ids"])?;
        let mut rows = Vec::new();
        let mut missing = Vec::new();
        for key in &keys {
            let found=self.rows(&format!("select id,document_id,body,source_id,start_ms,end_ms from \"{}\" where id=:id and id in ({})",c.documents,c.eligible()),named_params!{":id":key,":tenant":c.tenant,":as_of":c.as_of})?;
            if found.is_empty() {
                missing.push(*key);
            } else {
                rows.extend(found);
            }
        }
        Ok(
            json!({"returned":rows.len(),"rows":rows,"missing_ids":missing,"missing_reason":"absent_or_ineligible","requested":keys.len(),"truncated":false,"complete":true}),
        )
    }

    fn witness(&self, c: &Corpus, a: &Map<String, Value>) -> Result<Value> {
        only(
            a,
            &[
                "version",
                "node_type",
                "edge_type",
                "direction",
                "source",
                "target",
                "work",
                "bytes",
            ],
            "unknown witness argument",
        )?;
        for key in [
            "version",
            "node_type",
            "edge_type",
            "direction",
            "source",
            "target",
            "work",
            "bytes",
        ] {
            if !a.contains_key(key) {
                return Err(reject(
                    "invalid_input",
                    "witness requires all eight arguments",
                ));
            }
        }
        if a["version"] != "g2/1" {
            return Err(reject("unsupported", "unknown witness version"));
        }
        let source = integer(&a["source"], "source", i64::MIN, i64::MAX)?;
        let target = integer(&a["target"], "target", i64::MIN, i64::MAX)?;
        let work = integer(&a["work"], "work", 1, 1000000000)?;
        let bytes = integer(&a["bytes"], "bytes", 128, 67108864)?;
        let edge_id = c
            .graph
            .as_ref()
            .and_then(|g| g.edge_id.as_ref())
            .ok_or_else(|| {
                reject(
                    "unsupported",
                    "witness requires a configured stable edge_id column",
                )
            })?;
        // Reuse the typed binder and scope admission in this read transaction.
        let anchors = if source == target {
            vec![source]
        } else {
            vec![source, target]
        };
        let input = json!({"version":"g1/1","node_type":a["node_type"],"edge_type":a["edge_type"],"direction":a["direction"],"anchors":anchors,"max_depth":0,"limit":100});
        let admission = self.reach(c, input.as_object().unwrap())?;
        let columns = self.rows(
            "select name from pragma_table_xinfo(:table)",
            named_params! {":table":c.edges},
        )?;
        if !columns.iter().any(|r| r["name"] == *edge_id) {
            return Err(reject(
                "schema_error",
                "configured edge_id column is missing",
            ));
        }
        let missing = &admission["missing_ids"];
        let path: Value = if missing.as_array().is_some_and(|a| !a.is_empty()) {
            json!([])
        } else {
            let text: String = self.db.query_row(
                "select graph_witness(?1,'src','dst',?2,?3,?4,?5,?6)",
                rusqlite::params![c.edges, edge_id, source, target, work, bytes],
                |r| r.get(0),
            )?;
            serde_json::from_str(&text)
                .map_err(|_| reject("database_error", "invalid native witness JSON"))?
        };
        Ok(
            json!({"version":"g2/1","node_type":a["node_type"],"edge_type":a["edge_type"],"source":source,"target":target,
            "path":path,"missing_ids":missing,"complete":true,"truncated":false,"native_work_budget":true,"native_memory_budget":true,"cancellation":true,
            "work_limit":work,"memory_limit":bytes,"identity_domain":"configured corpus + edge ID"}),
        )
    }

    fn reach(&self, c: &Corpus, a: &Map<String, Value>) -> Result<Value> {
        let query: ReachQuery = serde_json::from_value(Value::Object(a.clone()))
            .map_err(|_| reject("invalid_input", "reach requires a valid g1/1 query"))?;
        let graph = c
            .graph
            .as_ref()
            .ok_or_else(|| reject("unsupported", "corpus has no configured graph types"))?;
        if query.version != "g1/1"
            || query.direction != "out"
            || query.node_type != graph.node_type
            || query.edge_type != graph.edge_type
        {
            return Err(reject("unsupported", "unknown version, type or direction"));
        }
        let columns = self.rows(
            "select name,type,pk from pragma_table_xinfo(:table)",
            named_params! {":table":c.documents},
        )?;
        let id = columns.iter().find(|r| r["name"] == "id");
        if !id.is_some_and(|r| {
            r["type"]
                .as_str()
                .is_some_and(|t| t.eq_ignore_ascii_case("INTEGER"))
                && r["pk"] == 1
        }) || columns
            .iter()
            .filter(|r| r["pk"].as_i64().unwrap_or(0) > 0)
            .count()
            != 1
            || ["tenant", "valid_from", "valid_to", "invalidated_at"]
                .iter()
                .any(|n| !columns.iter().any(|r| r["name"] == *n))
        {
            return Err(reject(
                "schema_error",
                "node mapping needs an INTEGER PRIMARY KEY id and eligibility columns",
            ));
        }
        let edges = self.rows(
            "select name from pragma_table_xinfo(:table)",
            named_params! {":table":c.edges},
        )?;
        if ["src", "dst"]
            .iter()
            .any(|n| !edges.iter().any(|r| r["name"] == *n))
        {
            return Err(reject(
                "schema_error",
                "edge mapping needs src and dst columns",
            ));
        }
        let input = json!({"ids":query.anchors,"depth":query.max_depth,"limit":query.limit});
        let mut result = self.traverse(c, input.as_object().expect("object"))?;
        result["version"] = json!(query.version);
        result["node_type"] = json!(graph.node_type);
        result["edge_type"] = json!(graph.edge_type);
        Ok(result)
    }

    fn traverse(&self, c: &Corpus, a: &Map<String, Value>) -> Result<Value> {
        only(a, &["ids", "depth", "limit"], "unknown traverse argument")?;
        let keys = ids(a.get("ids").unwrap_or(&Value::Null))?;
        let depth = bounded(a, "depth", 1, 0, 8)?;
        let limit = bounded(a, "limit", 50, 1, 100)? as usize;
        let n: i64 = self.db.query_row(
            &format!(
                "select count(*) from (select 1 from \"{}\" limit 20001)",
                c.edges
            ),
            [],
            |r| r.get(0),
        )?;
        if n > 20000 {
            return Err(reject("workload_limit", "graph source exceeds 20000 edges"));
        }
        let invalid=self.rows(&format!("with eligible as materialized ({}) select 1 from \"{}\" where typeof(src)!='integer' or typeof(dst)!='integer' or src not in (select id from eligible) or dst not in (select id from eligible) limit 1",c.eligible(),c.edges),named_params!{":tenant":c.tenant,":as_of":c.as_of})?;
        if !invalid.is_empty() {
            return Err(reject(
                "graph_scope",
                "configured edge source must contain only eligible integer endpoints",
            ));
        }
        let mut found: BTreeMap<i64, i64> = BTreeMap::new();
        let mut missing = Vec::new();
        for key in keys {
            let present = self.rows(
                &format!("select 1 from ({}) where id=:id", c.eligible()),
                named_params! {":tenant":c.tenant,":as_of":c.as_of,":id":key},
            )?;
            if present.is_empty() {
                missing.push(key);
                continue;
            }
            if depth == 0 {
                found.insert(key, 0);
                continue;
            }
            let rows = self.rows(
                "select node,depth from graph_bfs(:edges,'src','dst',:id,:depth)",
                named_params! {":edges":c.edges,":id":key,":depth":depth},
            )?;
            for row in rows {
                let node = row["node"].as_i64().expect("native node ID");
                let hops = row["depth"].as_i64().expect("native depth");
                found
                    .entry(node)
                    .and_modify(|d| *d = (*d).min(hops))
                    .or_insert(hops);
            }
        }
        let total = found.len();
        let mut ordered = found.into_iter().collect::<Vec<_>>();
        ordered.sort_by_key(|(n, d)| (*d, *n));
        let rows = ordered
            .into_iter()
            .take(limit)
            .map(|(id, depth)| json!({"id":id,"depth":depth}))
            .collect::<Vec<_>>();
        Ok(
            json!({"returned":rows.len(),"rows":rows,"missing_ids":missing,"missing_reason":"absent_or_ineligible","requested":limit,
            "total_reachable":total,"truncated":total>limit,"complete":total<=limit,"direction":"out","depth":depth,"graph_edges":n,
            "semantics":"distinct_nodes_minimum_hops_from_any_seed","native_work_budget":false,"native_memory_budget":false,"cancellation":false}),
        )
    }
}

pub fn cli(adapter: &Adapter, mut input: impl BufRead, mut output: impl Write) -> io::Result<i32> {
    let mut failed = false;
    loop {
        let mut record = Vec::new();
        let mut any = false;
        loop {
            let chunk = input.fill_buf()?;
            if chunk.is_empty() {
                break;
            }
            any = true;
            let end = chunk
                .iter()
                .position(|b| *b == b'\n')
                .map_or(chunk.len(), |i| i + 1);
            let take = end.min((INPUT_BYTES + 1).saturating_sub(record.len()));
            record.extend_from_slice(&chunk[..take]);
            let done = chunk[end - 1] == b'\n';
            input.consume(end);
            if done {
                break;
            }
        }
        if !any {
            break;
        }
        let result = if record.len() > INPUT_BYTES {
            error("input_limit", "record exceeds 65536 bytes")
        } else {
            match serde_json::from_slice::<Value>(&record) {
                Ok(value) => adapter.call(&value),
                Err(_) => error("invalid_json", "record must be one UTF-8 JSON object"),
            }
        };
        failed |= result["ok"] != true;
        serde_json::to_writer(&mut output, &result)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(i32::from(failed))
}

#[cfg(test)]
mod tests;
