"""Read-only named retrieval operations. Both transports call Adapter.call."""

import argparse
import asyncio
import json
import math
from pathlib import Path
import re
import sqlite3
import struct
import sys
import time


LIMITS = dict(input_bytes=65536, output_bytes=262144, candidates=4096,
              rows=100, ids=64, depth=8, graph_edges=20000, corpus_rows=10000,
              dimensions=4096, database_bytes=67108864)
OPERATIONS = ("describe", "search", "traverse", "fetch")
ID_SCHEMA = {"type": "integer", "minimum": -(2**63), "maximum": 2**63-1}
IDS_SCHEMA = {"type": "array", "items": ID_SCHEMA, "minItems": 1, "maxItems": 64, "uniqueItems": True}
ROW_SCHEMA = {"type": "integer", "minimum": 1, "maximum": 100}
ARGUMENTS = {
    "describe": {},
    "search": {
        "mode": {"type": "string", "enum": ["lexical", "dense", "hybrid"], "default": "hybrid"},
        "text": {"type": "string", "minLength": 1, "maxLength": 2048, "description": "FTS5 query syntax"},
        "vector": {"type": "array", "items": {"type": "number", "minimum": -1e10, "maximum": 1e10},
                   "minItems": 1, "maxItems": 4096, "description": "Must match configured dimensions"},
        "embedding_id": ID_SCHEMA,
        "candidates": {"type": "integer", "minimum": 1, "maximum": 4096, "default": 50},
        "limit": dict(ROW_SCHEMA, default=10)},
    "traverse": {"ids": IDS_SCHEMA, "depth": {"type": "integer", "minimum": 0, "maximum": 8, "default": 1},
                 "limit": dict(ROW_SCHEMA, default=50)},
    "fetch": {"ids": IDS_SCHEMA},
}


class Rejected(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def error(code, message):
    return {"ok": False, "error": {"code": code, "message": message}}


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise Rejected("invalid_input", f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        raise Rejected("configuration", "mapping names must be simple SQL identifiers")
    return '"' + value + '"'


def ids(value):
    if not isinstance(value, list) or not 1 <= len(value) <= LIMITS["ids"]:
        raise Rejected("invalid_input", "ids must contain 1 to 64 unique signed 64-bit integers")
    result = [integer(v, "id", -(2**63), 2**63 - 1) for v in value]
    if len(set(result)) != len(result):
        raise Rejected("invalid_input", "ids must be unique")
    return result


def authorizer(action, arg1, arg2, database, source):
    # URI mode=ro protects disk; query_only and this allowlist also protect TEMP.
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                  sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and arg2 not in ("load_extension", "readfile", "writefile"):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and arg1 in ("table_xinfo", "table_info", "data_version", "page_count", "page_size"):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class Adapter:
    def __init__(self, config_path):
        self.db = None
        config_path = Path(config_path).resolve()
        config = json.loads(config_path.read_text())
        if set(config) != {"database", "extension", "corpora"}:
            raise Rejected("configuration", "config requires database, extension and corpora")
        self.config = config
        path = (config_path.parent / config["database"]).resolve(strict=True)
        extension = (config_path.parent / config["extension"]).resolve(strict=True)
        if path.stat().st_size > LIMITS["database_bytes"]:
            raise Rejected("workload_limit", "database exceeds 64 MiB admission limit")
        if not isinstance(config["corpora"], dict) or not 1 <= len(config["corpora"]) <= 4:
            raise Rejected("configuration", "configure 1 to 4 corpora")
        for name, c in config["corpora"].items():
            identifier(name)
            if set(c) != {"documents", "vectors", "fts", "edges", "tenant", "as_of", "dimensions"}:
                raise Rejected("configuration", "each corpus needs documents/vectors/fts/edges, tenant/as_of/dimensions")
            for field in ("documents", "vectors", "fts", "edges"):
                identifier(c[field])
            if not isinstance(c["tenant"], str) or not 1 <= len(c["tenant"]) <= 128:
                raise Rejected("configuration", "tenant must contain 1 to 128 characters")
            integer(c["as_of"], "as_of", -(2**63), 2**63-1)
            integer(c["dimensions"], "dimensions", 1, LIMITS["dimensions"])
        try:
            self.db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True,
                                      isolation_level=None, timeout=1)
            self.db.row_factory = sqlite3.Row
            self.db.enable_load_extension(True)
            try:
                self.db.load_extension(str(extension))
            finally:
                self.db.enable_load_extension(False)
            self.db.execute("pragma query_only=on")
            # Connect virtual tables before installing the authorizer: SQLite's
            # declaration of a virtual schema invokes internal schema callbacks.
            for c in config["corpora"].values():
                for field in ("documents", "vectors", "fts", "edges"):
                    self.db.execute(f"select * from {identifier(c[field])} limit 0")
                ddl = self.db.execute("select sql from sqlite_schema where name=?", (c["vectors"],)).fetchone()[0]
                # shortcut: admit only one canonical flat float/L2 column; extend
                # this check when another metric or index has parity coverage.
                if not re.search(r"using\s+vec0\s*\(\s*embedding\s+float\s*\[\s*" +
                                 str(c["dimensions"]) + r"\s*\]\s*\)\s*$", ddl, re.I):
                    raise Rejected("configuration", "vectors must be vec0(embedding float[dimensions]) without options")
            self.db.set_authorizer(authorizer)
            self.extension_version = self.db.execute("select vec_version()").fetchone()[0]
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def eligible(self, c):
        return (f"select id from {identifier(c['documents'])} where tenant=:tenant "
                "and invalidated_at is null and valid_from<=:as_of "
                "and (valid_to is null or valid_to>:as_of)")

    def params(self, c):
        return {"tenant": c["tenant"], "as_of": c["as_of"]}

    def call(self, request):
        start = time.perf_counter()
        try:
            if len(encoded(request)) > LIMITS["input_bytes"]:
                raise Rejected("input_limit", "request exceeds 65536 encoded bytes")
            if not isinstance(request, dict) or set(request) - {"op", "corpus", "args"}:
                raise Rejected("invalid_input", "request accepts only op, corpus and args")
            op = request.get("op")
            if not isinstance(op, str) or op not in OPERATIONS:
                raise Rejected("unknown_operation", "supported operations: describe, search, traverse, fetch")
            args = request.get("args", {})
            if not isinstance(args, dict):
                raise Rejected("invalid_input", "args must be an object")
            name = request.get("corpus")
            if not isinstance(name, str) or name not in self.config["corpora"]:
                raise Rejected("unknown_corpus", "choose a host-configured corpus")
            c = self.config["corpora"][name]
            self.db.execute("begin")
            try:
                # All statements of one operation share this read transaction.
                pages = self.db.execute("pragma page_count").fetchone()[0]
                page_size = self.db.execute("pragma page_size").fetchone()[0]
                if pages * page_size > LIMITS["database_bytes"]:
                    raise Rejected("workload_limit", "database exceeds 64 MiB logical page limit")
                for field in ("documents", "vectors", "fts"):
                    n = self.db.execute(f"select count(*) from (select 1 from {identifier(c[field])} limit 10001)").fetchone()[0]
                    if n > LIMITS["corpus_rows"]:
                        raise Rejected("workload_limit", f"{field} exceeds 10000 rows")
                result = getattr(self, op)(c, args)
            finally:
                self.db.execute("rollback")
            result.update(ok=True, corpus=name, scope={"tenant": c["tenant"], "as_of": c["as_of"]},
                          operation=op, termination="row_limit" if result.get("truncated") else "complete",
                          consistency="one_read_transaction_per_operation",
                          elapsed_ms=(time.perf_counter()-start)*1000)
            if len(encoded(result)) > LIMITS["output_bytes"]:
                raise Rejected("output_limit", "result exceeds 262144 encoded bytes; request fewer rows")
            return result
        except Rejected as exc:
            return error(exc.code, exc.message)
        except sqlite3.Error as exc:
            return error("database_error", str(exc))
        except (ValueError, TypeError, OverflowError, RecursionError):
            return error("invalid_input", "request must contain finite JSON values")

    def describe(self, c, a):
        if a:
            raise Rejected("invalid_input", "describe takes no arguments")
        return dict(operations=list(OPERATIONS), toolset="retrieval", sqlite_version=sqlite3.sqlite_version,
                    extension_version=self.extension_version, dimensions=c["dimensions"], metric="l2",
                    available_corpora=list(self.config["corpora"]),
                    mappings={field:c[field] for field in ("documents","vectors","fts","edges")},
                    scope_is_authorization=False,
                    id_domain="corpus + signed_int64_document_id", limits=LIMITS,
                    approximation=False, score_directions={"l2": "lower", "bm25": "lower", "rrf": "higher"},
                    traversal="outgoing distinct nodes with minimum hops; anchor included if eligible",
                    cancellation=False, native_work_budget=False, native_memory_budget=False,
                    continuation=False, edge_witnesses=False, cross_call_snapshot=False)

    def vector(self, c, a):
        if ("vector" in a) == ("embedding_id" in a):
            raise Rejected("invalid_input", "supply exactly one of vector or embedding_id")
        if "embedding_id" in a:
            key = integer(a["embedding_id"], "embedding_id", -(2**63), 2**63-1)
            row = self.db.execute(f"select embedding from {identifier(c['vectors'])} where rowid=:id "
                                  f"and rowid in ({self.eligible(c)})", dict(self.params(c), id=key)).fetchone()
            if row is None:
                raise Rejected("missing_embedding", "embedding ID is absent or ineligible")
            v = list(struct.unpack(f"{c['dimensions']}f", row[0]))
        else:
            v = a["vector"]
        if not isinstance(v, list) or len(v) != c["dimensions"]:
            raise Rejected("invalid_vector", "vector length must match configured dimensions")
        if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e10 for x in v):
            raise Rejected("invalid_vector", "vector coordinates must be finite numbers with magnitude <= 1e10")
        return struct.pack(f"{len(v)}f", *v)

    def search(self, c, a):
        if set(a) - {"mode", "text", "vector", "embedding_id", "candidates", "limit"}:
            raise Rejected("invalid_input", "unknown search argument")
        mode = a.get("mode", "hybrid")
        if mode not in ("lexical", "dense", "hybrid"):
            raise Rejected("invalid_input", "mode must be lexical, dense or hybrid")
        cap = integer(a.get("candidates", 50), "candidates", 1, LIMITS["candidates"])
        limit = integer(a.get("limit", 10), "limit", 1, LIMITS["rows"])
        p = dict(self.params(c), cap=cap)
        eligible = self.eligible(c)
        branches = {}
        if mode != "lexical":
            p["vector"] = self.vector(c, a)
            # Scalar scoring admits all cutoff ties before the deterministic
            # branch limit; vec0 KNN cannot promise rowid order at its cutoff.
            rows = self.db.execute(f"with eligible as materialized ({eligible}) "
                f"select rowid id,vec_distance_l2(embedding,:vector) distance from {identifier(c['vectors'])} "
                "where rowid in (select id from eligible) order by distance,rowid limit :cap", p).fetchall()
            if any(r["distance"] is None or not math.isfinite(r["distance"]) for r in rows):
                raise Rejected("database_error", "stored vectors produced a non-finite L2 distance")
            branches["vector"] = [(r["id"], r["distance"]) for r in rows]
        elif "vector" in a or "embedding_id" in a:
            raise Rejected("invalid_input", "lexical mode does not accept a vector")
        if mode != "dense":
            text = a.get("text")
            if not isinstance(text, str) or not 1 <= len(text) <= 2048:
                raise Rejected("invalid_input", "text must contain 1 to 2048 characters of FTS5 query syntax")
            p["text"] = text
            fts = identifier(c["fts"])
            rows = self.db.execute(f"with eligible as materialized ({eligible}) select rowid id,rank "
                f"from {fts} where {fts} match :text and rowid in (select id from eligible) "
                "order by rank,rowid limit :cap", p).fetchall()
            branches["keyword"] = [(r["id"], r["rank"]) for r in rows]
        elif "text" in a:
            raise Rejected("invalid_input", "dense mode does not accept text")
        merged = {}
        for branch, rows in branches.items():
            for rank, (key, score) in enumerate(rows, 1):
                row = merged.setdefault(key, dict(id=key, vector_rank=None, keyword_rank=None,
                                                 vector_distance=None, keyword_bm25=None))
                row[branch + "_rank"] = rank
                row["vector_distance" if branch == "vector" else "keyword_bm25"] = score
        for row in merged.values():
            row["score"] = sum(1/(60+row[b+"_rank"]) for b in branches if row[b+"_rank"] is not None)
        ordered = sorted(merged.values(), key=lambda r: (-r["score"], r["id"]))
        rows = ordered[:limit]
        for row in rows:
            row["source_id"] = self.db.execute(f"select source_id from {identifier(c['documents'])} where id=?", (row["id"],)).fetchone()[0]
        return dict(rows=rows, mode=mode, requested=limit, returned=len(rows),
                    candidate_union=len(merged), candidates_requested=cap,
                    branches={b: dict(returned=len(v), boundary_hit=len(v)==cap) for b,v in branches.items()},
                    truncated=len(ordered)>limit, complete=len(ordered)<=limit, approximation=False,
                    completeness_domain="bounded_candidate_union", score="rrf_60_higher_is_better",
                    dense_strategy="exhaustive_scalar_l2", tie_policy="score desc,id asc; branch score asc,id asc")

    def fetch(self, c, a):
        if set(a) != {"ids"}:
            raise Rejected("invalid_input", "fetch requires ids only")
        keys = ids(a["ids"])
        rows, missing = [], []
        for key in keys:
            row = self.db.execute(f"select id,document_id,body,source_id,start_ms,end_ms from {identifier(c['documents'])} "
                                  f"where id=:id and id in ({self.eligible(c)})", dict(self.params(c), id=key)).fetchone()
            if row is None:
                missing.append(key)
            else:
                rows.append(dict(row))
        return dict(rows=rows, missing_ids=missing, missing_reason="absent_or_ineligible", requested=len(keys),
                    returned=len(rows), truncated=False, complete=True)

    def traverse(self, c, a):
        if set(a) - {"ids", "depth", "limit"}:
            raise Rejected("invalid_input", "unknown traverse argument")
        keys = ids(a.get("ids"))
        depth = integer(a.get("depth", 1), "depth", 0, LIMITS["depth"])
        limit = integer(a.get("limit", 50), "limit", 1, LIMITS["rows"])
        edges = identifier(c["edges"])
        n = self.db.execute(f"select count(*) from (select 1 from {edges} limit 20001)").fetchone()[0]
        if n > LIMITS["graph_edges"]:
            raise Rejected("workload_limit", "graph source exceeds 20000 edges")
        invalid = self.db.execute(f"with eligible as materialized ({self.eligible(c)}) select 1 from {edges} "
            "where typeof(src)!='integer' or typeof(dst)!='integer' or src not in (select id from eligible) "
            "or dst not in (select id from eligible) limit 1", self.params(c)).fetchone()
        if invalid:
            raise Rejected("graph_scope", "configured edge source must contain only eligible integer endpoints")
        found = {}
        missing = []
        for key in keys:
            present = self.db.execute(f"select 1 from ({self.eligible(c)}) where id=:id", dict(self.params(c), id=key)).fetchone()
            if present is None:
                missing.append(key)
                continue
            for node, hops in self.db.execute("select node,depth from graph_bfs(?,'src','dst',?,?)", (c["edges"], key, depth)):
                found[node] = min(found.get(node, hops), hops)
        ordered = sorted(found.items(), key=lambda r: (r[1], r[0]))
        return dict(rows=[dict(id=k, depth=d) for k,d in ordered[:limit]], missing_ids=missing,
                    missing_reason="absent_or_ineligible", requested=limit, returned=min(len(found),limit),
                    total_reachable=len(found), truncated=len(found)>limit, complete=len(found)<=limit,
                    direction="out", depth=depth, graph_edges=n, semantics="distinct_nodes_minimum_hops_from_any_seed",
                    native_work_budget=False, native_memory_budget=False, cancellation=False)


def cli(adapter, source, sink):
    failed = False
    while True:
        line = source.readline(LIMITS["input_bytes"]+1)
        if not line:
            break
        if len(line) > LIMITS["input_bytes"]:
            # Drain this record without retaining the rest of an oversized line.
            while not line.endswith(b"\n"):
                line = source.readline(LIMITS["input_bytes"]+1)
                if not line:
                    break
            result = error("input_limit", "record exceeds 65536 bytes")
        else:
            try:
                request = json.loads(line)
                result = adapter.call(request)
            except (ValueError, UnicodeError, RecursionError):
                result = error("invalid_json", "record must be one UTF-8 JSON object")
        failed |= not result["ok"]
        sink.write(encoded(result) + b"\n")
        sink.flush()
    return int(failed)


async def serve(adapter):
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

    server = Server("sqlite-vec-search-lab")

    @server.list_tools()
    async def list_tools():
        return [Tool(name=op, description=f"Read-only {op} in the configured retrieval toolset; use describe for limits.",
                     inputSchema={"type":"object", "properties":{
                         "corpus":{"type":"string", "enum":list(adapter.config["corpora"])}, "args":{"type":"object", "properties":ARGUMENTS[op],
                         "required":["ids"] if op in ("traverse","fetch") else [], "additionalProperties":False}},
                         "required":["corpus"], "additionalProperties":False},
                     annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False)) for op in OPERATIONS]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        # Shared validation preserves identical structured errors over CLI/MCP.
        if set(arguments) - {"corpus", "args"}:
            result = error("invalid_input", "request accepts only op, corpus and args")
        else:
            result = adapter.call(dict(arguments, op=name))
        return CallToolResult(structuredContent=result, content=[TextContent(type="text", text="See structuredContent for the result.")],
                              isError=not result["ok"])

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("transport", choices=("cli", "mcp"))
    opts = parser.parse_args()
    adapter = None
    try:
        adapter = Adapter(opts.config)
        if opts.transport == "cli":
            return cli(adapter, sys.stdin.buffer, sys.stdout.buffer)
        asyncio.run(serve(adapter))
        return 0
    except (Rejected, sqlite3.Error, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        # Startup failures go to stderr, including in MCP mode.
        print(encoded(error("startup_error", str(exc))).decode(), file=sys.stderr)
        return 2
    finally:
        if adapter:
            adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
