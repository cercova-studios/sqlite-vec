"""Measured host backend: ordered float64 cumulative sums, no Q×D matrix.

This is not the scalar oracle or a native MaxSim implementation. Numerical
agreement is qualified against the independent oracle, including real embeddings.
"""
import hashlib
import json

import numpy as np

WORK_LIMIT = 134217728


def pair_work(m, n, d):
    return m*d + n*d + m*n*d + m*n + m


def decode(blob, dimensions, maximum_rows, maximum_bytes):
    if type(blob) is not bytes or type(dimensions) is not int or not 1 <= dimensions <= 4096:
        raise ValueError("bytes matrix and admitted integer dimensions required")
    if not blob or len(blob) > maximum_bytes or len(blob) % (4*dimensions):
        raise ValueError("invalid matrix length")
    rows = len(blob) // (4*dimensions)
    if rows > maximum_rows:
        raise ValueError("too many matrix rows")
    matrix = np.frombuffer(blob, dtype="<f4").reshape(rows, dimensions).astype(np.float64)
    squares = np.square(matrix)
    np.cumsum(squares, axis=1, out=squares)
    if not np.isfinite(matrix).all() or np.any(np.abs(squares[:, -1] - 1) > 1e-4):
        raise ValueError("nonfinite or unnormalized matrix")
    return matrix


def score_pair(query, document):
    score = 0.0
    scratch = np.empty_like(document)
    for q in query:
        # cumsum preserves component order; sum/einsum may reassociate tiny terms.
        np.multiply(document, q, out=scratch)
        np.cumsum(scratch, axis=1, out=scratch)
        score += float(scratch[:, -1].max())
    return score


def request_work(m, dimensions, metadata, limit):
    if type(limit) is not int or not 1 <= limit <= WORK_LIMIT:
        raise ValueError("invalid request work budget")
    if len(metadata) > 100 or len({r["id"] for r in metadata}) != len(metadata):
        raise ValueError("candidate count/identity violation")
    total = m * dimensions
    payload = 0
    for row in metadata:
        if type(row["id"]) is not int or not -(2**63) <= row["id"] < 2**63:
            raise ValueError("invalid document identity")
        n, size = row["tokens"], row["bytes"]
        if type(n) is not int or not 1 <= n <= 4096 or size != n*dimensions*4 or size > 8388608:
            raise ValueError("invalid candidate shape")
        payload += size
        total += pair_work(m, n, dimensions)
    if payload > 33554432 or total > limit:
        raise ValueError("aggregate request byte/work limit")
    return total, payload


def digest_artifact(profile, role, source, count, dimensions, blob):
    shape = np.asarray([count, dimensions], dtype="<u4").tobytes()
    return hashlib.sha256(b"sqlite-vec/li-matrix/v1\0" + bytes.fromhex(profile) + bytes([role == "query"])
                          + bytes.fromhex(source) + shape + blob).hexdigest()


def rerank(db, qid, query_text, candidates, profile, work_limit=WORK_LIMIT, query_override=None):
    if not db.in_transaction:
        raise ValueError("one explicit read transaction required")
    if (len(candidates) > 100 or len(set(candidates)) != len(candidates)
            or any(type(key) is not int or not -(2**63) <= key < 2**63 for key in candidates)):
        raise ValueError("candidate count/identity violation")
    query_row = db.execute("""SELECT a.profile_id,a.token_count,a.dimensions,a.input_sha256,a.sha256,length(a.vectors)
        FROM li_refs r JOIN li_artifacts a ON a.sha256=r.artifact_sha256
        WHERE r.role='query' AND r.item_id=? AND a.role='query'""", (int(qid),)).fetchone()
    if not query_row:
        raise ValueError("missing query artifact")
    p, count, dimensions, input_sha, sha, size = query_row
    if (type(count) is not int or not 1 <= count <= 128
            or type(dimensions) is not int or not 1 <= dimensions <= 4096
            or size != count*dimensions*4 or size > 1048576):
        raise ValueError("query shape/byte limit")
    if p != profile or input_sha != hashlib.sha256(query_text.encode()).hexdigest():
        raise ValueError("query profile/source mismatch")
    rows = db.execute("""SELECT r.item_id,a.token_count,length(a.vectors),a.profile_id,a.dimensions,
         a.input_sha256,a.sha256,r.source_revision,s.source_revision,s.body
         FROM li_refs r JOIN li_artifacts a ON a.sha256=r.artifact_sha256
         JOIN sources s ON s.id=r.item_id
         WHERE r.role='document' AND a.role='document'
           AND r.item_id IN (SELECT value FROM json_each(?))""", (json.dumps(candidates),)).fetchall()
    metadata = [{"id": r[0], "tokens": r[1], "bytes": r[2], "profile": r[3], "dimensions": r[4],
                 "input": r[5], "sha": r[6], "source_epoch": r[7], "current_epoch": r[8], "body": r[9]} for r in rows]
    if len(candidates) != len(set(candidates)) or {r["id"] for r in metadata} != set(candidates):
        raise ValueError("missing/duplicate document artifact")
    for row in metadata:
        if row["profile"] != profile or row["dimensions"] != dimensions or row["source_epoch"] != row["current_epoch"]:
            raise ValueError("document profile/source epoch mismatch")
        if hashlib.sha256(row["body"].encode()).hexdigest() != row["input"]:
            raise ValueError("document source input mismatch")
    work, payload = request_work(count, dimensions, metadata, work_limit)
    blob = db.execute("SELECT vectors FROM li_artifacts WHERE sha256=?", (sha,)).fetchone()[0]
    if query_override is not None:
        if blob != query_override:
            raise ValueError("live query encoding differs from qualified artifact")
        blob = query_override
    if digest_artifact(p, "query", input_sha, count, dimensions, blob) != sha:
        raise ValueError("query digest mismatch")
    query = decode(blob, dimensions, 128, 1048576)
    scores = []
    products = 0
    read_bytes = len(blob)
    for row in metadata:
        document_blob = db.execute("SELECT vectors FROM li_artifacts WHERE sha256=?", (row["sha"],)).fetchone()[0]
        if digest_artifact(profile, "document", row["input"], row["tokens"], dimensions, document_blob) != row["sha"]:
            raise ValueError("document artifact digest mismatch")
        document = decode(document_blob, dimensions, 4096, 8388608)
        scores.append((row["id"], score_pair(query, document)))
        products += len(query)*len(document)*dimensions
        read_bytes += len(document_blob)
    scores.sort(key=lambda r: (-r[1], r[0]))
    return dict(scores=scores, work_used=work, component_products=products, candidate_bytes=payload,
                matrix_bytes_read=read_bytes, scored_candidates=len(scores), backend="numpy_ordered_cumsum_float64")
