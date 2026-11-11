"""Independent scalar MaxSim for SPEC L0; no model, ranking or SQLite integration.

Inputs are immutable row-major little-endian float32 bytes. Successful calls
validate every row's float64 squared norm within 1e-4 of one, without normalizing
or clamping. Dot products and the outer sum accumulate in input order as Python
float64 operations. Scratch is constant; no decoded matrix is retained.

The caller owns corpus/profile/epoch checks, request-wide admission, cancellation
and result publication. This pairwise oracle does not claim native cancellation
or process-memory bounds. The experimental runner must pass remaining work to
each call, including its separate initial query-admission charge from SPEC §6.
"""

from dataclasses import dataclass
import math
import struct

MAX_DIMENSIONS = 4096
MAX_QUERY_ROWS = 128
MAX_DOCUMENT_ROWS = 4096
MAX_QUERY_BYTES = 1 << 20
MAX_DOCUMENT_BYTES = 8 << 20
MAX_WORK = 134_217_728
NORM_TOLERANCE = 1e-4
_FLOAT = struct.Struct("<f")


class WorkLimitError(ValueError):
    """The admitted pair exceeds its supplied remaining work budget."""


@dataclass(frozen=True, slots=True)
class Score:
    score: float
    work_used: int
    query_rows: int
    document_rows: int
    dimensions: int


def _rows(blob: bytes, dimensions: int, rows_limit: int, bytes_limit: int, role: str) -> int:
    if type(blob) is not bytes:
        raise ValueError(f"{role} must be immutable bytes")
    if len(blob) > bytes_limit:
        raise ValueError(f"{role} bytes exceed {bytes_limit}")
    stride = dimensions * _FLOAT.size
    if not blob or len(blob) % stride:
        raise ValueError(f"{role} must contain nonempty complete rows")
    rows = len(blob) // stride
    if rows > rows_limit:
        raise ValueError(f"{role} rows exceed {rows_limit}")
    return rows


def work_required(query_blob: bytes, doc_blob: bytes, dimensions: int) -> int:
    """Check types/shapes/byte ceilings and estimate this pair's exact work.

    Float values and row norms are not inspected. Python integers do not wrap;
    the result can exceed MAX_WORK, allowing admission to report the requested
    cost without decoding. maxsim enforces the work ceiling before numerical work.
    """
    if type(dimensions) is not int or not 1 <= dimensions <= MAX_DIMENSIONS:
        raise ValueError(f"dimensions must be an integer in [1,{MAX_DIMENSIONS}]")
    m = _rows(query_blob, dimensions, MAX_QUERY_ROWS, MAX_QUERY_BYTES, "query")
    n = _rows(doc_blob, dimensions, MAX_DOCUMENT_ROWS, MAX_DOCUMENT_BYTES, "document")
    return m * dimensions + n * dimensions + m * n * dimensions + m * n + m


def _validate(blob: bytes, dimensions: int, role: str) -> int:
    used = 0
    for start in range(0, len(blob), dimensions * _FLOAT.size):
        norm = 0.0
        for component in range(dimensions):
            value = _FLOAT.unpack_from(blob, start + component * _FLOAT.size)[0]
            used += 1
            if not math.isfinite(value):
                raise ValueError(f"{role} contains a nonfinite component")
            norm += value * value
        if abs(norm - 1.0) > NORM_TOLERANCE:
            raise ValueError(f"{role} row squared norm is outside tolerance")
    return used


def maxsim(query_blob: bytes, doc_blob: bytes, dimensions: int, work_limit: int) -> Score:
    """Return a complete score or raise ValueError/WorkLimitError; never a prefix.

    Work counts validation components, scoring products (including decoding),
    pairwise maximum comparisons and outer additions. No score is published until
    both matrices validate and every admitted pair has been evaluated.
    """
    if type(work_limit) is not int or not 0 <= work_limit <= MAX_WORK:
        raise ValueError(f"work_limit must be an integer in [0,{MAX_WORK}]")
    required = work_required(query_blob, doc_blob, dimensions)
    if required > work_limit:
        raise WorkLimitError(f"pair requires {required} work units; remaining budget is {work_limit}")
    used = _validate(query_blob, dimensions, "query")
    used += _validate(doc_blob, dimensions, "document")
    stride = dimensions * _FLOAT.size
    score = 0.0
    for q_start in range(0, len(query_blob), stride):
        best = -math.inf
        for d_start in range(0, len(doc_blob), stride):
            dot = 0.0
            for component in range(dimensions):
                offset = component * _FLOAT.size
                q = _FLOAT.unpack_from(query_blob, q_start + offset)[0]
                d = _FLOAT.unpack_from(doc_blob, d_start + offset)[0]
                dot += q * d
                used += 1
            if dot > best:
                best = dot
            used += 1
        score += best
        used += 1
    assert used == required
    return Score(score, used, len(query_blob) // stride, len(doc_blob) // stride, dimensions)
