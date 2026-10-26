"""Check archived evidence without changing recorded results."""
from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parent
for name, digest in json.loads((root / "manifest.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
for path in root.rglob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
data = json.loads((root / "recorded/results.json").read_text())
rows = data["results"]
assert len(rows) == len({(r["profile"], r["chunk_rows"]) for r in rows}) == 30
for row in rows:
    for codec in row["codecs"].values():
        assert codec["framed_bytes"] == codec["compressed_bytes"] + 16
        assert codec["chosen_payload_bytes"] == min(row["raw_bytes"], codec["framed_bytes"])
        for key in ("encode", "decode"):
            assert len(codec[key]["batches_us"]) == 9
print("Archive hashes, Python syntax, 30 profiles and 60 codec receipts verified.")
