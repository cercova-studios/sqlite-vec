"""Validate retained baseline measurements and print descriptive latency summaries."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def validate(report):
    records = report["records"]
    expected = {(r, b, p) for r in range(3) for b, p in [(1, 0), (16, 0), (64, 0), (16, 512), (64, 128)]}
    assert len(records) == 15
    assert {(r["repeat"], r["batch"], r["payload_bytes"]) for r in records} == expected
    for record in records:
        assert record["oracle"] == "passed"
        assert record["rounds"] == 50 and record["history_rows"] == 1250
        assert record["peak_rss_bytes"] > 0 and record["capture_bytes"] > 0
        for name, samples in record["samples_ns"].items():
            assert len(samples) == (1 if name in {"create_fixture", "reopen"} else 50)
            assert all(isinstance(n, int) and n >= 0 for n in samples)
        assert set(record["samples_ns"]) == {"create_fixture", "create", "claim", "retry", "stale_conflict", "undo_claim", "delete", "undo_delete", "history_100", "reopen"}
        assert [p["row_history_entries"] for p in record["history_growth"]] == [7, 105, 1005]
        assert all(len(p["ns"]) == 30 for p in record["history_growth"])


def percentile(samples, p):
    values = sorted(samples)
    return values[min(len(values) - 1, int((len(values) - 1) * p))] / 1e6


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("baseline-locked.json")
    report = json.loads(path.read_text())
    validate(report)
    here = Path(__file__).resolve().parent
    assert hashlib.sha256((here / "main.rs").read_bytes()).hexdigest() == report["harness_sha256"]
    if "runner_sha256" in report:
        assert hashlib.sha256((here / "run.py").read_bytes()).hexdigest() == report["runner_sha256"]
    for name, digest in report["input_sha256"].items():
        source = subprocess.check_output(["git", "show", f"{report['revision']}:examples/write-prototype/{name}"], cwd=here.parents[2])
        assert hashlib.sha256(source).hexdigest() == digest
    # A missing observation and a failed oracle must both make offline acceptance fail.
    for mutate in [lambda r: r["records"][0].update(oracle="failed"),
                   lambda r: r["records"][0]["samples_ns"]["create"].pop()]:
        broken = copy.deepcopy(report)
        mutate(broken)
        try:
            validate(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("negative control was accepted")
    print(f"Revision: {report['revision']}; {report['platform']}; {report['rustc']}")
    print("All 15 records and both negative controls passed validation.")
    print("batch payload operation p50_ms p95_ms")
    for batch, payload in [(1, 0), (16, 0), (64, 0), (16, 512), (64, 128)]:
        rows = [r for r in report["records"] if (r["batch"], r["payload_bytes"]) == (batch, payload)]
        for name in ["create", "claim", "retry", "stale_conflict", "undo_claim", "delete", "undo_delete", "history_100"]:
            samples = [n for r in rows for n in r["samples_ns"][name]]
            print(batch, payload, name, f"{percentile(samples,.5):.6f}", f"{percentile(samples,.95):.6f}")
        print("peak_rss_mib", batch, payload, round(max(r["peak_rss_bytes"] for r in rows) / 1048576, 3))
        print("median_storage", batch, payload, json.dumps({name: int(statistics.median(r[name] for r in rows)) for name in ["capture_bytes", "canonical_request_bytes", "receipt_bytes", "reopened_db_bytes"]}))
    print("history_entries p50_ms p95_ms (batch=1,payload=0)")
    rows = [r for r in report["records"] if (r["batch"], r["payload_bytes"]) == (1, 0)]
    for index, entries in enumerate([7, 105, 1005]):
        samples = [n for r in rows for n in r["history_growth"][index]["ns"]]
        print(entries, f"{percentile(samples,.5):.6f}", f"{percentile(samples,.95):.6f}")


if __name__ == "__main__":
    main()
