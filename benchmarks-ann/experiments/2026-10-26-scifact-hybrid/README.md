# SciFact hybrid retrieval evaluation

Compare the current vector-only, FTS5-only and equal-weight RRF recipe on public
relevance labels and locally generated BGE embeddings. [REPORT.md](REPORT.md)
contains the measurements, interpretation and limits. No production defaults change.

## Reproduce

Run from the repository root on macOS arm64. The C build needs `cc`, SQLite headers
under `vendor/` and the generated `sqlite-vec.h`. Python commands use the existing
`tests` project; additional packages are temporary `uv` dependencies.

```sh
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests --with onnxruntime==1.23.2 --with tokenizers==0.22.1 python benchmarks-ann/experiments/2026-10-26-scifact-hybrid/prepare.py
python3 benchmarks-ann/experiments/2026-10-26-scifact-hybrid/build.py
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-10-26-scifact-hybrid/test_evaluate.py -v
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-10-26-scifact-hybrid/evaluate.py
```

Preparation downloads about 140 MB and embeds 5,183 documents and 300 queries
locally. The recorded model/CPU preparation took about four minutes; timings vary.
Full relevance evaluation can take tens of minutes because the unchanged filtered
FTS query is expensive. The timing sample is fixed independently of outcomes.
All downloaded data, arrays, compiled libraries and fresh results stay in ignored
`.work/`. The builder copies one committed C revision and excludes dirty C edits;
its generated header is copied from the checkout. Exact hashes are recorded.
Run Python with assertions enabled. Network access is needed only for preparation
and uncached temporary packages. No credentials or remote inference are required.

## Verify the saved run

```sh
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-10-26-scifact-hybrid/verify.py
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests --with pytrec-eval-terrier==0.5.11 python benchmarks-ann/experiments/2026-10-26-scifact-hybrid/check_metrics.py
UV_CACHE_DIR=/tmp/uv-cache-muse uv run --managed-python --project tests python benchmarks-ann/experiments/2026-10-26-scifact-hybrid/analyze.py
```

The verifier checks the frozen `recorded/` archive without model/data downloads.
The optional trec_eval command independently recomputes Recall@10, nDCG@10 and MRR@10
with trec_eval, writing its receipt in `recorded/metric-oracle.json`.
`analyze.py` regenerates the paired query-bootstrap differences. New benchmark
runs write `.work/results/`; copy nothing into the frozen archive until reviewed.
The manifest covers scripts, report, data/build manifests, guide snapshot and raw
ranking/timing receipts. Dataset/model bytes and generated arrays are identified
by SHA256 rather than committed. Different inference runtimes can produce small
floating-point differences; no bit-identical cross-platform replay is promised.

## Dataset and model attribution

SciFact: David Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (2020),
provided in BEIR by Nandan Thakur et al. (2021). [BEIR data and checksum](https://github.com/beir-cellar/beir/wiki/Datasets-available).
[Original SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md):
claims/annotations CC BY 4.0; abstracts ODC-By 1.0. Recorded lexical expressions
are tokenized transformations of the claims; qrels preserve the public labels.
The public BEIR test split is used as distributed; this is article retrieval,
not claim-veracity classification or a reproduction of the original blind test.

[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), MIT,
pinned revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The official ONNX
model is executed on CPU with documented CLS pooling, normalization and query
instruction. No remote model code is loaded. Model training/test overlap has
not been independently audited; this is not a contamination-free evaluation claim.
