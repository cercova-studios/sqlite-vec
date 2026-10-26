"""Keep replay output separate from the immutable recorded run."""
from pathlib import Path
import os

ARCHIVE = Path(__file__).resolve().parents[1]
REPO = ARCHIVE.parents[2]
ROOT = Path(os.environ.get("HNSW_ALPHA_WORKDIR", str(ARCHIVE / ".work"))).resolve()
for name in ("data", "builds", "results"):
    (ROOT / name).mkdir(parents=True, exist_ok=True)
