"""Explicitly acquire pinned public encoder data; never import downloaded source."""
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
MODEL = "colbert-ir/colbertv2.0"
REVISION = "c1e84128e85ef755c096a95bdb06b47793b13acf"
CODE = "cc4f3dc91c0b45d2d08c251d9d95178285c65f1c"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if expected and sha(temporary) != expected:
            raise ValueError(f"download hash mismatch: {path.name}")
        temporary.replace(path)
    if expected and sha(path) != expected:
        raise ValueError(f"existing hash mismatch: {path.name}")
    return dict(url=url, sha256=sha(path), bytes=path.stat().st_size)


def main():
    root = HERE / ".work/model"
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    assert metadata["sha"] == REVISION and metadata["cardData"]["license"] == "mit"
    (root / "hub-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = {}
    inventory = {row["rfilename"]: row for row in metadata["siblings"]}
    for name in ["README.md", "config.json", "artifact.metadata", "model.safetensors",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]:
        expected = inventory[name].get("lfs", {}).get("sha256")
        files[name] = download(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", root / name, expected)
        print(name, files[name]["bytes"], flush=True)
    sources = {}
    for name in ["LICENSE", "colbert/modeling/colbert.py", "colbert/modeling/hf_colbert.py",
                 "colbert/modeling/tokenization/query_tokenization.py",
                 "colbert/modeling/tokenization/doc_tokenization.py",
                 "colbert/modeling/tokenization/utils.py", "colbert/infra/config/settings.py"]:
        sources[name] = download(f"https://raw.githubusercontent.com/stanford-futuredata/ColBERT/{CODE}/{name}", root / "source" / name)
    manifest = dict(model=MODEL, revision=REVISION, license="mit", source_revision=CODE,
                    files=files, sources=sources, remote_code_executed=False)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
