"""Move sqlite-dist npm tarballs into an npm scope. See ARCHITECTURE.md, Releases."""

import argparse
import io
import json
import tarfile
from pathlib import Path

BASE = "sqlite-vec"


def scoped(name: str, scope: str) -> str:
    if name != BASE and not name.startswith(f"{BASE}-"):
        raise ValueError(f"unexpected npm package name: {name}")
    return f"{scope}/{name}"


def rewrite(member: str, data: bytes, scope: str) -> bytes:
    if member == "package/package.json":
        manifest = json.loads(data)
        manifest["name"] = scoped(manifest["name"], scope)
        manifest["optionalDependencies"] = {
            scoped(name, scope): version
            for name, version in manifest.get("optionalDependencies", {}).items()
        }
        return json.dumps(manifest, indent=2).encode() + b"\n"
    if member in ("package/index.cjs", "package/index.mjs"):
        loader = data.decode()
        assignment = "const packageName = platformPackageName(platform, arch);"
        if loader.count(assignment) != 1:
            raise ValueError(f"{member}: unexpected sqlite-dist loader; review package resolution")
        return loader.replace(
            assignment,
            f"const packageName = {json.dumps(scope + '/')} + platformPackageName(platform, arch);",
        ).encode()
    return data


def scope_tarball(source: Path, destination: Path, scope: str) -> str:
    out = io.BytesIO()
    name = ""
    with tarfile.open(source) as src, tarfile.open(fileobj=out, mode="w:gz") as dst:
        for info in src.getmembers():
            if not info.isfile():
                dst.addfile(info)
                continue
            data = rewrite(info.name, src.extractfile(info).read(), scope)
            if info.name == "package/package.json":
                name = json.loads(data)["name"]
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))
    if not name:
        raise ValueError(f"{source}: no package/package.json")
    target = destination / (name.replace("@", "").replace("/", "-") + ".tgz")
    target.write_bytes(out.getvalue())
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory of sqlite-dist npm tarballs")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--scope", required=True, help="npm scope, e.g. @cercova-studios")
    args = parser.parse_args()
    if not args.scope.startswith("@") or "/" in args.scope:
        parser.error("--scope must look like @owner")
    args.destination.mkdir(parents=True, exist_ok=True)
    tarballs = sorted(args.source.glob("*.tar.gz")) + sorted(args.source.glob("*.tgz"))
    if not tarballs:
        parser.error(f"no npm tarballs in {args.source}")
    for tarball in tarballs:
        print(scope_tarball(tarball, args.destination, args.scope))


if __name__ == "__main__":
    main()
