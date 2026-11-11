//! Compiles the sqlite-vec C core into this crate, so one loadable library
//! registers both vec0 and lance0. The C core never depends on Rust; this is
//! the only direction the two halves are linked.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("..");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    for f in ["sqlite-vec.h.tmpl", "VERSION"] {
        println!("cargo:rerun-if-changed={}", root.join(f).display());
    }
    println!("cargo:rerun-if-changed=src/vtab.c");
    for entry in fs::read_dir(&root).unwrap().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("sqlite-vec") && name.ends_with(".c") {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    // sqlite-vec.h, as the Makefile's envsubst rule generates it.
    let version = fs::read_to_string(root.join("VERSION")).unwrap().trim().to_owned();
    let mut parts = version.splitn(3, '.');
    let major = parts.next().unwrap_or("0").to_owned();
    let minor = parts.next().unwrap_or("0").to_owned();
    let patch = parts.next().unwrap_or("0").split('-').next().unwrap().to_owned();
    let header = fs::read_to_string(root.join("sqlite-vec.h.tmpl"))
        .unwrap()
        .replace("${VERSION_MAJOR}", &major)
        .replace("${VERSION_MINOR}", &minor)
        .replace("${VERSION_PATCH}", &patch)
        .replace("${VERSION}", &version)
        .replace("${DATE}", "")
        .replace("${SOURCE}", "");
    fs::write(out.join("sqlite-vec.h"), header).unwrap();

    let vendor = root.join("vendor");
    if !vendor.join("sqlite3ext.h").exists() {
        panic!(
            "{} is missing; run ./scripts/vendor.sh from the repository root first",
            vendor.join("sqlite3ext.h").display()
        );
    }

    // Same SIMD choices as the Makefile.
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let mut build = cc::Build::new();
    build
        .file(root.join("sqlite-vec.c"))
        .file(root.join("lance/src/vtab.c"))
        .include(&out)
        .include(&vendor)
        .opt_level(3)
        .warnings(false);
    if arch == "aarch64" {
        build.define("SQLITE_VEC_ENABLE_NEON", None);
        if os == "macos" {
            build.flag("-mcpu=apple-m1");
        }
    } else if arch == "x86_64" && features.split(',').any(|f| f == "avx2") {
        build.flag("-mavx").flag("-mavx2").define("SQLITE_VEC_ENABLE_AVX", None);
    }
    build.compile("sqlite_vec");
}
