fn main() {
    for file in ["sqlite-vec.c", "sqlite-vec.h", "sqlite3.h", "sqlite3ext.h"] {
        println!("cargo:rerun-if-changed={file}");
    }
    cc::Build::new()
        .file("sqlite-vec.c")
        .define("SQLITE_CORE", None)
        .compile("sqlite_vec0");
}
