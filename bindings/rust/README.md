# Rust registration

Link this fork from a local path after preparing its generated sources:

```sh
make sqlite-vec.h
make -C bindings/rust deps
cargo test --manifest-path bindings/rust/Cargo.toml
```

Register the extension on each open connection. This works with Apple's system
SQLite, which rejects process-global auto extensions:

```rust
let db = rusqlite::Connection::open_in_memory()?;
let rc = unsafe {
    sqlite_vec::sqlite3_vec_init(
        db.handle().cast(), std::ptr::null_mut(), std::ptr::null(),
    )
};
assert_eq!(rc, rusqlite::ffi::SQLITE_OK);
```

The caller must pass a live connection from the same SQLite library linked to
the binding. Do not pass a handle from another SQLite runtime. Check the returned
SQLite result code before using the extension. The FFI declaration now spells out
the C initializer's three arguments and return value; callers that cast its address
for `sqlite3_auto_extension` retain that usage on supporting hosts. Bare zero-argument
calls were not a valid way to invoke the C initializer.

`cargo test --features rusqlite/bundled` exercises bundled SQLite separately.
The existing [example](../../examples/simple-rust/demo.rs) selects this fork by path.
