# Historical backlog — superseded 2026-11-07

Preserved verbatim below for provenance. This is not the active work queue;
use the repository-root TODO.md. Historical publication/force-push tasks are not authorized.

---

- [ ] add `xyz_info` shadow table with version etc.

- later
  - [ ] partition: UPDATE support
  - [ ] skip invalid validity entries in knn filter?
  - [ ] nulls in metadata
  - [ ] partition `x in (...)` handling
  - [ ] blobs/date/datetime
  - [ ] uuid/ulid perf
  - [ ] Aux columns: `NOT NULL` constraint
  - [ ] Metadata columns: `NOT NULL` constraint
   - [ ] Partiion key: `NOT NULL` constraint
  - [ ] dictionary encoding?
  - [ ] properly sqlite3_vtab_nochange / sqlite3_value_nochange handling
  - [ ] perf
    - [ ] aux: cache INSERT
    - [ ] aux: LEFT JOIN on `_rowids` queries to avoid N lookup queries
