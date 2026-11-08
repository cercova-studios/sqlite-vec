# sqlite-vec hybrid extension (`lance0`)

The hybrid build of sqlite-vec: one loadable library with everything in the C core (`vec0`, the `vec_*` functions) plus `lance0`, a virtual table that stores embeddings and large blobs in a [Lance](https://lancedb.github.io/lance/) dataset. Relational data stays in ordinary SQLite tables, and one SQL statement can use all three.

```sql
.load ./dist/vec0_lance

create table users(id integer primary key, name text);
create virtual table emb using vec0(v float[768] indexed by hnsw());   -- fast ANN, in SQLite
create virtual table docs using lance0('/data/docs.lance', 768);      -- vectors + blobs, in Lance

-- vec0 finds the neighbors, lance0 returns their blobs, SQLite does the join
select u.name, length(d.payload), e.distance
from emb e
join docs d on d.rowid = e.rowid
join users u on u.id = e.rowid
where e.v match :q and e.k = 10
order by e.distance;

-- or search in Lance directly; equality filters on label run inside Lance
select rowid, distance from docs where query = :q and k = 10 and label = 'cat' order by distance;
```

Vectors use sqlite-vec's formats: a little-endian float32 BLOB (for example from `vec_f32()`) or JSON text such as `'[1, 2, 3]'`. Distances are L2, in the same units `vec0` reports.

## vec0 tables with Lance storage

A `vec0` table can keep its auxiliary columns (`+name type`) in Lance while its index stays in SQLite:

```sql
create virtual table items using vec0(
  embedding float[768] indexed by hnsw(),
  +title text, +image blob,
  storage=lance, storage_uri='/data/items.lance'
);

-- HNSW in SQLite finds the rows; their auxiliary values come from Lance in one batch
select rowid, title, length(image) from items where embedding match :q and k = 10;
```

- Without `storage_uri`, the dataset goes next to the database file, as `<db>.<table>.lance`. In-memory databases need `storage_uri`.
- Inserts, updates, deletes, transactions, savepoints and failed statements behave as they do with the default storage.
- `ALTER TABLE ... RENAME` keeps the data. `DROP TABLE` removes SQLite metadata
  and retains local and remote datasets, including after commit. Retention protects
  rolled-back drops and existing readers. Dataset files can continue to occupy disk
  space after the SQL table is gone; cleanup requires proving that no table,
  database or reader still needs them.
- When replacing a dropped `vec0` table, use a fresh `storage_uri`. Creation rejects
  a retained dataset at the same location, including the default location.
- Auxiliary values are read from Lance, which costs about 0.5 ms per query or point read (20k rows, measured on Apple M-series). Use it for large blobs that you don't want in the SQLite file. Small values that are read often are faster in the default storage.

## `lance0` columns

| Column | Type | Notes |
|---|---|---|
| `rowid` | integer | Required on insert. Use it to join to other tables. |
| `label` | text | Filterable metadata. `label = ?` runs inside Lance. |
| `vector` | blob | float32[dim]. |
| `payload` | blob | Large binary data such as images or audio. Stored only in Lance. |
| `distance` | real | Set by KNN queries. |
| `query`, `k` | hidden | KNN inputs. Supply both to run a vector search. |
| Table name | hidden | Maintenance command input, for example `docs` for a table named `docs`. |

## Maintenance

To merge small data files and update the Lance rowid index before the automatic
thresholds are reached, insert `optimize` into the hidden column named after the
table. This works for both `lance0` and `vec0` tables with `storage=lance`:

```sql
insert into docs(docs) values ('optimize');
insert into items(items) values ('optimize');
```

Maintenance runs when the SQLite transaction commits. A rollback, including
`ROLLBACK TO`, cancels requests made after that transaction or savepoint began.
Old Lance versions remain available for readers and failed-commit recovery.

For `lance0` tables with at least 16,384 rows, `optimize` also creates a Lance
IVF_HNSW_SQ vector index. **After this opt-in, nearest-neighbor results are
approximate.** Candidates are reranked using original vectors, so returned
distances retain their L2 units and values. Tables without this index use exact
vector scans. The index uses 8-bit scalar quantization and supports arbitrary
vector dimensions.

New or updated rows remain searchable before index maintenance: Lance scans
unindexed changes alongside the index. Run `optimize` again to update index
coverage. Building and maintaining the index adds commit time and disk use;
index creation can make queries slower on some datasets. Benchmark your workload
before opting in. `vec0 storage=lance` tables retain their C indexes and do not
build a second vector index in Lance.

## Transactions

`lance0` follows SQLite transactions, together with `vec0` and ordinary tables in the same database:

- **Commit.** During SQLite's commit (`xSync`), `lance0` writes to Lance and records the version Lance reached in the shadow table `<name>_lance`, inside the same transaction. That recorded version is the committed state.
- **Rollback and crashes.** Readers always open Lance at the recorded version. If SQLite rolls back after Lance committed, for example because a later table failed to commit or the process crashed, that Lance version is never visible. The next write restores Lance to the recorded version first.
- **Reading your own writes.** Inside a transaction, reads include the transaction's inserts, updates and deletes.
- **Savepoints.** `ROLLBACK TO` and failed statements undo only their own writes.
- **DROP.** Both `lance0` and `vec0 storage=lance` retain dataset files and remove
  their SQLite version metadata transactionally. A transaction that has written to
  that Lance table or requested maintenance must commit or roll back before DROP.
  Rejected DROP preserves its pending changes. SQLite can report this refusal as
  a generic SQL error rather than forwarding the module's detailed message.

This relies on a single writer, which SQLite already enforces.

## Limits

- Each SQLite transaction that writes to `lance0` creates at least one Lance version and data file. Batch writes into larger transactions.
- `UPDATE` of selected columns patches those columns; changing the rowid rewrites the row.
- Lance vector indexing requires an explicit `optimize` and at least 16,384 rows. Index quality and query speed depend on the data.
- Only `label`, `rowid`, `query` and `k` equality constraints run inside Lance. SQLite applies every other filter after Lance returns rows.
- The library is about 190 MB, because Lance includes DataFusion, Arrow and object storage support.

## Build and test

From the repository root, with Rust and `protoc` (the protobuf compiler) installed:

```sh
./scripts/vendor.sh          # once: SQLite headers for the C core
brew install protobuf        # or set PROTOC=/path/to/protoc
make hybrid                  # builds dist/vec0_lance.{dylib,so,dll}
make test-hybrid             # tests/test-hybrid.py
```

`build.rs` compiles the C core from `../sqlite-vec.c` into this crate. The dependency only goes that way: the C core never depends on Rust, and `make loadable` still builds the dependency-free `dist/vec0`.
