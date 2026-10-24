# Hybrid Search

Hybrid search combines keyword matches with vector similarity. `sqlite-vec`
does not include a keyword ranker of its own. Use SQLite's built-in
[FTS5](https://www.sqlite.org/fts5.html) for BM25 ranking, a `vec0` table for
KNN, and [`vec_rrf()`](../api-reference.md#vec_rrf) to merge the two ranked
lists with Reciprocal Rank Fusion.

## Tables

Give both tables the same rowid for the same document:

```sql
create virtual table fts_docs using fts5(body);
create virtual table vec_docs using vec0(embedding float[768]);

insert into fts_docs(rowid, body) values (:id, :body);
insert into vec_docs(rowid, embedding) values (:id, :embedding);
```

## Query

Rank each list with `row_number()`, join them with a full outer join, and score
each document with `vec_rrf()`. A document that appears in only one list has a
NULL rank in the other list, and `vec_rrf()` skips NULL ranks.

```sql
with knn as (
  select rowid, row_number() over (order by distance) as r
  from vec_docs where embedding match :q_vec and k = 50
),
fts as (
  select rowid, row_number() over (order by rank) as r
  from fts_docs where fts_docs match :q_text limit 50
)
select coalesce(knn.rowid, fts.rowid) as rowid,
       vec_rrf(knn.r, fts.r) as score
from knn full outer join fts on knn.rowid = fts.rowid
order by score desc
limit 10;
```

FTS5's `rank` column is its BM25 score, where lower values are better matches,
so `order by rank` gives rank 1 to the best keyword match. Full outer joins need
SQLite 3.39 or later.

## Weighting

`vec_rrf()` weights every list equally with the constant 60. To weight one list
more, write the RRF terms in SQL:

```sql
select coalesce(knn.rowid, fts.rowid) as rowid,
       coalesce(0.7 / (60 + knn.r), 0) + coalesce(0.3 / (60 + fts.r), 0) as score
from knn full outer join fts on knn.rowid = fts.rowid
order by score desc
limit 10;
```
