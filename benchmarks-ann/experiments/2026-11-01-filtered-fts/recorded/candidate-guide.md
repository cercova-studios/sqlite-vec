# Hybrid Search

> Historical experiment: only written instructions and results remain here.
> Recover scripts/receipts from Git `4c5a5d1` and run the commands below
> in that isolated snapshot. See the [reproduction index](../../README.md).

Hybrid search combines keyword matches with vector similarity. `sqlite-vec`
does not include a keyword ranker of its own. Use SQLite's built-in
[FTS5](https://www.sqlite.org/fts5.html) for BM25 ranking, a `vec0` table for
KNN, and [`vec_rrf()`](https://github.com/cercova-studios/sqlite-vec/blob/4c5a5d1fcb3e97eaa5389761ca0df944c41cc2dd/site/api-reference.md#vec_rrf) to merge the two ranked
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

## Filter candidates by scope and validity

When documents belong to a tenant or have validity windows, apply eligibility
inside **both** candidate branches, before their limits. Filtering a global
top-k afterward can leave no results even when eligible documents exist farther
down each list.

Keep application attributes under the same document ID as the text and vector:

```sql
create table doc_scope(
  id integer primary key,
  tenant text not null,
  valid_from integer not null,
  valid_to integer,
  invalidated_at integer,
  source_id text not null
);
```

Use Unix seconds for the timestamps. A document is eligible from `valid_from`
inclusive to `valid_to` exclusive; NULL `valid_to` means no end. This example
excludes any record with a non-NULL `invalidated_at`. Bind `:tenant`, `:as_of`,
`:q_vec`, `:q_text`, and a positive `:branch_limit` of at most 4096.

<!-- scoped-hybrid-query -->
```sql
with eligible as materialized (
  select id from doc_scope
  where tenant = :tenant and invalidated_at is null
    and valid_from <= :as_of
    and (valid_to is null or valid_to > :as_of)
),
knn as (
  select rowid, distance, row_number() over (order by distance) as r
  from vec_docs
  where embedding match :q_vec and k = :branch_limit
    and rowid in (select id from eligible)
),
fts as (
  select rowid, row_number() over (order by rank, rowid) as r
  from fts_docs
  where fts_docs match :q_text
    and exists (select 1 from eligible where eligible.id = fts_docs.rowid)
  order by rank, rowid limit :branch_limit
)
select coalesce(knn.rowid, fts.rowid) as id,
       vec_rrf(knn.r, fts.r) as score,
       knn.r as vector_rank, fts.r as keyword_rank,
       knn.distance as vector_distance, scope.source_id
from knn full outer join fts on knn.rowid = fts.rowid
join doc_scope scope on scope.id = coalesce(knn.rowid, fts.rowid)
order by score desc, id
limit 10;
```

The keyword branch uses a correlated `EXISTS` check so SQLite can scan the
full-text matches and test each row's eligibility before the branch limit.
A rowid `IN` constraint here can instead restart the full-text query for every
eligible ID. Keep the vector branch's rowid `IN` constraint: `vec0` uses it to
apply the filter during KNN search. Both predicates preserve the integer-ID
membership semantics of these tables.

The branch ranks and source ID let callers explain each result. A keyword-only
result has NULL vector rank and distance. Selective ANN filters can return fewer
than `:branch_limit` candidates. Intersect graph-expanded rowids with the same
eligible set before applying candidate limits.

Update attributes, text and vectors in one SQLite transaction. Invalidating an
attribute row suppresses it from this query while retaining its stored data.
Redaction, retention and access enforcement belong in the application; these
predicates describe retrieval eligibility.

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
