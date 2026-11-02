"""Prepare pinned public SciFact inputs; no network or model inference."""
from pathlib import Path
import gzip
import hashlib
import json
import sqlite3

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '2026-10-26-scifact-hybrid'
data_path = PREVIOUS / '.work/data.json'
manifest = json.loads((PREVIOUS / 'recorded/data-manifest.json').read_text())
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest['files']['data.json']
data = json.loads(data_path.read_text())
receipts = json.loads(gzip.decompress((PREVIOUS / 'recorded/receipts.json.gz').read_bytes()))
gold = {(r['scenario'], r['query_id']): r['candidates'] for r in receipts['results'] if r['method'] == 'keyword'}
db = sqlite3.connect(':memory:')
db.executescript("create virtual table tokens using fts5(text); create virtual table terms using fts5vocab(tokens,'instance');")
queries = []
for query in data['queries']:
    db.execute('delete from tokens')
    db.execute('insert into tokens values (?)', (query['text'],))
    terms = list(dict.fromkeys(r[0] for r in db.execute('select term from terms order by offset')))
    expression = ' OR '.join('"' + term.replace('"', '""') + '"' for term in terms)
    queries.append(dict(id=query['_id'], text=expression, expected={s:gold[s,query['_id']] for s in ('full','scoped')}))
output = dict(documents=[dict(id=int(d['_id']), body=d['title']+' '+d['text'], bucket=hashlib.sha256(d['_id'].encode()).digest()[0]%10) for d in data['documents']],queries=queries,timed_ids=sorted(receipts['timed_query_ids'],key=int)[:8],data_sha256=manifest['files']['data.json'])
work = HERE / '.work'
work.mkdir(exist_ok=True)
payload = json.dumps(output).encode()
(work / 'input.json').write_bytes(payload)
print(json.dumps(dict(input_sha256=hashlib.sha256(payload).hexdigest(),data_sha256=output['data_sha256'],timed_ids=output['timed_ids']),indent=2))
