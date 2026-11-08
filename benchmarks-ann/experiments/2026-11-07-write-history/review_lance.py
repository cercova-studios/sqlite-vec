"""Record SQLite contract counterexamples on fresh fixtures using a trusted hybrid build."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    extension = args.extension.resolve(strict=True)
    args.output.mkdir(parents=True)
    findings = []
    with tempfile.TemporaryDirectory(prefix='vec-rust-contract-review-') as temporary:
        root = Path(temporary)

        def connect(name):
            directory = root / name
            directory.mkdir()
            db = sqlite3.connect(directory / 'fixture.db')
            db.enable_load_extension(True)
            db.load_extension(str(extension))
            db.enable_load_extension(False)
            return db, directory

        db, directory = connect('rollback')
        try:
            dataset = directory / 'rows.lance'
            db.execute(f"CREATE VIRTUAL TABLE t USING vec0(v float[2],+payload blob,storage=lance,storage_uri={literal(dataset)})")
            db.execute("INSERT INTO t(rowid,v,payload) VALUES(1,'[1,0]',X'00FF')")
            db.commit()
            before = db.execute('SELECT rowid,hex(payload) FROM t').fetchall()
            db.execute('BEGIN')
            db.execute('DROP TABLE t')
            exists = dataset.exists()
            db.rollback()
            try:
                after = db.execute('SELECT rowid,hex(payload) FROM t').fetchall()
                error = None
            except sqlite3.Error as failure:
                after, error = None, str(failure)
            findings.append(dict(case='provider_drop_rollback',correct=after==before,
                before=before,after=after,error=error,dataset_exists_after_drop=exists))
        finally:
            db.close()

        db, directory = connect('duplicate')
        try:
            db.execute(f"CREATE VIRTUAL TABLE t USING lance0({literal(directory/'rows.lance')},2)")
            db.execute("INSERT INTO t(rowid,label,vector) VALUES(1,'first','[1,0]')")
            db.commit()
            rejected = False
            try:
                db.execute("INSERT INTO t(rowid,label,vector) VALUES(1,'second','[2,0]')")
                db.commit()
            except sqlite3.Error:
                rejected = True
                db.rollback()
            rows = db.execute('SELECT rowid,label FROM t ORDER BY rowid,label').fetchall()
            findings.append(dict(case='lance0_duplicate_rowid',correct=rejected and rows==[(1,'first')],
                                 duplicate_rejected=rejected,rows=rows))
        finally:
            db.close()

        db, directory = connect('collation')
        try:
            db.execute(f"CREATE VIRTUAL TABLE t USING lance0({literal(directory/'rows.lance')},2)")
            db.execute("INSERT INTO t(rowid,label,vector) VALUES(1,'Cat','[1,0]')")
            db.commit()
            expected = db.execute("SELECT id FROM (SELECT 1 AS id,'Cat' AS label) WHERE label COLLATE NOCASE='cat'").fetchall()
            observed = db.execute("SELECT rowid FROM t WHERE label COLLATE NOCASE='cat'").fetchall()
            findings.append(dict(case='lance0_nocase',correct=observed==expected,expected=expected,observed=observed))
        finally:
            db.close()

        db, directory = connect('text')
        try:
            db.execute(f"CREATE VIRTUAL TABLE t USING vec0(v float[2],+payload text,storage=lance,storage_uri={literal(directory/'rows.lance')})")
            expected = db.execute("SELECT hex(CAST(X'FF' AS TEXT))").fetchone()[0]
            error, observed = None, None
            try:
                db.execute("INSERT INTO t(rowid,v,payload) VALUES(1,'[1,0]',CAST(X'FF' AS TEXT))")
                db.commit()
                observed = db.execute('SELECT hex(payload) FROM t').fetchone()[0]
            except sqlite3.Error as failure:
                db.rollback()
                error = str(failure)
            findings.append(dict(case='provider_invalid_utf8',correct=error is not None or observed==expected,
                                 expected=expected,observed=observed,error=error))
        finally:
            db.close()

        db, directory = connect('schema')
        try:
            db.execute('ATTACH ? AS other',(str(directory/'other.db'),))
            for schema, tag in [('main','a'),('other','b')]:
                db.execute(f"CREATE VIRTUAL TABLE {schema}.t USING vec0(v float[2],+payload text,storage=lance,storage_uri={literal(directory/(tag+'.lance'))})")
                db.commit()
            error, observed = None, None
            try:
                db.execute("INSERT INTO other.t(rowid,v,payload) VALUES(1,'[1,0]','b')")
                db.commit()
                observed = db.execute('SELECT rowid,payload FROM other.t').fetchall()
            except sqlite3.Error as failure:
                error = str(failure)
                db.rollback()
            shadow = db.execute("SELECT name FROM other.sqlite_schema WHERE name='t_storage'").fetchall()
            findings.append(dict(case='provider_attached_schema',correct=observed==[(1,'b')] and bool(shadow),
                                 observed=observed,error=error,other_shadow=shadow,
                                 other_dataset_exists=(directory/'b.lance').exists()))
        finally:
            db.close()
    with extension.open('rb') as stream:
        artifact_hash = hashlib.file_digest(stream,'sha256').hexdigest()
    result = dict(review_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  artifact=dict(path=str(extension),sha256=artifact_hash,provenance='existing build; fresh check blocked by missing protoc'),
                  sqlite=sqlite3.sqlite_version,findings=findings,
                  violations=sum(not r['correct'] for r in findings))
    (args.output/'lance-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(violations=result['violations'],cases=[r['case'] for r in findings])))


if __name__=='__main__':
    main()
