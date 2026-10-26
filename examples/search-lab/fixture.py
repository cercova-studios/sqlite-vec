"""Create a new, original-text deterministic fixture; never replace a database."""

import argparse
import json
from pathlib import Path
import sqlite3


def create(directory, extension):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "fixture.db"
    # Exclusive creation prevents accidentally replacing an existing corpus.
    with path.open("xb"):
        pass
    db = sqlite3.connect(path)
    try:
        db.enable_load_extension(True)
        db.load_extension(str(Path(extension).resolve()))
        db.enable_load_extension(False)
        db.executescript("""
          create table documents(id integer primary key,document_id text,tenant text,
            valid_from integer,valid_to integer,invalidated_at integer,source_id text,
            body text,start_ms integer,end_ms integer);
          create virtual table vectors using vec0(embedding float[2]);
          create virtual table lexical using fts5(body);
          create table edges(src integer,dst integer);
        """)
        records = [
            (1,"ours",0,None,None,"Deploy a small database safely.",[1,0]),
            (2,"ours",0,None,None,"Deploy a small database safely.",[1,0]),
            (3,"ours",0,None,None,"Graph expansion finds related functions.",[0,2]),
            (4,"ours",0,None,None,"A rollback retains earlier database evidence.",[3,0]),
            (5,"ours",0,None,None,"An isolated node has useful evidence.",[4,0]),
            (6,"other",0,None,None,"deploy deploy deploy",[0,0]),
            (7,"ours",0,100,None,"deploy",[0,0]),
            (8,"ours",101,None,None,"deploy",[0,0]),
            (9,"ours",0,None,90,"deploy",[0,0]),
            (10,"ours",100,101,None,"Deploy a service with evidence.",[2,0]),
        ]
        for key, tenant, start, end, invalid, text, vector in records:
            db.execute("insert into documents values (?,?,?,?,?,?,?,?,?,?)",
                       (key,f"doc:{key}",tenant,start,end,invalid,f"source:{key}",text,key*1000,key*1000+500))
            db.execute("insert into lexical(rowid,body) values (?,?)", (key,text))
            db.execute("insert into vectors(rowid,embedding) values (?,?)", (key,json.dumps(vector)))
        db.executemany("insert into edges values (?,?)", [(1,2),(1,3),(2,4),(3,4),(4,1),(1,2),(4,4),(6,1),(1,7)])
        # Configured graph source has the same fixed host scope as retrieval.
        db.executescript("""
          create view admitted_edges as select src,dst from edges
          where src in (select id from documents where tenant='ours' and invalidated_at is null
              and valid_from<=100 and (valid_to is null or valid_to>100))
            and dst in (select id from documents where tenant='ours' and invalidated_at is null
              and valid_from<=100 and (valid_to is null or valid_to>100));
        """)
        db.commit()
    finally:
        db.close()
    config = dict(database=str(path), extension=str(Path(extension).resolve()), corpora={"demo":dict(
        documents="documents", vectors="vectors", fts="lexical", edges="admitted_edges",
        tenant="ours", as_of=100, dimensions=2)})
    config_path = directory / "config.json"
    with config_path.open("x") as out:
        json.dump(config,out,indent=2)
        out.write("\n")
    return config_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--extension", required=True)
    a = p.parse_args()
    print(create(a.directory,a.extension))
