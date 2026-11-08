#include "sqlite3.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef NDEBUG
#error Assertions are required for this experiment
#endif

#define OK(call) do { int rc_ = (call); if (rc_ != SQLITE_OK) { \
  fprintf(stderr,"%s returned %d at line %d\n",#call,rc_,__LINE__); abort(); } } while(0)

static void sql(sqlite3 *db, const char *text) {
  char *error = 0;
  int rc = sqlite3_exec(db,text,0,0,&error);
  if (rc != SQLITE_OK) { fprintf(stderr,"%s: %s\n",text,error); sqlite3_free(error); abort(); }
}
static sqlite3_int64 scalar(sqlite3 *db, const char *text) {
  sqlite3_stmt *s = 0; OK(sqlite3_prepare_v2(db,text,-1,&s,0));
  assert(sqlite3_step(s)==SQLITE_ROW);
  sqlite3_int64 value = sqlite3_column_int64(s,0); OK(sqlite3_finalize(s)); return value;
}
static double now(void) {
  struct timespec t; assert(clock_gettime(CLOCK_MONOTONIC,&t)==0);
  return t.tv_sec + t.tv_nsec/1e9;
}
static sqlite3 *database(const char *path, int rows, int logging) {
  if (strcmp(path,":memory:")) { int fd=open(path,O_CREAT|O_EXCL|O_RDWR,0600); assert(fd>=0); close(fd); }
  sqlite3 *db=0; OK(sqlite3_open(path,&db));
  sql(db,"PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;"
         "CREATE TABLE tasks(id INTEGER PRIMARY KEY,rev INTEGER NOT NULL,body TEXT NOT NULL,payload BLOB);"
         "CREATE TABLE requests(batch INTEGER PRIMARY KEY,actor TEXT NOT NULL);"
         "CREATE TABLE context(batch INTEGER NOT NULL);"
         "CREATE TABLE captured(batch INTEGER PRIMARY KEY REFERENCES requests,change BLOB NOT NULL);"
         "CREATE TABLE audit(batch INTEGER REFERENCES requests,id INTEGER,old_rev,new_rev,old_body,new_body,"
         "old_blob,new_blob,PRIMARY KEY(batch,id)) WITHOUT ROWID; BEGIN;");
  sqlite3_stmt *insert=0; OK(sqlite3_prepare_v2(db,"INSERT INTO tasks VALUES(?,1,'initial',X'00FF')",-1,&insert,0));
  for (int i=1;i<=rows;i++) { OK(sqlite3_bind_int(insert,1,i)); assert(sqlite3_step(insert)==SQLITE_DONE); OK(sqlite3_reset(insert)); }
  OK(sqlite3_finalize(insert)); sql(db,"COMMIT");
  if (logging) sql(db,
    "CREATE TRIGGER capture AFTER UPDATE ON tasks BEGIN "
    "INSERT INTO audit SELECT batch,old.id,old.rev,new.rev,old.body,new.body,old.payload,new.payload FROM context WHERE 1 "
    "ON CONFLICT(batch,id) DO UPDATE SET new_rev=excluded.new_rev,new_body=excluded.new_body,new_blob=excluded.new_blob; END;");
  return db;
}
static sqlite3_session *session(sqlite3 *db) {
  sqlite3_session *s=0; OK(sqlite3session_create(db,"main",&s)); OK(sqlite3session_attach(s,"tasks")); return s;
}
static int persist(sqlite3 *db, sqlite3_session *s, int batch) {
  int bytes=0; void *blob=0; OK(sqlite3session_changeset(s,&bytes,&blob));
  sqlite3_stmt *p=0; OK(sqlite3_prepare_v2(db,"INSERT INTO captured VALUES(?,?)",-1,&p,0));
  OK(sqlite3_bind_int(p,1,batch)); OK(sqlite3_bind_blob(p,2,blob,bytes,SQLITE_TRANSIENT));
  assert(sqlite3_step(p)==SQLITE_DONE); OK(sqlite3_finalize(p)); sqlite3_free(blob); return bytes;
}
static int abort_conflict(void *context, int type, sqlite3_changeset_iter *iterator) {
  (void)iterator; assert(type==SQLITE_CHANGESET_DATA); (*(int *)context)++; return SQLITE_CHANGESET_ABORT;
}
static void verify(void) {
  sqlite3 *db=database(":memory:",2,0);
  sql(db,"CREATE TABLE nopk(value); CREATE TABLE nullable(id TEXT PRIMARY KEY,value); CREATE VIRTUAL TABLE lexical USING fts5(body);");
  sqlite3_session *s=session(db);
  OK(sqlite3session_attach(s,"nopk")); OK(sqlite3session_attach(s,"nullable")); OK(sqlite3session_attach(s,"lexical"));
  sql(db,"BEGIN; INSERT INTO requests VALUES(1,'actor'); INSERT INTO context VALUES(1);"
         "UPDATE tasks SET rev=2,body='middle',payload=X'FF00'; SAVEPOINT s;"
         "UPDATE tasks SET rev=99,body='discard'; ROLLBACK TO s; RELEASE s;"
         "UPDATE tasks SET rev=3,body='final'; INSERT INTO nopk VALUES('invisible');"
         "INSERT INTO nullable VALUES(NULL,'invisible'); INSERT INTO lexical VALUES('invisible');");
  assert(persist(db,s,1)>0); sql(db,"DELETE FROM context; COMMIT;");
  int size=0; void *data=0; OK(sqlite3session_changeset(s,&size,&data)); sqlite3session_delete(s);
  sqlite3_changeset_iter *it=0; OK(sqlite3changeset_start(&it,size,data)); int count=0,rc;
  while ((rc=sqlite3changeset_next(it))==SQLITE_ROW) {
    const char *table=0; int columns=0,op=0,indirect=0;
    OK(sqlite3changeset_op(it,&table,&columns,&op,&indirect));
    assert(!strcmp(table,"tasks") && columns==4 && op==SQLITE_UPDATE);
    sqlite3_value *old=0,*new=0; OK(sqlite3changeset_old(it,3,&old)); OK(sqlite3changeset_new(it,3,&new));
    assert(sqlite3_value_type(old)==SQLITE_BLOB && sqlite3_value_bytes(old)==2);
    assert(sqlite3_value_type(new)==SQLITE_BLOB && sqlite3_value_bytes(new)==2);
    assert(!memcmp(sqlite3_value_blob(old),"\x00\xff",2));
    assert(!memcmp(sqlite3_value_blob(new),"\xff\x00",2)); count++;
  }
  assert(rc==SQLITE_DONE && count==2); OK(sqlite3changeset_finalize(it));
  int inverse_size=0; void *inverse=0; OK(sqlite3changeset_invert(size,data,&inverse_size,&inverse));
  int conflicts=0;
  OK(sqlite3changeset_apply(db,inverse_size,inverse,0,abort_conflict,&conflicts));
  assert(scalar(db,"SELECT count(*) FROM tasks WHERE rev=1 AND body='initial' AND typeof(payload)='blob' AND hex(payload)='00FF'")==2);
  /* Raw inversion restores the old revision too: it is not a monotonic agent undo. */
  OK(sqlite3changeset_apply(db,size,data,0,abort_conflict,&conflicts));
  sql(db,"UPDATE tasks SET rev=4,body='another writer' WHERE id=1");
  assert(sqlite3changeset_apply(db,inverse_size,inverse,0,abort_conflict,&conflicts)==SQLITE_ABORT);
  assert(conflicts==1 && scalar(db,"SELECT rev FROM tasks WHERE id=1")==4);
  assert(scalar(db,"SELECT rev FROM tasks WHERE id=2")==3);
  sqlite3_free(data); sqlite3_free(inverse);
  s=session(db); sql(db,"BEGIN; UPDATE tasks SET rev=100; ROLLBACK;");
  OK(sqlite3session_changeset(s,&size,&data)); assert(size==0); sqlite3_free(data); sqlite3session_delete(s);
  OK(sqlite3_close(db));
  puts("{\"checks\":[\"typed_blob\",\"net_changes\",\"savepoint\",\"rollback\",\"strict_conflict_atomicity\",\"inverse_rewinds_revision\",\"pk_null_virtual_exclusions\"]}");
}
int main(int argc, char **argv) {
  if (argc==2 && !strcmp(argv[1],"verify")) { verify(); return 0; }
  assert(argc==5);
  const char *mode=argv[1]; int rows=atoi(argv[3]),updates=atoi(argv[4]);
  assert(rows>0 && rows<=100000 && (updates==1 || updates==2));
  int logging=!strcmp(mode,"log") || strstr(mode,"_log")!=0;
  int capturing=!strcmp(mode,"session") || strstr(mode,"_session")!=0;
  int crash=!strncmp(mode,"crash_",6), abrupt_commit=!strncmp(mode,"commit_",7);
  assert(logging || capturing || !strcmp(mode,"off"));
  sqlite3 *db=database(argv[2],rows,logging);
  sqlite3_int64 used=0,peak=0,session_peak=0;
  OK(sqlite3_status64(SQLITE_STATUS_MEMORY_USED,&used,&peak,1));
  sqlite3_stmt *update=0; OK(sqlite3_prepare_v2(db,"UPDATE tasks SET rev=rev+1,body=? WHERE id=?",-1,&update,0));
  int bytes=0,batches=0; double start=now();
  for (int begin=1;begin<=rows;begin+=100) {
    batches++; sql(db,"BEGIN IMMEDIATE");
    char text[160]; snprintf(text,sizeof(text),"INSERT INTO requests VALUES(%d,'actor'); INSERT INTO context VALUES(%d);",batches,batches); sql(db,text);
    sqlite3_session *s=capturing?session(db):0;
    for (int i=begin;i<begin+100 && i<=rows;i++) for (int pass=0;pass<updates;pass++) {
      OK(sqlite3_bind_text(update,1,pass?"final":"middle",-1,SQLITE_STATIC)); OK(sqlite3_bind_int(update,2,i));
      assert(sqlite3_step(update)==SQLITE_DONE); OK(sqlite3_reset(update));
    }
    if (s) {
      sqlite3_int64 memory=sqlite3session_memory_used(s); if(memory>session_peak)session_peak=memory;
      bytes+=persist(db,s,batches); sqlite3session_delete(s);
    }
    sql(db,"DELETE FROM context");
    if (crash) _exit(17);
    sql(db,"COMMIT"); if(abrupt_commit)_exit(0);
  }
  double seconds=now()-start; OK(sqlite3_finalize(update));
  OK(sqlite3_status64(SQLITE_STATUS_MEMORY_USED,&used,&peak,0));
  OK(sqlite3_wal_checkpoint_v2(db,0,SQLITE_CHECKPOINT_TRUNCATE,0,0));
  printf("{\"sqlite\":\"%s\",\"mode\":\"%s\",\"rows\":%d,\"updates\":%d,\"seconds\":%.9f,\"changeset_bytes\":%d,\"session_peak_bytes\":%lld,\"sqlite_peak_bytes\":%lld,\"database_bytes\":%lld}\n",
    sqlite3_libversion(),mode,rows,updates,seconds,bytes,(long long)session_peak,(long long)peak,
    (long long)(scalar(db,"PRAGMA page_count")*scalar(db,"PRAGMA page_size")));
  OK(sqlite3_close(db)); return 0;
}
