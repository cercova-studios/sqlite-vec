/* Real SQLite conversion OOM against the registered provider, in a fresh host. */
#define SQLITE_CORE 1
#include "sqlite-vec.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NDEBUG
#error "The fault driver requires assertions"
#endif

static sqlite3_mem_methods original;
static int fail_next, failures;

static void *fault_malloc(int bytes) {
  if (fail_next) { fail_next = 0; failures++; return NULL; }
  return original.xMalloc(bytes);
}

static void *fault_realloc(void *pointer, int bytes) {
  if (fail_next) { fail_next = 0; failures++; return NULL; }
  return original.xRealloc(pointer, bytes);
}

static int count_row(void *context, sqlite3_int64 id, const void *data, int bytes) {
  (void)data;
  (void)bytes;
  assert(id == 3);
  (*(int *)context)++;
  return SQLITE_OK;
}

typedef struct {
  const sqlite3_vec_storage_methods *provider;
  sqlite3_vec_storage *store;
  int text_fault, inject, rc, calls;
} Fault;

static void fault_put(sqlite3_context *context, int argc, sqlite3_value **values) {
  Fault *fault = sqlite3_user_data(context);
  assert(argc == 2);
  assert(sqlite3_value_type(values[0]) == SQLITE_TEXT);
  assert(sqlite3_value_type(values[1]) == SQLITE_BLOB);
  if (fault->inject) {
    if (fault->text_fault) {
      assert(sqlite3_value_encoding(values[0]) == SQLITE_UTF16LE);
    } else {
      assert(sqlite3_value_text(values[0]) != NULL);
    }
    /* Buffers exceed lookaside slots, so conversion must reach xMalloc. */
    fail_next = 1;
  }
  fault->rc = fault->provider->xPut(fault->store, fault->inject ? 1 : 3, values, NULL);
  fault->calls++;
  sqlite3_result_int(context, fault->rc);
}

int main(int argc, char **argv) {
  assert(argc == 4);
  assert(sqlite3_config(SQLITE_CONFIG_GETMALLOC, &original) == SQLITE_OK);
  sqlite3_mem_methods methods = original;
  methods.xMalloc = fault_malloc;
  methods.xRealloc = fault_realloc;
  assert(sqlite3_config(SQLITE_CONFIG_MALLOC, &methods) == SQLITE_OK);
  sqlite3 *db = NULL;
  assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
  assert(sqlite3_exec(db, "PRAGMA encoding='UTF-16le'", NULL, NULL, NULL) == SQLITE_OK);
  assert(sqlite3_enable_load_extension(db, 1) == SQLITE_OK);
  char *error = NULL;
  assert(sqlite3_load_extension(db, argv[1], NULL, &error) == SQLITE_OK);
  assert(sqlite3_enable_load_extension(db, 0) == SQLITE_OK);
  const sqlite3_vec_storage_methods *provider = sqlite3_get_clientdata(db, SQLITE_VEC_STORAGE_KEY_PREFIX "lance");
  assert(provider && provider->iVersion >= 1);
  const char *names[] = {"text", "binary"};
  const int types[] = {SQLITE_TEXT, SQLITE_BLOB};
  sqlite3_vec_storage *store = NULL;
  assert(provider->xOpen(db, "main", "fault", argv[2], 1, 2, names, types,
                        0, NULL, NULL, NULL, &store, &error) == SQLITE_OK);
  Fault fault = {provider, store, strcmp(argv[3], "text") == 0, 1, -1, 0};
  assert(sqlite3_create_function(db, "fault_put", 2, SQLITE_UTF16LE, &fault,
                                 fault_put, NULL, NULL) == SQLITE_OK);
  sqlite3_stmt *stmt = NULL;
  /* A nonconstant expression avoids OP_Copy expanding the lazy zeroblob. */
  assert(sqlite3_prepare_v2(db, "SELECT fault_put(?,zeroblob(8192+(random()&0)))", -1, &stmt, NULL) == SQLITE_OK);
  unsigned short text16[4097];
  for (int i = 0; i < 4096; i++) text16[i] = 'a';
  text16[4096] = 0;
  if (fault.text_fault) {
    assert(sqlite3_bind_text16(stmt, 1, text16, 8192, SQLITE_STATIC) == SQLITE_OK);
  } else {
    assert(sqlite3_bind_text(stmt, 1, "plain", -1, SQLITE_STATIC) == SQLITE_OK);
  }
  assert(sqlite3_step(stmt) == SQLITE_NOMEM);
  fprintf(stderr, "fault mode=%s failures=%d armed=%d rc=%d\n", argv[3], failures, fail_next, fault.rc);
  assert(fault.calls == 1 && failures == 1 && !fail_next);
  if (fault.rc != SQLITE_NOMEM) {
    fprintf(stderr, "conversion OOM returned %d instead of SQLITE_NOMEM: %s\n", fault.rc, provider->xErrmsg(store));
    return 2;
  }
  sqlite3_finalize(stmt); /* clears the host's allocation-failure state */
  assert(provider->xSync(store) == SQLITE_OK);
  int count = 0;
  assert(provider->xScan(store, 0, count_row, &count) == SQLITE_OK && count == 0);
  fault.inject = 0;
  assert(sqlite3_prepare_v2(db, "SELECT fault_put('',x'')", -1, &stmt, NULL) == SQLITE_OK);
  assert(sqlite3_step(stmt) == SQLITE_ROW);
  assert(fault.calls == 2 && fault.rc == SQLITE_OK);
  assert(sqlite3_finalize(stmt) == SQLITE_OK);
  assert(provider->xSync(store) == SQLITE_OK);
  assert(provider->xCommit(store) == SQLITE_OK);
  assert(provider->xScan(store, 0, count_row, &count) == SQLITE_OK && count == 1);
  provider->xClose(store);
  assert(sqlite3_close(db) == SQLITE_OK);
  puts("conversion OOM rejected without staging a row; empty-value retry committed");
  return 0;
}
