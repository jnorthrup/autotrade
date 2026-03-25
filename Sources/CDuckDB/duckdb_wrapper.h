#ifndef DUCKDB_WRAPPER_H
#define DUCKDB_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <duckdb.h>

duckdb_state duckdb_wrapper_open(const char *path, duckdb_database *out_database);
duckdb_state duckdb_wrapper_connect(duckdb_database database, duckdb_connection *out_connection);
duckdb_state duckdb_wrapper_prepare(duckdb_connection connection, const char *query, duckdb_prepared_statement *out_prepared);
duckdb_state duckdb_wrapper_execute_prepared(duckdb_prepared_statement prepared, duckdb_result *out_result);
duckdb_state duckdb_wrapper_bind_varchar(duckdb_prepared_statement prepared, idx_t param_idx, const char *value);
void duckdb_wrapper_destroy_prepare(duckdb_prepared_statement *prepared);
void duckdb_wrapper_destroy_result(duckdb_result *result);
void duckdb_wrapper_disconnect(duckdb_connection *connection);
void duckdb_wrapper_close(duckdb_database *database);
idx_t duckdb_wrapper_column_count(duckdb_result result);
idx_t duckdb_wrapper_row_count(duckdb_result result);
char *duckdb_wrapper_value_varchar(duckdb_result result, idx_t col, idx_t row);
void duckdb_wrapper_free(void *ptr);

#endif
