#include "duckdb_wrapper.h"

duckdb_state duckdb_wrapper_open(const char *path, duckdb_database *out_database) {
    return duckdb_open(path, out_database);
}

duckdb_state duckdb_wrapper_connect(duckdb_database database, duckdb_connection *out_connection) {
    return duckdb_connect(database, out_connection);
}

duckdb_state duckdb_wrapper_query(duckdb_connection connection, const char *query, duckdb_result *out_result) {
    return duckdb_query(connection, query, out_result);
}

void duckdb_wrapper_destroy_result(duckdb_result *result) {
    duckdb_destroy_result(result);
}

void duckdb_wrapper_disconnect(duckdb_connection *connection) {
    duckdb_disconnect(connection);
}

void duckdb_wrapper_close(duckdb_database *database) {
    duckdb_close(database);
}

uint64_t duckdb_wrapper_column_count(duckdb_result result) {
    return duckdb_column_count(&result);
}

uint64_t duckdb_wrapper_row_count(duckdb_result result) {
    return duckdb_row_count(&result);
}

char *duckdb_wrapper_value_varchar(duckdb_result result, uint64_t col, uint64_t row) {
    return duckdb_value_varchar(&result, col, row);
}

void duckdb_wrapper_free(void *ptr) {
    duckdb_free(ptr);
}
