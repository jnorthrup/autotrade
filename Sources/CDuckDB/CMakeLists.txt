cmake_minimum_required(VERSION 3.20)
project(DuckDBWrapper C)

# Find DuckDB
set(DUCKDB_INCLUDE_DIR "/opt/homebrew/include")
set(DUCKDB_LIBRARY "/opt/homebrew/lib/libduckdb.dylib")

if(NOT EXISTS ${DUCKDB_INCLUDE_DIR}/duckdb.h)
    message(FATAL_ERROR "DuckDB header not found at ${DUCKDB_INCLUDE_DIR}/duckdb.h")
endif()

if(NOT EXISTS ${DUCKDB_LIBRARY})
    message(FATAL_ERROR "DuckDB library not found at ${DUCKDB_LIBRARY}")
endif()

add_library(duckdb_wrapper STATIC duckdb_wrapper.c)
target_link_libraries(duckdb_wrapper PRIVATE ${DUCKDB_LIBRARY})
target_include_directories(duckdb_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${DUCKDB_INCLUDE_DIR})