"""Get fixtures for the database connection."""

import os
from contextlib import contextmanager

import mariadb

from langchain_mariadb import MariaDBStore

USER = os.environ.get("MARIADB_USER", "langchain")
PASSWORD = os.environ.get("MARIADB_PASSWORD", "langchain")
HOST = os.environ.get("MARIADB_HOST", "localhost")
PORT = int(os.environ.get("MARIADB_PORT", "3306"))
DB = os.environ.get("MARIADB_DATABASE", "langchain")

default_conf = {
    "user": USER,
    "host": HOST,
    "database": DB,
    "port": PORT,
    "password": PASSWORD,
}

URL = f"mariadb+mariadbconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"


def url() -> str:
    return URL


@contextmanager
def pool() -> mariadb.ConnectionPool:
    # Establish a connection to your test database
    pool = mariadb.ConnectionPool(pool_name="mariadb_test", pool_size=1, **default_conf)
    try:
        yield pool
    finally:
        # Cleanup: close the pool after the test is done
        with pool.get_connection() as con:
            with con.cursor() as cursor:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS langchain_embedding")
                    cursor.execute(f"DROP TABLE IF EXISTS langchain_collection")
                except Exception as e:
                    pass
        pool.close()
