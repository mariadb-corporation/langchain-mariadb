"""Get fixtures for the database connection."""

import os

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
