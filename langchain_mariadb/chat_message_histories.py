"""Client for persisting chat message history in a MariaDB database."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import List, Optional, Sequence, Union, Any

import mariadb
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from sqlalchemy import Engine, create_engine

from langchain_mariadb._utils import enquote_identifier


logger = logging.getLogger(__name__)


def _create_table_and_index(table_name: str) -> List[str]:
    """Create SQL queries for table and index creation.

    Args:
        table_name: Name of the table to create

    Returns:
        List of SQL statements for creating table and index
    """
    escaped_table = enquote_identifier(table_name)
    index_name = f"idx_{table_name}_session_id"
    re.sub(r"[^0-9a-zA-Z_]", "", index_name)

    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS {escaped_table} (
            id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
            session_id UUID NOT NULL,
            message JSON NOT NULL,
            created_at TIMESTAMP(6) NOT NULL DEFAULT NOW()
        )
        """,
        f"CREATE INDEX IF NOT EXISTS {index_name} ON {escaped_table} (session_id)",
    ]
    return statements


def _set_datasource(
    datasource: Union[mariadb.ConnectionPool | Engine | str],
    engine_args: Optional[dict[str, Any]] = None,
) -> mariadb.ConnectionPool | Engine:
    if isinstance(datasource, str):
        return create_engine(url=datasource, **(engine_args or {}))
    elif isinstance(datasource, Engine):
        return datasource
    elif isinstance(datasource, mariadb.ConnectionPool):
        return datasource
    else:
        raise ValueError(
            "datasource should be a connection string, an instance of "
            "sqlalchemy.engine.Engine or a mariadb pool"
        )


class MariaDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that persists to a MariaDB database."""

    def __init__(
        self,
        table_name: str,
        session_id: str,
        /,
        *,
        datasource: Union[mariadb.ConnectionPool | Engine | str],
        engine_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a chat message history that persists to a MariaDB database.

        This client stores chat messages in a MariaDB table with the following schema:
        - id: Auto-incrementing primary key
        - session_id: UUID to group messages by conversation
        - message: JSON column storing the message content
        - created_at: Timestamp of message creation

        Args:
            table_name: Name of the database table to use (must be alphanumeric + underscores)
            session_id: UUID string to identify the chat session
            datasource: datasource (connection string, sqlalchemy engine or MariaDB connection pool)

        Example:
            ```python
            from langchain_core.messages import HumanMessage, AIMessage
            import mariadb
            import uuid

            # Setup connection pool
            pool = mariadb.ConnectionPool(
                pool_name="chat_pool",
                user="root",
                host="localhost",
                database="chatdb"
            )

            # Create tables if needed
            MariaDBChatMessageHistory.create_tables(pool, "chat_messages")

            # Initialize history for a session
            history = MariaDBChatMessageHistory(
                "chat_messages",
                str(uuid.uuid4()),
                datasource=pool
            )

            # Add messages
            history.add_messages([
                HumanMessage(content="Hello!"),
                AIMessage(content="Hi there!")
            ])

            # Retrieve messages
            messages = history.messages
            ```

        Raises:
            ValueError: If pool is not provided, session_id is not a valid UUID,
                       or table_name contains invalid characters
        """
        self._datasource = _set_datasource(datasource, engine_args)

        # Validate that session id is a UUID
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise ValueError(
                f"Invalid session id. Session id must be a valid UUID. Got {session_id}"
            )

        self._session_id = session_id

        if not re.match(r"^\w+$", table_name):
            raise ValueError(
                "Invalid table name. Table name must contain only alphanumeric "
                "characters and underscores."
            )
        self._table_name = table_name
        self._escaped_table = enquote_identifier(table_name)

    def get_connection(self):
        if isinstance(self._datasource, mariadb.ConnectionPool):
            return self._datasource.get_connection()
        else:
            return self._datasource.raw_connection()

    @staticmethod
    def create_tables(
        datasource: Union[mariadb.ConnectionPool | Engine | str],
        table_name: str,
        /,
    ) -> None:
        """Create the table schema in the database and create relevant indexes.

        Args:
            datasource: datasource (connection string, sqlalchemy engine or MariaDB connection pool)
            table_name: Name of the table to create
        """
        queries = _create_table_and_index(table_name)
        logger.info("Creating schema for table %s", table_name)
        ds = _set_datasource(datasource)
        if isinstance(ds, mariadb.ConnectionPool):
            con = ds.get_connection()
        else:
            con = ds.raw_connection()
        try:
            with con.cursor() as cursor:
                for query in queries:
                    cursor.execute(query)
                cursor.close()
            con.commit()
        finally:
            con.close()

    @staticmethod
    def drop_table(
        datasource: Union[mariadb.ConnectionPool | Engine | str], table_name: str, /
    ) -> None:
        """Delete the table schema from the database.

        WARNING: This will delete the table and all its data permanently.

        Args:
            datasource: datasource (connection string, sqlalchemy engine or MariaDB connection pool)
            table_name: Name of the table to drop
        """
        escaped_table = enquote_identifier(table_name)
        query = f"DROP TABLE IF EXISTS {escaped_table}"

        logger.info("Dropping table %s", table_name)
        ds = _set_datasource(datasource)
        if isinstance(ds, mariadb.ConnectionPool):
            con = ds.get_connection()
        else:
            con = ds.raw_connection()
        try:
            with con.cursor() as cursor:
                cursor.execute(query)
            con.commit()
        finally:
            con.close()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the chat history.

        Args:
            messages: Sequence of messages to add
        """
        values = [
            (self._session_id, json.dumps(message_to_dict(message)))
            for message in messages
        ]

        query = f"INSERT INTO {self._escaped_table} (session_id, message) VALUES (?, ?)"

        con = self.get_connection()
        try:
            with con.cursor() as cursor:
                cursor.executemany(query, values)
            con.commit()
        finally:
            con.close()

    def get_messages(self) -> List[BaseMessage]:
        """Retrieve messages from the chat history.

        Returns:
            List of messages in chronological order
        """
        query = f"SELECT message FROM {self._escaped_table} WHERE session_id = ? ORDER BY id"

        con = self.get_connection()
        try:
            with con.cursor() as cursor:
                cursor.execute(query, (self._session_id,))
                items = [json.loads(record[0]) for record in cursor.fetchall()]
        finally:
            con.close()
        return messages_from_dict(items)

    @property  # type: ignore[override]
    def messages(self) -> List[BaseMessage]:
        """Get all messages in the chat history."""
        return self.get_messages()

    def clear(self) -> None:
        """Clear all messages for the current session."""
        query = f"DELETE FROM {self._escaped_table} WHERE session_id = ?"

        con = self.get_connection()
        try:
            with con.cursor() as cursor:
                cursor.execute(query, (self._session_id,))
            con.commit()
        finally:
            con.close()
