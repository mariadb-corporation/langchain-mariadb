import uuid

import sqlalchemy
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_mariadb.chat_message_histories import MariaDBChatMessageHistory
from tests.utils import pool, url


def test_sync_chat_history() -> None:
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=123))
    with pool() as tmppool:
        MariaDBChatMessageHistory.drop_table(tmppool, table_name)
        MariaDBChatMessageHistory.create_tables(tmppool, table_name)

        chat_history = MariaDBChatMessageHistory(
            table_name, session_id, datasource=tmppool
        )
        run_test(chat_history)


def test_sync_chat_history_url() -> None:
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=123))
    url_value = url()
    MariaDBChatMessageHistory.drop_table(url_value, table_name)
    MariaDBChatMessageHistory.create_tables(url_value, table_name)

    chat_history = MariaDBChatMessageHistory(
        table_name, session_id, datasource=url_value
    )
    run_test(chat_history)


def test_sync_chat_history_sqlalchemy() -> None:
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=123))
    engine = sqlalchemy.create_engine(url())
    MariaDBChatMessageHistory.drop_table(engine, table_name)
    MariaDBChatMessageHistory.create_tables(engine, table_name)

    chat_history = MariaDBChatMessageHistory(table_name, session_id, datasource=engine)
    run_test(chat_history)


def run_test(chat_history: MariaDBChatMessageHistory) -> None:
    messages = chat_history.messages
    assert messages == []
    assert chat_history is not None

    # Get messages from the chat history
    messages = chat_history.messages
    assert messages == []

    chat_history.add_messages(
        [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]
    )

    # Get messages from the chat history
    messages = chat_history.messages
    assert len(messages) == 3
    assert messages == [
        SystemMessage(content="Meow"),
        AIMessage(content="woof"),
        HumanMessage(content="bark"),
    ]

    chat_history.add_messages(
        [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]
    )

    messages = chat_history.messages
    assert len(messages) == 6
    assert messages == [
        SystemMessage(content="Meow"),
        AIMessage(content="woof"),
        HumanMessage(content="bark"),
        SystemMessage(content="Meow"),
        AIMessage(content="woof"),
        HumanMessage(content="bark"),
    ]

    chat_history.clear()
    assert chat_history.messages == []
