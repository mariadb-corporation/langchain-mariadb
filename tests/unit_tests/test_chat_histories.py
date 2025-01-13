import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_mariadb.chat_message_histories import MariaDBChatMessageHistory
from tests.utils import pool


def test_sync_chat_history() -> None:
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=123))
    with pool() as tmppool:
        MariaDBChatMessageHistory.drop_table(tmppool, table_name)
        MariaDBChatMessageHistory.create_tables(tmppool, table_name)

        chat_history = MariaDBChatMessageHistory(table_name, session_id, pool=tmppool)

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
