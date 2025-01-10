# langchain-mariadb

[![Release Notes](https://img.shields.io/github/release/langchain-ai/langchain-mariadb)](https://github.com/langchain-ai/langchain-mariadb/releases)
[![CI](https://github.com/langchain-ai/langchain-mariadb/actions/workflows/ci.yml/badge.svg)](https://github.com/langchain-ai/langchain-mariadb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/6adMQxSpJS)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langchain-mariadb)](https://github.com/langchain-ai/langchain-mariadb/issues)

The `langchain-mariadb` package provides implementations of core LangChain abstractions using MariaDB's vector capabilities.

The package is released under the MIT license.

Feel free to use the abstractions as provided or modify/extend them as appropriate for your own application.

## Installation

```bash
pip install -U langchain-mariadb
```

## Usage

### ChatMessageHistory

The chat message history abstraction helps to persist chat message history in a MariaDB table.

MariaDBChatMessageHistory is parameterized using a `table_name` and a `session_id`.

The `table_name` is the name of the table in the database where 
the chat messages will be stored.

The `session_id` is a unique identifier for the chat session. It can be assigned
by the caller using `uuid.uuid4()`.

```python
import uuid

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_mariadb import MariaDBChatMessageHistory
import mariadb

# Establish a pool to the database
pool = mariadb.ConnectionPool(
    pool_name="chat_pool",
    user="root",
    host="localhost",
    database="chatdb"
)

# Create the table schema (only needs to be done once)
table_name = "chat_history"
MariaDBChatMessageHistory.create_tables(pool, table_name)

# Initialize the chat history manager
chat_history = MariaDBChatMessageHistory(
    table_name,
    str(uuid.uuid4()), # session_id
    pool=pool
)

# Add messages to the chat history
chat_history.add_messages([
    SystemMessage(content="Meow"),
    AIMessage(content="woof"),
    HumanMessage(content="bark"),
])

print(chat_history.messages)
```

### Vectorstore

See example for the [MariaDB vectorstore here](./examples/vectorstore.ipynb)
