# langchain-mariadb

[![CI](https://github.com/rusher/langchain-mariadb/actions/workflows/ci.yml/badge.svg)](https://github.com/rusher/langchain-mariadb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package provides LangChain integrations with MariaDB's vector capabilities, released under the MIT license. 
Users can use the implementations as provided or customize them for their specific applications.

## Installation

```bash
pip install -U langchain-mariadb
```

## Vector store

The package implements a LangChain vectorstore using MariaDB (version 11.7.1+ required) as the backend. Features include:

* Native vector similarity search
* Support for cosine and euclidean distance metrics
* Comprehensive metadata filtering
* Connection pooling for performance optimization
* Customizable table and column configurations

### Setup

You can start a MariaDB container with:

```shell
docker run --name mariadb-container -e MARIADB_ROOT_PASSWORD=langchain -e MARIADB_DATABASE=langchain -p 3306:3306 -d mariadb:11.7
```

The package uses SQLAlchemy but has only been tested with the MariaDB connector, which is strongly recommended. 
Since the Python mariadb connector relies on C/C components, you'll need to install them first:

```shell
# Debian, Ubuntu
sudo apt install libmariadb3 libmariadb-dev

# CentOS, RHEL, Rocky Linux
sudo yum install MariaDB-shared MariaDB-devel

# Then install the Python connector
pip install --quiet -U mariadb
```

#### Initialize the vectorstore

Installing `langchain_mariadb` along with an LLM model, here using `langchain_openai` as example. 
```shell
pip install --quiet -U langchain_openai langchain_mariadb
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain_mariadb import MariaDBStore
from langchain_core.documents import Document

# connection string
url = f"mariadb+mariadbconnector://myuser:mypassword@localhost/langchain"

# Create a new vector store
vectorstore = MariaDBStore(
    embeddings=OpenAIEmbeddings(),
    embedding_length=1536,
    datasource=url,
    collection_name="my_docs"
)
```

#### Add new data

```python
# adding documents
docs = [
    Document(page_content='there are cats in the pond', metadata={"id": 1, "location": "pond", "topic": "animals"}),
    Document(page_content='ducks are also found in the pond', metadata={"id": 2, "location": "pond", "topic": "animals"}),
    Document(page_content='fresh apples are available at the market', metadata={"id": 3, "location": "market", "topic": "food"}),
    Document(page_content='the market also sells fresh oranges', metadata={"id": 4, "location": "market", "topic": "food"}),
    Document(page_content='the new art exhibit is fascinating', metadata={"id": 5, "location": "museum", "topic": "art"}),
]
vectorstore.add_documents(docs)

# add from text
texts = [
    'a sculpture exhibit is also at the museum',
    'a new coffee shop opened on Main Street',
    'the book club meets at the library',
    'the library hosts a weekly story time for kids',
    'a cooking class for beginners is offered at the community center'
]

# optional metadata
metadatas = [
    {"id": 6, "location": "museum", "topic": "art"},
    {"id": 7, "location": "Main Street", "topic": "food"},
    {"id": 8, "location": "library", "topic": "reading"},
    {"id": 9, "location": "library", "topic": "reading"},
    {"id": 10, "location": "community center", "topic": "classes"}
]

vectorstore.add_texts(texts=texts, metadatas=metadatas)
```

#### Searching similarity

```python
# Search similar texts
results = vectorstore.similarity_search("Hello", k=2)

# Search with metadata filter
results = vectorstore.similarity_search(
"Hello",
filter={"category": "greeting"}
)
```

#### Filtering Support

The vectorstore supports a set of filters that can be applied against the metadata fields of the documents.

| Operator  | Meaning/Category        |
|-----------|-------------------------|
| \$eq      | Equality (==)           |
| \$ne      | Inequality (!=)         |
| \$lt      | Less than (<)           |
| \$lte     | Less than or equal (<=) |
| \$gt      | Greater than (>)        |
| \$gte     | Greater than or equal (>=) |
| \$in      | Special Cased (in)      |
| \$nin     | Special Cased (not in)  |
| \$like    | Text (like)             |
| \$nlike   | Text (not like)         |
| \$and     | Logical (and)           |
| \$or      | Logical (or)            |
| \$not     | Logical (not)           |

```python
# Search with simple filter
results = vectorstore.similarity_search('kitty', k=10, filter={
    'id': {'$in': [1, 5, 2, 9]}
})

# Search with multiple conditions (AND)
results = vectorstore.similarity_search('ducks', k=10, filter={
    'id': {'$in': [1, 5, 2, 9]},
    'location': {'$in': ["pond", "market"]}
})
```

## ChatMessageHistory

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

# connection string
url = f"mariadb+mariadbconnector://myuser:mypassword@localhost/chatdb"

# Create the table schema (only needs to be done once)
table_name = "chat_history"
MariaDBChatMessageHistory.create_tables(url, table_name)

# Initialize the chat history manager
chat_history = MariaDBChatMessageHistory(
    table_name,
    str(uuid.uuid4()), # session_id
    datasource=pool
)

# Add messages to the chat history
chat_history.add_messages([
    SystemMessage(content="Meow"),
    AIMessage(content="woof"),
    HumanMessage(content="bark"),
])

print(chat_history.messages)
```
