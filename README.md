# langchain-mariadb

[![CI](https://github.com/rusher/langchain-mariadb/actions/workflows/ci.yml/badge.svg)](https://github.com/rusher/langchain-mariadb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LangChain's MariaDB integration (langchain-mariadb) provides vector capabilities for working with MariaDB version 11.7.1 and above, distributed under the MIT license. Users can use the provided implementations as-is or customize them for specific needs.
Key features include:

* Built-in vector similarity search
* Support for cosine and euclidean distance metrics
* Robust metadata filtering options
* Performance optimization through connection pooling
* Configurable table and column settings

## Getting Started

### Setting Up MariaDB

Launch a MariaDB Docker container with:

```shell
docker run --name mariadb-container -e MARIADB_ROOT_PASSWORD=langchain -e MARIADB_DATABASE=langchain -p 3306:3306 -d mariadb:11.7
```

### Installing the Package

The package uses SQLAlchemy but works best with the MariaDB connector, which requires C/C++ components:
```shell
# Debian, Ubuntu
sudo apt install libmariadb3 libmariadb-dev

# CentOS, RHEL, Rocky Linux
sudo yum install MariaDB-shared MariaDB-devel

# Install Python connector
pip install --quiet -U mariadb
```

Then install `langchain-mariadb` package
```bash
pip install -U langchain-mariadb
```

VectorStore works along with an LLM model, here using `langchain-openai` as example. 
```shell
pip install langchain-openai
export OPENAI_API_KEY=...
```

#### Creating a Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_mariadb import MariaDBStore
from langchain_core.documents import Document

# connection string
url = f"mariadb+mariadbconnector://myuser:mypassword@localhost/langchain"

# Initialize vector store
vectorstore = MariaDBStore(
    embeddings=OpenAIEmbeddings(),
    embedding_length=1536,
    datasource=url,
    collection_name="my_docs"
)
```

#### Adding Data
You can add data as documents with metadata:
```python
# adding documents
docs = [
    Document(page_content='there are cats in the pond', metadata={"id": 1, "location": "pond", "topic": "animals"}),
    Document(page_content='ducks are also found in the pond', metadata={"id": 2, "location": "pond", "topic": "animals"}),
    # More documents...
]
vectorstore.add_documents(docs)
```


Or as plain text with optional metadata:
```python
texts = ['a sculpture exhibit is also at the museum', 'a new coffee shop opened on Main Street',]
metadatas = [
    {"id": 6, "location": "museum", "topic": "art"},
    {"id": 7, "location": "Main Street", "topic": "food"},
]

vectorstore.add_texts(texts=texts, metadatas=metadatas)
```

#### Searching

```python
# Basic similarity search
results = vectorstore.similarity_search("Hello", k=2)

# Search with metadata filtering
results = vectorstore.similarity_search(
    "Hello",
    filter={"category": "greeting"}
)
```

#### Filter Options

The system supports various filtering operations on metadata:

* Equality: $eq
* Inequality: $ne
* Comparisons: $lt, $lte, $gt, $gte
* List operations: $in, $nin
* Text matching: $like, $nlike
* Logical operations: $and, $or, $not

Example:
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

## Chat Message History

The package also provides a way to store chat message history in MariaDB:
```python
import uuid
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_mariadb import MariaDBChatMessageHistory

# Set up database connection
url = f"mariadb+mariadbconnector://myuser:mypassword@localhost/chatdb"

# Create table (one-time setup)
table_name = "chat_history"
MariaDBChatMessageHistory.create_tables(url, table_name)

# Initialize chat history manager
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
```_
