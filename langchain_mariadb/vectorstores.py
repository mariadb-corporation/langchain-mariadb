from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from langchain_mariadb._utils import enquote_identifier
from langchain_mariadb.expression_filter import MariaDBFilterExpressionConverter

"""
MariaDBStore is a vector store implementation that uses MariaDB database.

Example:
    Basic usage:
    ```python
    from langchain_mariadb import MariaDBStore
    from langchain_openai import OpenAIEmbeddings

    # Create a MariaDB connection pool
    url = f"mariadb+mariadbconnector://myuser:mypassword@localhost/mydatabase"

    # Initialize embeddings model
    embeddings = OpenAIEmbeddings()

    # Create a new vector store
    store = MariaDBStore.from_texts(
        texts=["Hello, world!", "Another text"],
        embedding=embeddings,
        datasource=url,
        collection_name="my_collection"  # Optional, defaults to "langchain"
    )

    # Search similar texts
    results = store.similarity_search("Hello", k=2)

    # Search with metadata filter
    results = store.similarity_search(
        "Hello",
        filter={"category": "greeting"}
    )

    # Search with complex filter
    results = store.similarity_search(
        "Hello",
        filter={
            "$and": [
                {"category": "greeting"},
                {"language": {"$in": ["en", "es"]}}
            ]
        }
    )
    ```

    Asynchronous usage:
    ```python
    import asyncio
    from langchain_mariadb import MariaDBStore
    from langchain_openai import OpenAIEmbeddings

    async def search_documents():
        # Create store as before
        url = f"mariadb+mariadbconnector://myuser:mypassword@localhost/mydatabase"
        
        embeddings = OpenAIEmbeddings()
        store = MariaDBStore.from_texts(
            texts=["Hello, world!", "Another text"],
            embedding=embeddings,
            datasource=url
        )

        # Perform async similarity search
        results = await store.amax_marginal_relevance_search(
            "Hello",
            k=2,
            fetch_k=10,
            lambda_mult=0.5
        )

        # Async search with scores
        results_with_scores = await store.amax_marginal_relevance_search_with_score(
            "Hello",
            k=2,
            filter={"category": "greeting"}
        )

        return results, results_with_scores

    # Run async function
    results, results_with_scores = asyncio.run(search_documents())
    ```

Advanced Usage:
    Custom configuration:
    ```python
    from langchain_mariadb import (
       MariaDBStore, MariaDBStoreSettings, TableConfig, ColumnConfig
    }

    # Configure custom table and column names
    config = MariaDBStoreSettings(
        tables=TableConfig(
            embedding_table="custom_embeddings",
            collection_table="custom_collections"
        ),
        columns=ColumnConfig(
            embedding_id="doc_id",
            content="text_content",
            metadata="doc_metadata"
        )
    )

    store = MariaDBStore.from_texts(
        texts=["Hello, world!"],
        embedding=embeddings,
        datasource=pool,
        config=config
    )
    ```

    Working with documents:
    ```python
    from langchain_core.documents import Document

    # Create from documents
    documents = [
        Document(page_content="Hello", metadata={"source": "greeting.txt"}),
        Document(page_content="World", metadata={"source": "greeting.txt"})
    ]

    store = MariaDBStore.from_documents(
        documents=documents,
        embedding=embeddings,
        datasource=pool
    )

    # Add more documents
    store.add_documents(documents)
    ```

Notes:
    - Requires MariaDB 11.7.1 or later
    - The database user needs permissions to create tables and indexes

Distance Strategies:
    - COSINE (default)
    - EUCLIDEAN
"""

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


# ------------------------------------------------------------------------------
# Helper Classes
# ------------------------------------------------------------------------------
class DistanceStrategy(str, Enum):
    """Distance strategies for vector similarity."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


# ------------------------------------------------------------------------------
# Configuration Classes
# ------------------------------------------------------------------------------
@dataclass
class TableConfig:
    """Configuration for database table names."""

    embedding_table: str
    collection_table: str

    def __init__(
        self,
        embedding_table: Optional[str] = None,
        collection_table: Optional[str] = None,
    ) -> None:
        """Initialize TableConfig with custom or default table names.

        Args:
            embedding_table: Name for embedding table (default: langchain_embedding)
            collection_table: Name for collection table (default: langchain_collection)
        """
        self.embedding_table = embedding_table or "langchain_embedding"
        self.collection_table = collection_table or "langchain_collection"

    @classmethod
    def default(cls) -> "TableConfig":
        """Create TableConfig with default values."""
        return cls()


@dataclass
class ColumnConfig:
    """Configuration for database column names."""

    # Embedding table columns
    embedding_id: str
    embedding: str
    content: str
    metadata: str

    # Collection table columns
    collection_id: str
    collection_label: str
    collection_metadata: str

    def __init__(
        self,
        # Embedding table columns
        embedding_id: Optional[str] = None,
        embedding: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[str] = None,
        # Collection table columns
        collection_id: Optional[str] = None,
        collection_label: Optional[str] = None,
        collection_metadata: Optional[str] = None,
    ) -> None:
        """Initialize ColumnConfig with custom or default column names.

        Args:
            embedding_id: Name for embedding ID column (default: id)
            embedding: Name for embedding vector column (default: embedding)
            content: Name for content column (default: content)
            metadata: Name for metadata column (default: metadata)
            collection_id: Name for collection ID column (default: id)
            collection_label: Name for collection label column (default: label)
            collection_metadata: Name for collection metadata column (default: metadata)
        """
        # Embedding table columns
        self.embedding_id = embedding_id or "id"
        self.embedding = embedding or "embedding"
        self.content = content or "content"
        self.metadata = metadata or "metadata"

        # Collection table columns
        self.collection_id = collection_id or "id"
        self.collection_label = collection_label or "label"
        self.collection_metadata = collection_metadata or "metadata"

    @classmethod
    def default(cls) -> "ColumnConfig":
        """Create ColumnConfig with default values."""
        return cls()


@dataclass
class MariaDBStoreSettings:
    """Configuration for MariaDBStore."""

    tables: TableConfig
    columns: ColumnConfig
    pre_delete_collection: bool
    lazy_init: bool

    def __init__(
        self,
        tables: Optional[TableConfig] = None,
        columns: Optional[ColumnConfig] = None,
        pre_delete_collection: bool = False,
        lazy_init: bool = False,
    ) -> None:
        """Initialize MariaDBStoreSettings with custom or default configurations.

        Args:
            tables: Table configuration
            columns: Column configuration
            pre_delete_collection: delete existing collection (default: False)
        """
        self.tables = tables or TableConfig.default()
        self.columns = columns or ColumnConfig.default()
        self.pre_delete_collection = pre_delete_collection
        self.lazy_init = lazy_init

    @classmethod
    def default(cls) -> "MariaDBStoreSettings":
        """Create MariaDBStoreSettings with default values."""
        return cls()


# ------------------------------------------------------------------------------
# Main VectorStore Implementation
# ------------------------------------------------------------------------------
class MariaDBStore(VectorStore):
    """MariaDB vector store integration."""

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------
    def __init__(
        self,
        embeddings: Embeddings,
        embedding_length: Optional[int] = None,
        *,
        datasource: Union[Engine | str],
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        config: MariaDBStoreSettings = MariaDBStoreSettings(),
        logger: Optional[logging.Logger] = None,
        engine_args: Optional[dict[str, Any]] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Initialize the MariaDB vector store.

        Args:
            embeddings: Embeddings object for creating embeddings
            embedding_length: Length of embedding vectors (default: 1536)
            datasource: datasource (connection string, sqlalchemy engine or
                        MariaDB connection pool)
            collection_name: Name of the collection to store vectors
            collection_metadata: Optional metadata for the collection
            distance_strategy: Strategy for distances (COSINE or EUCLIDEAN)
            config: Store configuration for tables and columns
            logger: Optional logger instance for debugging
            relevance_score_fn: function to override relevance score calculation
        """
        # Initialize core attributes
        self.embedding_function = embeddings
        self._embedding_length = embedding_length
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self._distance_strategy = distance_strategy
        self.pre_delete_collection = config.pre_delete_collection
        self.lazy_init = config.lazy_init
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        if isinstance(datasource, str):
            self._datasource = create_engine(url=datasource, **(engine_args or {}))
        elif isinstance(datasource, Engine):
            self._datasource = datasource
        else:
            raise ValueError(
                "datasource should be a connection string, an instance of "
                "sqlalchemy.engine.Engine or a mariadb pool"
            )
        # Initialize table and column names
        self._embedding_table_name = enquote_identifier(config.tables.embedding_table)
        self._embedding_id_col_name = enquote_identifier(config.columns.embedding_id)
        self._embedding_emb_col_name = enquote_identifier(config.columns.embedding)
        self._embedding_content_col_name = enquote_identifier(config.columns.content)
        self._embedding_meta_col_name = enquote_identifier(config.columns.metadata)

        self._collection_table_name = enquote_identifier(config.tables.collection_table)
        self._collection_id_col_name = enquote_identifier(config.columns.collection_id)
        self._collection_label_col_name = enquote_identifier(
            config.columns.collection_label
        )
        self._collection_meta_col_name = enquote_identifier(
            config.columns.collection_metadata
        )

        self._expression_converter = MariaDBFilterExpressionConverter(
            self._embedding_meta_col_name
        )

        # Initialize tables and collection
        self._collection_id: Optional[str] = None
        if not self.lazy_init:
            if self._embedding_length is None:
                self._embedding_length = 1536
            self._init_vectorstore()

    def _init_vectorstore(
        self,
    ) -> None:
        """Initialize the store."""
        self.create_tables_if_not_exists()
        self.create_collection()
        self.lazy_init = False

    # Core properties and utilities
    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def _embedding_to_binary(self, embedding: List[float]) -> bytes:
        """Convert embedding vector to binary format for storage.

        Args:
            embedding: List of floating point values

        Returns:
            Packed binary representation of the embedding
        """
        return np.array(embedding, np.float32).tobytes()

    def _binary_to_embedding(self, embedding: bytes) -> List[float] | None:
        """Convert binary data back to embedding vector.

        Args:
            embedding: Binary data to unpack

        Returns:
            List of floating point values, or None if input is None
        """
        if embedding is None:
            return None
        return cast(list[float], np.frombuffer(embedding, np.float32).tolist())

    def _validate_id(self, id_: str) -> None:
        """Validate document ID format.

        Args:
            id_: ID to validate

        Raises:
            ValueError: If ID format is invalid
        """
        if not re.match("^[a-zA-Z0-9_\\-]+$", id_):
            raise ValueError(
                f"ID format can only be alphanumeric with underscore and minus sign, "
                f"but got value: {id_}"
            )

    # Database management methods
    def create_tables_if_not_exists(self) -> None:
        """Create the necessary database tables if they don't exist."""
        # Create embedding table index name
        index_name = (
            f"idx_{self._embedding_table_name}_{self._embedding_emb_col_name}_idx"
        )
        index_name = re.sub(r"[^0-9a-zA-Z_]", "", index_name)

        # Create embedding table
        table_query = (
            f"CREATE TABLE IF NOT EXISTS {self._embedding_table_name} ("
            f"{self._embedding_id_col_name} VARCHAR(36)"
            f" NOT NULL DEFAULT UUID_v7() PRIMARY KEY,"
            f"{self._embedding_content_col_name} TEXT,"
            f"{self._embedding_meta_col_name} JSON,"
            f"{self._embedding_emb_col_name} VECTOR({self._embedding_length}) NOT NULL,"
            f"VECTOR INDEX {index_name} ({self._embedding_emb_col_name}) "
            f") ENGINE=InnoDB"
        )

        # Create collection table index names
        col_uniq_key_name = (
            f"idx_{self._collection_table_name}_{self._collection_label_col_name}"
        )
        col_index_name = f"{self._embedding_table_name}_collection_id_fkey"
        col_uniq_key_name = re.sub(r"[^0-9a-zA-Z_]", "", col_uniq_key_name)
        col_index_name = re.sub(r"[^0-9a-zA-Z_]", "", col_index_name)

        # Create collection table
        col_table_query = (
            f"CREATE TABLE IF NOT EXISTS {self._collection_table_name}("
            f"{self._collection_id_col_name} UUID"
            f" NOT NULL DEFAULT UUID_v7() PRIMARY KEY,"
            f"{self._collection_label_col_name} VARCHAR(256),"
            f"{self._collection_meta_col_name} JSON,"
            f"UNIQUE KEY {col_uniq_key_name} ({self._collection_label_col_name})"
            f")"
        )

        # Add foreign key constraint
        alter_query = (
            f"ALTER TABLE {self._embedding_table_name}"
            f" ADD COLUMN IF NOT EXISTS collection_id uuid,"
            f" ADD CONSTRAINT FOREIGN KEY IF NOT EXISTS"
            f" {col_index_name} (collection_id)"
            f" REFERENCES {self._collection_table_name}({self._collection_id_col_name})"
            f" ON DELETE CASCADE"
        )

        # Create collection ID index
        create_collection_id_idx = (
            f"CREATE INDEX IF NOT EXISTS coll_id_idx "
            f"ON {self._embedding_table_name} (collection_id)"
        )

        # Execute all queries
        con = self._datasource.raw_connection()
        cursor = con.cursor()
        try:
            cursor.execute(table_query)
            cursor.execute(col_table_query)
            cursor.execute(alter_query)
            cursor.execute(create_collection_id_idx)
            con.commit()
        finally:
            cursor.close()
            con.close()

    def drop_tables(self) -> None:
        """Drop all tables used by the vector store."""
        con = self._datasource.raw_connection()
        cursor = con.cursor()
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {self._embedding_table_name}")
            cursor.execute(f"DROP TABLE IF EXISTS {self._collection_table_name}")
            con.commit()
        finally:
            cursor.close()
            con.close()

    def _check_if_collection_exists(self) -> str | None:
        con = self._datasource.raw_connection()
        cursor = con.cursor()
        try:
            cursor.execute(
                f"SELECT {self._collection_id_col_name}"
                f" FROM {self._collection_table_name}"
                f" WHERE {self._collection_label_col_name}=?",
                (self.collection_name,),
            )
            row = cursor.fetchone()
            if row is not None:
                return row[0]
            return None

        finally:
            cursor.close()
            con.close()

    def create_collection(self) -> None:
        """Create a new collection or retrieve existing one."""
        if self.pre_delete_collection:
            self.delete_collection()

        con = self._datasource.raw_connection()
        cursor = con.cursor()
        try:
            # Check if collection exists
            collection_id = self._check_if_collection_exists()
            if collection_id is not None:
                self._collection_id = collection_id
                return
            # Create new collection
            query = (
                f"INSERT INTO {self._collection_table_name}"
                f"({self._collection_label_col_name},"
                f" {self._collection_meta_col_name})"
                f" VALUES (?,?) RETURNING {self._collection_id_col_name}"
            )
            cursor.execute(
                query, (self.collection_name, json.dumps(self.collection_metadata))
            )
            row = cursor.fetchone()
            if row is not None:
                self._collection_id = cast(str, row[0])
            con.commit()
        finally:
            cursor.close()
            con.close()

    def delete_collection(self) -> None:
        """Delete the current collection and its associated data."""
        con = self._datasource.raw_connection()
        cursor = con.cursor()
        try:
            try:
                # Find collection ID
                cursor.execute(
                    f"SELECT {self._collection_id_col_name}"
                    f" FROM {self._collection_table_name}"
                    f" WHERE {self._collection_label_col_name}=?",
                    (self.collection_name,),
                )
                row = cursor.fetchone()

                if row is not None:
                    collection_id = row[0]
                    # Delete associated embeddings and collection
                    query = (
                        f"DELETE FROM {self._embedding_table_name}"
                        f" WHERE collection_id = ?"
                    )
                    cursor.execute(
                        query,
                        (collection_id,),
                    )
                    cursor.execute(
                        f"DELETE FROM {self._collection_table_name} "
                        f"WHERE {self._collection_id_col_name} = ?",
                        (collection_id,),
                    )
            except Exception:
                self.logger.debug("Failed to delete previous collection")
            con.commit()
        finally:
            cursor.close()
            con.close()

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by their IDs.

        Args:
            ids: List of IDs to delete
            **kwargs: Additional arguments (not used)
        """
        con = self._datasource.raw_connection()
        cursor = con.cursor()
        try:
            filter_sql = self._create_filter_sql(kwargs.get("filter", None))

            # Return early if nothing to delete
            if not ids and not filter_sql:
                self.logger.debug("No IDs or filter provided, nothing to delete")
                return

            self.logger.debug("Deleting vectors by IDs")

            # Build the WHERE clause
            where_clauses = []
            data = []

            if ids:
                where_clauses.append(f"{self._embedding_id_col_name} = ?")
                data = [(i,) for i in ids]

            if filter_sql:
                where_clauses.append(filter_sql)

            if self._collection_id is not None:
                where_clauses.append(f"collection_id = '{self._collection_id}'")

            query = (
                f"DELETE FROM {self._embedding_table_name} "
                f"WHERE {' AND '.join(where_clauses)}"
            )

            # Use executemany if we have IDs, otherwise execute once
            if ids:
                cursor.executemany(query, data)
            else:
                cursor.execute(query)
            con.commit()
        except Exception as e:
            if hasattr(e, "errno") and e.errno == 1146:  # NO SUCH TABLE
                return
            else:
                raise e
        finally:
            cursor.close()
            con.close()

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by their IDs."""
        ids_ = []
        for _id in ids:
            if _id is not None:
                self._validate_id(_id)
                ids_.append(f"'{_id}'")

        if not ids_:
            return []

        # Build and execute query
        query = (
            f"SELECT {self._embedding_id_col_name}, "
            f"{self._embedding_content_col_name}, "
            f"{self._embedding_meta_col_name} "
            f"FROM {self._embedding_table_name} "
            f"WHERE {self._embedding_id_col_name} IN ({','.join(ids_)}) "
            f"AND collection_id = '{self._collection_id}' "
            f"ORDER BY {self._embedding_meta_col_name}"
        )

        documents = []
        con = self._datasource.raw_connection()
        cursor = con.cursor()
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            for row in rows:
                documents.append(
                    Document(
                        id=row[0],
                        page_content=row[1],
                        metadata=json.loads(row[2]),
                    )
                )
        finally:
            cursor.close()
            con.close()
        return documents

    def add_embeddings(
        self,
        texts: Sequence[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Sequence of strings to add
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts for each text
            ids: Optional list of IDs for the documents
            **kwargs: Additional arguments (not used)

        Returns:
            List of IDs for the added documents

        Raises:
            ValueError: If any provided ID contains invalid characters
        """
        # Generate or validate IDs
        if ids is None:
            ids_ = [str(uuid.uuid4()) for _ in texts]
        else:
            ids_ = []
            for _id in ids:
                if _id is None:
                    ids_.append(str(uuid.uuid4()))
                else:
                    if not re.match("^[a-zA-Z0-9_\\-]+$", _id):
                        raise ValueError(
                            f"ID format can only be alphanumeric with underscore "
                            f"and minus sign, but got value: {_id}"
                        )
                    ids_.append(_id)

        # Use empty metadata if none provided
        if not metadatas:
            metadatas = [{} for _ in texts]

        if len(embeddings) == 0:
            return []

        # Prepare data for insertion
        data = []
        for text, metadata, embedding, id_ in zip(texts, metadatas, embeddings, ids_):
            binary_emb = self._embedding_to_binary(embedding)
            data.append(
                (
                    id_,
                    text,
                    json.dumps(metadata),
                    binary_emb,
                    self._collection_id,
                )
            )

        # Insert embeddings into database in batches
        con = self._datasource.raw_connection()
        cursor = con.cursor()
        try:
            query = (
                f"INSERT INTO {self._embedding_table_name} ("
                f"{self._embedding_id_col_name}, "
                f"{self._embedding_content_col_name}, "
                f"{self._embedding_meta_col_name}, "
                f"{self._embedding_emb_col_name}, "
                f"collection_id"
                f") VALUES (?,?,?,?,?) "
                f"ON DUPLICATE KEY UPDATE "
                f"{self._embedding_content_col_name} = "
                f"VALUES({self._embedding_content_col_name}), "
                f"{self._embedding_meta_col_name} = "
                f"VALUES({self._embedding_meta_col_name}), "
                f"{self._embedding_emb_col_name} = "
                f"VALUES({self._embedding_emb_col_name})"
            )

            # Process in batches of 1000 parameters
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                cursor.executemany(query, batch)
            con.commit()

        finally:
            cursor.close()
            con.close()
        return ids_

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids for the texts.
                 If not provided, will generate a new id for each text.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts_ = list(texts)
        if not texts_:
            return []
        embeddings = self.embedding_function.embed_documents(texts_)
        if len(embeddings) == 0:
            return []
        if self.lazy_init and embeddings:
            self._embedding_length = len(embeddings[0])
            self._init_vectorstore()
        return self.add_embeddings(
            texts=texts_,
            embeddings=list(embeddings),
            metadatas=list(metadatas) if metadatas else None,
            ids=list(ids) if ids else None,
        )

    # Search methods
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with MariaDB.

        Args:
            query: Query text to search for
            k: Number of results to return (default: 4)
            filter: Optional filter by metadata
            **kwargs: Additional arguments passed to similarity_search_by_vector

        Returns:
            List of Documents most similar to the query
        """
        embedding = self.embeddings.embed_query(query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Union[None, dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query along with scores.

        Args:
            query: Text to look up documents similar to
            k: Number of Documents to return (default: 4)
            filter: Optional filter by metadata

        Returns:
            List of tuples of (Document, similarity_score)
        """
        embedding = self.embeddings.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Union[None, dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector along with scores.

        Args:
            embedding: Embedding vector to look up documents similar to
            k: Number of Documents to return (default: 4)
            filter: Optional filter by metadata

        Returns:
            List of tuples of (Document, similarity_score)
        """
        results = self.__query_collection(embedding=embedding, k=k, filter=filter)
        return self._results_to_docs_and_scores(results)

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        Args:
            query: Input text to search for
            k: Number of Documents to return (default: 4)
            **kwargs: Additional arguments including:
                score_threshold: Optional float between 0 and 1 to filter results

        Returns:
            List of tuples of (Document, relevance_score) where relevance_score
            is in the range [0, 1]. 0 is dissimilar, 1 is most similar.
        """
        if self.override_relevance_score_fn is None:
            embedding = self.embeddings.embed_query(query)
            results = self.__query_with_score_collection(
                embedding=embedding, k=k, filter=kwargs.get("filter", None)
            )
            docs = self._results_to_docs_and_scores(results)
            return docs

        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [
            (doc, self.override_relevance_score_fn(score))
            for doc, score in docs_and_scores
        ]

    async def _asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1] asynchronously.

        Args:
            query: Input text to search for
            k: Number of Documents to return (default: 4)
            **kwargs: Additional arguments including:
                score_threshold: Optional float between 0 and 1 to filter results

        Returns:
            List of tuples of (Document, relevance_score) where relevance_score
            is in the range [0, 1]. 0 is dissimilar, 1 is most similar.
        """
        if self.override_relevance_score_fn is None:
            embedding = self.embeddings.embed_query(query)
            results = self.__query_with_score_collection(
                embedding=embedding, k=k, filter=kwargs.get("filter", None)
            )
            docs = self._results_to_docs_and_scores(results)
            return docs

        docs_and_scores = await self.asimilarity_search_with_score(query, k, **kwargs)
        return [
            (doc, self.override_relevance_score_fn(score))
            for doc, score in docs_and_scores
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding vector to look up documents similar to
            k: Number of Documents to return (default: 4)
            filter: Optional metadata filter
            **kwargs: Additional arguments (not used)

        Returns:
            List of Documents most similar to the query vector
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return _results_to_docs(docs_and_scores)

    # MMR search methods
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using maximal marginal relevance.

        Args:
            query: Text to look up documents similar to
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch before selecting top-k (default: 20)
            lambda_mult: Balance between relevance and diversity, 0-1 (default: 0.5)
                0 = maximize diversity, 1 = maximize relevance
            filter: Optional metadata filter
            **kwargs: Additional arguments passed to search_by_vector

        Returns:
            List of Documents selected by maximal marginal relevance
        """
        embedding = self.embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using maximal marginal relevance asynchronously.

        Args:
            query: Text to look up documents similar to
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch before selecting top-k (default: 20)
            lambda_mult: Balance between relevance and diversity, 0-1 (default: 0.5)
                0 = maximize diversity, 1 = maximize relevance
            filter: Optional metadata filter
            **kwargs: Additional arguments passed to search_by_vector

        Returns:
            List of Documents selected by maximal marginal relevance
        """
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search,
            query,
            k,
            fetch_k,
            lambda_mult,
            filter,
            **kwargs,
        )

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using maximal marginal relevance with scores.

        Args:
            query: Text to look up documents similar to
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch before selecting top-k (default: 20)
            lambda_mult: Balance between relevance and diversity, 0-1 (default: 0.5)
                0 = maximize diversity, 1 = maximize relevance
            filter: Optional metadata filter
            **kwargs: Additional arguments passed to search_by_vector

        Returns:
            List of tuples of (Document, score) selected by maximal marginal relevance
        """
        embedding = self.embeddings.embed_query(query)
        return self.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using maximal marginal relevance with scores
        asynchronously.

        Args:
            query: Text to look up documents similar to
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch before selecting top-k (default: 20)
            lambda_mult: Balance between relevance and diversity, 0-1 (default: 0.5)
                0 = maximize diversity, 1 = maximize relevance
            filter: Optional metadata filter
            **kwargs: Additional arguments passed to search_by_vector

        Returns:
            List of tuples of (Document, score) selected by maximal marginal relevance
        """
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_with_score,
            query,
            k,
            fetch_k,
            lambda_mult,
            filter,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using maximal marginal relevance.

        Args:
            embedding: Query embedding vector
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch before selecting top-k (default: 20)
            lambda_mult: Balance between relevance and diversity, 0-1 (default: 0.5)
                0 = maximize diversity, 1 = maximize relevance
            filter: Optional metadata filter
            **kwargs: Additional arguments (not used)

        Returns:
            List of Documents selected by maximal marginal relevance
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return _results_to_docs(docs_and_scores)

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using maximal marginal relevance asynchronously.

        Args:
            embedding: Query embedding vector
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch before selecting top-k (default: 20)
            lambda_mult: Balance between relevance and diversity, 0-1 (default: 0.5)
                0 = maximize diversity, 1 = maximize relevance
            filter: Optional metadata filter
            **kwargs: Additional arguments (not used)

        Returns:
            List of Documents selected by maximal marginal relevance
        """
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k,
            fetch_k,
            lambda_mult,
            filter,
            **kwargs,
        )

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using maximal marginal relevance with scores.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Query embedding vector
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch before selecting top-k (default: 20)
            lambda_mult: Balance between relevance and diversity, 0-1 (default: 0.5)
                0 = maximize diversity, 1 = maximize relevance
            filter: Optional metadata filter
            **kwargs: Additional arguments (not used)

        Returns:
            List of tuples of (Document, score) selected by maximal marginal relevance
        """
        # Fetch candidates with embeddings
        results = self.__query_collection(
            embedding=embedding, k=fetch_k, filter=filter, need_embeddings=True
        )

        # Extract embeddings from results
        embedding_list = [self._binary_to_embedding(result[4]) for result in results]

        # Calculate MMR selection
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        # Convert results to documents with scores
        candidates = self._results_to_docs_and_scores(results)
        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Union[None, dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using maximal marginal relevance with scores
        asynchronously.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Query embedding vector
            k: Number of documents to return (default: 4)
            fetch_k: Number of documents to fetch before selecting top-k (default: 20)
            lambda_mult: Balance between relevance and diversity, 0-1 (default: 0.5)
                0 = maximize diversity, 1 = maximize relevance
            filter: Optional metadata filter
            **kwargs: Additional arguments (not used)

        Returns:
            List of tuples of (Document, score) selected by maximal marginal relevance
        """
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_with_score_by_vector,
            embedding,
            k,
            fetch_k,
            lambda_mult,
            filter,
            **kwargs,
        )

    # Query building methods
    def _build_base_select_query(
        self,
        distance_or_score_expr: str,
        need_embeddings: bool = False,
        filter: Union[None, dict] = None,
    ) -> Tuple[str, str]:
        """Build base SELECT query with common components.

        Args:
            distance_or_score_expr: Expression for distance/score calculation
            need_embeddings: Whether to include embeddings column
            filter: Optional filter expression

        Returns:
            Tuple of (base query, filter clause)
        """
        # Build filter clause
        filter_sql = self._create_filter_sql(filter)
        if filter_sql:
            filter_sql = f" AND {filter_sql}"

        # Build base query
        query = (
            f"SELECT {self._embedding_id_col_name}, "
            f"{self._embedding_content_col_name}, "
            f"{self._embedding_meta_col_name}, "
            f"{distance_or_score_expr}"
        )

        if need_embeddings:
            query += f", {self._embedding_emb_col_name}"

        return query, filter_sql

    def __query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Union[None, dict] = None,
        need_embeddings: bool = False,
    ) -> Sequence[Any]:
        """Query the collection for similar documents."""
        distance_expr = (
            f"vec_distance_{self._distance_strategy.value}({self._embedding_emb_col_name}"
            f", ?) as distance"
        )
        base_query, filter_sql = self._build_base_select_query(
            distance_expr, need_embeddings, filter
        )

        query = (
            f"{base_query} "
            f"FROM {self._embedding_table_name} "
            f"WHERE collection_id = ?{filter_sql} "
            f"ORDER BY distance ASC LIMIT ?"
        )

        if self.lazy_init and not self._embedding_length:
            self._embedding_length = len(embedding)
            self._init_vectorstore()

        return self.__inner_query_collection(embedding=embedding, k=k, query_=query)

    def __query_with_score_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Union[None, dict] = None,
    ) -> Sequence[Any]:
        """Query the collection and return results with similarity scores."""
        # Calculate similarity score based on distance strategy
        if self._distance_strategy == DistanceStrategy.COSINE:
            score_expr = (
                f"1.0 - vec_distance_cosine({self._embedding_emb_col_name}, ?) as score"
            )
        else:
            score_expr = (
                f"1.0 - vec_distance_cosine({self._embedding_emb_col_name}, ?)"
                f" / SQRT(2) as score"
            )

        base_query, filter_sql = self._build_base_select_query(
            score_expr, filter=filter
        )

        query = (
            f"{base_query} "
            f"FROM {self._embedding_table_name} "
            f"WHERE collection_id = ?{filter_sql} "
            f"ORDER BY score DESC LIMIT ?"
        )

        return self.__inner_query_collection(embedding=embedding, k=k, query_=query)

    def __inner_query_collection(
        self,
        embedding: List[float],
        query_: str,
        k: int = 4,
    ) -> Sequence[Any]:
        """Execute a collection query with the given parameters.

        Args:
            embedding: Query embedding vector
            query_: SQL query string
            k: Number of results to return

        Returns:
            Sequence of query results
        """
        con = self._datasource.raw_connection()
        cursor = con.cursor()
        try:
            binary_emb = self._embedding_to_binary(embedding)
            cursor.execute(query_, (binary_emb, self._collection_id, k))
            return cursor.fetchall()
        finally:
            cursor.close()
            con.close()

    def _create_filter_sql(self, filters: Union[None, dict] = None) -> str:
        if filters is None:
            return ""
        return self._expression_converter.convert_expression(filters)

    # Result processing methods
    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        """Convert raw results to documents and scores.

        Args:
            results: Raw query results from database

        Returns:
            List of tuples of (Document, similarity_score)
        """
        docs = [
            (
                Document(
                    id=str(result[0]),
                    page_content=result[1],
                    metadata=json.loads(result[2]),
                ),
                result[3],
            )
            for result in results
        ]
        return docs

    # Class methods for construction
    @classmethod
    def __from(
        cls: Type[MariaDBStore],
        texts: List[str],
        embeddings: list[list[float]],
        ids: Optional[List[str]] = None,
        *,
        metadatas: Optional[List[dict]] = None,
        datasource: Union[Engine | str],
        embedding: Embeddings,
        embedding_length: Optional[int] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        config: MariaDBStoreSettings = MariaDBStoreSettings(),
        **kwargs: Any,
    ) -> MariaDBStore:
        """Internal method to create a MariaDBStore instance from texts and embeddings.

        Args:
            texts: List of text strings to store
            embeddings: List of embedding vectors
            ids: Optional list of IDs for the documents
            metadatas: Optional list of metadata dicts
            datasource: datasource (connection string, sqlalchemy engine or
                        MariaDB connection pool)
            embedding: Embeddings object for creating embeddings
            embedding_length: Optional length of embedding vectors
            collection_name: Name of collection (default: langchain)
            collection_metadata: Optional metadata for the collection
            distance_strategy: Strategy for computing distances
            logger: Optional logger instance
            relevance_score_fn: Optional function to compute relevance scores
            **kwargs: Additional arguments passed to constructor

        Returns:
            MariaDBStore instance
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Use empty metadata if none provided
        if not metadatas:
            metadatas = [{} for _ in texts]

        # Determine embedding length if not specified
        emb_len = embedding_length
        if embedding_length is None and embeddings and len(embeddings) > 0:
            emb_len = len(embeddings[0])

        config.lazy_init = False
        # Create store instance
        store = cls(
            embedding,
            emb_len,
            datasource=datasource,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            logger=logger,
            relevance_score_fn=relevance_score_fn,
            config=config,
            **kwargs,
        )

        # Add embeddings to store
        store.add_embeddings(texts, embeddings, metadatas, ids, **kwargs)

        return store

    @classmethod
    def from_texts(
        cls: Type[MariaDBStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MariaDBStore:
        """Create a MariaDBStore instance from texts.

        Args:
            texts: List of text strings to store
            embedding: Embeddings object for creating embeddings
            metadatas: Optional list of metadata dicts for each text
            ids: Optional list of unique IDs for each text
            datasource: datasource (connection string, sqlalchemy engine or
                        MariaDB connection pool)
            collection_name: Name of the collection to store vectors
            distance_strategy: Strategy for distances (COSINE or EUCLIDEAN)
            embedding_length: Length of embedding vectors (default: 1536)
            config: Store configuration for tables and columns
            logger: Optional logger instance for debugging
            relevance_score_fn: override function relevance score calculation
            **kwargs: Additional arguments passed to add_embeddings

        Returns:
            MariaDBStore instance initialized with the provided texts
        """
        embeddings = embedding.embed_documents(list(texts))

        # Create store instance
        return cls.__from(
            texts,
            embeddings,
            ids,
            metadatas=metadatas,
            embedding=embedding,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls: Type[MariaDBStore],
        text_embeddings: List[Tuple[str, List[float]]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        *,
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        config: MariaDBStoreSettings = MariaDBStoreSettings(),
        **kwargs: Any,
    ) -> MariaDBStore:
        """Create a MariaDBStore instance from text-embedding pairs.

        Args:
            text_embeddings: List of (text, embedding) tuples
            ids: Optional list of IDs for the documents
            metadatas: Optional list of metadata dicts
            embedding: Embeddings object for creating embeddings
            collection_name: Name of collection (default: langchain)
            distance_strategy: Strategy for computing distances
            relevance_score_fn: Optional function to compute relevance scores
            **kwargs: Additional arguments passed to constructor

        Returns:
            MariaDBStore instance

        Example:
            .. code-block:: python

                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                vectorstore = MariaDBStore.from_embeddings(
                  text_embedding_pairs,
                  embeddings
                )
        """
        # Split text-embedding pairs
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        # Create store instance
        return cls.__from(
            texts,
            embeddings,
            ids,
            embedding=embedding,
            metadatas=metadatas,
            distance_strategy=distance_strategy,
            relevance_score_fn=relevance_score_fn,
            config=config,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[MariaDBStore],
        embedding: Embeddings,
        *,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        datasource: Union[Engine | str],
        config: MariaDBStoreSettings = MariaDBStoreSettings(),
        **kwargs: Any,
    ) -> MariaDBStore:
        """Create a MariaDBStore instance from an existing index.

        This method returns an instance of the store without inserting any new
        embeddings.

        Args:
            embedding: Embeddings object for creating embeddings
            collection_name: Name of collection (default: langchain)
            distance_strategy: Strategy for computing distances
            datasource: datasource (connection string, sqlalchemy engine or MariaDB
                        connection pool)
            **kwargs: Additional arguments passed to constructor

        Returns:
            MariaDBStore instance connected to existing index
        """
        store = cls(
            embedding,
            datasource=datasource,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            config=config,
            **kwargs,
        )
        return store

    @classmethod
    def from_documents(
        cls: Type[MariaDBStore],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> MariaDBStore:
        """Create a MariaDBStore instance from documents.

        Args:
            documents: List of Document objects to store
            embedding: Embeddings object for creating embeddings
            datasource: datasource (connection string, sqlalchemy engine or MariaDB
                        connection pool)
            collection_name: Name of collection (default: langchain)
            distance_strategy: Strategy for computing distances
            ids: Optional list of IDs for the documents
            **kwargs: Additional arguments passed to from_texts

        Returns:
            MariaDBStore instance
        """
        # Extract text content and metadata from documents
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        # Create store instance using from_texts
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            **kwargs,
        )
