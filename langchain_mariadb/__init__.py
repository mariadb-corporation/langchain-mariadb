from importlib import metadata

from langchain_mariadb.chat_message_histories import MariaDBChatMessageHistory
from langchain_mariadb.vectorstores import MariaDBStore
from langchain_mariadb.expression_filter import FilterExpressionBuilder

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [
    "__version__",
    "MariaDBChatMessageHistory",
    "MariaDBStore",
    "FilterExpressionBuilder",
]
