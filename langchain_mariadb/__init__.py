from importlib import metadata

from langchain_mariadb.chat_message_histories import MariaDBChatMessageHistory
from langchain_mariadb.translator import MariaDBTranslator
from langchain_mariadb.vectorstores import MariaDBStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [
    "__version__",
    "MariaDBChatMessageHistory",
    "MariaDBStore",
    "MariaDBTranslator",
]
