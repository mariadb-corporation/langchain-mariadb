# Documentation

This directory contains the GitBook documentation export tool for the langchain-mariadb package.

## Generate GitBook Documentation

Generate plain Markdown API documentation compatible with GitBook:

```bash
# Using make
make docs_export

# Or directly
poetry run python docs/export_gitbook.py
```

Output files will be in `docs/gitbook_export/`:
- `index.md` - API overview
- `vectorstores.md` - MariaDBStore API
- `chat_message_histories.md` - Chat history API  
- `expression_filter.md` - Filter expressions API
- `translator.md` - Translator API

## Features

- ✅ **Pure Markdown** - Works natively in GitBook  
- ✅ **Version badges** - Shows package version on each page  
- ✅ **Complete API docs** - Auto-generated from Python docstrings  
- ✅ **Proper formatting** - Clean parameter lists, examples, and signatures  

## Copy to GitBook

The `gitbook_export/` directory is gitignored and generated on-demand.

**For Pull Requests:**

1. Generate the export:
   ```bash
   make docs_export
   ```

2. Copy files to your GitBook repository:
   ```bash
   cp docs/gitbook_export/*.md /path/to/mariadb-gitbook/docs/langchain/
   ```

3. Create PR in the GitBook repository with the updated files

**Note:** The `gitbook_export/` folder is not committed to this repository - it's generated fresh each time.

## Updating Documentation

When you update Python docstrings, regenerate the export:

```bash
make docs_export
```

All documentation is auto-generated from Python source code docstrings (Google style).

## Writing Docstrings

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """Short description of the function.
    
    Longer description with more details.
    
    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 10.
    
    Returns:
        Description of the return value
    
    Examples:
        >>> my_function("test", 5)
        True
    """
    pass
```

## Deployment

The documentation is automatically built and deployed to GitHub Pages via GitHub Actions when changes are merged to the main branch.

**URL**: `https://mariadb-corporation.github.io/langchain-mariadb/`

## Troubleshooting

**Import errors**: Ensure all dependencies are installed
```bash
poetry install --with docs
```

**Build errors**: Check docstring formatting and RST syntax

**Missing docs**: Ensure classes/functions are public (don't start with `_`) and not marked `:private:`

**Build warnings**: You may see ~8 warnings from `langchain_core.vectorstores.base.VectorStore.as_retriever` - these are from the external langchain-core library and can be safely ignored. They don't affect the generated documentation.
