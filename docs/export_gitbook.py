#!/usr/bin/env python3
"""Export API documentation as plain Markdown for GitBook using griffe."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import toml
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "toml"])
    import toml

try:
    from griffe import Attribute, Class, Docstring, Function, Module, load
except ImportError:
    print("Installing griffe...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "griffe"])
    from griffe import Attribute, Class, Function, load


def get_version():
    """Get version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path) as f:
        pyproject = toml.load(f)

    if "project" in pyproject:
        return pyproject["project"]["version"]
    elif "tool" in pyproject and "poetry" in pyproject["tool"]:
        return pyproject["tool"]["poetry"]["version"]
    return "unknown"


def format_parameter(param: Any) -> str:
    """Format a parameter with its type and default."""
    parts = [param.name]

    if param.annotation:
        # Clean up annotation
        annotation = str(param.annotation).replace("typing.", "")
        parts.append(f": {annotation}")

    if param.default:
        default = str(param.default)
        if len(default) > 40:
            parts.append(" = ...")
        else:
            parts.append(f" = {default}")

    return "".join(parts)


def format_signature(func: Any) -> str:
    """Format function signature nicely."""
    name = func.name
    params = []

    for param in func.parameters:
        if param.name in ("self", "cls"):
            continue
        params.append(format_parameter(param))

    # Format return type
    returns = ""
    if func.returns:
        returns = f" -> {str(func.returns).replace('typing.', '')}"

    # Multi-line if long
    if len(params) > 3 or sum(len(p) for p in params) > 80:
        sig = f"{name}(\n"
        for i, param in enumerate(params):
            sig += f"    {param}"
            if i < len(params) - 1:
                sig += ","
            sig += "\n"
        sig += f"){returns}"
        return sig
    else:
        # Single line
        return f"{name}({', '.join(params)}){returns}"


def render_docstring_section(section: Any) -> str:
    """Render a docstring section (Args, Returns, etc.)."""
    lines = []

    if section.kind.value == "text":
        # Only include text sections that are descriptions, not raw docstring with Args/Returns/etc
        text = section.value.strip() if section.value else ""
        # Skip if it contains raw docstring sections (Args:, Returns:, etc.)
        if text and not any(
            keyword in text
            for keyword in [
                "Args:",
                "Returns:",
                "Raises:",
                "Examples:",
                "Attributes:",
                "Note:",
            ]
        ):
            lines.append(text)
            lines.append("")

    elif section.kind.value == "parameters":
        lines.append("**Parameters:**\n")
        for param in section.value:
            name = param.name
            desc = param.description.strip() if param.description else ""
            annotation = f" (`{param.annotation}`)" if param.annotation else ""
            # Handle multi-line descriptions with proper indentation
            if "\n" in desc:
                desc_lines = desc.split("\n")
                lines.append(f"- **{name}**{annotation}: {desc_lines[0]}")
                for desc_line in desc_lines[1:]:
                    lines.append(f"  {desc_line.strip()}")
            else:
                lines.append(f"- **{name}**{annotation}: {desc}")
        lines.append("")

    elif section.kind.value == "returns":
        lines.append("**Returns:**\n")
        # section.value can be a list or a single object
        if isinstance(section.value, list):
            for ret in section.value:
                annotation = f" `{ret.annotation}`" if ret.annotation else ""
                desc = ret.description if ret.description else ""
                if annotation or desc:
                    lines.append(
                        f"{annotation} - {desc}"
                        if annotation and desc
                        else (annotation or desc)
                    )
        else:
            annotation = (
                f" `{section.value.annotation}`"
                if section.value and section.value.annotation
                else ""
            )
            desc = section.value.description if section.value else ""
            if annotation or desc:
                lines.append(
                    f"{annotation} - {desc}"
                    if annotation and desc
                    else (annotation or desc)
                )
        lines.append("")

    elif section.kind.value == "raises":
        lines.append("**Raises:**\n")
        for exc in section.value:
            lines.append(f"- **{exc.annotation}**: {exc.description}")
        lines.append("")

    elif section.kind.value == "examples":
        lines.append("**Examples:**\n")
        for example in section.value:
            # example is a tuple of (kind, description)
            if hasattr(example, "description"):
                desc = example.description
            else:
                desc = str(example[1]) if isinstance(example, tuple) else str(example)

            # Format as code block if it looks like code
            if desc.strip():
                lines.append("```python")
                lines.append(desc.strip())
                lines.append("```")
        lines.append("")

    elif section.kind.value == "attributes":
        lines.append("**Attributes:**\n")
        for attr in section.value:
            annotation = f" (`{attr.annotation}`)" if attr.annotation else ""
            lines.append(f"- **{attr.name}**{annotation}: {attr.description}")
        lines.append("")

    return "\n".join(lines)


def document_function(func: Function, level: int = 2) -> str:
    """Generate markdown for a function."""
    lines = []
    heading = "#" * level

    lines.append(f"{heading} `{func.name}`\n")

    # Signature
    sig = format_signature(func)
    lines.append("```python")
    lines.append(sig)
    lines.append("```\n")

    # Docstring
    if func.docstring:
        # Get the summary (first line/paragraph)
        if func.docstring.value:
            # Extract just the first sentence or paragraph before Args:
            summary = func.docstring.value.split("\n\n")[0].split("Args:")[0].strip()
            if summary:
                lines.append(summary)
                lines.append("")

        # Sections (Args, Returns, etc.) - skip text sections
        for section in func.docstring.parsed:
            if section.kind.value != "text":
                section_text = render_docstring_section(section)
                if section_text:
                    lines.append(section_text)

    return "\n".join(lines)


def document_class(cls: Class, level: int = 2) -> str:
    """Generate markdown for a class."""
    lines = []
    heading = "#" * level

    lines.append(f"{heading} {cls.name}\n")

    # Class docstring
    if cls.docstring:
        # Get the summary (first line/paragraph)
        if cls.docstring.value:
            summary = (
                cls.docstring.value.split("\n\n")[0]
                .split("Args:")[0]
                .split("Example:")[0]
                .strip()
            )
            if summary:
                lines.append(summary)
                lines.append("")

        # Sections (Examples, Attributes, etc.) - skip text sections
        for section in cls.docstring.parsed:
            if section.kind.value != "text":
                section_text = render_docstring_section(section)
                if section_text:
                    lines.append(section_text)

    # __init__ method
    if "__init__" in cls.members:
        init = cls.members["__init__"]
        lines.append(f"{heading}# Constructor\n")
        sig = format_signature(init)
        lines.append("```python")
        lines.append(sig)
        lines.append("```\n")

        if init.docstring:
            # Get the summary (first line/paragraph)
            if init.docstring.value:
                summary = (
                    init.docstring.value.split("\n\n")[0].split("Args:")[0].strip()
                )
                if summary:
                    lines.append(summary)
                    lines.append("")

            # Sections (Args, Returns, etc.) - skip text sections
            for section in init.docstring.parsed:
                if section.kind.value != "text":
                    section_text = render_docstring_section(section)
                    if section_text:
                        lines.append(section_text)

    # Public methods
    methods = [
        (name, member)
        for name, member in cls.members.items()
        if isinstance(member, Function)
        if not name.startswith("_") or name == "__init__"
        if name != "__init__"  # Already documented
    ]

    if methods:
        lines.append(f"{heading}# Methods\n")
        for name, method in methods:
            lines.append(document_function(method, level + 2))

    # Properties/Attributes
    attrs = [
        (name, member)
        for name, member in cls.members.items()
        if isinstance(member, Attribute)
        if not name.startswith("_")
    ]

    if attrs:
        lines.append(f"{heading}# Attributes\n")
        for name, attr in attrs:
            annotation = f" (`{attr.annotation}`)" if attr.annotation else ""
            desc = attr.docstring.value if attr.docstring else ""
            lines.append(f"- **{name}**{annotation}: {desc}")
        lines.append("")

    return "\n".join(lines)


def export_module(module_name: str, title: str) -> str:
    """Export a module's documentation."""
    version = get_version()
    lines = []

    lines.append(f"# {title}\n")
    lines.append(f"> **Version:** langchain-mariadb v{version}\n")

    # Load module with Google-style docstring parsing
    module = load(module_name, docstring_parser="google")

    # Module docstring
    if module.docstring:
        # Get the summary (first paragraph before Example:)
        if module.docstring.value:
            # Split on double newline or Example:
            parts = module.docstring.value.split("Example:")
            summary = parts[0].strip()
            if summary:
                lines.append(summary)
                lines.append("")

            # If there's an example, format it properly
            if len(parts) > 1:
                lines.append("**Example:**\n")
                lines.append("```python")
                # Clean up the example code
                example_code = parts[1].strip()
                # Remove >>> and ... prefixes
                cleaned_lines = []
                for line in example_code.split("\n"):
                    line = line.strip()
                    if line.startswith(">>> ") or line.startswith("... "):
                        cleaned_lines.append(line[4:])
                    elif line.startswith(">>>") or line.startswith("..."):
                        cleaned_lines.append(line[3:])
                    elif line and not line.startswith("#"):
                        # Output lines or comments
                        cleaned_lines.append(line)
                lines.append("\n".join(cleaned_lines))
                lines.append("```")
                lines.append("")

    # Document classes
    for name, member in module.members.items():
        if isinstance(member, Class) and not name.startswith("_"):
            lines.append(document_class(member, level=2))
            lines.append("---\n")

    # Document functions
    for name, member in module.members.items():
        if isinstance(member, Function) and not name.startswith("_"):
            lines.append(document_function(member, level=2))

    return "\n".join(lines)


def main():
    """Generate GitBook-compatible markdown using griffe."""
    output_dir = Path(__file__).parent / "gitbook_export"
    output_dir.mkdir(exist_ok=True)

    modules = {
        "vectorstores.md": ("langchain_mariadb.vectorstores", "Vector Stores"),
        "chat_message_histories.md": (
            "langchain_mariadb.chat_message_histories",
            "Chat Message History",
        ),
        "expression_filter.md": (
            "langchain_mariadb.expression_filter",
            "Expression Filters",
        ),
        "translator.md": ("langchain_mariadb.translator", "Translator"),
    }

    for filename, (module_name, title) in modules.items():
        try:
            content = export_module(module_name, title)
            filepath = output_dir / filename
            filepath.write_text(content)
            print(f"✓ Exported {filename}")
        except Exception as e:
            print(f"✗ Failed to export {filename}: {e}")

    # Create index
    version = get_version()
    index_content = f"""# API Reference

> **Version:** langchain-mariadb v{version}

Complete API reference for the langchain-mariadb package.

## Modules

- **[Vector Stores](vectorstores.md)** - MariaDB vector store for embeddings
- **[Chat Message History](chat_message_histories.md)** - Persistent chat history
- **[Expression Filters](expression_filter.md)** - Metadata filtering utilities
- **[Translator](translator.md)** - Natural language to SQL translation
"""
    (output_dir / "index.md").write_text(index_content)
    print("✓ Exported index.md")

    print("\n✓ GitBook export complete!")
    print(f"  Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
