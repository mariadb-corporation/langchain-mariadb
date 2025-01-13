import re

identifier_pattern = re.compile(r"[0-9a-zA-Z$_\u0080-\uFFFF]*", re.UNICODE)


def is_simple_identifier(identifier: str) -> bool:
    return (
        identifier is not None
        and len(identifier) > 0
        and identifier_pattern.match(identifier) is not None
    )


def enquote_identifier(identifier: str, always_quote: bool = False) -> str:
    if identifier.find("\u0000") != -1:
        raise ValueError("Invalid name - containing u0000 character")

    if is_simple_identifier(identifier):
        if len(identifier) < 1 or len(identifier) > 64:
            raise ValueError("Invalid identifier length")

        if always_quote:
            return f"`{identifier}`"

        # Check if identifier contains only digits
        if identifier.isdigit():
            return f"`{identifier}`"

        return identifier
    else:
        if identifier.startswith("`") and identifier.endswith("`"):
            identifier = identifier[1:-1]

        if len(identifier) < 1 or len(identifier) > 64:
            raise ValueError("Invalid identifier length")

        return f"`{identifier.replace('`', '``')}`"
