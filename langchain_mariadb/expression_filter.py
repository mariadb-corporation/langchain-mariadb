"""
A flexible and composable filter expression system for building SQL-like queries.

This module provides a builder pattern implementation for creating complex filter
expressions that can be converted to various query formats.

Example:
    >>> # Dict filter value
    >>> filter = {
    ...    '$or': [{'status': {'$eq': 'active'}}, {'status': {'$eq': 'pending'}}],
    ...    'age': {'$gte': 18}
    ...    'country': {'$in': ['US', 'CA', 'UK']}
    ...}
    >>>
    >>> # Convert to SQL-like string (with a proper converter implementation)
    >>> converter = SQLFilterExpressionConverter()  # Some converter
    >>> sql_where = converter.convert_expression(filter)
    >>> print(sql_where)
    >>> # Output:
    >>> # (status = 'active' OR status = 'pending')
    >>> # AND age >= 18 AND country IN ['US','CA','UK']
"""
import collections
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum, auto
from typing import Any, List, Optional, Union

# Type aliases
ValueType = Union[
    int,
    str,
    bool,
    float,
    Sequence[int],
    Sequence[str],
    Sequence[bool],
    Sequence[float],
]
Operand = Union["Key", "Value", "Expression", "Group"]


class Operator(Enum):
    """Enumeration of supported filter operations"""

    AND = auto()
    OR = auto()
    EQ = auto()
    NE = auto()
    GT = auto()
    GTE = auto()
    LT = auto()
    LTE = auto()
    LIKE = auto()
    NLIKE = auto()
    IN = auto()
    NIN = auto()
    NOT = auto()


# Operator negation mapping
TYPE_NEGATION_MAP = {
    Operator.AND: Operator.OR,
    Operator.OR: Operator.AND,
    Operator.EQ: Operator.NE,
    Operator.LIKE: Operator.NLIKE,
    Operator.NE: Operator.EQ,
    Operator.GT: Operator.LTE,
    Operator.GTE: Operator.LT,
    Operator.LT: Operator.GTE,
    Operator.LTE: Operator.GT,
    Operator.IN: Operator.NIN,
    Operator.NIN: Operator.IN,
    Operator.NOT: Operator.NOT,
}


class Key:
    """Represents a key in a filter expression"""

    def __init__(self, key: str):
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}")
        if not key.strip():
            raise ValueError("Key cannot be empty")
        self.key = key


class Value:
    """Represents a value in a filter expression"""

    def __init__(self, value: ValueType):
        if not isinstance(value, (int, str, float, bool, collections.abc.Sequence)):
            raise TypeError(f"Unsupported value type: {type(value)}")
        self.value = value


class Expression:
    """
    Represents a boolean filter expression with a specific structure:
    - Consists of a left operand, an operator, and an optional right operand
    - Enables construction of complex filtering logic using different types of
      comparisons
    """

    def __init__(self, type_: Operator, left: Operand, right: Optional[Operand] = None):
        self.type = type_
        self.left = left
        self.right = right


class Group:
    """
    Represents a grouped collection of filter expressions that should be evaluated
    together
    - Enables creating complex, nested filtering logic with specific evaluation
      precedence
    - Analogous to parentheses in mathematical or logical expressions
    """

    def __init__(self, content: Expression):
        self.content = content


class StringBuilder:
    """Simple StringBuilder implementation for efficient string concatenation"""

    def __init__(self) -> None:
        self.buffer: List[str] = []
        self._length: int = 0

    def append(self, string: str) -> None:
        if not isinstance(string, str):
            raise TypeError(f"Can only append strings, got {type(string)}")
        self.buffer.append(string)
        self._length += len(string)

    def __str__(self) -> str:
        return "".join(self.buffer)

    def __len__(self) -> int:
        return self._length


def eq(key: str, value: ValueType) -> Expression:
    return Expression(
        Operator.EQ, Key(key), Value(value) if value is not None else None
    )


def ne(key: str, value: ValueType) -> Expression:
    return Expression(
        Operator.NE, Key(key), Value(value) if value is not None else None
    )


def gt(key: str, value: Union[int, str, float]) -> Expression:
    return Expression(
        Operator.GT, Key(key), Value(value) if value is not None else None
    )


def gte(key: str, value: Union[int, str, float]) -> Expression:
    return Expression(
        Operator.GTE, Key(key), Value(value) if value is not None else None
    )


def lt(key: str, value: Union[int, str, float]) -> Expression:
    return Expression(
        Operator.LT, Key(key), Value(value) if value is not None else None
    )


def lte(key: str, value: Union[int, str, float]) -> Expression:
    return Expression(
        Operator.LTE, Key(key), Value(value) if value is not None else None
    )


def like(key: str, value: Union[int, str, float]) -> Expression:
    return Expression(
        Operator.LIKE, Key(key), Value(value) if value is not None else None
    )


def nlike(key: str, value: Union[int, str, float]) -> Expression:
    return Expression(
        Operator.NLIKE, Key(key), Value(value) if value is not None else None
    )


def includes(
    key: str, values: Union[List[int], List[str], List[bool], List[float]]
) -> Expression:
    """Check if a key's value is in a list of values (formerly in_)"""
    return Expression(
        Operator.IN, Key(key), Value(values) if values is not None else None
    )


def excludes(
    key: str, values: Union[List[int], List[str], List[bool], List[float]]
) -> Expression:
    """Check if a key's value is not in a list of values (formerly nin)"""
    return Expression(
        Operator.NIN, Key(key), Value(values) if values is not None else None
    )


def both(left: Operand, right: Operand) -> Expression:
    """Combine two expressions with AND"""
    return Expression(Operator.AND, left, right)


def either(left: Operand, right: Operand) -> Expression:
    """Combine two expressions with OR"""
    return Expression(Operator.OR, left, right)


def negate(content: Expression) -> Expression:
    """Negate an expression (i.e. NOT)"""
    return Expression(Operator.NOT, content)


def group(content: Expression) -> Group:
    return Group(content)


# Operator mappings
STANDARD_SIMPLE_OPERATOR = {
    "$eq": eq,
    "$ne": ne,
    "$lt": lt,
    "$lte": lte,
    "$gt": gt,
    "$gte": gte,
}

STANDARD_LIST_OPERATOR = {
    "$in": includes,
    "$nin": excludes,
}

STANDARD_BETWEEN_OPERATOR = {
    "$like": like,
    "$nlike": nlike,
}

STANDARD_STRING_ONLY_OPERATOR = {
    "$like": like,
    "$nlike": nlike,
}

GROUP_OPERATORS = {"$and": both, "$or": either, "$not": negate}

SUPPORTED_OPERATORS = (
    set(STANDARD_SIMPLE_OPERATOR)
    .union(STANDARD_LIST_OPERATOR)
    .union(GROUP_OPERATORS)
    .union(STANDARD_STRING_ONLY_OPERATOR)
    .union(
        {
            "$between": None,
        }
    )
)


class FilterExpressionConverter(ABC):
    """
    Abstract base class defining the interface for converting filter expressions
    into various string-based query representations
    """

    @abstractmethod
    def convert_expression(self, filters: dict) -> str:
        """Convert a complete expression into its string representation"""
        pass

    @abstractmethod
    def convert_symbol_to_context(
        self, exp: Expression, context: StringBuilder
    ) -> None:
        """Determine the appropriate operation symbol for a given expression"""
        pass

    @abstractmethod
    def convert_operand_to_context(
        self, operand: Operand, context: StringBuilder
    ) -> None:
        """Convert an operand into a string representation within a given context"""
        pass

    @abstractmethod
    def convert_expression_to_context(
        self, expression: Expression, context: StringBuilder
    ) -> None:
        """Convert an expression to its string representation in the given context"""
        pass

    @abstractmethod
    def convert_key_to_context(self, filter_key: Key, context: StringBuilder) -> None:
        """Convert a key to its string representation in the given context"""
        pass

    @abstractmethod
    def convert_value_to_context(
        self, filter_value: Value, context: StringBuilder
    ) -> None:
        """Convert a value to its string representation in the given context"""
        pass

    @abstractmethod
    def convert_single_value_to_context(
        self, value: ValueType, context: StringBuilder
    ) -> None:
        """Convert a single value to its string representation in the given context"""
        pass

    @abstractmethod
    def write_group_start(self, group: Group, context: StringBuilder) -> None:
        """Write the start of a group in the given context"""
        pass

    @abstractmethod
    def write_group_end(self, group: Group, context: StringBuilder) -> None:
        """Write the end of a group in the given context"""
        pass

    @abstractmethod
    def write_value_range_start(
        self, list_value: Value, context: StringBuilder
    ) -> None:
        """Write the start of a value range in the given context"""
        pass

    @abstractmethod
    def write_value_range_end(self, list_value: Value, context: StringBuilder) -> None:
        """Write the end of a value range in the given context"""
        pass

    @abstractmethod
    def write_value_range_separator(
        self, list_value: Value, context: StringBuilder
    ) -> None:
        """Write the separator between values in a range in the given context"""
        pass


class BaseFilterExpressionConverter(FilterExpressionConverter):
    """
    Base implementation of the FilterExpressionConverter interface providing
    common functionality for converting filter expressions to string representations
    """

    def _validate_expression(self, expression: Expression) -> None:
        """Validate expression structure before conversion"""
        if not isinstance(expression, Expression):
            raise TypeError(f"Expected Expression, got {type(expression)}")
        if expression.type not in Operator:
            raise ValueError(f"Invalid operator type: {expression.type}")
        if expression.left is None:
            raise ValueError("Expression must have a left operand")
        if expression.type not in (Operator.NOT,) and expression.right is None:
            raise ValueError(
                f"Expression with operator {expression.type} must have a right operand"
            )

    def convert_expression(self, filters: dict) -> str:
        exp = _transform_to_expression(filters)
        if exp is None:
            return ""
        self._validate_expression(exp)
        return self._convert_operand(exp)

    def _convert_operand(self, operand: Operand) -> str:
        context = StringBuilder()
        self.convert_operand_to_context(operand, context)
        return str(context)

    def convert_symbol_to_context(
        self, exp: Expression, context: StringBuilder
    ) -> None:
        symbol_map = {
            Operator.AND: " AND ",
            Operator.OR: " OR ",
            Operator.EQ: " = ",
            Operator.NE: " != ",
            Operator.LT: " < ",
            Operator.LTE: " <= ",
            Operator.GT: " > ",
            Operator.GTE: " >= ",
            Operator.IN: " IN ",
            Operator.NOT: " NOT IN ",
            Operator.NIN: " NOT IN ",
            Operator.LIKE: " LIKE ",
            Operator.NLIKE: " NOT LIKE ",
        }
        if exp.type not in symbol_map:
            raise ValueError(f"Unsupported expression type: {exp.type}")
        context.append(symbol_map[exp.type])

    def convert_operand_to_context(
        self, operand: Operand, context: StringBuilder
    ) -> None:
        if isinstance(operand, Group):
            self._convert_group_to_context(operand, context)
        elif isinstance(operand, Key):
            self.convert_key_to_context(operand, context)
        elif isinstance(operand, Value):
            self.convert_value_to_context(operand, context)
        elif isinstance(operand, Expression):
            if (
                operand.type != Operator.NOT
                and operand.type != Operator.AND
                and operand.type != Operator.OR
                and not isinstance(operand.right, Value)
            ):
                raise ValueError(
                    "Non AND/OR expression must have Value right argument!"
                )

            if operand.type == Operator.NOT:
                self._convert_not_expression_to_context(operand, context)
            else:
                self.convert_expression_to_context(operand, context)
        else:
            raise ValueError(f"Unexpected operand type: {type(operand)}")

    def _convert_not_expression_to_context(
        self, expression: Expression, context: StringBuilder
    ) -> None:
        self.convert_operand_to_context(self._negate_operand(expression), context)

    def _negate_operand(self, operand: Operand) -> Operand:
        if isinstance(operand, Group):
            in_ex = self._negate_operand(operand.content)
            if isinstance(in_ex, Group):
                in_ex = in_ex.content
                return Group(in_ex)
            raise ValueError(f"Unexpected operand type: {type(operand)}")
        elif isinstance(operand, Expression):
            if operand.type == Operator.NOT:
                return self._negate_operand(operand.left)
            elif operand.type in (Operator.AND, Operator.OR):
                if operand.right is None:
                    raise ValueError("Unexpected None value")
                return Expression(
                    TYPE_NEGATION_MAP[operand.type],
                    self._negate_operand(operand.left),
                    self._negate_operand(operand.right),
                )
            elif operand.type in TYPE_NEGATION_MAP:
                return Expression(
                    TYPE_NEGATION_MAP[operand.type], operand.left, operand.right
                )
            else:
                raise ValueError(f"Unknown expression type: {operand.type}")
        else:
            raise ValueError(f"Cannot negate operand of type: {type(operand)}")

    def convert_value_to_context(
        self, filter_value: Value, context: StringBuilder
    ) -> None:
        if isinstance(filter_value.value, (list, tuple)):
            self.write_value_range_start(filter_value, context)
            for i, value in enumerate(filter_value.value):
                self.convert_single_value_to_context(value, context)
                if i < len(filter_value.value) - 1:
                    self.write_value_range_separator(filter_value, context)
            self.write_value_range_end(filter_value, context)
        else:
            self.convert_single_value_to_context(filter_value.value, context)

    def convert_single_value_to_context(
        self, value: ValueType, context: StringBuilder
    ) -> None:
        if isinstance(value, str):
            context.append(f"'{value}'")
        else:
            context.append(str(value))

    def _convert_group_to_context(self, group: Group, context: StringBuilder) -> None:
        self.write_group_start(group, context)
        self.convert_operand_to_context(group.content, context)
        self.write_group_end(group, context)

    def write_value_range_start(
        self, list_value: Value, context: StringBuilder
    ) -> None:
        context.append("[")

    def write_value_range_end(self, list_value: Value, context: StringBuilder) -> None:
        context.append("]")

    def write_value_range_separator(
        self, list_value: Value, context: StringBuilder
    ) -> None:
        context.append(",")


def _transform_to_expression(
    filters: Union[None, dict] = None,
) -> Union[Expression, None]:
    """Create an Expression from a dictionary filter.

    Args:
        filters: Dictionary of filters

    Returns:
        Expression object representing the filter clause, or None if no filters

    Raises:
        ValueError: If filter specification is invalid
    """
    if filters is None:
        return None

    if isinstance(filters, dict):
        if len(filters) == 1:
            # Check for top-level operators ($AND, $OR, $NOT)
            key, value = list(filters.items())[0]
            if key.startswith("$"):
                # Validate operator
                if key.lower() not in GROUP_OPERATORS.keys():
                    raise ValueError(
                        f"Invalid filter condition. Expected $and, $or or $not "
                        f"but got: {key}"
                    )
            else:
                # Single field filter
                return _handle_field_filter(key, filters[key])

            # Handle logical operators
            if key.lower() == "$and" or key.lower() == "$or":
                if not isinstance(value, list) or len(value) < 2:
                    raise ValueError(
                        f"Expected a list of at least 2 elements for $and/$or, "
                        f"but got: {value}"
                    )
                    # Build AND chain
                val0 = _ensureValue(_transform_to_expression(value[0]))
                exp = _ensureValue(_transform_to_expression(value[1]))

                _len = len(value)
                while _len > 2:
                    v1 = _transform_to_expression(value[_len - 1])
                    v2 = _transform_to_expression(value[_len - 2])
                    if v1 is None:
                        if v2 is not None:
                            exp = v2
                    else:
                        if v2 is None:
                            exp = v1
                        else:
                            if key.lower() == "$and":
                                exp = both(
                                    _ensureValue(
                                        _transform_to_expression(value[_len - 1])
                                    ),
                                    _ensureValue(
                                        _transform_to_expression(value[_len - 2])
                                    ),
                                )
                            else:
                                exp = either(
                                    _ensureValue(
                                        _transform_to_expression(value[_len - 1])
                                    ),
                                    _ensureValue(
                                        _transform_to_expression(value[_len - 2])
                                    ),
                                )
                    _len = _len - 1
                if key.lower() == "$and":
                    return both(val0, exp)
                else:
                    return either(val0, exp)

            else:  # key.lower() == "$not":
                # Handle NOT operator
                if isinstance(value, Expression):
                    return negate(value)
                if isinstance(value, dict):
                    return negate(_ensureValue(_transform_to_expression(value)))
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                    if isinstance(value, dict):
                        return negate(_ensureValue(_transform_to_expression(value)))

                raise ValueError(
                    f"Invalid filter condition for $not. Expected Expression, dict, "
                    f"or list with single item, but got: {type(value)}"
                )

        elif len(filters) > 1:
            # Multiple field filters - combine with AND
            for key in filters:
                if key.startswith("$"):
                    raise ValueError(
                        f"Invalid filter condition. Expected a field but got: {key}"
                    )
            expressions = [_handle_field_filter(k, v) for k, v in filters.items()]
            if len(expressions) > 1:
                return both(expressions[0], expressions[1])
            elif expressions:
                return expressions[0]
            else:
                raise ValueError("No valid expressions in filter")
        else:
            raise ValueError("Got an empty dictionary for filters")
    else:
        raise ValueError(
            f"Invalid filter type: Expected dict or Expression but got {type(filters)}"
        )


def _ensureValue(val: Union[Expression, None]) -> Expression:
    if val is None:
        raise ValueError("Invalid filter value: Expected Expression, but got None")
    return val

    # Filter methods


def _handle_field_filter(
    field: str,
    value: Any,
) -> Expression:
    """Create a filter for a specific field.

    Args:
        field: Name of field to filter on
        value: Value to filter by. Can be:
            - Direct value for equality filter
            - Dict with operator and value for other filters

    Returns:
        Filter expression

    Raises:
        ValueError: If field name or filter specification is invalid
    """
    if not isinstance(field, str):
        raise ValueError(
            f"Field should be a string but got: {type(field)} with value: {field}"
        )

    if field.startswith("$"):
        raise ValueError(
            f"Invalid filter condition. Expected a field but got an operator: {field}"
        )

    # Allow [a-zA-Z0-9_] only
    if not field.isidentifier():
        raise ValueError(f"Invalid field name: {field}. Expected a valid identifier.")

    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(
                "Invalid filter condition. Expected a dictionary with a single key "
                f"that corresponds to an operator but got {len(value)} keys. "
                f"The first few keys are: {list(value.keys())[:3]}"
            )
        operator, filter_value = list(value.items())[0]

        # Verify operator is valid
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Invalid operator: {operator}. Expected one of {SUPPORTED_OPERATORS}"
            )
    else:
        # Default to equality filter
        operator = "$eq"
        filter_value = value

    if operator in STANDARD_SIMPLE_OPERATOR:
        return STANDARD_SIMPLE_OPERATOR[operator](field, filter_value)
    elif operator == "$between":
        # Use AND with two comparisons
        low, high = filter_value
        return both(gte(field, low), lte(field, high))
    elif operator in STANDARD_STRING_ONLY_OPERATOR:
        for val in filter_value:
            if not isinstance(val, str):
                raise NotImplementedError(
                    f"Unsupported type: {type(val)} for value: {val}"
                )
        return STANDARD_STRING_ONLY_OPERATOR[operator](field, filter_value)
    elif operator in STANDARD_LIST_OPERATOR:
        for val in filter_value:
            if not isinstance(val, (str, int, float)):
                raise NotImplementedError(
                    f"Unsupported type: {type(val)} for value: {val}"
                )
            if isinstance(val, bool):
                raise NotImplementedError(
                    f"Unsupported type: {type(val)} for value: {val}"
                )
        if operator == "$in":
            return includes(field, filter_value)
        else:
            return excludes(field, filter_value)
    else:
        raise NotImplementedError(f"Operator {operator} not implemented")


class MariaDBFilterExpressionConverter(BaseFilterExpressionConverter):
    """Converter for MariaDB filter expressions."""

    def __init__(self, metadata_field_name: str):
        super().__init__()
        self.metadata_field_name = metadata_field_name

    def convert_expression_to_context(
        self, expression: Expression, context: StringBuilder
    ) -> None:
        super().convert_operand_to_context(expression.left, context)
        super().convert_symbol_to_context(expression, context)
        if expression.right:
            super().convert_operand_to_context(expression.right, context)

    def convert_key_to_context(self, key: Key, context: StringBuilder) -> None:
        context.append(f"JSON_VALUE({self.metadata_field_name}, '$.{key.key}')")

    def write_value_range_start(
        self, _list_value: Value, context: StringBuilder
    ) -> None:
        context.append("(")

    def write_value_range_end(self, _list_value: Value, context: StringBuilder) -> None:
        context.append(")")

    def write_group_start(self, _group: Group, context: StringBuilder) -> None:
        context.append("(")

    def write_group_end(self, _group: Group, context: StringBuilder) -> None:
        context.append(")")
