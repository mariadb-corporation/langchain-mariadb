"""
A flexible and composable filter expression system for building SQL-like queries.

This module provides a builder pattern implementation for creating complex filter
expressions that can be converted to various query formats.

Example:
    >>> # Create a builder instance
    >>> builder = FilterExpressionBuilder()
    >>> 
    >>> # Build a complex filter expression
    >>> filter_exp = builder.both(
    ...     builder.either(
    ...         builder.eq("status", "active"),
    ...         builder.eq("status", "pending")
    ...     ),
    ...     builder.both(
    ...         builder.gte("age", 18),
    ...         builder.includes("country", ["US", "CA", "UK"])
    ...     )
    ... )
    >>> 
    >>> # Convert to SQL-like string (with a proper converter implementation)
    >>> converter = SQLFilterExpressionConverter()  # Example converter
    >>> sql_where = converter.convert_expression(filter_exp)
    >>> print(sql_where)
    >>> # Output: (status = 'active' OR status = 'pending') AND (age >= 18 AND country IN ['US','CA','UK'])
    
The expression above would match records where:
- status is either 'active' or 'pending' AND
- age is 18 or greater AND
- country is one of US, CA, or UK

Features:
    - Composable filter expressions
    - Support for common operators (=, !=, >, >=, <, <=)
    - List operations (includes, excludes)
    - Logical operations (both, either, negate)
    - Pattern matching (like, nlike)
    - Grouping for complex conditions
    - Extensible conversion system for different output formats

Available Methods:
    Comparison:
        - eq(key, value)      # Equal to
        - ne(key, value)      # Not equal to
        - gt(key, value)      # Greater than
        - gte(key, value)     # Greater than or equal
        - lt(key, value)      # Less than
        - lte(key, value)     # Less than or equal
        
    Pattern Matching:
        - like(key, pattern)  # Pattern matching
        - nlike(key, pattern) # Negative pattern matching
        
    List Operations:
        - includes(key, list) # Check if value is in list
        - excludes(key, list) # Check if value is not in list
        
    Logical Operations:
        - both(left, right)   # Logical AND
        - either(left, right) # Logical OR
        - negate(expression)  # Logical NOT
        
    Grouping:
        - group(expression)   # Group expressions together
"""

from enum import Enum, auto
from typing import Union, List, Optional, Any
from abc import ABC, abstractmethod
from collections.abc import Sequence

# Type aliases
ValueType = Union[int, str, bool, Sequence[int], Sequence[str], Sequence[bool]]
Operand = Union['Key', 'Value', 'Expression', 'Group']

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
        if not isinstance(value, (int, str, float, bool, Sequence)):
            raise TypeError(f"Unsupported value type: {type(value)}")
        self.value = value

class Expression:
    """
    Represents a boolean filter expression with a specific structure:
    - Consists of a left operand, an operator, and an optional right operand
    - Enables construction of complex filtering logic using different types of comparisons
    """
    def __init__(self, type_: Operator, left: Operand, right: Optional[Operand] = None):
        self.type = type_
        self.left = left
        self.right = right

class Group:
    """
    Represents a grouped collection of filter expressions that should be evaluated together
    - Enables creating complex, nested filtering logic with specific evaluation precedence
    - Analogous to parentheses in mathematical or logical expressions
    """
    def __init__(self, content: Expression):
        self.content = content

class StringBuilder:
    """Simple StringBuilder implementation for efficient string concatenation"""
    def __init__(self):
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

class FilterExpressionBuilder:
    """
    Fluent builder for creating flexible and composable filter expressions

    Features:
    - Equality and inequality checks
    - Numeric comparisons (greater than, less than, etc.)
    - Logical combinations (AND, OR, NOT)
    - Collection membership tests (IN, NOT IN)
    - Expression grouping for complex nested conditions
    """

    def eq(self, key: str, value: ValueType) -> Expression:
        return Expression(Operator.EQ, Key(key), Value(value) if value is not None else None)

    def ne(self, key: str, value: ValueType) -> Expression:
        return Expression(Operator.NE, Key(key), Value(value) if value is not None else None)

    def gt(self, key: str, value: Union[int, str, float]) -> Expression:
        return Expression(Operator.GT, Key(key), Value(value) if value is not None else None)

    def gte(self, key: str, value: Union[int, str, float]) -> Expression:
        return Expression(Operator.GTE, Key(key), Value(value) if value is not None else None)

    def lt(self, key: str, value: Union[int, str, float]) -> Expression:
        return Expression(Operator.LT, Key(key), Value(value) if value is not None else None)

    def lte(self, key: str, value: Union[int, str, float]) -> Expression:
        return Expression(Operator.LTE, Key(key), Value(value) if value is not None else None)

    def like(self, key: str, value: Union[int, str, float]) -> Expression:
        return Expression(Operator.LIKE, Key(key), Value(value) if value is not None else None)

    def nlike(self, key: str, value: Union[int, str, float]) -> Expression:
        return Expression(Operator.NLIKE, Key(key), Value(value) if value is not None else None)

    def includes(self, key: str, values: Union[List[int], List[str], List[bool], List[float]]) -> Expression:
        """Check if a key's value is in a list of values (formerly in_)"""
        return Expression(Operator.IN, Key(key), Value(values) if values is not None else None)

    def excludes(self, key: str, values: Union[List[int], List[str], List[bool], List[float]]) -> Expression:
        """Check if a key's value is not in a list of values (formerly nin)"""
        return Expression(Operator.NIN, Key(key), Value(values) if values is not None else None)

    def both(self, left: Operand, right: Operand) -> Expression:
        """Combine two expressions with AND (formerly and_)"""
        return Expression(Operator.AND, left, right)

    def either(self, left: Operand, right: Operand) -> Expression:
        """Combine two expressions with OR (formerly or_)"""
        return Expression(Operator.OR, left, right)

    def negate(self, content: Expression) -> Expression:
        """Negate an expression (formerly not_)"""
        return Expression(Operator.NOT, content)

    def group(self, content: Expression) -> Group:
        return Group(content)

class FilterExpressionConverter(ABC):
    """
    Abstract base class defining the interface for converting filter expressions
    into various string-based query representations
    """

    @abstractmethod
    def convert_expression(self, expression: Expression) -> str:
        """Convert a complete expression into its string representation"""
        pass

    @abstractmethod
    def convert_symbol_to_context(self, exp: Expression, context: StringBuilder) -> None:
        """Determine the appropriate operation symbol for a given expression"""
        pass

    @abstractmethod
    def convert_operand_to_context(self, operand: Operand, context: StringBuilder) -> None:
        """Convert an operand into a string representation within a given context"""
        pass

    @abstractmethod
    def convert_expression_to_context(self, expression: Expression, context: StringBuilder) -> None:
        """Convert an expression to its string representation in the given context"""
        pass

    @abstractmethod
    def convert_key_to_context(self, filter_key: Key, context: StringBuilder) -> None:
        """Convert a key to its string representation in the given context"""
        pass

    @abstractmethod
    def convert_value_to_context(self, filter_value: Value, context: StringBuilder) -> None:
        """Convert a value to its string representation in the given context"""
        pass

    @abstractmethod
    def convert_single_value_to_context(self, value: ValueType, context: StringBuilder) -> None:
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
    def write_value_range_start(self, list_value: Value, context: StringBuilder) -> None:
        """Write the start of a value range in the given context"""
        pass

    @abstractmethod
    def write_value_range_end(self, list_value: Value, context: StringBuilder) -> None:
        """Write the end of a value range in the given context"""
        pass

    @abstractmethod
    def write_value_range_separator(self, list_value: Value, context: StringBuilder) -> None:
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
            raise ValueError(f"Expression with operator {expression.type} must have a right operand")

    def convert_expression(self, expression: Expression) -> str:
        self._validate_expression(expression)
        return self._convert_operand(expression)

    def _convert_operand(self, operand: Operand) -> str:
        context = StringBuilder()
        self.convert_operand_to_context(operand, context)
        return str(context)

    def convert_symbol_to_context(self, exp: Expression, context: StringBuilder) -> None:
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

    def convert_operand_to_context(self, operand: Operand, context: StringBuilder) -> None:
        if isinstance(operand, Group):
            self._convert_group_to_context(operand, context)
        elif isinstance(operand, Key):
            self.convert_key_to_context(operand, context)
        elif isinstance(operand, Value):
            self.convert_value_to_context(operand, context)
        elif isinstance(operand, Expression):
            if (operand.type != Operator.NOT and
                    operand.type != Operator.AND and
                    operand.type != Operator.OR and
                    not isinstance(operand.right, Value)):
                raise ValueError("Non AND/OR expression must have Value right argument!")

            if operand.type == Operator.NOT:
                self._convert_not_expression_to_context(operand, context)
            else:
                self.convert_expression_to_context(operand, context)
        else:
            raise ValueError(f"Unexpected operand type: {type(operand)}")

    def _convert_not_expression_to_context(self, expression: Expression, context: StringBuilder) -> None:
        self.convert_operand_to_context(self._negate_operand(expression), context)

    def _negate_operand(self, operand: Operand) -> Operand:
        if isinstance(operand, Group):
            in_ex = self._negate_operand(operand.content)
            if isinstance(in_ex, Group):
                in_ex = in_ex.content
            return Group(in_ex)

        elif isinstance(operand, Expression):
            if operand.type == Operator.NOT:
                return self._negate_operand(operand.left)
            elif operand.type in (Operator.AND, Operator.OR):
                return Expression(
                    TYPE_NEGATION_MAP[operand.type],
                    self._negate_operand(operand.left),
                    self._negate_operand(operand.right)
                )
            elif operand.type in TYPE_NEGATION_MAP:
                return Expression(
                    TYPE_NEGATION_MAP[operand.type],
                    operand.left,
                    operand.right
                )
            else:
                raise ValueError(f"Unknown expression type: {operand.type}")
        else:
            raise ValueError(f"Cannot negate operand of type: {type(operand)}")

    def convert_value_to_context(self, filter_value: Value, context: StringBuilder) -> None:
        if isinstance(filter_value.value, (list, tuple)):
            self.write_value_range_start(filter_value, context)
            for i, value in enumerate(filter_value.value):
                self.convert_single_value_to_context(value, context)
                if i < len(filter_value.value) - 1:
                    self.write_value_range_separator(filter_value, context)
            self.write_value_range_end(filter_value, context)
        else:
            self.convert_single_value_to_context(filter_value.value, context)

    def convert_single_value_to_context(self, value: ValueType, context: StringBuilder) -> None:
        if isinstance(value, str):
            context.append(f"'{value}'")
        else:
            context.append(str(value))

    def _convert_group_to_context(self, group: Group, context: StringBuilder) -> None:
        self.write_group_start(group, context)
        self.convert_operand_to_context(group.content, context)
        self.write_group_end(group, context)

    def write_value_range_start(self, list_value: Value, context: StringBuilder) -> None:
        context.append("[")

    def write_value_range_end(self, list_value: Value, context: StringBuilder) -> None:
        context.append("]")

    def write_value_range_separator(self, list_value: Value, context: StringBuilder) -> None:
        context.append(",")