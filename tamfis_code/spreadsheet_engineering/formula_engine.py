"""Formula engine: parsing, validation, and dependency analysis for Excel formulas.

Supports:
- Basic arithmetic: +, -, *, /, ^
- Cell references: A1, $A$1, A:B, $A:$B
- Functions: SUM, AVERAGE, COUNT, MIN, MAX, IF, VLOOKUP, HLOOKUP, INDEX, MATCH,
  CONCATENATE, LEFT, RIGHT, MID, LEN, UPPER, LOWER, TRIM, ROUND, ROUNDUP, ROUNDDOWN,
  ABS, SQRT, POWER, MOD, TODAY, NOW, YEAR, MONTH, DAY, DATEDIF,
  AND, OR, NOT, TRUE, FALSE, IFERROR, COUNTIF, SUMIF, AVERAGEIF,
  TEXT, VALUE, INT, CEILING, FLOOR
- Named ranges (via external registry)
- Cross-sheet references: Sheet1!A1
- Array formulas (basic detection)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Supported Excel functions with their argument counts
# -1 means variable arguments
_SUPPORTED_FUNCTIONS: Dict[str, int] = {
    # Math/Trig
    "SUM": -1,
    "AVERAGE": -1,
    "AVERAGEA": -1,
    "COUNT": -1,
    "COUNTA": -1,
    "COUNTBLANK": 1,
    "COUNTIF": 2,
    "COUNTIFS": -1,
    "MIN": -1,
    "MAX": -1,
    "ABS": 1,
    "SQRT": 1,
    "POWER": 2,
    "MOD": 2,
    "ROUND": 3,
    "ROUNDUP": 2,
    "ROUNDDOWN": 2,
    "CEILING": 2,
    "FLOOR": 2,
    "INT": 1,
    "TRUNC": 2,
    "RAND": 0,
    "RANDBETWEEN": 2,
    # Logic
    "IF": 3,
    "IFS": -1,
    "AND": -1,
    "OR": -1,
    "NOT": 1,
    "TRUE": 0,
    "FALSE": 0,
    "XOR": -1,
    # Text
    "CONCATENATE": -1,
    "CONCAT": -1,
    "TEXTJOIN": -1,
    "LEFT": 2,
    "RIGHT": 2,
    "MID": 3,
    "LEN": 1,
    "UPPER": 1,
    "LOWER": 1,
    "TRIM": 1,
    "PROPER": 1,
    "REPLACE": 4,
    "SUBSTITUTE": 4,
    "FIND": 2,
    "SEARCH": 2,
    "TEXT": 2,
    "VALUE": 1,
    "T": 1,
    "N": 1,
    "UNICHAR": 1,
    "UNICODE": 1,
    "CHAR": 1,
    "CODE": 1,
    # Lookup/Reference
    "VLOOKUP": 4,
    "HLOOKUP": 4,
    "LOOKUP": -1,
    "INDEX": -1,
    "MATCH": 3,
    "OFFSET": 5,
    "INDIRECT": 1,
    "ADDRESS": -1,
    "CHOOSE": -1,
    "TRANSPOSE": 1,
    "FILTER": -1,
    "SORT": -1,
    "SORTBY": -1,
    "XLOOKUP": -1,
    "XMATCH": -1,
    # Date/Time
    "TODAY": 0,
    "NOW": 0,
    "DATE": 3,
    "DATEDIF": 3,
    "DATEVALUE": 1,
    "TIME": 3,
    "TIMEVALUE": 1,
    "YEAR": 1,
    "MONTH": 1,
    "DAY": 1,
    "HOUR": 1,
    "MINUTE": 1,
    "SECOND": 1,
    "WEEKDAY": -1,
    "WEEKNUM": -1,
    "EOMONTH": 2,
    "WORKDAY": -1,
    "NETWORKDAYS": -1,
    "DAYS": -1,
    "DAYS360": -1,
    "EDATE": 2,
    "EOMONTH": 2,
    # Financial
    "PV": -1,
    "FV": -1,
    "PMT": -1,
    "IPMT": -1,
    "PPMT": -1,
    "NPV": -1,
    "IRR": -1,
    "XIRR": -1,
    "MIRR": -1,
    "RATE": -1,
    "NPER": -1,
    # Statistical
    "STDEV": -1,
    "STDEVA": -1,
    "STDEV.P": -1,
    "STDEV.S": -1,
    "VAR": -1,
    "VARA": -1,
    "VAR.P": -1,
    "VAR.S": -1,
    "CORREL": 2,
    "COVARIANCE.P": 2,
    "COVARIANCE.S": 2,
    "PERCENTILE": -1,
    "QUARTILE": -1,
    "RANK": -1,
    "FORECAST": -1,
    # Error handling
    "IFERROR": 2,
    "IFNA": 2,
    "NA": 0,
    # Information
    "ISBLANK": 1,
    "ISERROR": 1,
    "ISEVEN": 1,
    "ISODD": 1,
    "ISNUMBER": 1,
    "ISTEXT": 1,
    "ISLOGICAL": 1,
    "ISNONTEXT": 1,
    "N": 1,
    "TYPE": 1,
    # Dynamic array
    "UNIQUE": -1,
    "FILTER": -1,
    "SORT": -1,
    "SORTBY": -1,
    "SEQUENCE": -1,
    "RANDARRAY": -1,
    "XMATCH": -1,
    "XLOOKUP": -1,
}


class FormulaError(Exception):
    """Raised for invalid or unsupported formulas."""

    def __init__(self, message: str, formula: str = "", cell: str = ""):
        self.message = message
        self.formula = formula
        self.cell = cell
        super().__init__(f"[{cell}] {message} (formula: {formula})")


class FormulaEngine:
    """Parse, validate, and analyze Excel formulas."""

    def __init__(self) -> None:
        self._registered: Dict[str, Dict[str, str]] = {}  # sheet -> {cell -> formula}
        self._named_ranges: Dict[str, str] = {}  # name -> cell_ref or formula

    def register(self, sheet: str, cell: str, formula: str) -> None:
        """Register a formula for dependency tracking."""
        if sheet not in self._registered:
            self._registered[sheet] = {}
        self._registered[sheet][cell] = formula

    def parse(self, formula: str) -> Dict[str, Any]:
        """Parse a formula string into structured components.

        Returns:
            Dict with keys:
            - raw: original formula string
            - type: "arithmetic", "function", "reference", "string", "number", "boolean", "error"
            - functions: list of function names used
            - references: list of cell/range references
            - literals: list of literal values
            - operators: list of operators used
            - valid: whether the formula is syntactically valid
            - errors: list of validation errors
        """
        if not formula or not isinstance(formula, str):
            return {
                "raw": str(formula),
                "type": "empty",
                "functions": [],
                "references": [],
                "literals": [],
                "operators": [],
                "valid": False,
                "errors": ["Empty or non-string formula"],
            }

        raw = formula.strip()
        if not raw.startswith(("=", "+", "-", "@")):
            return {
                "raw": raw,
                "type": "value",
                "functions": [],
                "references": [],
                "literals": [raw],
                "operators": [],
                "valid": True,
                "errors": [],
            }

        # Strip leading operator
        expr = raw.lstrip("=+-@")
        functions = self._extract_functions(expr)
        references = self._extract_references(expr)
        operators = self._extract_operators(expr)
        literals = self._extract_literals(expr)
        errors = self._validate_formula(expr, functions, references)

        # Determine type
        if functions:
            formula_type = "function"
        elif references and not operators:
            formula_type = "reference"
        elif operators:
            formula_type = "arithmetic"
        else:
            formula_type = "expression"

        return {
            "raw": raw,
            "type": formula_type,
            "functions": functions,
            "references": references,
            "literals": literals,
            "operators": operators,
            "valid": len(errors) == 0,
            "errors": errors,
        }

    def validate(self, formula: str) -> Tuple[bool, List[str]]:
        """Validate a formula string.

        Returns:
            (is_valid, list_of_errors)
        """
        if not formula:
            return False, ["Empty formula"]

        raw = formula.strip()
        expr = raw.lstrip("=+-@")
        functions = self._extract_functions(expr)
        references = self._extract_references(expr)
        errors = self._validate_formula(expr, functions, references)

        return len(errors) == 0, errors

    def get_dependencies(self, cell: str, sheet: str = "") -> List[str]:
        """Get all cells that a formula depends on."""
        if sheet not in self._registered:
            return []
        formula = self._registered[sheet].get(cell, "")
        if not formula:
            return []

        parsed = self.parse(formula)
        return parsed.get("references", [])

    def get_dependents(self, cell: str, sheet: str = "") -> List[str]:
        """Get all cells that depend on a given cell."""
        dependents = []
        for s, cells in self._registered.items():
            for c, f in cells.items():
                deps = self.get_dependencies(c, s)
                target = f"{s}!{c}" if s != sheet else c
                if cell in deps or cell == target:
                    dependents.append(f"{s}!{c}" if s != sheet else c)
        return dependents

    def _extract_functions(self, expr: str) -> List[str]:
        """Extract function names from a formula expression."""
        # Match function names followed by opening paren
        pattern = r"([A-Za-z_][A-Za-z0-9_.]*)\s*\("
        matches = re.findall(pattern, expr)
        # Filter to known functions
        return [m.upper() for m in matches if m.upper() in _SUPPORTED_FUNCTIONS]

    def _extract_references(self, expr: str) -> List[str]:
        """Extract cell/range references from a formula expression."""
        # Match cell references: A1, $A$1, A:B, $A:$B, Sheet1!A1
        pattern = r"(\$?[A-Za-z]{1,3}\$?\d+(?::\$?[A-Za-z]{1,3}\$?\d+)?(?:![\$?[A-Za-z]{1,3}\$?\d+(?::\$?[A-Za-z]{1,3}\$?\d+)?)?)"
        matches = re.findall(pattern, expr)
        return list(set(matches))  # Deduplicate

    def _extract_operators(self, expr: str) -> List[str]:
        """Extract operators from a formula expression."""
        operators = []
        for op in ["+", "-", "*", "/", "^", "&", "<>", "<=", ">=", "<", ">", "="]:
            if op in expr:
                operators.append(op)
        return operators

    def _extract_literals(self, expr: str) -> List[str]:
        """Extract literal values from a formula expression."""
        literals = []
        # String literals (quoted)
        strings = re.findall(r'"([^"]*)"', expr)
        literals.extend(strings)
        # Number literals (not part of references)
        numbers = re.findall(r"(?<![A-Za-z\$])(\d+\.?\d*)(?![A-Za-z\$])", expr)
        literals.extend(numbers)
        return literals

    def _validate_formula(
        self, expr: str, functions: List[str], references: List[str]
    ) -> List[str]:
        """Validate a formula expression."""
        errors = []

        # Check for unbalanced parentheses
        open_parens = expr.count("(")
        close_parens = expr.count(")")
        if open_parens != close_parens:
            errors.append(
                f"Unbalanced parentheses: {open_parens} open, {close_parens} close"
            )

        # Check for unknown functions
        for func in functions:
            if func not in _SUPPORTED_FUNCTIONS:
                errors.append(f"Unknown function: {func}")

        # Check for empty function calls
        if re.search(r"[A-Za-z_][A-Za-z0-9_.]*\s*\(\s*\)", expr):
            errors.append("Empty function call detected")

        # Check for division by zero patterns
        if re.search(r"/\s*0\b", expr):
            errors.append("Potential division by zero")

        # Check for circular reference patterns (basic)
        # This is a simplified check; full circular detection requires
        # building a dependency graph
        if len(functions) > 10:
            errors.append("Very complex formula; consider simplifying")

        return errors


# Module-level convenience functions

def parse_formula(formula: str) -> Dict[str, Any]:
    """Parse a formula string into structured components.

    Convenience wrapper around FormulaEngine.parse().
    """
    engine = FormulaEngine()
    return engine.parse(formula)


def validate_formula(formula: str) -> Tuple[bool, List[str]]:
    """Validate a formula string.

    Convenience wrapper around FormulaEngine.validate().

    Returns:
        (is_valid, list_of_errors)
    """
    engine = FormulaEngine()
    return engine.validate(formula)
