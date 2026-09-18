"""Spreadsheet engineering module for tamfis-code.

Provides formula-aware xlsx creation, analysis, and manipulation
on top of openpyxl. Integrates with the existing artifacts.py
pipeline and MCP tool layer.
"""
from __future__ import annotations

from .engine import SpreadsheetEngine, create_spreadsheet, analyze_spreadsheet
from .xlsx_builder import create_workbook
from .xlsx_inspector import inspect_workbook
from .formula_engine import FormulaEngine, parse_formula, validate_formula
from .chart_generator import ChartGenerator, create_chart
from .templates import BUDGET_TEMPLATE, INVOICE_TEMPLATE, KPI_DASHBOARD_TEMPLATE

__all__ = [
    "SpreadsheetEngine",
    "create_spreadsheet",
    "analyze_spreadsheet",
    "create_workbook",
    "inspect_workbook",
    "FormulaEngine",
    "parse_formula",
    "validate_formula",
    "ChartGenerator",
    "create_chart",
    "BUDGET_TEMPLATE",
    "INVOICE_TEMPLATE",
    "KPI_DASHBOARD_TEMPLATE",
]
