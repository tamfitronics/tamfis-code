"""Core spreadsheet engine: high-level API for creation and analysis.

This module provides the public-facing functions that integrate with
artifacts.py and the MCP tool layer. It delegates to the lower-level
modules (xlsx_builder, xlsx_inspector, formula_engine, chart_generator).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .xlsx_builder import create_workbook
from .xlsx_inspector import inspect_workbook
from .formula_engine import FormulaEngine


def create_spreadsheet(
    path: Path,
    content: Dict[str, Any],
    *,
    allow_formulas: bool = False,
    theme: str = "executive",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create an xlsx file from a structured content dict.

    Accepts the same format as artifacts.create_artifact for backwards
    compatibility, plus extended fields:

    - sheets: list of {name, rows, headers, formulas, freeze_panes,
      conditional_formatting, charts}
    - title: workbook title
    - theme: color theme (executive, modern, creative, natural)
    - allow_formulas: if True, cells starting with =/+/-/@ are written
      as formulas instead of escaped strings
    - charts: list of chart specs for embedded charts

    Returns metadata dict with success, path, size_bytes, sheets_created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = create_workbook(path, content, theme=theme)
        return {
            "success": True,
            "artifact_type": "xlsx",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sheets_created": written,
        }
    except Exception as exc:
        return {
            "success": False,
            "artifact_type": "xlsx",
            "path": str(path),
            "error": str(exc),
        }


def analyze_spreadsheet(
    path: Path,
    *,
    include_formulas: bool = True,
    include_styles: bool = False,
    include_charts: bool = True,
) -> Dict[str, Any]:
    """Analyze an existing xlsx file and return structured metadata.

    Returns:
    - workbook_info: title, sheets, dimensions
    - sheet_details: per-sheet row/column counts, formulas, data types
    - charts: list of chart descriptions if include_charts
    - summary: overall statistics
    """
    return inspect_workbook(
        path,
        include_formulas=include_formulas,
        include_styles=include_styles,
        include_charts=include_charts,
    )


class SpreadsheetEngine:
    """Stateful spreadsheet engine for multi-step operations.

    Maintains an in-memory representation of a workbook and provides
    methods for incremental modifications.
    """

    def __init__(self) -> None:
        self._workbook_data: Optional[Dict[str, Any]] = None
        self._formula_engine = FormulaEngine()

    def load(self, path: Path) -> None:
        """Load a workbook from file."""
        self._workbook_data = inspect_workbook(path)

    def create(self, content: Dict[str, Any]) -> None:
        """Create a new workbook from content dict."""
        self._workbook_data = {
            "content": content,
            "formulas": {},
            "charts": [],
        }

    def add_sheet(
        self,
        name: str,
        rows: List[List[Any]],
        *,
        formulas: Optional[Dict[str, str]] = None,
        freeze_panes: Optional[str] = None,
    ) -> None:
        """Add a sheet to the current workbook."""
        if self._workbook_data is None:
            raise RuntimeError("No workbook loaded. Call create() or load() first.")
        if "sheets" not in self._workbook_data:
            self._workbook_data["sheets"] = []
        sheet_spec = {
            "name": name,
            "rows": rows,
            "freeze_panes": freeze_panes,
        }
        if formulas:
            sheet_spec["formulas"] = formulas
            for cell_ref, formula_text in formulas.items():
                self._formula_engine.register(name, cell_ref, formula_text)
        self._workbook_data["sheets"].append(sheet_spec)

    def save(self, path: Path, *, theme: str = "executive") -> Dict[str, Any]:
        """Save the current workbook to file."""
        if self._workbook_data is None:
            raise RuntimeError("No workbook loaded. Call create() or load() first.")
        return create_spreadsheet(path, self._workbook_data["content"], theme=theme)
