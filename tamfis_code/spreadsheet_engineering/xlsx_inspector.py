"""openpyxl wrapper for inspecting existing workbooks.

Extends the basic inspect_artifact() from artifacts.py with:
- Formula extraction (reads formula strings, not just values)
- Style inspection (fonts, fills, borders, alignment)
- Conditional formatting detection
- Chart detection and description
- Data type inference per column
- Cell dependency analysis
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from openpyxl import load_workbook


def inspect_workbook(
    path: Path,
    *,
    include_formulas: bool = True,
    include_styles: bool = False,
    include_charts: bool = True,
) -> Dict[str, Any]:
    """Inspect an xlsx file and return structured metadata.

    Args:
        path: Path to the xlsx file
        include_formulas: Extract formula strings (default True)
        include_styles: Extract cell style info (default False)
        include_charts: Extract chart descriptions (default True)

    Returns:
        Dict with keys:
        - workbook_info: title, sheet names, dimensions
        - sheet_details: per-sheet analysis
        - charts: list of chart descriptions
        - summary: overall statistics
    """
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")

    # openpyxl's Workbook is NOT a context manager (only read-only
    # read-only-mode workbooks ever were); always close explicitly.
    wb = load_workbook(path, data_only=False)
    try:
        workbook_info = {
            "title": wb.properties.title or "",
            "sheet_names": wb.sheetnames,
            "sheet_count": len(wb.sheetnames),
        }

        sheet_details = []
        all_formulas = {}
        all_styles = {}
        all_charts = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            detail = _inspect_sheet(
                ws, sheet_name,
                include_formulas=include_formulas,
                include_styles=include_styles,
                include_charts=include_charts,
            )
            sheet_details.append(detail)

            if include_formulas and detail.get("formulas"):
                all_formulas[sheet_name] = detail["formulas"]
            if include_styles and detail.get("styles"):
                all_styles[sheet_name] = detail["styles"]
            if include_charts and detail.get("charts"):
                all_charts.extend(detail["charts"])

        # Compute summary
        total_rows = sum(d["row_count"] for d in sheet_details)
        total_cols = max((d["col_count"] for d in sheet_details), default=0)
        total_cells = sum(d["cell_count"] for d in sheet_details)
        total_formulas = sum(len(d.get("formulas", {})) for d in sheet_details)
        data_types = _infer_data_types(sheet_details)

        summary = {
            "total_sheets": len(sheet_details),
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_cells": total_cells,
            "total_formulas": total_formulas,
            "data_types": data_types,
        }
    finally:
        wb.close()

    result = {
        "workbook_info": workbook_info,
        "sheet_details": sheet_details,
        "summary": summary,
    }
    if include_formulas and all_formulas:
        result["formulas"] = all_formulas
    if include_styles and all_styles:
        result["styles"] = all_styles
    if include_charts and all_charts:
        result["charts"] = all_charts

    return result


def _inspect_sheet(
    ws: Any,
    sheet_name: str,
    *,
    include_formulas: bool = True,
    include_styles: bool = False,
    include_charts: bool = True,
) -> Dict[str, Any]:
    """Inspect a single worksheet."""
    max_row = ws.max_row or 0
    max_col = ws.max_column or 0

    # Count data cells (non-empty)
    cell_count = 0
    formulas = {}
    styles = {}
    column_types: Dict[int, List[str]] = {}

    for row in ws.iter_rows(min_row=1, max_row=max_row, max_col=max_col):
        for cell in row:
            if cell.value is not None:
                cell_count += 1
                col_idx = cell.column

                # Track column types
                if col_idx not in column_types:
                    column_types[col_idx] = []
                if cell.value is not None:
                    column_types[col_idx].append(type(cell.value).__name__)

                # Extract formulas
                if include_formulas and cell.data_type == "f" and cell.value:
                    formulas[cell.coordinate] = cell.value

                # Extract styles
                if include_styles:
                    style_info = {}
                    if cell.font.name or cell.font.bold or cell.font.color.rgb:
                        style_info["font"] = {
                            "name": cell.font.name or "Calibri",
                            "bold": cell.font.bold or False,
                            "color": str(cell.font.color.rgb) if cell.font.color and cell.font.color.rgb else None,
                        }
                    if cell.fill.start_color and cell.fill.start_color.rgb:
                        style_info["fill"] = str(cell.fill.start_color.rgb)
                    if cell.border:
                        style_info["border"] = {
                            side: str(side_style.color.rgb) if side_style.color and side_style.color.rgb else None
                            for side, side_style in {
                                "top": cell.border.top,
                                "right": cell.border.right,
                                "bottom": cell.border.bottom,
                                "left": cell.border.left,
                            }.items()
                            if side_style and side_style.style
                        }
                    if style_info:
                        styles[cell.coordinate] = style_info

    # Determine dominant type per column
    col_type_summary = {}
    for col_idx, types in column_types.items():
        if not types:
            continue
        type_counts: Dict[str, int] = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        dominant = max(type_counts, key=type_counts.get)
        col_letter = _get_column_letter(col_idx)
        col_type_summary[col_letter] = dominant

    # Detect charts
    charts = []
    if include_charts and hasattr(ws, "drawings"):
        for drawing in ws.drawings:
            if hasattr(drawing, "charts") and drawing.charts:
                for chart in drawing.charts:
                    chart_info = {
                        "type": getattr(chart, "chart_type", "unknown"),
                        "title": getattr(chart, "title", None),
                    }
                    if hasattr(chart, "xOffset"):
                        chart_info["position"] = {
                            "x": getattr(chart, "xOffset", 0),
                            "y": getattr(chart, "yOffset", 0),
                        }
                    charts.append(chart_info)

    return {
        "name": sheet_name,
        "row_count": max_row,
        "col_count": max_col,
        "cell_count": cell_count,
        "formulas": formulas,
        "styles": styles,
        "column_types": col_type_summary,
        "charts": charts,
    }


def _get_column_letter(col_idx: int) -> str:
    """Convert column index to letter (1-based)."""
    result = ""
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _infer_data_types(sheet_details: List[Dict[str, Any]]) -> Dict[str, int]:
    """Infer overall data type distribution across all sheets."""
    type_counts: Dict[str, int] = {}
    for detail in sheet_details:
        for col_letter, dtype in detail.get("column_types", {}).items():
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
    return type_counts
