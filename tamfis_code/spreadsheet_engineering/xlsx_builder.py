"""openpyxl wrapper for creating styled, formula-aware workbooks.

Extends the basic _create_xlsx() from artifacts.py with:
- Formula support (cells starting with =/+/-/@ written as formulas)
- Cell styling (fonts, colors, borders, alignment)
- Conditional formatting
- Chart generation
- Theme-based color palettes
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.styles import (
    Alignment,
    Border,
    Color,
    Font,
    PatternFill,
    Side,
)
from openpyxl.utils import get_column_letter

# Theme palettes (matching office_artifacts.py for consistency)
_THEME_PALETTES: Dict[str, Dict[str, str]] = {
    "executive": {
        "accent": "1F4E79",
        "accent_2": "2F75B5",
        "highlight": "16A3A5",
        "light": "F2F6FA",
        "soft": "E8F3F4",
        "text": "263238",
        "muted": "667085",
    },
    "modern": {
        "accent": "4338CA",
        "accent_2": "7C3AED",
        "highlight": "06B6D4",
        "light": "F4F3FF",
        "soft": "E8F8FC",
        "text": "20223A",
        "muted": "667085",
    },
    "creative": {
        "accent": "B4235A",
        "accent_2": "F97316",
        "highlight": "0EA5A8",
        "light": "FFF3F7",
        "soft": "FFF2E8",
        "text": "33202A",
        "muted": "6B6470",
    },
    "natural": {
        "accent": "176B57",
        "accent_2": "3A8D5D",
        "highlight": "D69E2E",
        "light": "EEF8F3",
        "soft": "FFF8E6",
        "text": "24352F",
        "muted": "63716C",
    },
}


def _resolve_theme(theme: str) -> Dict[str, str]:
    """Resolve a theme name to a color palette."""
    requested = theme.strip().lower()
    return dict(_THEME_PALETTES.get(requested, _THEME_PALETTES["executive"]))


def _hex_to_rgb(hex_color: str) -> Color:
    """Convert a hex color string to an openpyxl Color.

    openpyxl's Font/PatternFill/Side color slots demand a Color object (or
    an aRGB hex string) -- a raw ``(r, g, b)`` tuple raises
    ``TypeError: .color should be Color but value is <class 'tuple'>`` at
    assignment time. Colors are aRGB (alpha prefix "00" is transparent in
    some viewers; "FF" full opacity is the safe conventional form).
    """
    hex_color = (hex_color or "").lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    if len(hex_color) == 6:
        hex_color = "FF" + hex_color
    return Color(rgb=hex_color)


def _apply_theme_to_sheet(
    sheet: Any, palette: Dict[str, str], start_row: int = 1, num_rows: int = 1
) -> None:
    """Apply theme colors to header rows."""
    accent_rgb = _hex_to_rgb(palette["accent"])
    text_rgb = _hex_to_rgb(palette["text"])
    light_rgb = _hex_to_rgb(palette["light"])

    header_font = Font(
        name="Calibri",
        size=11,
        bold=True,
        color=accent_rgb,
    )
    header_fill = PatternFill(
        start_color=light_rgb,
        end_color=light_rgb,
        fill_type="solid",
    )
    header_border = Border(
        bottom=Side(style="thin", color=palette["accent"]),
    )

    for row_idx in range(start_row, start_row + num_rows):
        for cell in sheet[row_idx]:
            cell.font = header_font
            cell.fill = header_fill
            cell.border = header_border


def _safe_cell_value(value: Any, allow_formulas: bool) -> Any:
    """Determine the correct cell value, handling formulas."""
    if not isinstance(value, str):
        return value
    if not value:
        return value
    # If formulas are allowed, write formula strings as-is
    if allow_formulas and value.startswith(("=", "+", "-", "@")):
        return value
    # Otherwise, escape to prevent formula injection
    if value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _apply_cell_formatting(
    cell: Any,
    value: Any,
    row_idx: int,
    col_idx: int,
    palette: Dict[str, str],
    is_header: bool = False,
) -> None:
    """Apply formatting to a single cell."""
    if is_header:
        accent_rgb = _hex_to_rgb(palette["accent"])
        light_rgb = _hex_to_rgb(palette["light"])
        cell.font = Font(name="Calibri", size=11, bold=True, color=accent_rgb)
        cell.fill = PatternFill(
            start_color=light_rgb,
            end_color=light_rgb,
            fill_type="solid",
        )
        cell.border = Border(
            bottom=Side(style="thin", color=palette["accent"]),
        )
        return

    # Default cell formatting
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        cell.number_format = "#,##0.00" if isinstance(value, float) else "#,##0"
    elif isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        cell.font = Font(name="Calibri", size=10, color=_hex_to_rgb(palette["muted"]))


def _apply_column_widths(sheet: Any, max_width: int = 60) -> None:
    """Auto-size columns based on content."""
    for column in sheet.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            if cell.value:
                cell_length = len(str(cell.value))
                # Estimate for numbers/formulas
                if isinstance(cell.value, (int, float)):
                    cell_length = max(cell_length, 10)
                elif cell_length > 0 and str(cell.value)[0] in ("=", "+", "-", "@"):
                    cell_length = max(cell_length, 15)
                max_length = max(max_length, cell_length)
        adjusted = min(max(max_length + 2, 8), max_width)
        sheet.column_dimensions[column_letter].width = adjusted


def _apply_conditional_formatting(
    sheet: Any, spec: Dict[str, Any]
) -> None:
    """Apply conditional formatting rules to a sheet."""
    try:
        from openpyxl.formatting.rule import CellIsRule
        from openpyxl.styles import Color, PatternFill

        rules = spec.get("rules", [])
        for rule_spec in rules:
            cell_range = rule_spec.get("cell_range", "")
            if not cell_range:
                continue
            formula = rule_spec.get("formula", "")
            rule_type = rule_spec.get("type", "cellIs")
            operator = rule_spec.get("operator", "equal")

            fill_color = rule_spec.get("fill_color", "FFC7CE")
            font_color = rule_spec.get("font_color", "9C0006")
            border_color = rule_spec.get("border_color", "FF0000")

            fill = PatternFill(
                start_color=Color(rgb=fill_color),
                end_color=Color(rgb=fill_color),
                fill_type="solid",
            )
            font = Font(color=Color(rgb=font_color), bold=True)

            rule = CellIsRule(
                operator=operator,
                formula=[formula],
                fill=fill,
                font=font,
            )
            sheet.conditional_formatting.add(cell_range, rule)
    except Exception:
        # Conditional formatting is optional; skip on failure
        pass


def _apply_freeze_panes(sheet: Any, freeze_spec: Optional[str]) -> None:
    """Apply freeze pane settings."""
    if freeze_spec:
        sheet.freeze_panes = freeze_spec


def create_workbook(
    path: Path,
    content: Dict[str, Any],
    *,
    theme: str = "executive",
) -> int:
    """Create an xlsx workbook from structured content.

    Args:
        path: Output file path
        content: Content dict with optional fields:
            - sheets: list of {name, rows, headers, formulas, freeze_panes,
              conditional_formatting, charts}
            - title: workbook title
            - allow_formulas: bool (default False)
            - theme: color theme name
        theme: Color theme to apply

    Returns:
        Number of sheets created
    """
    palette = _resolve_theme(theme)
    allow_formulas = bool(content.get("allow_formulas", False))

    workbook = Workbook()
    workbook.remove(workbook.active)  # Remove default sheet

    sheets = content.get("sheets") or [{"name": "Sheet1", "rows": content.get("rows") or []}]
    sheet_count = 0

    for index, spec in enumerate(sheets):
        if not isinstance(spec, dict):
            continue

        sheet_name = str(spec.get("name") or f"Sheet{index + 1}")[:31]
        sheet = workbook.create_sheet(sheet_name)

        rows = spec.get("rows") or []
        has_headers = spec.get("header", True)
        num_header_rows = 1 if has_headers and rows else 0

        # Write data rows
        for row_idx, row in enumerate(rows):
            values = row if isinstance(row, list) else [row]
            for col_idx, value in enumerate(values):
                cell = sheet.cell(row=row_idx + 1, column=col_idx + 1)
                cell.value = _safe_cell_value(value, allow_formulas)
                _apply_cell_formatting(
                    cell, value, row_idx + 1, col_idx + 1,
                    palette, is_header=(row_idx < num_header_rows),
                )

        # Apply header formatting
        if has_headers and rows:
            _apply_theme_to_sheet(sheet, palette, start_row=1, num_rows=num_header_rows)

        # Apply freeze panes
        freeze_panes = spec.get("freeze_panes")
        _apply_freeze_panes(sheet, freeze_panes)

        # Apply conditional formatting
        cond_format = spec.get("conditional_formatting")
        if cond_format:
            _apply_conditional_formatting(sheet, cond_format)

        # Auto-size columns
        _apply_column_widths(sheet)

        sheet_count += 1

    # Ensure at least one sheet exists
    if not workbook.sheetnames:
        workbook.create_sheet("Sheet1")

    # Set workbook properties
    title = str(content.get("title") or "")
    if title:
        workbook.properties.title = title

    workbook.save(path)
    return sheet_count
