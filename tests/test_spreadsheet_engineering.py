"""Tests for the spreadsheet_engineering module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tamfis_code.spreadsheet_engineering import (
    BUDGET_TEMPLATE,
    INVOICE_TEMPLATE,
    KPI_DASHBOARD_TEMPLATE,
    ChartGenerator,
    FormulaEngine,
    analyze_spreadsheet,
    create_spreadsheet,
    create_workbook,
    inspect_workbook,
    parse_formula,
    validate_formula,
)


class TestCreateSpreadsheet:
    """Test create_spreadsheet() function."""

    def test_simple_spreadsheet(self, tmp_path: Path) -> None:
        """Create a simple spreadsheet with rows and headers."""
        content = {
            "title": "Test",
            "sheets": [
                {
                    "name": "Data",
                    "rows": [
                        ["Name", "Value"],
                        ["A", 1],
                        ["B", 2],
                        ["C", 3],
                    ],
                    "header": True,
                }
            ],
        }
        result = create_spreadsheet(tmp_path / "test.xlsx", content)
        assert result["success"] is True
        assert result["artifact_type"] == "xlsx"
        assert result["sheets_created"] == 1
        assert (tmp_path / "test.xlsx").exists()

    def test_formulas_enabled(self, tmp_path: Path) -> None:
        """Create a spreadsheet with formulas when allow_formulas=True."""
        content = {
            "title": "Formulas",
            "allow_formulas": True,
            "sheets": [
                {
                    "name": "Sheet1",
                    "rows": [
                        ["A", "B", "Sum"],
                        [1, 2, "=A2+B2"],
                        [3, 4, "=A3+B3"],
                    ],
                    "header": True,
                }
            ],
        }
        result = create_spreadsheet(tmp_path / "formulas.xlsx", content)
        assert result["success"] is True

    def test_formulas_disabled(self, tmp_path: Path) -> None:
        """Formulas should be escaped when allow_formulas=False."""
        content = {
            "title": "No Formulas",
            "allow_formulas": False,
            "sheets": [
                {
                    "name": "Sheet1",
                    "rows": [
                        ["Formula"],
                        ["=1+1"],
                    ],
                    "header": True,
                }
            ],
        }
        result = create_spreadsheet(tmp_path / "no_formulas.xlsx", content)
        assert result["success"] is True
        # Verify the file was created
        assert (tmp_path / "no_formulas.xlsx").exists()

    def test_multiple_sheets(self, tmp_path: Path) -> None:
        """Create a spreadsheet with multiple sheets."""
        content = {
            "title": "Multi-sheet",
            "sheets": [
                {"name": "Sheet1", "rows": [["A", "B"]]},
                {"name": "Sheet2", "rows": [["C", "D"]]},
                {"name": "Sheet3", "rows": [["E", "F"]]},
            ],
        }
        result = create_spreadsheet(tmp_path / "multi.xlsx", content)
        assert result["success"] is True
        assert result["sheets_created"] == 3

    def test_freeze_panes(self, tmp_path: Path) -> None:
        """Create a spreadsheet with freeze panes."""
        content = {
            "title": "Frozen",
            "sheets": [
                {
                    "name": "Data",
                    "rows": [[f"Col{i}" for i in range(10)] for _ in range(5)],
                    "freeze_panes": "A2",
                }
            ],
        }
        result = create_spreadsheet(tmp_path / "frozen.xlsx", content)
        assert result["success"] is True


class TestCreateWorkbook:
    """Test create_workbook() function directly."""

    def test_basic_workbook(self, tmp_path: Path) -> None:
        """Create a basic workbook."""
        content = {
            "sheets": [
                {
                    "name": "Test",
                    "rows": [["Header"], ["Row1"], ["Row2"]],
                }
            ],
        }
        path = tmp_path / "basic.xlsx"
        count = create_workbook(path, content)
        assert count == 1
        assert path.exists()

    def test_theme_applied(self, tmp_path: Path) -> None:
        """Verify theme is applied to workbook."""
        content = {
            "title": "Themed",
            "theme": "modern",
            "sheets": [
                {
                    "name": "Data",
                    "rows": [["Name", "Value"], ["A", 100]],
                    "header": True,
                }
            ],
        }
        path = tmp_path / "themed.xlsx"
        count = create_workbook(path, content)
        assert count == 1
        assert path.exists()


class TestInspectWorkbook:
    """Test inspect_workbook() function."""

    def test_inspect_simple(self, tmp_path: Path) -> None:
        """Inspect a simple workbook."""
        content = {
            "sheets": [
                {
                    "name": "Data",
                    "rows": [["A", "B"], [1, 2], [3, 4]],
                }
            ],
        }
        path = tmp_path / "inspect.xlsx"
        create_workbook(path, content)

        result = inspect_workbook(path, include_formulas=True, include_styles=False, include_charts=True)
        assert "workbook_info" in result
        assert "sheet_details" in result
        assert "summary" in result
        assert result["workbook_info"]["sheet_count"] == 1
        assert result["summary"]["total_rows"] == 3
        assert result["summary"]["total_columns"] == 2

    def test_inspect_with_formulas(self, tmp_path: Path) -> None:
        """Inspect a workbook with formulas."""
        content = {
            "allow_formulas": True,
            "sheets": [
                {
                    "name": "Calc",
                    "rows": [
                        ["A", "B", "Sum"],
                        [1, 2, "=A2+B2"],
                    ],
                }
            ],
        }
        path = tmp_path / "formula_inspect.xlsx"
        create_workbook(path, content)

        result = inspect_workbook(path, include_formulas=True)
        assert "formulas" in result
        assert "Calc" in result["formulas"]
        assert "C2" in result["formulas"]["Calc"]


class TestFormulaEngine:
    """Test FormulaEngine class."""

    def test_parse_simple_arithmetic(self) -> None:
        """Parse a simple arithmetic formula."""
        engine = FormulaEngine()
        result = engine.parse("=A1+B1")
        assert result["valid"] is True
        assert result["type"] == "arithmetic"
        assert "+" in result["operators"]

    def test_parse_function(self) -> None:
        """Parse a function formula."""
        engine = FormulaEngine()
        result = engine.parse("=SUM(A1:A10)")
        assert result["valid"] is True
        assert result["type"] == "function"
        assert "SUM" in result["functions"]

    def test_parse_reference(self) -> None:
        """Parse a cell reference."""
        engine = FormulaEngine()
        result = engine.parse("=A1")
        assert result["valid"] is True
        assert "A1" in result["references"]

    def test_validate_invalid(self) -> None:
        """Validate an invalid formula."""
        engine = FormulaEngine()
        is_valid, errors = engine.validate("=SUM(A1:A10")
        assert is_valid is False
        assert len(errors) > 0

    def test_register_and_dependencies(self) -> None:
        """Register a formula and get dependencies."""
        engine = FormulaEngine()
        engine.register("Sheet1", "C1", "=A1+B1")
        deps = engine.get_dependencies("C1", "Sheet1")
        assert "A1" in deps or "B1" in deps

    def test_module_level_parse(self) -> None:
        """Test module-level parse_formula function."""
        result = parse_formula("=AVERAGE(B1:B10)")
        assert result["valid"] is True
        assert "AVERAGE" in result["functions"]

    def test_module_level_validate(self) -> None:
        """Test module-level validate_formula function."""
        is_valid, errors = validate_formula("=1+1")
        assert is_valid is True
        assert errors == []


class TestChartGenerator:
    """Test ChartGenerator class."""

    def test_chart_generator_creation(self) -> None:
        """Create a ChartGenerator instance."""
        from openpyxl import Workbook
        wb = Workbook()
        generator = ChartGenerator(wb)
        assert generator._workbook is wb


class TestTemplates:
    """Test pre-built templates."""

    def test_budget_template(self) -> None:
        """Budget template has correct structure."""
        assert BUDGET_TEMPLATE["title"] == "Budget Report"
        assert BUDGET_TEMPLATE["theme"] == "executive"
        assert len(BUDGET_TEMPLATE["sheets"]) == 3
        sheet_names = [s["name"] for s in BUDGET_TEMPLATE["sheets"]]
        assert "Summary" in sheet_names
        assert "Monthly Detail" in sheet_names
        assert "Assumptions" in sheet_names

    def test_invoice_template(self) -> None:
        """Invoice template has correct structure."""
        assert INVOICE_TEMPLATE["title"] == "Invoice"
        assert INVOICE_TEMPLATE["theme"] == "modern"
        assert len(INVOICE_TEMPLATE["sheets"]) == 1

    def test_kpi_template(self) -> None:
        """KPI dashboard template has correct structure."""
        assert KPI_DASHBOARD_TEMPLATE["title"] == "KPI Dashboard"
        assert KPI_DASHBOARD_TEMPLATE["theme"] == "creative"
        assert len(KPI_DASHBOARD_TEMPLATE["sheets"]) == 2


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_create_and_inspect_roundtrip(self, tmp_path: Path) -> None:
        """Create a spreadsheet and inspect it."""
        content = {
            "title": "Roundtrip Test",
            "theme": "modern",
            "allow_formulas": True,
            "sheets": [
                {
                    "name": "Data",
                    "rows": [
                        ["Product", "Qty", "Price", "Total"],
                        ["Widget A", 10, 25.00, "=C2*B2"],
                        ["Widget B", 5, 50.00, "=C3*B3"],
                    ],
                    "header": True,
                }
            ],
        }
        path = tmp_path / "roundtrip.xlsx"
        create_result = create_spreadsheet(path, content)
        assert create_result["success"] is True

        inspect_result = inspect_workbook(path, include_formulas=True)
        assert inspect_result["summary"]["total_rows"] == 3
        assert inspect_result["summary"]["total_columns"] == 4
        assert "formulas" in inspect_result
        assert "Data" in inspect_result["formulas"]

    def test_budget_template_creation(self, tmp_path: Path) -> None:
        """Create a spreadsheet from the budget template."""
        path = tmp_path / "budget.xlsx"
        result = create_spreadsheet(path, BUDGET_TEMPLATE)
        assert result["success"] is True
        assert result["sheets_created"] == 3
        assert path.exists()
        assert path.stat().st_size > 0

    def test_analyze_spreadsheet(self, tmp_path: Path) -> None:
        """Test analyze_spreadsheet convenience function."""
        content = {
            "sheets": [
                {
                    "name": "Analysis",
                    "rows": [["X", "Y"], [1, 2], [3, 4]],
                }
            ],
        }
        path = tmp_path / "analyze.xlsx"
        create_workbook(path, content)

        result = analyze_spreadsheet(path, include_formulas=True)
        assert "workbook_info" in result
        assert "sheet_details" in result
        assert "summary" in result
