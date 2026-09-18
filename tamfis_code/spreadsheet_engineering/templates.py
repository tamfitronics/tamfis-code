"""Pre-built spreadsheet templates for common use cases.

Templates provide structured content dicts that can be passed directly
to create_spreadsheet() or SpreadsheetEngine.create().
"""
from __future__ import annotations

from typing import Any, Dict, List


# ============================================================================
# BUDGET TEMPLATE
# ============================================================================

BUDGET_TEMPLATE: Dict[str, Any] = {
    "title": "Budget Report",
    "theme": "executive",
    "sheets": [
        {
            "name": "Summary",
            "rows": [
                ["Category", "Budgeted", "Actual", "Variance", "Variance %"],
                ["Revenue", 100000, 95000, "=C2-B2", "=(D2/B2)"],
                ["Cost of Goods", 40000, 38000, "=C3-B3", "=(D3/B3)"],
                ["Operating Expenses", 25000, 27000, "=C4-B4", "=(D4/B4)"],
                ["Tax", 10000, 9500, "=C5-B5", "=(D5/B5)"],
                ["Net Income", 25000, 20500, "=C6-B6", "=(D6/B6)"],
            ],
            "header": True,
            "freeze_panes": "A2",
        },
        {
            "name": "Monthly Detail",
            "rows": [
                ["Month", "Revenue", "COGS", "OpEx", "Tax", "Net Income"],
                ["January", 8000, 3200, 2000, 800, 2000],
                ["February", 8500, 3400, 2100, 850, 2150],
                ["March", 9000, 3600, 2200, 900, 2300],
                ["April", 8200, 3300, 2050, 820, 2030],
                ["May", 8800, 3500, 2150, 880, 2270],
                ["June", 9500, 3800, 2300, 950, 2450],
                ["July", 9200, 3700, 2250, 920, 2330],
                ["August", 8600, 3450, 2100, 860, 2190],
                ["September", 9100, 3650, 2200, 910, 2340],
                ["October", 8900, 3550, 2150, 890, 2310],
                ["November", 9300, 3750, 2250, 930, 2370],
                ["December", 9900, 3950, 2400, 990, 2560],
            ],
            "header": True,
            "freeze_panes": "A2",
        },
        {
            "name": "Assumptions",
            "rows": [
                ["Assumption", "Value", "Notes"],
                ["Growth Rate", "5%", "Year-over-year revenue growth"],
                ["COGS %", "40%", "Cost of goods as % of revenue"],
                ["OpEx %", "25%", "Operating expenses as % of revenue"],
                ["Tax Rate", "10%", "Effective tax rate"],
                ["Inflation", "2%", "Annual inflation rate"],
            ],
            "header": True,
        },
    ],
}


# ============================================================================
# INVOICE TEMPLATE
# ============================================================================

INVOICE_TEMPLATE: Dict[str, Any] = {
    "title": "Invoice",
    "theme": "modern",
    "sheets": [
        {
            "name": "Invoice",
            "rows": [
                ["INVOICE", "", "", "", ""],
                ["Invoice #", "INV-001", "", "Date", "2026-01-15"],
                ["", "", "", "Due Date", "2026-02-14"],
                ["", "", "", "", ""],
                ["Bill To:", "", "", "Ship To:", ""],
                ["Acme Corp", "", "", "Acme Corp Warehouse", ""],
                ["123 Main St", "", "", "456 Oak Ave", ""],
                ["City, State 12345", "", "", "City, State 67890", ""],
                ["", "", "", "", ""],
                ["Description", "Qty", "Unit Price", "Total"],
                ["Widget A", 10, 25.00, "=C11*B11"],
                ["Widget B", 5, 50.00, "=C12*B12"],
                ["Service Fee", 1, 100.00, "=C13*B13"],
                ["", "", "", ""],
                ["Subtotal", "", "", "=SUM(D11:D13)"],
                ["Tax (10%)", "", "", "=D16*0.10"],
                ["Total", "", "", "=D16+D17"],
            ],
            "header": True,
        },
    ],
}


# ============================================================================
# KPI DASHBOARD TEMPLATE
# ============================================================================

KPI_DASHBOARD_TEMPLATE: Dict[str, Any] = {
    "title": "KPI Dashboard",
    "theme": "creative",
    "sheets": [
        {
            "name": "Dashboard",
            "rows": [
                ["Key Performance Indicators", "", "", "", ""],
                ["", "", "", "", ""],
                ["Metric", "Current", "Target", "Status", "Trend"],
                ["Revenue", 125000, 150000, "=IF(B10>=C10,\"On Track\",\"Behind\")", "↑"],
                ["Customers", 1200, 1500, "=IF(B11>=C11,\"On Track\",\"Behind\")", "↑"],
                ["NPS Score", 72, 75, "=IF(B12>=C12,\"On Track\",\"Behind\")", "→"],
                ["Churn Rate", 3.2, 2.5, "=IF(B13<=C13,\"On Track\",\"Behind\")", "↓"],
                ["CAC", 45, 50, "=IF(B14<=C14,\"On Track\",\"Behind\")", "↓"],
                ["LTV", 500, 450, "=IF(B15>=C15,\"On Track\",\"Behind\")", "↑"],
                ["LTV:CAC", 11.1, 9.0, "=IF(B16>=C16,\"On Track\",\"Behind\")", "↑"],
                ["MRR", 85000, 100000, "=IF(B17>=C17,\"On Track\",\"Behind\")", "↑"],
                ["ARR", 1020000, 1200000, "=IF(B18>=C18,\"On Track\",\"Behind\")", "↑"],
                ["Gross Margin", 68, 70, "=IF(B19>=C19,\"On Track\",\"Behind\")", "→"],
                ["EBITDA", 25000, 30000, "=IF(B20>=C20,\"On Track\",\"Behind\")", "↑"],
            ],
            "header": True,
            "freeze_panes": "A2",
        },
        {
            "name": "Monthly Trends",
            "rows": [
                ["Month", "Revenue", "Customers", "NPS", "Churn %", "MRR"],
                ["Jan", 80000, 1000, 68, 3.5, 68000],
                ["Feb", 85000, 1050, 69, 3.4, 72000],
                ["Mar", 90000, 1100, 70, 3.3, 76000],
                ["Apr", 95000, 1120, 70, 3.3, 80000],
                ["May", 100000, 1150, 71, 3.2, 84000],
                ["Jun", 105000, 1170, 71, 3.2, 87000],
                ["Jul", 110000, 1180, 71, 3.1, 90000],
                ["Aug", 115000, 1190, 72, 3.1, 94000],
                ["Sep", 118000, 1195, 72, 3.1, 96000],
                ["Oct", 120000, 1200, 72, 3.2, 98000],
                ["Nov", 122000, 1205, 72, 3.2, 100000],
                ["Dec", 125000, 1200, 72, 3.2, 102000],
            ],
            "header": True,
            "freeze_panes": "A2",
        },
    ],
}


# ============================================================================
# HELPER: Generate a simple data table template
# ============================================================================

def data_table_template(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
    *,
    theme: str = "executive",
    formulas: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Generate a simple data table spreadsheet.

    Args:
        title: Spreadsheet title
        headers: Column header names
        rows: Data rows (list of lists)
        theme: Color theme
        formulas: Optional dict mapping cell references to formula strings

    Returns:
        Content dict for create_spreadsheet()
    """
    content = {
        "title": title,
        "theme": theme,
        "sheets": [
            {
                "name": "Data",
                "rows": [headers] + rows,
                "header": True,
            }
        ],
    }
    if formulas:
        content["sheets"][0]["formulas"] = formulas
    return content


# ============================================================================
# HELPER: Generate a pivot-style summary template
# ============================================================================

def pivot_summary_template(
    title: str,
    categories: List[str],
    values: List[List[float]],
    *,
    theme: str = "modern",
    chart_type: str = "bar",
) -> Dict[str, Any]:
    """Generate a pivot-style summary with embedded chart.

    Args:
        title: Spreadsheet title
        categories: Category labels
        values: List of value series (each series is a list of values)
        theme: Color theme
        chart_type: Chart type for embedded chart

    Returns:
        Content dict for create_spreadsheet()
    """
    num_categories = len(categories)
    num_series = len(values)

    # Build data rows
    data_rows = [
        ["Category"] + [f"Series {i+1}" for i in range(num_series)]
    ]
    for i, cat in enumerate(categories):
        row = [cat]
        for s in range(num_series):
            if s < len(values) and i < len(values[s]):
                row.append(values[s][i])
            else:
                row.append(0)
        data_rows.append(row)

    return {
        "title": title,
        "theme": theme,
        "sheets": [
            {
                "name": "Summary",
                "rows": data_rows,
                "header": True,
            }
        ],
    }
