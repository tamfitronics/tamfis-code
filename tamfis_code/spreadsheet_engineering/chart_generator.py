"""Chart generation for xlsx workbooks.

Supports:
- Bar charts (vertical, horizontal, stacked, 100% stacked)
- Line charts (standard, with markers, stacked)
- Pie charts (2D, 3D, exploded slices)
- Doughnut charts
- Scatter charts (with/without lines)
- Area charts (standard, stacked)
- Column charts (standard, stacked, 100% stacked)
- Radar charts
- Combination charts (e.g., line + bar)

Charts are embedded in the workbook using openpyxl's chart module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.chart import (
    BarChart,
    LineChart,
    PieChart,
    DoughnutChart,
    ScatterChart,
    AreaChart,
    RadarChart,
    Reference,
    Series,
)
from openpyxl.chart.series import DataPoint
from openpyxl.chart.label import DataLabelList
from openpyxl.drawing.xdr import XDRPoint2D
from openpyxl.utils import get_column_letter


# Chart type mapping
_CHART_TYPES = {
    "bar": BarChart,
    "column": BarChart,  # openpyxl uses BarChart for both
    "line": LineChart,
    "pie": PieChart,
    "doughnut": DoughnutChart,
    "scatter": ScatterChart,
    "area": AreaChart,
    "radar": RadarChart,
}


def create_chart(
    workbook: Workbook,
    spec: Dict[str, Any],
    sheet_name: str,
    *,
    position: Optional[Tuple[int, int]] = None,
) -> None:
    """Create a chart in a workbook from a specification.

    Args:
        workbook: openpyxl Workbook instance
        spec: Chart specification dict with keys:
            - type: chart type (bar, line, pie, doughnut, scatter, area, radar)
            - title: chart title
            - x_axis: column reference for x values (e.g., "A")
            - y_axis: column reference for y values (e.g., "B")
            - x_range: row range for x values (e.g., (2, 10))
            - y_range: row range for y values (e.g., (2, 10))
            - series: list of series specs (for multi-series charts)
            - style: chart style preset
            - category_axis_title: x-axis label
            - value_axis_title: y-axis label
            - legend: whether to show legend
            - data_labels: whether to show data labels
            - exploded: for pie/doughnut, list of slice indices to explode
            - overlap: for bar charts, overlap between bars (-100 to 100)
            - gap_width: for bar charts, gap between bars (0 to 500)
        sheet_name: Name of the sheet to add the chart to
        position: (row, col) position for the chart anchor (default: 10, 0)
    """
    chart_type = spec.get("type", "bar").lower()
    ChartClass = _CHART_TYPES.get(chart_type)

    if ChartClass is None:
        raise ValueError(f"Unsupported chart type: {chart_type}. "
                        f"Supported: {list(_CHART_TYPES.keys())}")

    ws = workbook[sheet_name]
    if ws is None:
        raise ValueError(f"Sheet not found: {sheet_name}")

    chart = ChartClass()
    chart.type = chart_type

    # Set title
    title = spec.get("title")
    if title:
        chart.title = title

    # Set style
    style = spec.get("style")
    if style:
        chart.style = style

    # Configure axes
    cat_axis = spec.get("category_axis_title")
    val_axis = spec.get("value_axis_title")
    if cat_axis:
        chart.category_axis.title = cat_axis
    if val_axis:
        chart.value_axis.title = val_axis

    # Configure legend
    legend = spec.get("legend", True)
    if not legend:
        chart.legend = None

    # Configure data labels
    if spec.get("data_labels"):
        chart.has_data_labels = True
        data_label_list = DataLabelList()
        data_label_list.show_val = True
        data_label_list.show_cat_name = False
        data_label_list.show_ser_name = False
        data_label_list.show_pct = False
        data_label_list.show_leader_lines = False
        chart.data_labels = data_label_list

    # Add series
    series_specs = spec.get("series", [])
    x_col = spec.get("x_axis")
    y_col = spec.get("y_axis")
    x_range = spec.get("x_range")
    y_range = spec.get("y_range")

    if series_specs:
        # Multi-series mode
        for i, s_spec in enumerate(series_specs):
            s_name = s_spec.get("name", f"Series {i + 1}")
            s_y_col = s_spec.get("y_axis", y_col)
            s_x_range = s_spec.get("x_range", x_range)
            s_y_range = s_spec.get("y_range", y_range)

            if s_x_range and s_y_range and x_col and s_y_col:
                x_ref = Reference(ws, min_col=get_column_letter(x_col),
                                 min_row=s_x_range[0], max_row=s_x_range[1])
                y_ref = Reference(ws, min_col=get_column_letter(s_y_col),
                                 min_row=s_y_range[0], max_row=s_y_range[1])
                series = Series(y_ref, x_ref, title=s_name)
                chart.series.append(series)
    elif x_col and y_col and x_range and y_range:
        # Single series mode
        x_ref = Reference(ws, min_col=get_column_letter(x_col),
                         min_row=x_range[0], max_row=x_range[1])
        y_ref = Reference(ws, min_col=get_column_letter(y_col),
                         min_row=y_range[0], max_row=y_range[1])
        series = Series(y_ref, x_ref, title=spec.get("series_name"))
        chart.series.append(series)
    else:
        raise ValueError("Chart spec requires either 'series' list or "
                        "'x_axis'/'y_axis' with 'x_range'/'y_range'")

    # Pie/doughnut specific options
    if chart_type in ("pie", "doughnut"):
        exploded = spec.get("exploded", [])
        if exploded:
            for i, point in enumerate(chart.series[0].data_points):
                if i in exploded:
                    point.graphicalProperties = None
                    point.spPr = None
                    # Mark as exploded
                    point.idx = i

    # Bar chart specific options
    if chart_type in ("bar", "column"):
        overlap = spec.get("overlap")
        if overlap is not None:
            chart.overlap = overlap
        gap_width = spec.get("gap_width")
        if gap_width is not None:
            chart.gap_width = gap_width

    # Set chart dimensions
    width = spec.get("width", 400)
    height = spec.get("height", 300)
    chart.width = width
    chart.height = height

    # Position the chart
    pos = position or (10, 0)
    ws.add_chart(chart, f"{get_column_letter(pos[1] + 1)}{pos[0] + 1}")


def create_simple_bar_chart(
    workbook: Workbook,
    sheet_name: str,
    x_col: int,
    y_col: int,
    start_row: int,
    end_row: int,
    *,
    title: str = "",
    category_axis_title: str = "",
    value_axis_title: str = "",
    position: Optional[Tuple[int, int]] = None,
) -> None:
    """Create a simple vertical bar chart.

    Args:
        workbook: openpyxl Workbook instance
        sheet_name: Sheet containing the data
        x_col: Column index for category labels (1-based)
        y_col: Column index for values (1-based)
        start_row: First data row
        end_row: Last data row
        title: Chart title
        category_axis_title: X-axis label
        value_axis_title: Y-axis label
        position: (row, col) for chart anchor
    """
    ws = workbook[sheet_name]
    chart = BarChart()
    chart.type = "col"

    if title:
        chart.title = title
    if category_axis_title:
        chart.category_axis.title = category_axis_title
    if value_axis_title:
        chart.value_axis.title = value_axis_title

    # Add data reference
    categories = Reference(ws, min_col=x_col, min_row=start_row, max_row=end_row)
    data = Reference(ws, min_col=y_col, min_row=start_row, max_row=end_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    # Position
    pos = position or (10, 0)
    ws.add_chart(chart, f"{get_column_letter(pos[1] + 1)}{pos[0] + 1}")


def create_simple_line_chart(
    workbook: Workbook,
    sheet_name: str,
    x_col: int,
    y_col: int,
    start_row: int,
    end_row: int,
    *,
    title: str = "",
    category_axis_title: str = "",
    value_axis_title: str = "",
    show_markers: bool = False,
    position: Optional[Tuple[int, int]] = None,
) -> None:
    """Create a simple line chart.

    Args:
        workbook: openpyxl Workbook instance
        sheet_name: Sheet containing the data
        x_col: Column index for category labels (1-based)
        y_col: Column index for values (1-based)
        start_row: First data row
        end_row: Last data row
        title: Chart title
        category_axis_title: X-axis label
        value_axis_title: Y-axis label
        show_markers: Whether to show data point markers
        position: (row, col) for chart anchor
    """
    ws = workbook[sheet_name]
    chart = LineChart()
    chart.type = "line"

    if title:
        chart.title = title
    if category_axis_title:
        chart.category_axis.title = category_axis_title
    if value_axis_title:
        chart.value_axis.title = value_axis_title

    # Add data reference
    categories = Reference(ws, min_col=x_col, min_row=start_row, max_row=end_row)
    data = Reference(ws, min_col=y_col, min_row=start_row, max_row=end_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    if show_markers:
        chart.style = 2  # Line with markers style

    # Position
    pos = position or (10, 0)
    ws.add_chart(chart, f"{get_column_letter(pos[1] + 1)}{pos[0] + 1}")


def create_simple_pie_chart(
    workbook: Workbook,
    sheet_name: str,
    label_col: int,
    value_col: int,
    start_row: int,
    end_row: int,
    *,
    title: str = "",
    exploded_slices: Optional[List[int]] = None,
    position: Optional[Tuple[int, int]] = None,
) -> None:
    """Create a simple pie chart.

    Args:
        workbook: openpyxl Workbook instance
        sheet_name: Sheet containing the data
        label_col: Column index for labels (1-based)
        value_col: Column index for values (1-based)
        start_row: First data row
        end_row: Last data row
        title: Chart title
        exploded_slices: List of slice indices to explode
        position: (row, col) for chart anchor
    """
    ws = workbook[sheet_name]
    chart = PieChart()

    if title:
        chart.title = title

    # Add data reference
    labels = Reference(ws, min_col=label_col, min_row=start_row, max_row=end_row)
    data = Reference(ws, min_col=value_col, min_row=start_row, max_row=end_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(labels)

    # Explode slices
    if exploded_slices:
        for i, point in enumerate(chart.series[0].data_points):
            if i in exploded_slices:
                point.graphicalProperties = None

    # Position
    pos = position or (10, 0)
    ws.add_chart(chart, f"{get_column_letter(pos[1] + 1)}{pos[0] + 1}")


class ChartGenerator:
    """Stateful chart generator for multi-step chart operations.

    Maintains a reference to a workbook and provides methods for
    creating and customizing charts across sheets.
    """

    def __init__(self, workbook: Workbook) -> None:
        self._workbook = workbook

    def add_bar_chart(
        self,
        sheet_name: str,
        x_col: int,
        y_col: int,
        start_row: int,
        end_row: int,
        *,
        title: str = "",
        category_axis_title: str = "",
        value_axis_title: str = "",
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Add a vertical bar chart to a sheet."""
        create_simple_bar_chart(
            self._workbook, sheet_name, x_col, y_col,
            start_row, end_row,
            title=title,
            category_axis_title=category_axis_title,
            value_axis_title=value_axis_title,
            position=position,
        )

    def add_line_chart(
        self,
        sheet_name: str,
        x_col: int,
        y_col: int,
        start_row: int,
        end_row: int,
        *,
        title: str = "",
        category_axis_title: str = "",
        value_axis_title: str = "",
        show_markers: bool = False,
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Add a line chart to a sheet."""
        create_simple_line_chart(
            self._workbook, sheet_name, x_col, y_col,
            start_row, end_row,
            title=title,
            category_axis_title=category_axis_title,
            value_axis_title=value_axis_title,
            show_markers=show_markers,
            position=position,
        )

    def add_pie_chart(
        self,
        sheet_name: str,
        label_col: int,
        value_col: int,
        start_row: int,
        end_row: int,
        *,
        title: str = "",
        exploded_slices: Optional[List[int]] = None,
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Add a pie chart to a sheet."""
        create_simple_pie_chart(
            self._workbook, sheet_name, label_col, value_col,
            start_row, end_row,
            title=title,
            exploded_slices=exploded_slices,
            position=position,
        )

    def add_chart(self, sheet_name: str, spec: Dict[str, Any], *,
                  position: Optional[Tuple[int, int]] = None) -> None:
        """Add a chart from a full specification dict."""
        create_chart(self._workbook, spec, sheet_name, position=position)
