"""Native Office/PDF artifact creation and inspection."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path
from typing import Any


SUPPORTED_ARTIFACTS = {"docx", "xlsx", "pptx", "pdf"}


def _sections(content: dict[str, Any]) -> list[dict[str, Any]]:
    raw = content.get("sections") or []
    return [item for item in raw if isinstance(item, dict)]


def create_artifact(path: Path, kind: str, content: dict[str, Any]) -> dict[str, Any]:
    kind = kind.lower().lstrip(".")
    if kind not in SUPPORTED_ARTIFACTS:
        raise ValueError(f"Unsupported artifact format: {kind}")
    if path.suffix.lower() != f".{kind}":
        raise ValueError(f"Output path must end in .{kind}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=f".{kind}", dir=path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        if kind == "docx":
            _create_docx(temp_path, content)
        elif kind == "xlsx":
            _create_xlsx(temp_path, content)
        elif kind == "pptx":
            _create_pptx(temp_path, content)
        else:
            _create_pdf(temp_path, content)
        # See fs_atomic.preserve_existing_metadata: os.replace() swaps
        # inodes, so regenerating an existing artifact would otherwise
        # silently drop its original mode/owner in favor of mkstemp's
        # 0600 + the running process's uid/gid.
        from .fs_atomic import preserve_existing_metadata
        preserve_existing_metadata(temp_path, path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return {
        "success": True, "artifact_type": kind, "path": str(path),
        "size_bytes": path.stat().st_size,
    }


def _create_docx(path: Path, content: dict[str, Any]) -> None:
    from docx import Document
    document = Document()
    title = str(content.get("title") or "")
    if title:
        document.add_heading(title, 0)
    for section in _sections(content):
        heading = str(section.get("heading") or "")
        if heading:
            document.add_heading(heading, level=min(max(int(section.get("level") or 1), 1), 9))
        body = section.get("content") or section.get("body") or ""
        paragraphs = body if isinstance(body, list) else str(body).split("\n\n")
        for paragraph in paragraphs:
            document.add_paragraph(str(paragraph))
    document.core_properties.title = title
    document.save(path)


def _safe_cell(value: Any, allow_formulas: bool) -> Any:
    if not allow_formulas and isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _create_xlsx(path: Path, content: dict[str, Any]) -> None:
    """Create an xlsx file, delegating to spreadsheet_engineering when
    the content contains advanced fields (formulas, styles, charts).

    Falls back to the legacy inline implementation for simple content
    dicts that only have ``rows``/``header``/``freeze_panes`` so that
    existing callers are not affected by the new module's API.
    """
    # Check if content uses advanced features that require the new module
    has_advanced = any(
        key in content
        for key in ("formulas", "conditional_formatting", "charts", "palette", "theme")
    )
    if has_advanced:
        from .spreadsheet_engineering.xlsx_builder import create_workbook
        create_workbook(path, content)
        return

    # Legacy path for simple content dicts
    from openpyxl import Workbook
    from openpyxl.styles import Font
    workbook = Workbook()
    workbook.remove(workbook.active)
    allow_formulas = bool(content.get("allow_formulas", False))
    sheets = content.get("sheets") or [{"name": "Sheet1", "rows": content.get("rows") or []}]
    for index, spec in enumerate(sheets):
        if not isinstance(spec, dict):
            continue
        sheet = workbook.create_sheet(str(spec.get("name") or f"Sheet{index + 1}")[:31])
        rows = spec.get("rows") or []
        for row in rows:
            values = row if isinstance(row, list) else [row]
            sheet.append([_safe_cell(value, allow_formulas) for value in values])
        if rows and bool(spec.get("header", True)):
            for cell in sheet[1]:
                cell.font = Font(bold=True)
        sheet.freeze_panes = spec.get("freeze_panes")
        for column in sheet.columns:
            width = min(max((len(str(cell.value or "")) for cell in column), default=8) + 2, 60)
            sheet.column_dimensions[column[0].column_letter].width = width
    if not workbook.sheetnames:
        workbook.create_sheet("Sheet1")
    workbook.save(path)


def _create_pptx(path: Path, content: dict[str, Any]) -> None:
    from pptx import Presentation
    presentation = Presentation()
    title = str(content.get("title") or "")
    slides = content.get("slides") or []
    if title:
        slide = presentation.slides.add_slide(presentation.slide_layouts[0])
        slide.shapes.title.text = title
        if len(slide.placeholders) > 1:
            slide.placeholders[1].text = str(content.get("subtitle") or "")
    for spec in slides:
        if not isinstance(spec, dict):
            continue
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = str(spec.get("title") or "")
        body = spec.get("body") or spec.get("bullets") or ""
        items = body if isinstance(body, list) else str(body).splitlines()
        frame = slide.placeholders[1].text_frame
        frame.clear()
        for index, item in enumerate(items):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = str(item)
    presentation.save(path)


def _create_pdf(path: Path, content: dict[str, Any]) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    styles = getSampleStyleSheet()
    story = []
    title = str(content.get("title") or "")
    if title:
        story.extend((Paragraph(html.escape(title), styles["Title"]), Spacer(1, 12)))
    for section in _sections(content):
        heading = str(section.get("heading") or "")
        if heading:
            story.append(Paragraph(html.escape(heading), styles["Heading2"]))
        body = section.get("content") or section.get("body") or ""
        paragraphs = body if isinstance(body, list) else str(body).split("\n\n")
        for paragraph in paragraphs:
            story.extend((Paragraph(html.escape(str(paragraph)).replace("\n", "<br/>"), styles["BodyText"]), Spacer(1, 8)))
    SimpleDocTemplate(str(path), pagesize=A4, title=title).build(story)


def inspect_artifact(
    path: Path, *, max_chars: int = 30_000, offset: int = 0,
) -> dict[str, Any]:
    """Extract text from an Office/PDF artifact for the model to read.

    `offset` skips the first `offset` characters of the FULL extracted
    text (confirmed live 2026-09: a model reading a long .docx got a
    truncated view and naturally asked for "the rest" with offset -- the
    tool rejected the argument it needed, and the turn stalled). The
    response reports both `offset` and the full `total_chars` so the
    caller can page through deterministically.
    """
    kind = path.suffix.lower().lstrip(".")
    try:
        offset = max(int(offset), 0)
    except (TypeError, ValueError):
        offset = 0
    if kind == "docx":
        from docx import Document
        doc = Document(path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        details = {"paragraphs": len(doc.paragraphs)}
    elif kind == "xlsx":
        # Try the rich inspector first; fall back to legacy if unavailable
        try:
            from .spreadsheet_engineering.xlsx_inspector import inspect_workbook as _rich_inspect
            rich = _rich_inspect(path, include_formulas=True, include_styles=False, include_charts=True)
            # Build a text representation for the caller
            chunks = []
            for sheet_info in rich.get("sheet_details", []):
                chunks.append(f"## {sheet_info['name']}")
                for row in sheet_info.get("formulas", {}):
                    chunks.append(f"  {row}: {sheet_info['formulas'][row]}")
                # Also include raw values for non-formula cells
                from openpyxl import load_workbook
                book = load_workbook(path, read_only=True, data_only=False)
                ws = book[sheet_info["name"]]
                for row in ws.iter_rows(values_only=True):
                    chunks.append("\t".join("" if value is None else str(value) for value in row))
                book.close()
            text = "\n".join(chunks)
            details = dict(rich.get("summary") or {})
            # Contract parity with the legacy inspector (and the API test
            # surface): `sheets` = the ordered sheet-name list. The rich
            # summary alone carries counts (total_sheets), not the names.
            details.setdefault(
                "sheets", rich.get("workbook_info", {}).get("sheet_names", []),
            )
        except Exception:
            # Legacy fallback
            from openpyxl import load_workbook
            book = load_workbook(path, read_only=True, data_only=False)
            chunks = []
            details = {"sheets": book.sheetnames}
            for sheet in book.worksheets:
                chunks.append(f"## {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    chunks.append("\t".join("" if value is None else str(value) for value in row))
            text = "\n".join(chunks)
            book.close()
    elif kind == "pptx":
        from pptx import Presentation
        deck = Presentation(path)
        chunks = []
        for number, slide in enumerate(deck.slides, 1):
            chunks.append(f"## Slide {number}")
            chunks.extend(shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text)
        text = "\n".join(chunks)
        details = {"slides": len(deck.slides)}
    elif kind == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        details = {"pages": len(reader.pages), "metadata": {str(k): str(v) for k, v in (reader.metadata or {}).items()}}
    else:
        raise ValueError("inspect_artifact supports .docx, .xlsx, .pptx, and .pdf")
    return {
        "success": True, "artifact_type": kind, "path": str(path), **details,
        "total_chars": len(text),
        "offset": offset,
        "text": text[offset:offset + max_chars],
        "truncated": len(text) > offset + max_chars,
    }
