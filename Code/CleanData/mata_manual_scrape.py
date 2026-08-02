# Author: Alden Porter
# Extract Mata manual text and write an extraction-quality report.

import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextLine
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_PDF_PATH = REPOSITORY_ROOT / "Input/Raw/m-2.pdf"
TEXT_OUTPUT_PATH = REPOSITORY_ROOT / "Input/Intermediate/mata_manual.txt"
SECTIONS_OUTPUT_PATH = REPOSITORY_ROOT / "Input/Intermediate/mata_manual_sections.jsonl"
REPORT_OUTPUT_PATH = REPOSITORY_ROOT / "Input/Intermediate/mata_manual_scrape_report.json"
CATALOG_PATH = REPOSITORY_ROOT / "Input/Clean/function_catalog.jsonl"

TITLE_PATTERN = re.compile(
    r"^(?P<function>.+?\(\s*\))\s+[—-]\s+"
    r"(?P<purpose>.+?)(?:\s+(?P<printed_page>\d+))?$"
)
FUNCTION_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\*?\(\)$")
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE = re.compile(r"[ \t]+")
PAGE_NUMBER_TEXTS = frozenset(str(number) for number in range(1, 1200))
PAGE_NUMBER_TEXTS |= frozenset({"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"})
NAVIGATION_HEADER_TEXTS = frozenset(
    {
        "Acknowledgments References Also see",
        "Also see",
        "Conformability",
        "Conformability References Also see",
        "Contents",
        "Contents Also see",
        "Contents Description Also see",
        "Contents Description Remarks and examples Also see",
        "Contents Description Remarks and examples Reference Also see",
        "Contents Description Remarks and examples References Also see",
        "Description",
        "Description Remarks and examples Also see",
        "Description Remarks and examples Reference Also see",
        "Description Syntax Conformability Also see",
        "Description Syntax Conformability Diagnostics Also see",
        "Description Syntax Option Remarks and examples",
        "Description Syntax Option Remarks and examples Also see",
        "Description Syntax Options Remarks and examples Also see",
        "Description Syntax Remarks and examples",
        "Description Syntax Remarks and examples Also see",
        "Description Syntax Remarks and examples Conformability",
        "Description Syntax Remarks and examples Conformability Also see",
        "Description Syntax Remarks and examples Conformability Diagnostics Also see",
        "Description Syntax Remarks and examples Diagnostics Also see",
        "Description Syntax Remarks and examples Diagnostics References Also see",
        "Description Syntax Remarks and examples Reference Also see",
        "Description Syntax Remarks and examples References Also see",
        "Diagnostics",
        "Diagnostics Also see",
        "Diagnostics Methods and formulas References Also see",
        "Diagnostics Reference Also see",
        "Diagnostics References Also see",
        "Error codes",
        "Introduction",
        "Methods and formulas",
        "Reference",
        "References",
        "Remarks and examples",
        "Syntax",
    }
)
EXPECTED_SECTION_LABELS = (
    "Description",
    "Syntax",
    "Remarks and examples",
    "Also see",
)
EXTRACTED_SECTION_LABELS = ("Description", "Syntax", "Remarks and examples")


@dataclass(frozen=True)
class ExtractedLine:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float


@dataclass(frozen=True)
class RenderedRow:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float


@dataclass(frozen=True)
class FunctionEntry:
    function: str
    purpose: str
    pdf_page: int
    printed_page: str


@dataclass(frozen=True)
class RenderedPage:
    pdf_page: int
    printed_page: str
    lines: tuple[str, ...]


def iter_text_lines(item) -> Iterable[LTTextLine]:
    if isinstance(item, LTTextLine):
        yield item
    elif hasattr(item, "__iter__"):
        for child in item:
            yield from iter_text_lines(child)


def normalize_text(text: str) -> str:
    return WHITESPACE.sub(" ", CONTROL_CHARACTERS.sub("", text)).strip()


def canonicalize_function(function: str) -> str:
    """Convert PDF typography such as ``st data( )`` to Mata syntax."""
    name, separator, arguments = function.partition("(")
    if not separator:
        return function
    return f"{WHITESPACE.sub('_', name.strip())}({WHITESPACE.sub('', arguments)}"


def title_from_line(text: str):
    match = TITLE_PATTERN.match(text)
    if not match:
        return None, None

    function = canonicalize_function(normalize_text(match.group("function")))
    function = re.sub(r"^\[M-5\]\s*", "", function)
    purpose = normalize_text(match.group("purpose")).rstrip(".")
    if not FUNCTION_IDENTIFIER.fullmatch(function):
        return None, {
            "text": text,
            "reason": f"malformed function identifier {function!r}",
        }
    if not purpose:
        return None, {"text": text, "reason": "missing title purpose"}
    return (function, purpose, match.group("printed_page") or ""), None


def rendered_rows(
    lines: Sequence[ExtractedLine], tolerance: float = 2.0
) -> List[RenderedRow]:
    """Order text by baseline and combine fragments that share a visual row."""
    rows: List[List[ExtractedLine]] = []
    for line in sorted(lines, key=lambda value: (-value.y1, value.x0)):
        if rows and abs(rows[-1][0].y1 - line.y1) <= tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])

    result = []
    for row in rows:
        fragments = sorted(row, key=lambda value: value.x0)
        text = ""
        previous_x1 = None
        for fragment in fragments:
            if previous_x1 is not None:
                gap = fragment.x0 - previous_x1
                # Mathematical fragments are typically nearly touching; prose
                # columns and table cells need a separator to remain readable.
                separator = "" if gap <= 3 else " "
                text += separator
            text += fragment.text
            previous_x1 = fragment.x1
        normalized = normalize_text(text)
        if normalized:
            result.append(
                RenderedRow(
                    normalized,
                    min(fragment.x0 for fragment in fragments),
                    max(fragment.x1 for fragment in fragments),
                    min(fragment.y0 for fragment in fragments),
                    max(fragment.y1 for fragment in fragments),
                )
            )
    return result


def line_rows(lines: Sequence[ExtractedLine], tolerance: float = 2.0) -> List[str]:
    return [row.text for row in rendered_rows(lines, tolerance)]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_file.write(text)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def atomic_write_jsonl(path: Path, records: Sequence[dict]) -> None:
    atomic_write_text(
        path, "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )


def extract_pdf_pages(path: Path) -> List[Tuple[int, List[ExtractedLine]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Source PDF does not exist: {path}")

    resource_manager = PDFResourceManager()
    device = PDFPageAggregator(resource_manager, laparams=LAParams())
    interpreter = PDFPageInterpreter(resource_manager, device)
    pages = []

    with path.open("rb") as pdf_file:
        for page_number, page in enumerate(
            PDFPage.get_pages(pdf_file, check_extractable=True), start=1
        ):
            interpreter.process_page(page)
            layout = device.get_result()
            lines = [
                ExtractedLine(
                    normalize_text(line.get_text()),
                    line.x0,
                    line.x1,
                    line.y0,
                    line.y1,
                )
                for line in iter_text_lines(layout)
                if normalize_text(line.get_text())
            ]
            pages.append((page_number, lines))
    device.close()
    return pages


def page_furniture_category(row: RenderedRow) -> str | None:
    """Classify only fixed-position, exact-text manual furniture."""
    if row.text in NAVIGATION_HEADER_TEXTS and 560 <= row.y0 <= 580:
        return "headers"
    if (
        row.text in PAGE_NUMBER_TEXTS
        and 200 <= row.x0 <= 225
        and 30 <= row.y0 <= 35
        and row.y1 <= 47
    ):
        return "page_numbers"
    return None


def render_manual(
    pages: Sequence[Tuple[int, List[ExtractedLine]]],
) -> Tuple[str, List[FunctionEntry], Counter, List[dict], List[RenderedPage]]:
    output = []
    entries = []
    extraction_metrics = Counter()
    retained_section_labels = Counter()
    rejected_headings = []
    rendered_pages = []
    for page_number, extracted_lines in pages:
        rows = rendered_rows(extracted_lines)
        page_lines = []
        printed_page = ""
        for row in rows:
            title, rejection = title_from_line(row.text)
            if rejection and row.y1 >= 590:
                rejection["source_pdf_page"] = page_number
                rejected_headings.append(rejection)
            if title:
                function, purpose, title_page = title
                # M-5 documentation entries are page-top titles. Restricting
                # parsing to this band avoids collecting function calls from
                # examples such as ``return(foo()) -- bar`` in body text.
                if row.y1 >= 590:
                    printed_page = title_page or printed_page
                    entries.append(
                        FunctionEntry(function, purpose, page_number, title_page)
                    )
            category = page_furniture_category(row)
            if category:
                extraction_metrics[category] += 1
                continue
            page_lines.append(row.text)
            extraction_metrics["retained_body_lines"] += 1
            if row.text in EXPECTED_SECTION_LABELS:
                retained_section_labels[row.text] += 1

        marker = f"<!-- source-pdf-page: {page_number}"
        if printed_page:
            marker += f"; printed-page: {printed_page}"
        marker += " -->"
        output.extend((marker, *page_lines, ""))
        rendered_pages.append(
            RenderedPage(page_number, printed_page, tuple(page_lines))
        )
    for label in EXPECTED_SECTION_LABELS:
        extraction_metrics[f"retained_{label}"] = retained_section_labels[label]
    return (
        "\n".join(output).rstrip() + "\n",
        entries,
        extraction_metrics,
        rejected_headings,
        rendered_pages,
    )


def extract_sections(
    pages: Sequence[RenderedPage], headings: Sequence[FunctionEntry]
) -> list[dict]:
    """Split the retained manual text at valid M-5 page-top function headings."""
    sections = []
    current = None
    current_section = None
    valid_headings = {
        (heading.pdf_page, heading.function, heading.purpose) for heading in headings
    }
    for page in pages:
        chunk_lines = []
        for line in page.lines:
            title, _ = title_from_line(line)
            if title and (page.pdf_page, title[0], title[1]) in valid_headings:
                function, purpose, title_page = title
                if current:
                    sections.append(current)
                current = {
                    "function": function,
                    "purpose": purpose,
                    "source_pdf_pages": [],
                    "printed_pages": [],
                    "description": "",
                    "syntax": "",
                    "remarks_and_examples": "",
                    "raw_text": "",
                    "page_chunks": [],
                }
                current_section = None
            if not current:
                continue
            if line in EXTRACTED_SECTION_LABELS:
                current_section = line
            elif line in EXPECTED_SECTION_LABELS:
                current_section = None
            chunk_lines.append(line)
            if current_section:
                field = {
                    "Description": "description",
                    "Syntax": "syntax",
                    "Remarks and examples": "remarks_and_examples",
                }[current_section]
                if line != current_section:
                    current[field] += (("\n" if current[field] else "") + line)
        if current and chunk_lines:
            current["source_pdf_pages"].append(page.pdf_page)
            if page.printed_page:
                current["printed_pages"].append(page.printed_page)
            marker = f"<!-- source-pdf-page: {page.pdf_page}"
            if page.printed_page:
                marker += f"; printed-page: {page.printed_page}"
            marker += " -->"
            chunk_text = "\n".join(chunk_lines)
            current["page_chunks"].append(
                {
                    "source_pdf_page": page.pdf_page,
                    "printed_page": page.printed_page,
                    "text": chunk_text,
                }
            )
            current["raw_text"] += (("\n" if current["raw_text"] else "") + marker + "\n" + chunk_text)
    if current:
        sections.append(current)
    for section in sections:
        section["source_pdf_pages"] = sorted(set(section["source_pdf_pages"]))
        section["printed_pages"] = list(dict.fromkeys(section["printed_pages"]))
    return sections


def section_quality_checks(sections: Sequence[dict]) -> dict:
    """Reject unusable records and surface extraction artifacts for review."""
    errors = []
    flags = []
    coverage = Counter()
    for index, section in enumerate(sections, 1):
        identity = f"entry {index} ({section['function']!r})"
        if not FUNCTION_IDENTIFIER.fullmatch(section["function"]):
            errors.append(f"{identity}: missing or malformed function identifier")
        if not section["source_pdf_pages"]:
            errors.append(f"{identity}: missing source PDF page")
        if len(section["purpose"]) < 3 or not re.search(r"[A-Za-z]", section["purpose"]):
            errors.append(f"{identity}: missing or unreadable purpose")
        has_syntax = bool(section["syntax"].strip())
        has_examples = bool(section["remarks_and_examples"].strip())
        coverage[
            "both" if has_syntax and has_examples else "syntax" if has_syntax else "examples" if has_examples else "neither"
        ] += 1
        raw = section["raw_text"]
        if CONTROL_CHARACTERS.search(raw):
            flags.append(f"{identity}: control characters")
        if re.search(r"\b[A-Za-z_]+\s+[A-Za-z_]+\s*\(\s*\)", raw):
            flags.append(f"{identity}: possible broken identifier")
        if any(header in raw for header in NAVIGATION_HEADER_TEXTS):
            flags.append(f"{identity}: repeated navigation header")
        if has_syntax and len(section["syntax"].strip()) < 12:
            flags.append(f"{identity}: suspiciously short syntax")
        if has_examples and len(section["remarks_and_examples"].strip()) < 20:
            flags.append(f"{identity}: suspiciously short remarks/examples")
    if errors:
        raise ValueError("Section extraction failed:\n" + "\n".join(errors))
    return {"coverage": dict(coverage), "flags": flags}


def catalog_comparison(sections: Sequence[dict]) -> dict:
    catalog_names = set()
    catalog_records = 0
    for line in CATALOG_PATH.read_text(encoding="utf-8").splitlines():
        catalog_records += 1
        catalog_names.add(json.loads(line)["function"])
    extracted_names = {section["function"] for section in sections}
    return {
        "catalog_record_count": catalog_records,
        "catalog_unique_count": len(catalog_names),
        "extraction_unique_count": len(extracted_names),
        "record_vs_unique_discrepancy": catalog_records - len(extracted_names),
        "catalog_only_names": sorted(catalog_names - extracted_names),
        "extraction_only_names": sorted(extracted_names - catalog_names),
    }


def validation_report(
    pages: Sequence[Tuple[int, List[ExtractedLine]]],
    entries: Sequence[FunctionEntry],
    extraction_metrics: Counter,
    rejected_headings: Sequence[dict],
) -> dict:
    title_pages = {entry.pdf_page for entry in entries}
    return {
        "source_pdf": str(RAW_PDF_PATH.relative_to(REPOSITORY_ROOT)),
        "pages_extracted": len(pages),
        "extraction_qc": {
            "total_pages_processed": len(pages),
            "removed_page_furniture": {
                "headers": extraction_metrics["headers"],
                "footers": extraction_metrics["footers"],
                "page_numbers": extraction_metrics["page_numbers"],
            },
            "retained_body_lines": extraction_metrics["retained_body_lines"],
            "retained_section_labels": {
                label: extraction_metrics[f"retained_{label}"]
                for label in EXPECTED_SECTION_LABELS
            },
        },
        "function_heading_occurrences": len(entries),
        "unique_function_names": len({entry.function for entry in entries}),
        "function_heading_pages": len(title_pages),
        "rejected_heading_candidates": list(rejected_headings),
        "function_headings": [
            {
                "function": entry.function,
                "purpose": entry.purpose,
                "source_pdf_page": entry.pdf_page,
                "printed_page": entry.printed_page,
            }
            for entry in entries
        ],
        "sample_entries": [
            {
                "function": entry.function,
                "purpose": entry.purpose,
                "source_pdf_page": entry.pdf_page,
                "printed_page": entry.printed_page,
            }
            for entry in entries[:10]
        ],
    }


def main() -> None:
    print("Extracting Mata manual text with page coordinates...")
    pages = extract_pdf_pages(RAW_PDF_PATH)
    manual_text, entries, extraction_metrics, rejected_headings, rendered_pages = render_manual(pages)
    sections = extract_sections(rendered_pages, entries)
    section_qc = section_quality_checks(sections)
    report = validation_report(pages, entries, extraction_metrics, rejected_headings)
    report["section_extraction"] = {
        "occurrences": len(sections),
        "coverage": section_qc["coverage"],
        "corruption_flags": section_qc["flags"],
        "catalog_comparison": catalog_comparison(sections),
    }

    atomic_write_text(TEXT_OUTPUT_PATH, manual_text)
    atomic_write_jsonl(SECTIONS_OUTPUT_PATH, sections)
    atomic_write_text(REPORT_OUTPUT_PATH, json.dumps(report, indent=2) + "\n")
    print(
        f"Extracted {len(entries)} clean headings and rejected {len(rejected_headings)} "
        f"malformed candidates; wrote the validation report "
        f"to {REPORT_OUTPUT_PATH}."
    )


if __name__ == "__main__":
    main()
