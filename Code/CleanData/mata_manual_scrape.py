# Author: Alden Porter
# Extract Mata manual text and create function question-and-answer training records.

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
TRAINING_OUTPUT_PATH = REPOSITORY_ROOT / "Input/Clean/training_data.jsonl"
REPORT_OUTPUT_PATH = REPOSITORY_ROOT / "Input/Intermediate/mata_manual_scrape_report.json"

TITLE_PATTERN = re.compile(
    r"^(?P<function>.+?\(\s*\))\s+[—-]\s+"
    r"(?P<purpose>.+?)(?:\s+(?P<printed_page>\d+))?$"
)
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
        return None

    function = canonicalize_function(normalize_text(match.group("function")))
    purpose = normalize_text(match.group("purpose")).rstrip(".")
    if not function or not purpose or len(function) > 100:
        return None
    return function, purpose, match.group("printed_page") or ""


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
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    atomic_write_text(path, text)


def load_training_records(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Curated training data does not exist: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Curated training record {line_number} is invalid JSON: {error.msg}"
            ) from error
        if set(record) != {"prompt", "completion"}:
            raise ValueError(
                f"Curated training record {line_number} must contain prompt and completion."
            )
        records.append(record)
    return records


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
) -> Tuple[str, List[FunctionEntry], Counter]:
    output = []
    entries = []
    extraction_metrics = Counter()
    retained_section_labels = Counter()
    for page_number, extracted_lines in pages:
        rows = rendered_rows(extracted_lines)
        page_lines = []
        printed_page = ""
        for row in rows:
            title = title_from_line(row.text)
            if title:
                function, purpose, title_page = title
                # M-5 documentation entries are page-top titles. Restricting
                # parsing to this band avoids collecting function calls from
                # examples such as ``return(foo()) -- bar`` in body text.
                if row.y1 >= 590:
                    function = re.sub(r"^\[M-5\]\s*", "", function)
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
    for label in EXPECTED_SECTION_LABELS:
        extraction_metrics[f"retained_{label}"] = retained_section_labels[label]
    return "\n".join(output).rstrip() + "\n", entries, extraction_metrics


def create_training_records(entries: Sequence[FunctionEntry]) -> List[dict]:
    seen = set()
    records = []
    for entry in entries:
        key = (entry.function, entry.purpose)
        if key in seen:
            continue
        seen.add(key)
        purpose = entry.purpose.lower()
        records.append(
            {
                "prompt": f"What Mata function is used for {purpose}?",
                "completion": f"{entry.function} is used for {purpose}.",
            }
        )
    if not records:
        raise ValueError("Could not create Mata function training records.")
    return records


def validation_report(
    pages: Sequence[Tuple[int, List[ExtractedLine]]],
    entries: Sequence[FunctionEntry],
    records: Sequence[dict],
    extraction_metrics: Counter,
) -> dict:
    missing_prompts = []
    missing_completions = []
    incomplete_prompts = []
    incomplete_completions = []
    duplicate_prompts = []
    seen_prompts = set()

    for number, record in enumerate(records, start=1):
        prompt = record.get("prompt", "")
        completion = record.get("completion", "")
        if not prompt.strip():
            missing_prompts.append(number)
        if not completion.strip():
            missing_completions.append(number)
        if prompt and (not prompt.endswith("?") or len(prompt.split()) < 6):
            incomplete_prompts.append(number)
        if completion and (not completion.endswith(".") or " is used for " not in completion):
            incomplete_completions.append(number)
        if prompt in seen_prompts:
            duplicate_prompts.append(number)
        seen_prompts.add(prompt)

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
        "unique_function_records": len(records),
        "function_heading_pages": len(title_pages),
        "records": {
            "missing_prompts": {"count": len(missing_prompts), "records": missing_prompts},
            "missing_completions": {
                "count": len(missing_completions),
                "records": missing_completions,
            },
            "incomplete_prompts": {
                "count": len(incomplete_prompts),
                "records": incomplete_prompts,
            },
            "incomplete_completions": {
                "count": len(incomplete_completions),
                "records": incomplete_completions,
            },
            "duplicate_prompts": {"count": len(duplicate_prompts), "records": duplicate_prompts},
        },
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
    manual_text, entries, extraction_metrics = render_manual(pages)
    records = load_training_records(TRAINING_OUTPUT_PATH)
    report = validation_report(pages, entries, records, extraction_metrics)

    atomic_write_text(TEXT_OUTPUT_PATH, manual_text)
    atomic_write_text(REPORT_OUTPUT_PATH, json.dumps(report, indent=2) + "\n")
    print(
        f"Created {len(records)} training records, {len(entries)} headings, "
        f"and validation report in {REPORT_OUTPUT_PATH}."
    )


if __name__ == "__main__":
    main()
