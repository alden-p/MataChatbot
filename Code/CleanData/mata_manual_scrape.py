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
HEADER_PAGE_NUMBER = re.compile(r"\s+\d+$")


@dataclass(frozen=True)
class ExtractedLine:
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


def line_rows(lines: Sequence[ExtractedLine], tolerance: float = 2.0) -> List[str]:
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
            result.append(normalized)
    return result


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


def is_repeated_running_header(text: str, page_number: int, repeated_headers: set) -> bool:
    if title_from_line(text):
        return False
    key = HEADER_PAGE_NUMBER.sub("", text)
    return page_number > 1 and key in repeated_headers


def render_manual(
    pages: Sequence[Tuple[int, List[ExtractedLine]]],
) -> Tuple[str, List[FunctionEntry]]:
    top_line_counts = Counter()
    for _, extracted_lines in pages:
        for line in line_rows(extracted_lines)[:3]:
            top_line_counts[HEADER_PAGE_NUMBER.sub("", line)] += 1
    repeated_headers = {line for line, count in top_line_counts.items() if count > 1}

    output = []
    entries = []
    for page_number, extracted_lines in pages:
        lines = line_rows(extracted_lines)
        page_lines = []
        printed_page = ""
        for line in lines:
            title = title_from_line(line)
            if title:
                function, purpose, title_page = title
                title_line = next(
                    (
                        extracted
                        for extracted in extracted_lines
                        if normalize_text(extracted.text) == line
                    ),
                    None,
                )
                # M-5 documentation entries are page-top titles. Restricting
                # parsing to this band avoids collecting function calls from
                # examples such as ``return(foo()) -- bar`` in body text.
                if title_line and title_line.y1 >= 590:
                    function = re.sub(r"^\[M-5\]\s*", "", function)
                    printed_page = title_page or printed_page
                    entries.append(
                        FunctionEntry(function, purpose, page_number, title_page)
                    )
            if not is_repeated_running_header(line, page_number, repeated_headers):
                page_lines.append(line)

        marker = f"<!-- source-pdf-page: {page_number}"
        if printed_page:
            marker += f"; printed-page: {printed_page}"
        marker += " -->"
        output.extend((marker, *page_lines, ""))

    return "\n".join(output).rstrip() + "\n", entries


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
    manual_text, entries = render_manual(pages)
    records = create_training_records(entries)
    report = validation_report(pages, entries, records)

    atomic_write_text(TEXT_OUTPUT_PATH, manual_text)
    atomic_write_jsonl(TRAINING_OUTPUT_PATH, records)
    atomic_write_text(REPORT_OUTPUT_PATH, json.dumps(report, indent=2) + "\n")
    print(
        f"Created {len(records)} training records, {len(entries)} headings, "
        f"and validation report in {REPORT_OUTPUT_PATH}."
    )


if __name__ == "__main__":
    main()
