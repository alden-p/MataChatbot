"""Validate the Mata manual extraction outputs produced by mata_manual_scrape.py."""

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANUAL_PATH = REPOSITORY_ROOT / "Input/Intermediate/mata_manual.txt"
RECORDS_PATH = REPOSITORY_ROOT / "Input/Clean/training_data.jsonl"
REPORT_PATH = REPOSITORY_ROOT / "Input/Intermediate/mata_manual_scrape_report.json"
REQUIRED_FUNCTIONS = {"abbrev()", "docx*()", "Pdf*()", "solve_tol()", "st_data()"}
LATER_FUNCTION_PAGE = 327
KNOWN_NAVIGATION_HEADERS = {
    "Description Syntax Conformability Diagnostics Also see",
    "Description Syntax Remarks and examples Conformability Diagnostics Also see",
}


def load_records(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Record {line_number} is not a JSON object.")
        records.append(record)
    return records


def main() -> None:
    for path in (MANUAL_PATH, RECORDS_PATH, REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.relative_to(REPOSITORY_ROOT)}; run mata_manual_scrape.py first."
            )

    manual = MANUAL_PATH.read_text(encoding="utf-8")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    records = load_records(RECORDS_PATH)

    page_markers = manual.count("<!-- source-pdf-page:")
    if page_markers != report["pages_extracted"]:
        raise ValueError(
            f"Expected {report['pages_extracted']} source-page markers, found {page_markers}."
        )
    if "\f" in manual:
        raise ValueError("Extracted manual contains form-feed page artifacts.")
    if any(ord(character) < 32 and character not in "\n\t" for character in manual):
        raise ValueError("Extracted manual contains control characters.")
    extraction_qc = report["extraction_qc"]
    if extraction_qc["total_pages_processed"] != page_markers:
        raise ValueError("Extraction QC page total does not match source-page markers.")
    if extraction_qc["removed_page_furniture"]["headers"] == 0:
        raise ValueError("Extraction QC did not remove any navigation headers.")
    if extraction_qc["removed_page_furniture"]["page_numbers"] == 0:
        raise ValueError("Extraction QC did not remove any page numbers.")
    for label in ("Description", "Syntax", "Remarks and examples", "Also see"):
        if extraction_qc["retained_section_labels"][label] == 0:
            raise ValueError(f"Extraction QC did not retain any {label!r} section labels.")

    page_start = f"<!-- source-pdf-page: {LATER_FUNCTION_PAGE}"
    page_end = f"<!-- source-pdf-page: {LATER_FUNCTION_PAGE + 1}"
    later_page = manual.split(page_start, 1)[1].split(page_end, 1)[0]
    for label in ("Description", "Syntax"):
        if label not in later_page.splitlines():
            raise ValueError(
                f"Later function entry on PDF page {LATER_FUNCTION_PAGE} is missing {label}."
            )
    if "315" in later_page.splitlines():
        raise ValueError("Known printed page number remains in the later function entry.")
    for header in KNOWN_NAVIGATION_HEADERS:
        if header in manual.splitlines():
            raise ValueError(f"Known navigation header remains in extracted text: {header!r}")

    if len(records) != report["unique_function_records"]:
        raise ValueError("Training-record count does not match the validation report.")
    if not records:
        raise ValueError("No training records were generated.")
    for number, record in enumerate(records, 1):
        if set(record) != {"prompt", "completion"}:
            raise ValueError(f"Record {number} does not have exactly prompt and completion.")
        if not isinstance(record["prompt"], str) or not record["prompt"].strip():
            raise ValueError(f"Record {number} has a missing prompt.")
        if not isinstance(record["completion"], str) or not record["completion"].strip():
            raise ValueError(f"Record {number} has a missing completion.")

    validation = report["records"]
    invalid_counts = {
        category: result["count"]
        for category, result in validation.items()
        if result["count"] != 0
    }
    if invalid_counts:
        raise ValueError(f"Validation report contains record issues: {invalid_counts}")

    functions = {
        record["completion"].split(" is used for ", 1)[0]
        for record in records
        if " is used for " in record["completion"]
    }
    missing_functions = sorted(REQUIRED_FUNCTIONS - functions)
    if missing_functions:
        raise ValueError(f"Required function entries are missing: {', '.join(missing_functions)}")
    spaced_functions = sorted(
        function for function in functions if " " in function.partition("(")[0]
    )
    if spaced_functions:
        raise ValueError(
            "Function identifiers contain spaces: " + ", ".join(spaced_functions)
        )

    print(
        "Scraped-data inspection passed: "
        f"{report['pages_extracted']} pages, "
        f"{report['unique_function_names']} function names, "
        f"{len(records)} training records."
    )


if __name__ == "__main__":
    main()
