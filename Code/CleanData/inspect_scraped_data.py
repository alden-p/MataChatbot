"""Validate Mata extraction outputs, the curated catalog, and generated examples."""

import json
from pathlib import Path

from generate_training_data import (
    CATALOG_PATH,
    PURPOSE_ONLY_TEMPLATE_IDS,
    TEMPLATES,
    TRAINING_DATA_PATH,
    ambiguous_purposes,
    generate_records,
    load_catalog,
    normalize_purpose,
    render_jsonl,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANUAL_PATH = REPOSITORY_ROOT / "Input/Intermediate/mata_manual.txt"
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


def template_matches(
    record: dict, catalog: list[dict[str, str]], ambiguous: set[str]
) -> list[tuple[tuple[str, str], str]]:
    matches = []
    for entry in catalog:
        for template_id, prompt_template, completion_template in TEMPLATES:
            if (
                normalize_purpose(entry["purpose"]) in ambiguous
                and template_id in PURPOSE_ONLY_TEMPLATE_IDS
            ):
                continue
            expected = {
                "prompt": prompt_template.format(**entry),
                "completion": completion_template.format(**entry),
            }
            if record == expected:
                matches.append(((entry["function"], entry["purpose"]), template_id))
    return matches


def validate_catalog_and_records() -> tuple[int, int]:
    catalog = load_catalog(CATALOG_PATH)
    records = load_records(TRAINING_DATA_PATH)
    expected_count = len(catalog) * 3
    if len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} generated records for {len(catalog)} catalog "
            f"entries, found {len(records)}."
        )

    pair_counts = {(entry["function"], entry["purpose"]): 0 for entry in catalog}
    ambiguous = ambiguous_purposes(catalog)
    prompts = set()
    for number, record in enumerate(records, 1):
        if set(record) != {"prompt", "completion"}:
            raise ValueError(f"Record {number} does not have exactly prompt and completion.")
        for field in ("prompt", "completion"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"Record {number} has a missing {field}.")
        if record["prompt"] in prompts:
            raise ValueError(f"Record {number} duplicates a generated prompt.")
        prompts.add(record["prompt"])

        matches = template_matches(record, catalog, ambiguous)
        if len(matches) != 1:
            raise ValueError(
                f"Record {number} does not match exactly one approved catalog template."
            )
        pair, template_id = matches[0]
        if normalize_purpose(pair[1]) in ambiguous and template_id in PURPOSE_ONLY_TEMPLATE_IDS:
            raise ValueError(
                f"Record {number} uses purpose-only template {template_id!r} for "
                f"ambiguous purpose {pair[1]!r}."
            )
        pair_counts[pair] += 1

    incorrect_counts = {
        pair: count for pair, count in pair_counts.items() if count != 3
    }
    if incorrect_counts:
        raise ValueError(
            "Each catalog pair must appear in exactly three generated records; "
            f"violations: {incorrect_counts!r}"
        )

    expected_bytes = render_jsonl(generate_records(catalog))
    actual_bytes = TRAINING_DATA_PATH.read_bytes()
    if actual_bytes != expected_bytes:
        raise ValueError(
            "Generated training data does not match byte-for-byte regeneration "
            "with the fixed seed."
        )
    return len(catalog), len(records)


def validate_catalog_provenance(catalog: list[dict[str, str]], report: dict) -> None:
    headings = {
        (entry["function"], normalize_purpose(entry["purpose"]))
        for entry in report["function_headings"]
    }
    missing = [
        (entry["function"], entry["purpose"])
        for entry in catalog
        if (entry["function"], normalize_purpose(entry["purpose"])) not in headings
    ]
    if missing:
        raise ValueError(
            "Curated catalog entries are not present in clean PDF headings: "
            + ", ".join(repr(pair) for pair in missing)
        )


def main() -> None:
    for path in (MANUAL_PATH, CATALOG_PATH, TRAINING_DATA_PATH, REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.relative_to(REPOSITORY_ROOT)}; run mata_manual_scrape.py first."
            )

    manual = MANUAL_PATH.read_text(encoding="utf-8")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    catalog_count, record_count = validate_catalog_and_records()
    validate_catalog_provenance(load_catalog(CATALOG_PATH), report)

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
    for rejected in report["rejected_heading_candidates"]:
        if not rejected["reason"].startswith("malformed function identifier "):
            raise ValueError(f"Unexpected rejected heading: {rejected!r}")
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

    functions = {entry["function"] for entry in load_catalog(CATALOG_PATH)}
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
        f"{catalog_count} catalog entries, {record_count} training records."
    )


if __name__ == "__main__":
    main()
