"""Create a deterministic review queue from provenance-preserving manual sections."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS_PATH = ROOT / "Input/Intermediate/mata_manual_sections.jsonl"
QUEUE_PATH = ROOT / "Input/Intermediate/mata_example_review_queue.jsonl"


def category(section: dict) -> str:
    text = (section["function"] + " " + section["purpose"] + " " + section["syntax"]).casefold()
    if section["function"].startswith("st_"):
        return "stata-interaction"
    if "view" in text:
        return "views-and-data-mutation"
    if section["function"] in {"printf()", "display()"}:
        return "output"
    if any(word in text for word in ("matrix", "vector", "scalar")):
        return "mata-values"
    return "function-reference"


def main() -> None:
    records = []
    for line in SECTIONS_PATH.read_text(encoding="utf-8").splitlines():
        section = json.loads(line)
        records.append({
            "function": section["function"], "purpose": section["purpose"],
            "source_pdf_pages": section["source_pdf_pages"], "syntax": section["syntax"],
            "remarks_and_examples": section["remarks_and_examples"],
            "proposed_task_category": category(section),
        })
    QUEUE_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} review records to {QUEUE_PATH}.")


if __name__ == "__main__":
    main()
