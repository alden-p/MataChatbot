"""Generate deterministic grouped chats from curated facts and validated do-files."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "Input/Clean/function_catalog.jsonl"
EXAMPLES_PATH = ROOT / "Input/Clean/mata_code_examples.jsonl"
VALIDATION_PATH = ROOT / "Input/Intermediate/code_example_validation.json"
TRAINING_DATA_PATH = ROOT / "Input/Clean/training_data.jsonl"
CATALOG_CONVERSATION_LIMIT = 20


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completion_hash(completion: str) -> str:
    return hashlib.sha256(completion.encode("utf-8")).hexdigest()


def load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    catalog = load_jsonl(path)
    for number, entry in enumerate(catalog, 1):
        if set(entry) != {"function", "purpose"} or not all(isinstance(entry[key], str) and entry[key].strip() for key in entry):
            raise ValueError(f"Invalid catalog entry {number}.")
    return catalog


def validated_examples() -> list[dict]:
    results = {record["id"]: record for record in json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))}
    approved = []
    for example in load_jsonl(EXAMPLES_PATH):
        result = results.get(example["id"])
        if not result or result.get("status") != "passed" or result.get("source_hash") != completion_hash(example["completion"]):
            continue
        approved.append(example)
    return approved


def message(role: str, content: str, chat_id: int, chat_index: int, kind: str, pages: list[int], tags: list[str], subject: str) -> dict:
    return {
        "role": role, "content": content, "chat_id": chat_id, "chat_index": chat_index,
        "kind": kind, "source_pdf_pages": ";".join(map(str, pages)),
        "tags": ";".join(tags), "subject": subject,
    }


def generate_records(catalog: list[dict], examples: list[dict]) -> list[dict]:
    records = []
    chat_id = 0
    for entry in catalog[:CATALOG_CONVERSATION_LIMIT]:
        chat_id += 1
        tags = ["function-lookup", entry["function"].rstrip("()")]
        records.extend((
            message("user", f"What is the purpose of {entry['function']}?", chat_id, 1, "function-lookup", [], tags, entry["function"]),
            message("assistant", f"{entry['function']} is used for {entry['purpose']}.", chat_id, 2, "function-lookup", [], tags, entry["function"]),
        ))
    for example in examples:
        chat_id += 1
        source = example["source"]
        records.extend((
            message("user", example["prompt"], chat_id, 1, "executable-code", source["source_pdf_pages"], example["features"], source["function"]),
            message("assistant", example["completion"], chat_id, 2, "executable-code", source["source_pdf_pages"], example["features"], source["function"]),
        ))
    return records


def render_jsonl(records: list[dict]) -> bytes:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records).encode()


def main() -> None:
    records = generate_records(load_catalog(), validated_examples())
    TRAINING_DATA_PATH.write_bytes(render_jsonl(records))
    print(f"Generated {len(records) // 2} chats ({len(records)} messages).")


if __name__ == "__main__":
    main()
