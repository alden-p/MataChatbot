"""Inspect generated chats, provenance, validation evidence, and coverage."""

import json
from collections import Counter

from generate_training_data import EXAMPLES_PATH, TRAINING_DATA_PATH, completion_hash, load_jsonl
from validate_code_examples import RESULTS_PATH

MINIMUM_FEATURES = {
    "do-structure", "output", "scalar", "vector", "matrix", "loop", "conditional",
    "function-definition", "st_data", "st_store", "st_view", "view", "validation",
}


def main() -> None:
    examples = {example["id"]: example for example in load_jsonl(EXAMPLES_PATH)}
    results = {result["id"]: result for result in json.loads(RESULTS_PATH.read_text(encoding="utf-8"))}
    messages = load_jsonl(TRAINING_DATA_PATH)
    if len(messages) % 2:
        raise ValueError("Training data has an incomplete conversation.")
    feature_counts = Counter()
    kind_counts = Counter()
    for offset in range(0, len(messages), 2):
        user, assistant = messages[offset:offset + 2]
        if user["role"] != "user" or assistant["role"] != "assistant" or user["chat_index"] != 1 or assistant["chat_index"] != 2 or user["chat_id"] != assistant["chat_id"]:
            raise ValueError(f"Invalid message order at chat {offset // 2 + 1}.")
        if not user["content"].strip() or not assistant["content"].strip():
            raise ValueError(f"Empty content at chat {offset // 2 + 1}.")
        if user["kind"] != assistant["kind"] or user["source_pdf_pages"] != assistant["source_pdf_pages"]:
            raise ValueError(f"Inconsistent chat metadata at chat {offset // 2 + 1}.")
        kind_counts[user["kind"]] += 1
        feature_counts.update(filter(None, user["tags"].split(";")))
        if user["kind"] == "executable-code":
            matches = [example for example in examples.values() if example["completion"] == assistant["content"]]
            if len(matches) != 1:
                raise ValueError(f"Executable chat {user['chat_id']} is not a curated example.")
            result = results.get(matches[0]["id"])
            if not result or result["status"] != "passed" or result["source_hash"] != completion_hash(assistant["content"]):
                raise ValueError(f"Executable chat {user['chat_id']} lacks matching passing validation.")
            if not user["source_pdf_pages"]:
                raise ValueError(f"Executable chat {user['chat_id']} lacks provenance.")
    missing = sorted(MINIMUM_FEATURES - set(feature_counts))
    if missing:
        raise ValueError("Missing required feature coverage: " + ", ".join(missing))
    print(f"Inspection passed: {dict(kind_counts)}; features: {dict(feature_counts)}")


if __name__ == "__main__":
    main()
