"""Generate chat-template messages from the curated catalog with fixed seed 20260726.

The fixed seed makes template selection reproducible while retaining catalog order.
"""

import json
import random
import re
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPOSITORY_ROOT / "Input/Clean/function_catalog.jsonl"
TRAINING_DATA_PATH = REPOSITORY_ROOT / "Input/Clean/training_data.jsonl"
SEED = 20260726
TEMPLATES = (
    ("purpose_lookup", "What Mata function is used for {purpose}?", "{function} is the Mata function used for {purpose}."),
    ("purpose_name", "Name the Mata function for {purpose}.", "The Mata function is {function}."),
    ("purpose_association", "Which Mata function is associated with {purpose}?", "{function} is associated with {purpose}."),
    ("purpose_reference", "Find the Mata function described as: {purpose}.", "The reference is {function}, for {purpose}."),
    ("function_definition", "What does {function} do in Mata?", "{function} is used for {purpose}."),
    ("function_purpose", "What is the purpose of {function}?", "{function} is used for {purpose}."),
    ("function_description", "Describe the Mata function {function}.", "{function} is used for {purpose}."),
    ("reference_completion", "Complete this Mata reference: {function} --", "{function} is used for {purpose}."),
)
PURPOSE_ONLY_TEMPLATE_IDS = frozenset(template_id for template_id, *_ in TEMPLATES[:4])
FUNCTION_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\*?\(\)$")
WHITESPACE = re.compile(r"\s+")


def normalize_purpose(purpose: str) -> str:
    return WHITESPACE.sub(" ", purpose).strip().casefold()


def load_catalog(path: Path = CATALOG_PATH) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Function catalog does not exist: {path}")

    catalog = []
    pairs = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Catalog line {line_number} is invalid JSON: {error.msg}") from error
        if not isinstance(entry, dict) or set(entry) != {"function", "purpose"}:
            raise ValueError(
                f"Catalog line {line_number} must contain exactly function and purpose."
            )
        function = entry["function"]
        purpose = entry["purpose"]
        if not isinstance(function, str) or not function.strip():
            raise ValueError(f"Catalog line {line_number} has an empty function.")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError(f"Catalog line {line_number} has an empty purpose.")
        if not FUNCTION_IDENTIFIER.fullmatch(function):
            raise ValueError(
                f"Catalog line {line_number} has malformed function identifier {function!r}."
            )
        pair = (function, purpose)
        if pair in pairs:
            raise ValueError(f"Catalog line {line_number} duplicates {pair!r}.")
        pairs.add(pair)
        catalog.append({"function": function, "purpose": purpose})
    if not catalog:
        raise ValueError("Function catalog is empty.")
    return catalog


def ambiguous_purposes(catalog: Sequence[dict[str, str]]) -> set[str]:
    functions_by_purpose: dict[str, set[str]] = {}
    for entry in catalog:
        functions_by_purpose.setdefault(normalize_purpose(entry["purpose"]), set()).add(
            entry["function"]
        )
    return {
        purpose
        for purpose, functions in functions_by_purpose.items()
        if len(functions) > 1
    }


def generate_records(catalog: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    randomizer = random.Random(SEED)
    ambiguous = ambiguous_purposes(catalog)
    records = []
    prompts = set()

    for entry_number, entry in enumerate(catalog, 1):
        function = entry["function"]
        purpose = entry["purpose"]
        safe_templates = [
            template
            for template in TEMPLATES
            if normalize_purpose(purpose) not in ambiguous
            or template[0] not in PURPOSE_ONLY_TEMPLATE_IDS
        ]
        if len(safe_templates) < 3:
            raise ValueError(
                f"Catalog entry {entry_number} ({function!r}, {purpose!r}) has fewer "
                "than three safe templates."
            )
        selected = []
        for template in randomizer.sample(safe_templates, len(safe_templates)):
            _, prompt, completion = template
            record = {
                "prompt": prompt.format(function=function, purpose=purpose),
                "completion": completion.format(function=function, purpose=purpose),
            }
            if record["prompt"] not in prompts:
                selected.append(record)
            if len(selected) == 3:
                break
        if len(selected) != 3:
            raise ValueError(
                f"Catalog entry {entry_number} ({function!r}, {purpose!r}) cannot "
                "produce three distinct prompts."
            )
        prompts.update(record["prompt"] for record in selected)
        records.extend(selected)

    if len(records) != len(catalog) * 3:
        raise ValueError("Generator did not produce exactly three records per catalog entry.")
    return records


def render_jsonl(records: Sequence[dict[str, str]]) -> bytes:
    messages = []
    for record in records:
        messages.extend(
            (
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": record["completion"]},
            )
        )
    return "".join(
        json.dumps(message, ensure_ascii=False) + "\n" for message in messages
    ).encode("utf-8")


def main() -> None:
    catalog = load_catalog()
    records = generate_records(catalog)
    TRAINING_DATA_PATH.write_bytes(render_jsonl(records))
    print(
        f"Generated {len(records)} conversations ({len(records) * 2} messages) from "
        f"{len(catalog)} catalog entries using seed {SEED}."
    )


if __name__ == "__main__":
    main()
