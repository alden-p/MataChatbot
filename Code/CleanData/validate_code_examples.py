"""Run curated Mata do-file examples and record non-corpus validation evidence."""

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PATH = ROOT / "Input/Clean/mata_code_examples.jsonl"
RESULTS_PATH = ROOT / "Input/Intermediate/code_example_validation.json"
LOG_DIRECTORY = ROOT / "Input/Intermediate/code_example_logs"
ERROR_PATTERN = re.compile(r"(?im)(?:^|\n)(?:r\(\d+\);|.*\b(?:mata|stata)\b.*\berror\b)")


def content_hash(completion: str) -> str:
    return hashlib.sha256(completion.encode("utf-8")).hexdigest()


def load_examples() -> list[dict]:
    examples = []
    for number, line in enumerate(EXAMPLES_PATH.read_text(encoding="utf-8").splitlines(), 1):
        example = json.loads(line)
        required = {"id", "prompt", "completion", "features", "source"}
        if not isinstance(example, dict) or not required <= example.keys():
            raise ValueError(f"Example {number} does not have the required fields.")
        if not all(isinstance(example.get(key), str) and example[key].strip() for key in ("id", "prompt", "completion")):
            raise ValueError(f"Example {number} has empty id, prompt, or completion.")
        if not isinstance(example["features"], list) or not example["features"]:
            raise ValueError(f"Example {number} needs nonempty feature tags.")
        source = example["source"]
        if not isinstance(source, dict) or not source.get("function") or not source.get("source_pdf_pages"):
            raise ValueError(f"Example {number} lacks manual provenance.")
        examples.append(example)
    return examples


def validate_example(example: dict, command: str, version: str) -> dict:
    fixture = example.get("fixture")
    if fixture is not None and (not isinstance(fixture, dict) or not fixture.get("setup")):
        raise ValueError(f"Example {example['id']} has an invalid fixture.")
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    digest = content_hash(example["completion"])
    log_path = LOG_DIRECTORY / f"{example['id']}-{digest[:12]}.log"
    with tempfile.NamedTemporaryFile("w", suffix=".do", prefix=f"mata-example-{example['id']}-", delete=False, encoding="utf-8") as file:
        do_path = Path(file.name)
        file.write(f'log using "{str(log_path).replace(chr(34), chr(34) * 2)}", replace text\n')
        if fixture:
            file.write(fixture["setup"].rstrip() + "\n")
        file.write(example["completion"].rstrip() + "\n")
        file.write("capture log close\n")
    try:
        run = subprocess.run([command, "do", str(do_path.resolve())], text=True, capture_output=True, timeout=120)
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        return {
            "id": example["id"], "source_hash": digest, "stata_version": version,
            "date": dt.date.today().isoformat(),
            "status": "passed" if run.returncode == 0 and not ERROR_PATTERN.search(log) else "failed",
            "exit_status": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
            "log_path": str(log_path.relative_to(ROOT)), "log": log,
        }
    finally:
        do_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stata-command", default="stata-se")
    args = parser.parse_args()
    examples = load_examples()
    version_run = subprocess.run([args.stata_command, "-q"], text=True, capture_output=True, timeout=30)
    version = (version_run.stdout + version_run.stderr).splitlines()[0] if version_run.returncode == 0 else "unavailable"
    results = [validate_example(example, args.stata_command, version) for example in examples]
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    failures = [result["id"] for result in results if result["status"] != "passed"]
    if failures:
        raise SystemExit("Validation failed: " + ", ".join(failures))
    print(f"Validated {len(results)} executable examples.")


if __name__ == "__main__":
    main()
