# Author: Alden Porter
# Extract Mata manual text and create function question-and-answer training records.

import json
import re
from io import StringIO
from pathlib import Path

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


RAW_PDF_PATH = Path("../Input/Raw/m-2.pdf")
TEXT_OUTPUT_PATH = Path("../Output/mata_manual.txt")
TRAINING_OUTPUT_PATH = Path("../Input/Clean/training_data.jsonl")

FUNCTION_ENTRY_PATTERN = re.compile(
    r"^[\f \t]*([A-Za-z_][A-Za-z0-9_]*\([ \t]*\))[ \t]*"
    r"—[ \t]*(.+?)(?:[ \t]+\d+)?[ \t]*$",
    re.MULTILINE,
)


def convert_pdf_to_text(path):
    resource_manager = PDFResourceManager()
    text_buffer = StringIO()
    device = TextConverter(resource_manager, text_buffer, laparams=LAParams())

    try:
        with path.open("rb") as pdf_file:
            interpreter = PDFPageInterpreter(resource_manager, device)
            for page in PDFPage.get_pages(pdf_file, check_extractable=True):
                interpreter.process_page(page)
        return text_buffer.getvalue()
    finally:
        device.close()
        text_buffer.close()


def save_text_to_file(text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def create_training_records(manual_text):
    function_manual_start = manual_text.find("\fabs( ) —")
    if function_manual_start == -1:
        raise ValueError("Could not find the alphabetical Mata function index.")

    records_by_function = {}
    for match in FUNCTION_ENTRY_PATTERN.finditer(manual_text[function_manual_start:]):
        function_name = re.sub(r"\s+", "", match.group(1))
        purpose = re.sub(r"\s+", " ", match.group(2)).strip().rstrip(".")
        records_by_function[function_name] = {
            "prompt": f"What Mata function is used for {purpose.lower()}?",
            "completion": f"{function_name} is used for {purpose.lower()}.",
        }

    if not records_by_function:
        raise ValueError("Could not create Mata function training records.")

    return list(records_by_function.values())


def save_training_records(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("Extracting Mata manual text...")
    manual_text = convert_pdf_to_text(RAW_PDF_PATH)
    save_text_to_file(manual_text, TEXT_OUTPUT_PATH)

    records = create_training_records(manual_text)
    save_training_records(records, TRAINING_OUTPUT_PATH)
    print(f"Created {len(records)} training records in {TRAINING_OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
