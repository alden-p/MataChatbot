# MataChatbot Copilot Instructions

## Commands

This repository has no committed dependency manifest, build system, linter, or unit-test runner. The Python environment must supply the libraries imported by the scripts: `pdfminer`, `torch`, `transformers`, `peft`, `datasets`, and `trl`.

- Check all Python scripts for syntax errors: `python3 -m py_compile Code/*.py`
- Run the tokenizer connectivity smoke check (the only test script): `python3 Code/test.py`
- Extract manual text: `cd Code && python3 mata_manual_scrape.py`
- Train the LoRA adapter: `cd Code && python3 matatrain_llm.py`

`test.py` downloads the gated `meta-llama/Llama-3.2-3B` tokenizer, so it requires Hugging Face access approved for that model.

## Architecture and data flow

The project is a script-driven MATA manual fine-tuning workflow:

1. `Code/mata_manual_scrape.py` converts the source manual PDF into `Output/mata_manual.txt` using `pdfminer`.
2. Training examples are curated independently as JSONL. `Code/matatrain_llm.py` consumes only `Input/Clean/training_data.jsonl`; it does not transform the extracted manual text into examples.
3. The training script loads `meta-llama/Llama-3.2-3B`, prepares it for k-bit training, applies LoRA to `q_proj` and `v_proj`, and uses TRL's `SFTTrainer`.
4. The resulting adapter and tokenizer are saved directly under `Output/`, alongside the extracted manual text.

## Repository-specific conventions

- Training records must be one JSON object per line with `prompt` and `completion` fields. The trainer formats each record as `### Human: {prompt}\n### Assistant: {completion}`; preserve this contract if data loading or prompt formatting changes.
- Scripts use hard-coded paths relative to the `Code/` working directory rather than paths derived from `__file__`. Run training and scraping from `Code/`, or update all affected paths together.
- `mata_manual_scrape.py` currently expects `../Input/m-2.pdf`, while the checked-in PDF is `Input/Raw/m-2.pdf`. Resolve that path deliberately before running the scraper; it is not discovered automatically.
- Training configuration is local to `main()` in `matatrain_llm.py`: model ID, LoRA parameters, dataset location, output directory, and `TrainingArguments` are changed in that script rather than through a CLI or configuration file.
