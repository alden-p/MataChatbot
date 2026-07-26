"""Run the scraper without allowing it to alter curated data."""

import hashlib

from generate_training_data import CATALOG_PATH, TRAINING_DATA_PATH, load_catalog
from inspect_scraped_data import main as inspect_scraped_data
from mata_manual_scrape import main as scrape_manual


def main() -> None:
    catalog_before = CATALOG_PATH.read_bytes()
    training_before = TRAINING_DATA_PATH.read_bytes()
    catalog_count = len(load_catalog(CATALOG_PATH))
    generated_count = len(TRAINING_DATA_PATH.read_text(encoding="utf-8").splitlines())
    if generated_count != catalog_count * 3:
        raise ValueError(
            f"Expected {catalog_count * 3} generated records, found {generated_count}."
        )

    scrape_manual()

    if hashlib.sha256(CATALOG_PATH.read_bytes()).digest() != hashlib.sha256(
        catalog_before
    ).digest():
        raise ValueError("Scraper modified the curated function catalog.")
    if hashlib.sha256(TRAINING_DATA_PATH.read_bytes()).digest() != hashlib.sha256(
        training_before
    ).digest():
        raise ValueError("Scraper modified generated training data.")
    inspect_scraped_data()
    print("Scraper regression check passed: curated data are unchanged.")


if __name__ == "__main__":
    main()
