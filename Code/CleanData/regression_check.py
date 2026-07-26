"""Run the scraper without allowing it to alter curated training records."""

import hashlib

from inspect_scraped_data import main as inspect_scraped_data
from mata_manual_scrape import TRAINING_OUTPUT_PATH, main as scrape_manual


def main() -> None:
    before = TRAINING_OUTPUT_PATH.read_bytes()
    if before.count(b"\n") != 245:
        raise ValueError("Expected exactly 245 curated training records before scraping.")

    scrape_manual()

    after = TRAINING_OUTPUT_PATH.read_bytes()
    if hashlib.sha256(after).digest() != hashlib.sha256(before).digest():
        raise ValueError("Scraper modified curated training records.")
    inspect_scraped_data()
    print("Scraper regression check passed: curated records are unchanged.")


if __name__ == "__main__":
    main()
