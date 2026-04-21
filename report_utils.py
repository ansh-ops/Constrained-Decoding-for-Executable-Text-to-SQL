import csv
import json
from datetime import datetime
from pathlib import Path


REPORTS_DIR = Path("reports")


def make_run_dir(prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPORTS_DIR / prefix / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
