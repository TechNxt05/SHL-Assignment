"""
Utility to view the current state of the scraped catalog.
"""
import json
from pathlib import Path

CATALOG_FILE = Path("data/processed/catalog.json")

def main():
    if not CATALOG_FILE.exists():
        print("Catalog file not found.")
        return

    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total valid assessments in catalog: {len(data)}")
    if data:
        print("\nLast 5 entries:")
        for item in data[-5:]:
            print(f"- {item['name']} ({item['test_type']})")
            print(f"  URL: {item['url']}")
            print(f"  Levels: {item['job_levels']}")
            print(f"  Duration: {item['duration_minutes']} min")
            print("-" * 20)

if __name__ == "__main__":
    main()
