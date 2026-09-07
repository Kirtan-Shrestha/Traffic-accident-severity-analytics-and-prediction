"""
Extract all field code/label lookups from the official STATS19 data guide
into a single JSON reference file, used by staging scripts to decode
coded fields (severity, road_type, weather_conditions, etc.) consistently.
"""

import json
from pathlib import Path

import pandas as pd

DATA_GUIDE_PATH = Path("docs/data_guide.xlsx")
OUTPUT_PATH = Path("docs/code_lookups.json")


def build_lookups() -> dict:
    df = pd.read_excel(DATA_GUIDE_PATH, sheet_name="2024_code_list")

    # Keep only rows that actually have a code and a label (skip descriptive-only rows)
    coded = df.dropna(subset=["code/format", "label"])

    lookups: dict[str, dict[str, str]] = {}

    for _, row in coded.iterrows():
        field = row["field name"]
        code = str(row["code/format"]).strip()
        label = str(row["label"]).strip()

        if field not in lookups:
            lookups[field] = {}
        lookups[field][code] = label

    return lookups


if __name__ == "__main__":
    lookups = build_lookups()

    OUTPUT_PATH.write_text(json.dumps(lookups, indent=2))

    print(f"Built lookups for {len(lookups)} fields")
    print(f"Saved to {OUTPUT_PATH}")

    # Show a couple of examples to confirm correctness
    for field in ["collision_severity", "weather_conditions"]:
        if field in lookups:
            print(f"\n{field}: {lookups[field]}")