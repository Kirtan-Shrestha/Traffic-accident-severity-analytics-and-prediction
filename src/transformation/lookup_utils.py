"""Shared utilities for decoding STATS19 coded fields using the code lookup JSON."""

import json
import logging
from pathlib import Path

import pandas as pd

LOOKUPS_PATH = Path("docs/code_lookups.json")
logger = logging.getLogger(__name__)


def load_lookups() -> dict:
    return json.loads(LOOKUPS_PATH.read_text())


def decode_field(df: pd.DataFrame, field: str, lookups: dict) -> pd.DataFrame:
    """Add a `{field}_label` column decoded from the code lookups, if available."""
    if field not in lookups:
        logger.warning(f"No lookup found for field '{field}', skipping decode")
        return df

    field_lookup = lookups[field]
    df[f"{field}_label"] = df[field].astype(str).map(field_lookup)
    df[f"{field}_label"] = df[f"{field}_label"].fillna("Unmapped code")
    return df