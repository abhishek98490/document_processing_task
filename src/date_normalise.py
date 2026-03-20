import re
from typing import Optional
from src.logging import logging
from src.config import MONTH_MAP


def to_iso(raw: Optional[str]) -> Optional[str]:

    if not raw:
        return None

    raw = raw.strip()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw

    if re.fullmatch(r"\d{4}-\d{2}", raw):
        return raw

    m = re.fullmatch(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})", raw)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # if first part > 12 it must be day
        if d > 12:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        return f"{y:04d}-{d:02d}-{mo:02d}"  # DD/MM assumed

    # MM/YYYY or MM-YYYY
    m = re.fullmatch(r"(\d{1,2})[\/\-](\d{4})", raw)
    if m:
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}"

    # Month name DD, YYYY  →  "January 5, 2020"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", raw)
    if m:
        mon = MONTH_MAP.get(m.group(1).lower())
        if mon:
            return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}"

    # DD Month YYYY  →  "05 Jan 2020"
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        mon = MONTH_MAP.get(m.group(2).lower())
        if mon:
            return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(1)):02d}"

    # YYYY only
    m = re.fullmatch(r"(\d{4})", raw)
    if m:
        return raw

    logging.warning(f"Could not normalise date: '{raw}' — returning as-is")
    return raw


def normalise_dates(result: dict) -> dict:
    """Apply to_iso to every date field in the LLM date extraction response."""
    result["expiry_date"]     = to_iso(result.get("expiry_date"))
    result["activation_date"] = to_iso(result.get("activation_date"))
    result["other_dates"]     = {
        k: to_iso(v) for k, v in result.get("other_dates", {}).items()
    }
    return result