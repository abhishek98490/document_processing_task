import os
from concurrent.futures import ThreadPoolExecutor


DISTANCE_METRIC = os.environ.get("CHROMA_DISTANCE", "cosine")

MONTH_MAP = {m: i for i, m in enumerate(
    ["january","february","march","april","may","june",
     "july","august","september","october","november","december"], 1
)}


CHUNK_SIZE       = 200
OVERLAP          = 30
N_RESULTS        = 5
SUMMARY_CHAR_CAP = 2000

EXECUTOR = ThreadPoolExecutor(max_workers=3)

ANALYSE_PROMPT = """\
You will be given a document. Do two things and return ONLY valid JSON — no prose, no markdown fences.

1. Classify into exactly one of: Invoice, Identity Document, Contract, Medical Record, Resume, Report, Other
2. Summarise in 3-5 concise factual sentences

Return format:
{
  "doc_type": "<category>",
  "summary":  "<summary>"
}"""

DATE_EXTRACTION_QUERY  = "expiry date activation date issue date valid from valid until date of birth"

DATE_EXTRACTION_PROMPT = """\
Extract all dates from the following document context.
Return ONLY valid JSON — no prose, no markdown fences.

Return format:
{
  "expiry_date":      "<date or null>",
  "activation_date":  "<date or null>",
  "other_dates":      { "<label>": "<date>" },
  "confidence":       <float 0.0-1.0>,
}

Rules:
- Return dates exactly as found — the caller will normalise to ISO 8601.
- If a date is not found return null.
- confidence reflects how certain you are about the extracted dates."""


