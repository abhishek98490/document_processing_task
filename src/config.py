import os

DISTANCE_METRIC = os.environ.get("CHROMA_DISTANCE", "cosine")

MONTH_MAP = {m: i for i, m in enumerate(
    ["january","february","march","april","may","june",
     "july","august","september","october","november","december"], 1
)}
