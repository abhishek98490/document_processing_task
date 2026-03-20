from src.logging import logging
import cv2
import pytesseract
import re


def clean_text(text):
    text = text.strip()
    text = re.sub(r"[^\w\s\-\(\)]", "", text) 
    return text


def OCREngine(img_path: str) -> str:
    logging.info(f"Processing: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"cv2.imread returned None for: {img_path}")
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    data = pytesseract.image_to_data(
        thresh,
        lang="eng",
        config="--oem 1 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    n = len(data['text'])
    lines = {}

    for j in range(n):
        raw_text = data['text'][j]
        text = clean_text(raw_text)
        conf = int(data['conf'][j]) if data['conf'][j] != '-1' else 0

        if text != "" and conf > 60:
            line_id = (data['block_num'][j], data['line_num'][j])

            if line_id not in lines:
                lines[line_id] = {"words": []}

            lines[line_id]["words"].append(text)

    return "\n".join(" ".join(line["words"]) for line in lines.values())