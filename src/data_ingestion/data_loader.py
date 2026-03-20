import tempfile
from pathlib import Path
from typing import Callable
from dataclasses import dataclass

import PyPDF2
from src.logging import logging
from pdf2image import convert_from_path

from src.data_ingestion.ocr import  OCREngine

 
 
@dataclass
class ExtractedDocument:
    text: str
    source_path: Path
    extraction_method: str
    ocr_used: bool = False
 

_PDF_TEXT_THRESHOLD = 20



def _load_pdf(file_path: Path) -> ExtractedDocument:
    logging.info(f"[PDF] Loading: {file_path.name}")

    
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages_text: list[str] = []
            needs_ocr = False

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text) < _PDF_TEXT_THRESHOLD:
                    logging.debug(
                        f"[PDF] Page {page_num + 1}: only {len(text)} chars — "
                        "marking for OCR fallback"
                    )
                    needs_ocr = True
                    break
                pages_text.append(text)

            if not needs_ocr:
                full_text = "\n\n".join(pages_text).strip()
                if full_text:
                    logging.info(f"[PDF] Extracted via PyPDF2: {len(full_text)} chars")
                    return ExtractedDocument(
                        text=full_text,
                        source_path=file_path,
                        extraction_method="pypdf2",
                        ocr_used=False,
                    )

    except Exception as e:
        logging.warning(f"[PDF] PyPDF2 failed on {file_path.name}: {e} — falling back to OCR")

    logging.info(f"[PDF] Falling back to  OCR for {file_path.name}")
    return _pdf_via_ocr(file_path)


def _pdf_via_ocr(file_path: Path) -> ExtractedDocument:
    page_texts: list[str] = []

    try:
        images = convert_from_path(str(file_path), dpi=300)
    except Exception as e:
        logging.error(f"pdf2image failed for {file_path.name}: {e}")
        return ExtractedDocument(
            text="",
            source_path=file_path,
            extraction_method="  OCR",
            ocr_used=True,
        )

    with tempfile.TemporaryDirectory() as tmp:
        for idx, image in enumerate(images):
            img_path = f"{tmp}/page_{idx}.png"
            image.save(img_path, "PNG")
            text = OCREngine(img_path)
            if text:
                page_texts.append(text)
            else:
                logging.warning(f"Page {idx + 1} yielded no text")

    full_text = "\n\n".join(page_texts).strip()
    if not full_text:
        logging.error(f"OCR returned empty for {file_path.name}")

    return ExtractedDocument(
        text=full_text,
        source_path=file_path,
        extraction_method="  OCR",
        ocr_used=True,
    )


def _load_image(file_path: Path) -> ExtractedDocument:
    logging.info(f"[IMG]  OCR: {file_path.name}")
    text = OCREngine(str(file_path))

    if not text:
        logging.error(f"[IMG]  OCR returned empty for {file_path.name}")

    return ExtractedDocument(
        text=text,
        source_path=file_path,
        extraction_method="  OCR",
        ocr_used=True,
    )


_EXTENSION_REGISTRY: dict[str, Callable[[Path], ExtractedDocument]] = {
    ".pdf": _load_pdf,
    ".jpg": _load_image,
    ".jpeg": _load_image,
    ".png": _load_image
}


def load(file_path: str | Path) -> ExtractedDocument:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    handler = _EXTENSION_REGISTRY.get(ext)

    if handler is None:
        supported = ", ".join(sorted(_EXTENSION_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported file type '{ext}' for '{path.name}'. "
            f"Supported: {supported}"
        )

    return handler(path)


def supported_extensions() -> list[str]:
    return sorted(_EXTENSION_REGISTRY.keys())


if __name__=="__main__":
    file_path = "/Users/abhisheksharma/Desktop/Important/Documents/GOV_ID/adhar card.pdf"
    data = load(file_path)
