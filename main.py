import sys
import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional 

from src.logging import logging
from src.chunking import Chunking
from src.vector_store import Chroma_database
from src.LLM_gateway.LLM_Call import LLM
from src.data_ingestion.data_loader import load, supported_extensions
from src.date_normalise import normalise_dates
from src.config import CHUNK_SIZE, OVERLAP, N_RESULTS, SUMMARY_CHAR_CAP, EXECUTOR, ANALYSE_PROMPT, DATE_EXTRACTION_PROMPT, DATE_EXTRACTION_QUERY





# ── Result containers ───────────────────────────────────────────────────────────

@dataclass
class DateResult:
    doc_type:        str
    summary:         str
    expiry_date:     Optional[str]
    activation_date: Optional[str]
    other_dates:     dict
    confidence:      float
    sources:         list


@dataclass
class QueryResult:
    doc_type: str
    summary:  str
    LLM_Response:   str
    sources:  list



def _run_analyse(llm: LLM, text: str) -> tuple[str, str]:
    snippet = text[:SUMMARY_CHAR_CAP]
    raw     = llm.chat(context=snippet, query=ANALYSE_PROMPT)
    cleaned = raw.strip().strip("```").replace("json", "", 1).strip()
    try:
        parsed   = json.loads(cleaned)
        doc_type = parsed.get("doc_type", "Unknown").strip()
        summary  = parsed.get("summary", "").strip()
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse analyse response: {raw}")
        doc_type, summary = "Unknown", raw
    logging.info(f"Analyse complete | doc_type={doc_type}")
    return doc_type, summary



def _run_date_extraction(
    text: str, filename: str,
    db: Chroma_database, chunker: Chunking, llm: LLM,
) -> tuple[dict, list]:
    chunks = chunker.sliding_window_chunking(text)
    if not chunks:
        raise ValueError(f"No chunks produced from '{filename}'")

    db.process_and_add_documents(chunks, filename)
    logging.info(f"Stored {len(chunks)} chunks for '{filename}'")

    context, sources = db.retrive_text(
        query=DATE_EXTRACTION_QUERY,
        filename=filename,
        n_results=N_RESULTS,
    )

    if not context:
        logging.warning("No chunks found for date extraction")
        return {
            "expiry_date": None, "activation_date": None,
            "other_dates": {}, "confidence": 0.0, "raw_dates_found": []
        }, []

    raw     = llm.chat(context=context, query=DATE_EXTRACTION_PROMPT)
    cleaned = raw.strip().strip("```").replace("json", "", 1).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse date extraction response: {raw}")
        result = {
            "expiry_date": None, "activation_date": None,
            "other_dates": {}, "confidence": 0.0, "raw_dates_found": []
        }

    result = normalise_dates(result)   
    logging.info(f"Dates extracted | expiry={result.get('expiry_date')} | confidence={result.get('confidence')}")
    return result, sources


def _run_rag(
    text: str, user_query: str, filename: str,
    db: Chroma_database, chunker: Chunking, llm: LLM,
) -> tuple[str, list]:
    chunks = chunker.sliding_window_chunking(text)
    if not chunks:
        raise ValueError(f"No chunks produced from '{filename}'")

    db.process_and_add_documents(chunks, filename)
    logging.info(f"Stored {len(chunks)} chunks for '{filename}'")

    context, sources = db.retrive_text(
        query=user_query,
        filename=filename,
        n_results=N_RESULTS,
    )

    if not context:
        logging.warning("No relevant chunks found")
        return "No relevant context found for this query.", []

    answer = llm.chat(context, user_query)
    logging.info("RAG answer generated")
    return answer, sources



async def _parallel_date_mode(
    text: str, filename: str,
    db: Chroma_database, chunker: Chunking, llm: LLM,
) -> DateResult:
    loop = asyncio.get_event_loop()
    logging.info("Mode: date extraction | Launching: analyse + date extraction")

    task_analyse = loop.run_in_executor(EXECUTOR, _run_analyse, llm, text)
    task_dates   = loop.run_in_executor(
        EXECUTOR, _run_date_extraction, text, filename, db, chunker, llm
    )

    (doc_type, summary), (date_result, sources) = await asyncio.gather(
        task_analyse, task_dates
    )

    return DateResult(
        doc_type        = doc_type,
        summary         = summary,
        expiry_date     = date_result.get("expiry_date"),
        activation_date = date_result.get("activation_date"),
        other_dates     = date_result.get("other_dates", {}),
        confidence      = date_result.get("confidence", 0.0),
        sources         = sources,
    )


async def _parallel_query_mode(
    text: str, user_query: str, filename: str,
    db: Chroma_database, chunker: Chunking, llm: LLM,
) -> QueryResult:
    loop = asyncio.get_event_loop()
    logging.info("Mode: user query | Launching: analyse + RAG")

    task_analyse = loop.run_in_executor(EXECUTOR, _run_analyse, llm, text)
    task_rag     = loop.run_in_executor(
        EXECUTOR, _run_rag, text, user_query, filename, db, chunker, llm
    )

    (doc_type, summary), (answer, sources) = await asyncio.gather(
        task_analyse, task_rag
    )

    return QueryResult(
        doc_type = doc_type,
        summary  = summary,
        LLM_Response   = answer,
        sources  = sources,
    )


def run(filepath: str, user_query: Optional[str] = None) -> None:
    doc      = load(filepath)
    filename = doc.source_path.name

    logging.info(
        f"Loaded '{filename}' | method='{doc.extraction_method}' | "
        f"ocr={doc.ocr_used} | chars={len(doc.text)}"
    )

    if not doc.text.strip():
        raise ValueError(f"No text extracted from '{filename}'")

    chunker = Chunking(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    db      = Chroma_database()
    llm     = LLM()

    if user_query:
        result = asyncio.run(
            _parallel_query_mode(doc.text, user_query, filename, db, chunker, llm)
        )

        print(f"\n=== Document Type ===\n{result.doc_type}")
        print(f"\n=== Summary ===\n{result.summary}")
        print(f"\n=== Answer ===\n{result.LLM_Response}")
        print(f"\n=== Sources ===\n{', '.join(result.sources)}")

        return result

    else:
        result = asyncio.run(
            _parallel_date_mode(doc.text, filename, db, chunker, llm)
        )
        print(f"\n=== Document Type ===\n{result.doc_type}")
        print(f"\n=== Summary ===\n{result.summary}")
        print(f"\n=== Expiry Date ===\n{result.expiry_date}")
        print(f"\n=== Activation Date ===\n{result.activation_date}")
        print(f"\n=== Other Dates ===\n{json.dumps(result.other_dates, indent=2)}")
        print(f"\n=== Confidence ===\n{result.confidence}")
        print(f"\n=== Sources ===\n{', '.join(result.sources)}")

        return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python main.py <filepath> Optional[query]")
        print(f"Supported types: {', '.join(supported_extensions())}")
        print('Example:  python main.py "docs/adhar card.pdf"')
        print('Example (query mode): python main.py "docs/adhar card.pdf" "What is the address?"')
        sys.exit(1)

    filepath   = sys.argv[1]
    user_query = sys.argv[2] if len(sys.argv) > 2 else None

    response = run(filepath=filepath, user_query=user_query)

    print("+"*30)
    print(response)