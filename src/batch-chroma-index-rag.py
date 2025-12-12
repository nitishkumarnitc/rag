#!/usr/bin/env python3
"""
rag_batch_chroma.py
Minimal, production-oriented RAG pipeline with:
 - parallel PDF extraction (pdfplumber + ThreadPoolExecutor)
 - chunking (langchain_text_splitters.RecursiveCharacterTextSplitter)
 - batched embeddings (sentence-transformers.SentenceTransformer)
 - persistent Chroma (chromadb, duckdb+parquet) with incremental adds
 - query API + saving CSV of responses

Requirements (install in your venv / environment):
pip install pdfplumber sentence-transformers chromadb langchain-text-splitters tqdm python-dotenv

Notes:
- On macOS M1/M2 set HF_DEVICE=mps
- On Linux with CUDA set HF_DEVICE=cuda
- If chromadb isn't desired, you can replace the chromadb section with FAISS or Chroma via LangChain
"""

import os
import csv
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party libs (assume installed)
import pdfplumber
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
# ---------------- Config (tweak these via env vars or here) ----------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "documents"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "db" / "chroma"))
OUT_CSV = Path(os.getenv("OUT_CSV", BASE_DIR / "FinalResponse_Chroma_Batch.csv"))

HF_DEVICE = os.getenv("HF_DEVICE", "cpu")        # "mps", "cuda", or "cpu"
MODEL_NAME = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 3000))        # larger -> fewer chunks
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))         # embeddings per batch
EMBEDDING_DIM = 384                                    # approx for all-MiniLM-L6-v2
TOP_K = int(os.getenv("TOP_K", 3))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 1200))
PDF_WORKERS = int(os.getenv("PDF_WORKERS", 3))         # parallel PDF extraction threads

USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")       # only needed if CALLING LLM

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag_batch_chroma")


# ---------------- Helpers ----------------
def extract_text_from_pdf(path: Path) -> List[Dict[str, Any]]:
    """
    Return list of documents (one entry per page) with metadata.
    """
    docs = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({
                        "page_content": text,
                        "metadata": {"source": str(path.name), "path": str(path), "page": i}
                    })
    except Exception as e:
        logger.warning("Failed to extract %s: %s", path, e)
    return docs


def load_documents(folder: Path = DOCS_DIR) -> List[Dict]:
    """
    Load PDFs and TXT files from folder in parallel, return list of {'page_content', 'metadata'}.
    """
    docs: List[Dict] = []
    if not folder.exists():
        logger.error("Documents folder not found: %s", folder)
        return docs

    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".pdf", ".txt")])
    logger.info("Found %d files in %s", len(files), folder)

    # Process txt files synchronously (cheap)
    for f in files:
        if f.suffix.lower() == ".txt":
            try:
                txt = f.read_text(encoding="utf-8")
            except Exception:
                txt = f.read_text(encoding="latin-1")
            docs.append({"page_content": txt, "metadata": {"source": f.name, "path": str(f), "page": 0}})

    # Process PDFs in parallel
    pdfs = [f for f in files if f.suffix.lower() == ".pdf"]
    if pdfs:
        logger.info("Extracting %d PDF(s) using %d workers...", len(pdfs), PDF_WORKERS)
        with ThreadPoolExecutor(max_workers=PDF_WORKERS) as ex:
            futures = {ex.submit(extract_text_from_pdf, p): p for p in pdfs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="PDF extract"):
                res = fut.result()
                if res:
                    docs.extend(res)

    logger.info("Loaded %d document pages/segments", len(docs))
    return docs


def chunk_documents(raw_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # ✅ Convert dicts → LangChain Documents
    lc_docs = [
        Document(
            page_content=d["page_content"],
            metadata=d.get("metadata", {})
        )
        for d in raw_docs
        if d.get("page_content")
    ]

    chunks = splitter.split_documents(lc_docs)

    # Normalize back to dicts (optional, but consistent)
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "id": f"chunk-{i}",
            "text": c.page_content,
            "metadata": c.metadata
        })

    return out

# ---------------- Embedding + Chroma builder ----------------

def build_chroma_collection(
    chunks,
    chroma_dir,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    batch_size=64
):
    logger.info("STEP 3 — build/append Chroma vector index at %s", chroma_dir)

    # ---- Embeddings (HF, free, local) ----
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )

    # ---- NEW Chroma client (FIX) ----
    client = chromadb.PersistentClient(path=chroma_dir)

    collection = client.get_or_create_collection(
        name="rag_docs",
        metadata={"hnsw:space": "cosine"}
    )

    # ---- Batch insert ----
    texts, metadatas, ids = [], [], []

    for i, c in enumerate(chunks):
        texts.append(c["text"])
        metadatas.append(c.get("metadata", {}))
        ids.append(c["id"])

        if len(texts) >= batch_size:
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embedder.embed_documents(texts)
            )
            texts, metadatas, ids = [], [], []

    if texts:
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embedder.embed_documents(texts)
        )

    logger.info("Chroma index build complete")
    return collection



# ---------------- Retrieval & optional LLM ----------------
def retrieve_topk(collection, query: str, k: int = TOP_K) -> List[Tuple[Dict, float]]:
    """
    Returns list of (metadata+text dict, score)
    """
    if collection.count() == 0:
        return []

    # chroma query returns dict with 'ids','distances','documents','metadatas'
    res = collection.query(query_texts=[query], n_results=k)
    # result format: each field is a list of lists (per query)
    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    out = []
    for doc_text, meta, dist in zip(documents, metadatas, distances):
        out.append(({"text": doc_text, "metadata": meta}, float(dist)))
    return out


def build_context_from_topk(topk: List[Tuple[Dict, float]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    used = 0
    for item, score in topk:
        text = item.get("text", "") or ""
        src = item.get("metadata", {}).get("source", "unknown")
        if not text:
            continue
        remain = max_chars - used
        if remain <= 0:
            break
        piece = text if len(text) <= remain else text[:remain].rsplit(" ", 1)[0]
        parts.append(f"[{src}] {piece}")
        used += len(piece)
    return "\n\n".join(parts)


# Optional LLM call — placeholder (use OpenAI or other LLM here)
def call_llm_stub(context: str, question: str) -> str:
    # Keep minimal: user can swap with ChatOpenAI / other LLM calls.
    if not context:
        return "No context found."
    return f"ANSWER (mock): Based on provided context, concise answer to: {question}"


# ---------------- Batch run + CSV ----------------
def run_pipeline_and_save(queries: List[str], docs_folder: Path = DOCS_DIR,
                          chroma_dir: Path = CHROMA_DIR, out_csv: Path = OUT_CSV):
    t0 = time.time()
    logger.info("STEP 1 — load documents from %s", docs_folder)
    raw_docs = load_documents(docs_folder)
    if not raw_docs:
        logger.error("No documents loaded — exiting.")
        return

    logger.info("STEP 2 — chunk documents (chunk_size=%d, overlap=%d)", CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunk_documents(raw_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logger.info("Created %d chunks", len(chunks))

    logger.info("STEP 3 — build/append Chroma vector index at %s", chroma_dir)
    collection = build_chroma_collection(chunks, chroma_dir, model_name=MODEL_NAME, device=HF_DEVICE, batch_size=BATCH_SIZE)

    logger.info("STEP 4 — run queries and write CSV")
    rows = []
    for q in queries:
        topk = retrieve_topk(collection, q, k=TOP_K)
        ctx = build_context_from_topk(topk)
        answer = call_llm_stub(ctx, q) if USE_LLM else (ctx or f"No context for {q}")
        # write top_docs as repr for test compatibility
        top_docs_repr = repr(tuple((t[0].get("metadata", {}).get("source", ""), float(t[1])) for t in topk))
        rows.append({"top_docs": top_docs_repr, "response": answer})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["top_docs", "response"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    logger.info("DONE — wrote %s (%d queries) in %.1fs", out_csv, len(queries), elapsed)


# ---------------- Main (example usage) ----------------
if __name__ == "__main__":
    example_queries = [
        "Give a one-paragraph summary of the document.",
        "List the main topics discussed.",
        "Find any mention of 'limited liability' and return the snippet.",
        "What numeric facts or statistics are present?",
        "Extract recommendations described in the report."
    ]

    run_pipeline_and_save(example_queries)
