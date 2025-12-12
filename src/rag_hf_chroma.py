# rag_hf_chroma_batch.py
"""
Minimal, robust RAG pipeline using HuggingFaceEmbeddings + Chroma (batch indexing).
- Uses langchain_huggingface.HuggingFaceEmbeddings (avoids deprecation).
- Indexes in batches (safe for millions of chunks).
- Prints concise progress logs and small previews for docs/chunks.
- Disables Chroma telemetry for clean logs.

Usage:
    python rag_hf_chroma_batch.py

Configure via environment (.env):
    OPENAI_API_KEY (optional, if USE_LLM True)
    HF_DEVICE ("cpu", "mps", "cuda") optional; default "cpu"
"""

import os
import csv
import time
import logging
from dotenv import load_dotenv
from typing import List, Tuple

# ----- IMPORTANT: Assume these packages installed -----
# pip install langchain langchain-huggingface langchain-community chromadb sentence-transformers pypdf python-dotenv
# ----------------------------------------------------

# LangChain / embeddings / vectorstore imports (assume installed)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use the new huggingface wrapper (no deprecation warning)
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma vectorstore wrapper (langchain-community)
from langchain_community.vectorstores import Chroma

# Optional LLM (only if USE_LLM=True)
from langchain_openai import ChatOpenAI

# ---------------- CONFIG ----------------
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 2000))        # larger to reduce chunk count
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150))
TOP_K = int(os.environ.get("TOP_K", 3))
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT", 1200))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1000))       # tune to memory
USE_LLM = os.environ.get("USE_LLM", "False").lower() in ("1", "true", "yes")
HF_MODEL = os.environ.get("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_DEVICE = os.environ.get("HF_DEVICE", "cpu")             # "cpu", "mps", or "cuda"
CHROMA_DIR = os.environ.get("CHROMA_DIR", "db/chroma")
OUT_CSV = os.environ.get("OUT_CSV", "FinalResponseHFChroma.csv")
DOCS_DIR = os.environ.get("DOCS_DIR", "documents")
# ----------------------------------------

# disable chroma telemetry for cleaner logs
os.environ.setdefault("CHROMA_TELEMETRY", "0")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag")

# load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # used only if USE_LLM True

# ---------------- HELPERS ----------------
def print_section(title: str):
    print("\n" + title)
    print("=" * max(len(title), 40))


# 1) LOAD documents
def load_documents(folder: str = DOCS_DIR) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(folder):
        logger.warning("Documents folder not found: %s", folder)
        return docs

    logger.info("STEP 1 — LOADING DOCUMENTS from %s", folder)
    start = time.time()
    filenames = sorted(os.listdir(folder))
    for fname in filenames:
        full = os.path.join(folder, fname)
        if not os.path.isfile(full):
            continue
        try:
            if fname.lower().endswith(".pdf"):
                loaded = PyPDFLoader(full).load()
                docs.extend(loaded)
            elif fname.lower().endswith(".txt"):
                loaded = TextLoader(full).load()
                docs.extend(loaded)
            else:
                # attempt text loader for other text-like files
                loaded = TextLoader(full).load()
                docs.extend(loaded)
        except Exception as e:
            logger.warning("Failed to load %s: %s", full, e)

    elapsed = time.time() - start
    logger.info("Loaded %d documents in %.2fs", len(docs), elapsed)

    # Print small preview for clarity
    print_section("DOCUMENTS LOADED (preview)")
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        preview = (d.page_content or "").replace("\n", " ")[:120]
        print(f"{i:03d}. {src} → {preview}{'...' if len(preview) == 120 else ''}")
        if i >= 30:
            print(f"... ({len(docs)-30} more documents hidden)")
            break

    return docs


# 2) CHUNK documents
def chunk_documents(docs: List[Document]) -> List[Document]:
    logger.info("STEP 2 — CHUNKING documents (chunk_size=%d, overlap=%d)", CHUNK_SIZE, CHUNK_OVERLAP)
    start = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    elapsed = time.time() - start
    logger.info("Created %d chunks in %.2fs", len(chunks), elapsed)

    # print a compact preview (first 20)
    print_section("CHUNKS CREATED (first 20)")
    for i, c in enumerate(chunks[:20], start=1):
        preview = (c.page_content or "").replace("\n", " ")[:120]
        src = c.metadata.get("source", "unknown")
        print(f"{i:03d}. source={src} | {preview}{'...' if len(preview) == 120 else ''}")
    if len(chunks) > 20:
        print(f"... ({len(chunks)-20} more chunks hidden)")

    return chunks


# 3) BUILD Chroma index in batches
def build_chroma_index(chunks: List[Document], persist_dir: str = CHROMA_DIR):
    logger.info("STEP 3 — BUILDING CHROMA VECTOR INDEX")
    start_total = time.time()

    # init HF embeddings (force device via model_kwargs)
    logger.info("Initializing HuggingFaceEmbeddings model=%s device=%s", HF_MODEL, HF_DEVICE)
    hf = HuggingFaceEmbeddings(
        model_name=HF_MODEL,
        model_kwargs={"device": HF_DEVICE},
    )

    os.makedirs(persist_dir, exist_ok=True)

    vstore = None
    created = 0
    start = time.time()

    # index in batches to avoid memory spikes
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_start = time.time()
        logger.info("Indexing batch %d — chunks %d..%d (size=%d)", (i // BATCH_SIZE) + 1, i + 1, i + len(batch), len(batch))

        if vstore is None:
            # first batch creates collection
            vstore = Chroma.from_documents(batch, hf, persist_directory=persist_dir, collection_name="default")
            created += len(batch)
        else:
            # prefer add_documents if available
            try:
                vstore.add_documents(batch)
                created += len(batch)
            except Exception as e:
                # If add_documents not available or failed, re-open collection and try
                logger.warning("vstore.add_documents failed: %s — attempting to reopen/add", e)
                try:
                    tmp = Chroma(persist_directory=persist_dir, embedding=hf, collection_name="default")
                    tmp.add_documents(batch)
                    # assign back to vstore to keep same client wrapper
                    vstore = tmp
                    created += len(batch)
                except Exception as e2:
                    logger.error("Failed to append batch to Chroma: %s", e2)
                    raise

        batch_elapsed = time.time() - batch_start
        logger.info("Batched indexed %d chunks in %.2fs (total indexed=%d)", len(batch), batch_elapsed, created)

    # persist defensively
    try:
        vstore.persist()
    except Exception:
        pass

    total_elapsed = time.time() - start_total
    logger.info("Chroma index built: total chunks indexed=%d in %.2fs", created, total_elapsed)
    return vstore


# 4) RETRIEVE top-k
def retrieve(vstore, query: str, k: int = TOP_K) -> List[Tuple[str, float, str]]:
    results = vstore.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in results:
        src = doc.metadata.get("source") or doc.metadata.get("source_document") or "unknown"
        out.append((src, float(score), doc.page_content or ""))
    return out


# 5) BUILD context (concat top-k snippets up to MAX_CONTEXT chars)
def build_context(top_docs: List[Tuple[str, float, str]], limit: int = MAX_CONTEXT) -> str:
    parts = []
    used = 0
    for src, score, text in top_docs:
        if not text:
            continue
        remain = limit - used
        if remain <= 0:
            break
        piece = text if len(text) <= remain else text[:remain].rsplit(" ", 1)[0]
        parts.append(f"[{src}] {piece}")
        used += len(piece)
    return "\n\n".join(parts)


# 6) OPTIONAL LLM call (uses OpenAI Chat wrapper)
def call_llm(context: str, question: str) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set. Cannot call LLM."
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
    prompt = f"Use ONLY the context to answer concisely.\n\nContext:\n{context}\n\nQuestion: {question}"
    resp = llm.invoke(prompt)
    return str(resp)


# 7) RUN queries and write CSV
def run(vstore, queries: List[str], out_csv: str = OUT_CSV):
    logger.info("STEP 4 — RUNNING %d queries and writing CSV to %s", len(queries), out_csv)
    rows = []
    for q in queries:
        topk = retrieve(vstore, q)
        top_docs_repr = repr(tuple((t[0], float(t[1])) for t in topk))
        ctx = build_context(topk)
        if USE_LLM:
            try:
                ans = call_llm(ctx, q)
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
                ans = f"LLM failed: {e}\n\nContext:\n{ctx}"
        else:
            ans = ctx or f"No context found for query: {q}"
        rows.append({"top_docs": top_docs_repr, "response": ans})

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["top_docs", "response"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), out_csv)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # SIMPLE INFORMATION RETRIEVAL
    print("RAG PIPELINE STARTED")

    # sample_queries = [
    #     "What is the main topic of the document?",
    #     "Summarize the key points in one paragraph.",
    #     "List any definitions found in the document.",
    #     "What problem does this document try to solve?",
    #     "Give a short summary in bullet points."
    # ]
    #FACT & NUMBER EXTRACTION
    sample_queries = [
        "What numeric values or statistics are mentioned?",
        "Extract all dates referenced in the document.",
        "What financial figures appear in the text?",
        "List the percentages mentioned along with their context.",
        "What are the top three key metrics discussed?"
    ]

    # DEEP RETRIEVAL / ANALYSIS
    # sample_queries = [
    #     "Explain the core argument made in the document.",
    #     "What assumptions does the document rely on?",
    #     "What are the risks highlighted by the author?",
    #     "Compare two key ideas discussed in the document.",
    #     "What recommendations does the document propose?"
    # ]

    # RAG STRESS TEST (SEARCHING HARD TERMS)
    # sample_queries = [
    #     "Give a detailed explanation of the concept of income measurement.",
    #     "What challenges are discussed regarding implementation?",
    #     "What steps are proposed to handle these challenges?",
    #     "Identify three important sections relevant to policy analysts.",
    #     "Summarize the main conclusions of the report."
    # ]

    # MULTI-CHUNK RETRIEVAL (GOOD FOR CHROMA/FAISS TESTING)
    # sample_queries = [
    #     "Find anything related to ‘limited liability’.",
    #     "Return text containing the phrase ‘operational efficiency’.",
    #     "Which part discusses economic uncertainties?",
    #     "What does the document say about climate change?",
    #     "Find statements related to government spending."
    # ]

    # TEXTBOOK / EDUCATION STYLE QUERIES
    # sample_queries = [
    #     "Return the two most relevant text chunks about entrepreneurship.",
    #     "Show the closest chunks to the idea of ‘financial stability’.",
    #     "Extract all paragraphs related to technology or innovation.",
    #     "Find three chunks that explain regulatory challenges.",
    #     "Retrieve content mentioning future projections or forecasts."
    # ]

    # sample_queries = [
    #     "Define the main concept discussed in the chapter.",
    #     "Explain the advantages and disadvantages of the topic.",
    #     "Give examples mentioned in the material.",
    #     "Summarize the theoretical framework provided.",
    #     "Describe any diagrams or figures referenced."
    # ]

    print_section("RAG RUN — HF + CHROMA (batch)")
    docs = load_documents()
    if not docs:
        logger.error("No documents found at %s. Place PDFs/TXT under that folder and re-run.", DOCS_DIR)
        raise SystemExit(1)

    chunks = chunk_documents(docs)
    if not chunks:
        logger.error("No chunks created. Check CHUNK_SIZE and input documents.")
        raise SystemExit(1)

    vstore = build_chroma_index(chunks, persist_dir=CHROMA_DIR)
    run(vstore, sample_queries, out_csv=OUT_CSV)
    print("\nDONE — results written to:", OUT_CSV)

