# simplest_rag.py — Minimal RAG with OpenAI + FAISS (LangChain 1.x)

import os, csv, logging
from dotenv import load_dotenv
from typing import List, Tuple

# ---- LangChain Imports (Assume installed) ----
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3
MAX_CONTEXT = 1200
USE_LLM = True     # Set False during testing
# ----------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")

# ----------------------------------------------------
# 1. Load documents
#
def load_documents(folder=DOCS_DIR):
    docs = []
    for file in os.listdir(folder):
        full = os.path.join(folder, file)

        if file.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(full).load())
        elif file.lower().endswith(".txt"):
            docs.extend(TextLoader(full).load())

    logger.info("Loaded %d documents", len(docs))

    # --- CLEAN PRINT ---
    print("\n===== DOCUMENTS LOADED =====")
    for i, d in enumerate(docs, 1):
        preview = d.page_content[:80].replace("\n", " ")
        print(f"{i:02d}. {d.metadata.get('source', 'unknown')} → {preview}...")

    return docs

# ----------------------------------------------------
# 2. Chunk documents
# ----------------------------------------------------
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    logger.info("Created %d chunks", len(chunks))

    # --- CLEAN PRINT ---
    print("\n===== CHUNKS CREATED =====")
    for i, c in enumerate(chunks[:20], 1):  # show first 20 chunks only
        preview = c.page_content[:80].replace("\n", " ")
        print(f"{i:03d}. {preview}...  | source={c.metadata.get('source')}")

    if len(chunks) > 20:
        print(f"... ({len(chunks)-20} more chunks hidden)\n")

    return chunks



# ----------------------------------------------------
# 3. Build FAISS vector index
# ----------------------------------------------------
def build_index(chunks: List[Document]):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vstore = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built")
    return vstore


# ----------------------------------------------------
# 4. Retrieve top-k documents
# ----------------------------------------------------
def retrieve(vstore, query: str, k: int = TOP_K) -> List[Tuple[str, float, str]]:
    results = vstore.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in results:
        src = doc.metadata.get("source", "unknown")
        out.append((src, float(score), doc.page_content))
    return out


# ----------------------------------------------------
# 5. Build context string
# ----------------------------------------------------
def build_context(top_docs, limit=MAX_CONTEXT) -> str:
    ctx = ""
    for src, score, text in top_docs:
        if not text:
            continue
        space_left = limit - len(ctx)
        if space_left <= 0:
            break
        snippet = text[:space_left]
        ctx += f"[{src}] {snippet}\n\n"
    return ctx


# ----------------------------------------------------
# 6. Call LLM
# ----------------------------------------------------
def call_llm(context: str, question: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    prompt = f"Use ONLY the context to answer.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = llm.invoke(prompt)
    return str(response)


# ----------------------------------------------------
# 7. Batch processing + save CSV
# ----------------------------------------------------
def run(vstore, queries, out_csv="src/FinalResponseOpenAI.csv"):
    rows = []
    for q in queries:
        top_docs = retrieve(vstore, q)
        context = build_context(top_docs)
        if USE_LLM:
            answer = call_llm(context, q)
        else:
            answer = context or "No context found."
        rows.append({"query": q, "answer": answer})

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "answer"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved results to %s", out_csv)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    queries = [
        "What is nitish kumar?",
        "Top Skills of Nitish Kumar"
    ]

    docs = load_documents()
    chunks = chunk_documents(docs)
    # vstore = build_index(chunks)
    # run(vstore, queries)
    print("DONE")
