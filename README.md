

## RAG (Retrieval-Augmented Generation) — Code-Level Learning

### 1. **RAG is a pipeline, not a single model**

In code, RAG is always split into **clear stages**:

```text
Load → Chunk → Embed → Index → Retrieve → Prompt → Generate
```

Each stage is **replaceable and testable**.

---

### 2. **Document loading must be streaming & incremental**

**Learning**

* Never load huge files fully into memory
* Use directory loaders + lazy iteration

**Code idea**

```python
for doc in loader.lazy_load():
    process(doc)
```

---

### 3. **Chunking controls recall vs latency**

**Key trade-off**

* Small chunks → better recall, slower
* Large chunks → faster, less precise

**Best practice**

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
```

---

### 4. **Embeddings are the most expensive step**

**Learning**

* Embedding time dominates ingestion
* Must be:

  * Batched
  * Cached
  * Device-aware (CPU / MPS / GPU)

**Code pattern**

```python
embeddings = embedder.embed_documents(batch)
```


### 6. **Retrieval should be Top-K + filtering**

**Baseline**

```python
docs = retriever.get_relevant_documents(query)
```

**Better**

* Top-K
* Metadata filters
* Score threshold

---

### 7. **Hybrid search improves quality**

**Learning**

* Vector search alone misses keyword-heavy queries
* Combine:

  * BM25 (keyword)
  * Vector similarity

**Conceptual code**

```python
results = merge(bm25_results, vector_results)
```

---

### 8. **Prompt = instructions + context + question**

**Correct structure**

```text
SYSTEM: You are a domain expert
CONTEXT: <retrieved chunks>
QUESTION: <user query>
```

**Rule**

* Never let the model answer without context

---

### 9. **Hallucination control is enforced in code**

**Learning**

* Hallucination is prevented by:

  * Retrieval
  * Prompt constraints
  * Fallback logic

**Example**

```python
if not docs:
    return "I don’t have enough information."
```

---

### 10. **Index rebuild must be automated**

**Production rule**

* Index is **reproducible**
* Code > data

**Startup logic**

```python
def init_rag():
    if not vector_db_exists():
        ingest_docs()
```

---

### 11. **Observability matters**

**Track**

* Retrieval latency
* Chunk hit-rate
* Token usage
* Answer confidence

---

