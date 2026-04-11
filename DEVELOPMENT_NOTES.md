# LLM-Powered Analytics Assistant — Development Notes

## Project Overview
GUVI/HCL Final Project — LLM-Powered Analytics Assistant with RAG  
Dataset: Brazilian E-Commerce Public Dataset by Olist (~100k orders, 8 tables)  
Stack: Python, SQLite, FAISS, Sentence-Transformers, Ollama (codellama), Streamlit, Plotly

---

## Folder Structure

```
LLM-PoweredAnalytics/
├── DATA/                        # Raw Olist CSV files
├── database/                    # SQLite DB + FAISS index (git-ignored)
│   ├── olist.db
│   ├── faiss.index
│   └── chunks.pkl
├── rag/
│   ├── embedder.py              # Text chunking + embedding + FAISS index builder
│   └── retriever.py             # FAISS similarity search
├── sql/
│   └── nl_to_sql.py             # NL-to-SQL engine + SQL executor
├── llm/
│   ├── router.py                # Query classifier (SQL / RAG / HYBRID)
│   ├── sentiment.py             # Sentiment + complaint theme extractor
│   ├── synthesizer.py           # Hybrid answer combiner
│   └── chart_generator.py       # Auto Plotly chart generator
├── notebooks/
│   └── data.ipynb               # EDA + pipeline evaluation
├── olist_loader.py              # CSV → SQLite ingestion
├── app.py                       # Streamlit UI
├── requirements.txt
├── .env                         # API keys (git-ignored)
└── .gitignore
```

---

## .env File Structure

```
embedding_model=sentence-transformers/all-MiniLM-L6-v2
chunk_size=1000
chunk_overlap=200
model=codellama
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
HF_TOKEN=...
```

---

## Step-by-Step Build Guide

### Step 0 — Load Data into SQLite
```bash
python olist_loader.py
```
- Reads all 9 CSVs from `DATA/`
- Creates `database/olist.db` with 9 tables
- Uses relative paths via `os.path.dirname(os.path.abspath(__file__))`

---

### Step 1 — Build FAISS Index
```bash
python -m rag.embedder
```
- Reads `order_reviews` from SQLite
- Chunks review text (~1000 chars with 200 overlap)
- Embeds using `sentence-transformers/all-MiniLM-L6-v2`
- Saves `database/faiss.index` + `database/chunks.pkl`
- Output: `Embedding matrix shape: (40950, 384)` and `FAISS index size: 40950 vectors`

**Key design decisions:**
- `get_embedding(text: str)` — single string → single vector (used by retriever)
- `embed_documents(texts)` called directly inside `build_faiss_index()` for batch embedding
- Do NOT pass a list to `get_embedding()` — it uses `embed_query()` not `embed_documents()`

---

### Step 2 — NL-to-SQL Engine
```bash
python -m sql.nl_to_sql
```
- Uses Ollama `codellama` model
- Schema string passed in system prompt
- `nl_to_sql(question, retry_error=None)` — supports retry on SQL failure
- `run_sql(sql)` — returns DataFrame, catches exceptions gracefully
- `clean_sql(text)` — strips markdown code fences from LLM response

**Common LLM mistakes to guard against:**
- Hallucinating columns (e.g. `quantity` — does not exist in `order_items`)
- Wrong table aliases (e.g. `T3.price` when `price` is in `order_items` not `order_payments`)
- Wrapping SQL in markdown fences ` ```sql ``` `

**Schema notes:**
- Revenue = `SUM(order_items.price)` — no quantity column, each row = 1 item
- `price` is in `order_items`, `payment_value` is in `order_payments`

---

### Step 3 — Query Router
```bash
python -m llm.router
```
- Classifies question as `SQL`, `RAG`, or `HYBRID`
- Uses Ollama `codellama` model
- Safety fallback: extracts first valid word from response, defaults to `SQL`

**Routing logic:**
| Route | When to use |
|-------|------------|
| SQL | Numbers, counts, totals, averages, rankings |
| RAG | Reviews, sentiment, complaints, opinions |
| HYBRID | Needs both structured data AND review analysis |

---

### Step 4 — Sentiment Analysis
```bash
python -m llm.sentiment
```
- Retrieves top-5 FAISS chunks for the query
- Passes review text to LLM for analysis
- Returns `{"sentiment": "positive/negative/mixed", "themes": [...]}`
- `parse_sentiment()` extracts structured output from LLM response

---

### Step 5 — Synthesizer
```bash
python -m llm.synthesizer
```
- Used for HYBRID queries only
- Takes SQL result (as string) + RAG result (dict) + original question
- Returns 3-5 sentence business insight

---

### Step 6 — Chart Generator
```bash
python -m llm.chart_generator
```
- `pick_chart_type(df)` — asks LLM to choose `bar`, `line`, or `pie`
- `build_chart(df, question)` — auto-detects x (text) and y (numeric) columns
- Returns Plotly figure ready for `st.plotly_chart()`
- Defaults to `bar` if LLM returns unexpected output

---

### Step 7 — Streamlit App
```bash
streamlit run app.py
```
Three pipelines wired together:

| Route | Pipeline |
|-------|---------|
| SQL | `nl_to_sql` → `run_sql` → DataFrame + Plotly chart |
| RAG | `retriever.retrieve` → `analyse_sentiment` → sentiment + themes |
| HYBRID | SQL pipeline + RAG pipeline → `synthesize` → merged answer |

---

## Errors Encountered & Fixes

### 1. `ValueError: too many values to unpack` in FAISS search
**Cause:** Query embedding had 3 dimensions `(1, 1, 384)` instead of `(1, 384)`  
**Fix:** Use `.reshape(1, -1)` instead of `np.array([embedding])`

### 2. `AssertionError: d == self.d` in FAISS search
**Cause:** `get_embedding()` was calling `embed_documents()` (returns list of vectors) instead of `embed_query()` (returns single vector)  
**Fix:** Keep `get_embedding()` using `embed_query()` for single text; use `embedder.embed_documents()` directly in `build_faiss_index()`

### 3. `DatabaseError: no such column: T3.price`
**Cause:** LLM (llama3.2 / codellama) hallucinated wrong column names in JOINs  
**Fix:** Add retry mechanism — send error back to LLM for self-correction; add explicit schema notes about which columns belong to which table

### 4. LLM wraps SQL in markdown fences
**Cause:** codellama often responds with ` ```sql ... ``` ` blocks  
**Fix:** `clean_sql()` function strips code fences before executing

### 5. `IndentationError` on `def` keyword
**Cause:** Extra spaces before `def` at module level  
**Fix:** Function definitions at module level must start at column 0

### 6. `LangChainDeprecationWarning` for HuggingFaceEmbeddings
**Cause:** `langchain_community.embeddings.HuggingFaceEmbeddings` is deprecated  
**Fix:** Change import to `from langchain_huggingface import HuggingFaceEmbeddings`

---

## LLM Model Decisions

| Model tried | Result |
|-------------|--------|
| `llama3.2` (3B) | Too small — hallucinated columns, poor SQL |
| `codellama` | Better — still needs retry mechanism for complex joins |
| Claude Haiku | Best for SQL — but requires paid API |
| Groq (llama3-70b) | Best free option — 70B model, accurate SQL |

**Current setup:** `codellama` via Ollama (local, free, no internet needed)  
**Recommended for production:** Groq free API with `llama3-70b-8192`

---

## How to Run Full Pipeline from Scratch

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama
ollama serve
ollama pull codellama

# 3. Load data into SQLite
python olist_loader.py

# 4. Build FAISS index (takes 5-10 mins)
python -m rag.embedder

# 5. Run the app
streamlit run app.py
```

---

## Remaining Deliverables
- [ ] Complete `notebooks/data.ipynb` (EDA + evaluation)
- [ ] Deploy on Streamlit Community Cloud
- [ ] Add Google Drive link to dataset in README
- [ ] Record 3-5 min demo video (SQL query, RAG query, HYBRID query)
- [ ] Complete project report document
