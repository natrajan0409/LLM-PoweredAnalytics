# LLM-Powered Analytics Assistant — Development Notes

## Project Overview
**GUVI/HCL Final Project** — LLM-Powered Analytics Assistant with RAG  
**Dataset:** Brazilian E-Commerce Public Dataset by Olist (~100k orders, 8 tables, ~40k reviews)  
**Stack:** Python · SQLite · FAISS · Sentence-Transformers · Ollama (codellama) · Streamlit · Plotly  
**Date:** April 2026

---

## Folder Structure

```
LLM-PoweredAnalytics/
├── DATA/                        # Raw Olist CSV files (git-ignored)
├── database/                    # SQLite DB + FAISS index (git-ignored)
│   ├── olist.db                 # 9-table SQLite database
│   ├── faiss.index              # FAISS vector index (40,950 vectors)
│   └── chunks.pkl               # Review chunk texts + order_id metadata
├── rag/
│   ├── embedder.py              # Text chunking + embedding + FAISS index builder
│   └── retriever.py             # FAISS similarity search
├── sql/
│   └── nl_to_sql.py             # NL-to-SQL engine + SQL executor + retry logic
├── llm/
│   ├── router.py                # Query classifier (SQL / RAG / HYBRID)
│   ├── sentiment.py             # Sentiment + complaint theme extractor
│   ├── synthesizer.py           # Hybrid answer combiner
│   └── chart_generator.py       # Auto Plotly chart generator
├── notebooks/
│   └── data.ipynb               # EDA + pipeline evaluation
├── .streamlit/
│   └── config.toml              # Disables file watcher (fixes torchvision warnings)
├── olist_loader.py              # CSV → SQLite ingestion
├── app.py                       # Streamlit UI (tabbed: Chat + Architecture flowchart)
├── requirements.txt
├── .env                         # API keys — git-ignored
├── .gitignore
├── README.md
└── DEVELOPMENT_NOTES.md         # This file
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

## Application Flow

```
User Question
     │
     ▼
Query Router (codellama LLM)
     │
     ├──── SQL ────► nl_to_sql() ──► clean_sql() ──► run_sql()
     │                                                    │
     │                                              on error: retry
     │                                                    │
     │                                             DataFrame ──► build_chart() ──► Plotly fig
     │
     ├──── RAG ────► get_embedding() ──► FAISS.search() ──► analyse_sentiment()
     │                                                            │
     │                                               sentiment + themes dict
     │
     └── HYBRID ──► Both SQL + RAG run ──► synthesize() ──► Combined insight
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
- Fixed bug: wrong CSV filename for product category (`product_category_name_translation.csv`)

---

### Step 1 — Build FAISS Index
```bash
python -m rag.embedder
```
- Reads `order_reviews` from SQLite — filters nulls and empty strings
- Chunks review text (~1000 chars with 200 overlap) keeping `order_id` metadata
- Embeds using `sentence-transformers/all-MiniLM-L6-v2` via `embed_documents()`
- Builds `faiss.IndexFlatL2(384)` — L2 distance, 384 dimensions
- Saves `database/faiss.index` + `database/chunks.pkl`
- **Output:** `Embedding matrix shape: (40950, 384)` · `FAISS index size: 40950 vectors`

**Key design decisions:**
- `get_embedding(text: str)` → uses `embed_query()` — single string → single vector (used by retriever)
- `embed_documents(texts)` called directly inside `build_faiss_index()` — batch embedding (faster)
- Never pass a list to `get_embedding()` — it only handles one string

---

### Step 2 — NL-to-SQL Engine
```bash
python -m sql.nl_to_sql
```
- Uses Ollama `codellama` model
- Full schema string embedded in system prompt
- `nl_to_sql(question, retry_error=None)` — sends error back to LLM for self-correction on failure
- `run_sql(sql)` — returns DataFrame, catches all exceptions gracefully (never crashes)
- `clean_sql(text)` — strips markdown code fences ` ```sql ``` ` from LLM response

**Schema hints added to prompt:**
- `price` lives in `order_items` — NOT in `order_payments`
- No `quantity` column — each row in `order_items` = 1 unit; revenue = `SUM(price)`

---

### Step 3 — Query Router
```bash
python -m llm.router
```
- Classifies question as `SQL`, `RAG`, or `HYBRID`
- Uses Ollama `codellama` — strict one-word reply prompt
- Safety fallback: scans response words for valid route, defaults to `SQL`

| Route | Triggered by |
|-------|-------------|
| SQL | Numbers, counts, totals, averages, rankings, dates |
| RAG | Reviews, sentiment, complaints, opinions, feedback |
| HYBRID | Needs both structured data AND review analysis |

**Test results (all 5 correct):**
```
SQL      ← What are the top 5 product categories by revenue?
RAG      ← What do customers complain about most?
RAG      ← What are the reviews saying about top selling products?
SQL      ← How many orders were delivered late?
RAG      ← What is the sentiment around electronics products?
```

---

### Step 4 — Sentiment Analysis
```bash
python -m llm.sentiment
```
- Retrieves top-5 FAISS chunks for the query text
- Joins chunk texts and passes to `codellama` with structured output prompt
- `parse_sentiment()` extracts sentiment label and theme list from response
- Returns `{"sentiment": "positive/negative/mixed", "themes": ["...", "...", "..."]}`

**Test output:**
```
Retrieved chunks: 5
Sentiment: mixed
Themes:
 - Received incorrect product
 - Poor customer service
 - Credit issue
```

---

### Step 5 — Synthesizer
```bash
python -m llm.synthesizer
```
- Used for HYBRID queries only
- Accepts: `question`, `sql_result` (string), `rag_result` (dict)
- Builds a combined context block and asks LLM for a 3–5 sentence business insight
- Keeps both structured numbers and review themes in one coherent answer

---

### Step 6 — Chart Generator
```bash
python -m llm.chart_generator
```
- `pick_chart_type(df)` — passes column names + sample rows to LLM, gets `bar`/`line`/`pie`
- `build_chart(df, question)` — auto-detects x (first text column) and y (first numeric column)
- Dark theme applied: `paper_bgcolor="#1e2130"`, `plot_bgcolor="#1e2130"`
- Returns `None` if no numeric column found (safe — app checks before rendering)
- Defaults to `bar` if LLM returns unexpected output

---

### Step 7 — Streamlit App
```bash
streamlit run app.py
```

**UI Features:**
- Dark theme via custom CSS (`#0f1117` background)
- Two tabs: `💬 Ask Assistant` + `🗺️ App Architecture`
- Sidebar: how-it-works cards, sample questions, dataset info
- Colour-coded route badges (blue=SQL, green=RAG, orange=HYBRID)
- `st.status()` live progress indicators per pipeline
- Sentiment displayed as colour-coded cards (green/red/orange)
- Hybrid layout: side-by-side columns for SQL data + review sentiment
- Architecture tab: interactive Graphviz flowchart of all pipelines

**Three pipelines wired together:**

| Route | Pipeline |
|-------|---------|
| SQL | `nl_to_sql` → `run_sql` → retry on error → DataFrame + `build_chart` |
| RAG | `retriever.retrieve` → `analyse_sentiment` → sentiment + themes + excerpts |
| HYBRID | Both above → `synthesize` → combined insight card |

---

## Errors Encountered & Fixes

### 1. `ValueError: too many values to unpack` in FAISS search
**Cause:** Query embedding had 3 dimensions `(1, 1, 384)` instead of `(1, 384)`  
**Fix:** Changed `np.array([embedding])` → `np.array(embedding).reshape(1, -1)`

### 2. `AssertionError: d == self.d` in FAISS search
**Cause:** `get_embedding()` was changed to call `embed_documents()` which returns a list of vectors — wrong type for single query lookup  
**Fix:** Restored `get_embedding()` to use `embed_query()` (single vector); batch embedding handled separately in `build_faiss_index()`

### 3. `DatabaseError: no such column: T3.price`
**Cause:** `codellama` hallucinated wrong column in JOIN alias  
**Fix:** Added retry mechanism (sends error to LLM for correction) + explicit schema hints in prompt about which columns belong to which table

### 4. LLM wraps SQL in markdown fences
**Cause:** `codellama` responds with ` ```sql ... ``` ` blocks instead of raw SQL  
**Fix:** `clean_sql()` strips code fences before executing

### 5. `TypeError: nl_to_sql() got unexpected keyword 'return_error'`
**Cause:** Typo in `app.py` — used `return_error` instead of `retry_error`  
**Fix:** Corrected kwarg name to `retry_error`

### 6. `ModuleNotFoundError: No module named 'torchvision'` flood in Streamlit
**Cause:** Streamlit file watcher scans all `transformers` model files, many need `torchvision`  
**Fix:** Created `.streamlit/config.toml` with `fileWatcherType = "none"`

### 7. `IndentationError` on `def` keyword
**Cause:** Extra leading spaces before `def` at module level  
**Fix:** Module-level functions must start at column 0

### 8. `LangChainDeprecationWarning` for HuggingFaceEmbeddings
**Cause:** `langchain_community.embeddings.HuggingFaceEmbeddings` deprecated since LangChain 0.2.2  
**Fix:** Changed import to `from langchain_huggingface import HuggingFaceEmbeddings`

### 9. `os.path("model")` crash in nl_to_sql.py
**Cause:** `os.path` is a module, not a callable function  
**Fix:** Changed to `os.getenv("model", "codellama")`

---

## LLM Model Decisions

| Model | Parameters | SQL Quality | Notes |
|-------|-----------|-------------|-------|
| `llama3.2` | 3B | Poor | Hallucinated columns, wrong JOIN aliases |
| `codellama` | 7B | Acceptable | Needs retry + prompt hints for complex joins |
| Claude Haiku | — | Excellent | Requires paid Anthropic API — token limit exceeded |
| Gemini Flash | — | Excellent | Requires Google subscription |
| Groq `llama3-70b` | 70B | Best free | Free API, 5-min signup, no credit card |

**Current setup:** `codellama` via Ollama — local, free, works offline  
**Recommended upgrade:** Groq free API → `llama3-70b-8192` for production-quality SQL

---

## UI Design Decisions

| Decision | Reason |
|----------|--------|
| Dark theme (`#0f1117`) | Matches analytics/dashboard aesthetic |
| Tabbed layout (Chat + Architecture) | Keeps UI clean while showing app design |
| `st.status()` instead of spinner | Shows live step-by-step progress |
| Graphviz flowchart in app | Built-in Streamlit — no extra package needed |
| Sidebar with sample questions | Helps evaluators test all 3 pipeline routes quickly |
| Colour-coded sentiment cards | Instant visual — green/red/orange without reading |

---

## How to Run Full Pipeline from Scratch

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama and pull model
ollama serve
ollama pull codellama

# 3. Place CSVs in DATA/ folder

# 4. Load data into SQLite
python olist_loader.py

# 5. Build FAISS index (5–10 mins first time)
python -m rag.embedder

# 6. Run the app
streamlit run app.py
```

---

## Remaining Deliverables

- [x] `olist_loader.py` — data ingestion
- [x] `rag/embedder.py` — FAISS index builder
- [x] `rag/retriever.py` — FAISS search
- [x] `sql/nl_to_sql.py` — NL-to-SQL engine
- [x] `llm/router.py` — query classifier
- [x] `llm/sentiment.py` — sentiment analysis
- [x] `llm/synthesizer.py` — hybrid combiner
- [x] `llm/chart_generator.py` — auto chart
- [x] `app.py` — Streamlit UI with dark theme + flowchart tab
- [x] `README.md` — full documentation
- [ ] `notebooks/data.ipynb` — EDA + pipeline evaluation
- [ ] Deploy on Streamlit Community Cloud
- [ ] Add Google Drive dataset link to README
- [ ] Record 3–5 min demo video (SQL, RAG, HYBRID)
- [ ] Complete project report document
