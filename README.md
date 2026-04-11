# 🛒 LLM-Powered Analytics Assistant with RAG

> Ask questions about Olist e-commerce data in plain English — no SQL required.

**GUVI / HCL Final Project** · Retail / E-Commerce Analytics Domain

---

## 📌 Overview

Business analysts spend significant time writing SQL queries and manually reading through thousands of customer reviews to extract insights. This project eliminates that friction.

The **LLM-Powered Analytics Assistant** enables any business user to:
- Query a structured e-commerce database using natural language
- Analyse unstructured customer review text for sentiment and themes
- Get combined insights that merge structured data with review analysis

Built on the **Brazilian Olist E-Commerce Dataset** (~100K orders, 8 relational tables, ~40K customer reviews).

---

## 🏗️ Application Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Query Router   │  ← LLM Classifier
                    │ (codellama LLM) │    (SQL / RAG / HYBRID)
                    └────────┬────────┘
           ┌─────────────────┼──────────────────┐
           │                 │                  │
      SQL Route         RAG Route          HYBRID Route
           │                 │                  │
    ┌──────▼──────┐   ┌──────▼──────┐    Both pipelines
    │  NL-to-SQL  │   │ Embed Query │    run in parallel
    │ (codellama) │   │ (MiniLM-L6) │          │
    └──────┬──────┘   └──────┬──────┘          │
           │                 │                  │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │ Execute SQL │   │ FAISS Search│   │ Synthesizer │
    │  (SQLite)   │   │  Top-5 docs │   │ (LLM merge) │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                  │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │ Auto Chart  │   │  Sentiment  │   │  Combined   │
    │  (Plotly)   │   │  + Themes   │   │   Answer    │
    └─────────────┘   └─────────────┘   └─────────────┘
```

### Three Core Pipelines

| Route | Triggered When | Output |
|-------|---------------|--------|
| **SQL** | Questions about numbers, counts, totals, rankings | DataFrame + auto Plotly chart |
| **RAG** | Questions about reviews, sentiment, complaints | Sentiment label + top 3 themes |
| **HYBRID** | Questions needing both data and reviews | Merged business insight |

---

## 🗂️ Project Structure

```
LLM-PoweredAnalytics/
├── DATA/                        # Raw Olist CSV files (git-ignored)
├── database/                    # SQLite DB + FAISS index (git-ignored)
│   ├── olist.db                 # 9-table SQLite database
│   ├── faiss.index              # FAISS vector index (40k vectors)
│   └── chunks.pkl               # Review chunk metadata
├── rag/
│   ├── embedder.py              # Text chunking + embedding + FAISS builder
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
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── olist_loader.py              # CSV → SQLite ingestion
├── app.py                       # Streamlit UI
├── requirements.txt
├── .env                         # API keys (git-ignored)
└── .gitignore
```

---

## 📊 Dataset

| Field | Detail |
|-------|--------|
| **Name** | Brazilian E-Commerce Public Dataset by Olist |
| **Source** | Kaggle (uploaded to Google Drive per guidelines) |
| **Size** | ~100,000 orders · 8 relational tables · ~40,000 review records |
| **Period** | 2016 – 2018 |
| **Format** | CSV files loaded into SQLite |

### Tables

| Table | Description |
|-------|-------------|
| `orders` | Order lifecycle — status, purchase date, delivery date |
| `order_items` | Product-level line items with price and freight value |
| `order_reviews` | Customer-written review text and star ratings (1–5) |
| `order_payments` | Payment type and instalment details |
| `products` | Product metadata including category |
| `sellers` | Seller location and ID |
| `customers` | Customer location and ID |
| `product_category` | Category name translations (Portuguese → English) |

> **Google Drive Dataset Link:** _[Add your link here]_

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Ollama `codellama` (local, free) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (`IndexFlatL2`, 384 dimensions) |
| **Database** | SQLite via pandas |
| **NLP Framework** | LangChain + LangChain-HuggingFace |
| **UI** | Streamlit |
| **Charts** | Plotly Express |
| **Data** | Pandas, NumPy |

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- `codellama` model pulled

### Step 1 — Clone and install dependencies
```bash
git clone https://github.com/natrajan0409/LLM-PoweredAnalytics.git
cd LLM-PoweredAnalytics
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Step 2 — Set up environment variables
Create a `.env` file in the project root:
```
embedding_model=sentence-transformers/all-MiniLM-L6-v2
chunk_size=1000
chunk_overlap=200
model=codellama
```

### Step 3 — Download and place the dataset
Download the Olist dataset CSVs and place them in the `DATA/` folder:
```
DATA/
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_orders_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

### Step 4 — Start Ollama and pull model
```bash
ollama serve
ollama pull codellama
```

### Step 5 — Load data into SQLite
```bash
python olist_loader.py
```

### Step 6 — Build FAISS index
```bash
python -m rag.embedder
```
> This takes 5–10 minutes on first run. Saves `database/faiss.index` and `database/chunks.pkl`.

### Step 7 — Launch the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 💡 Example Queries

### SQL Route
```
What are the top 5 product categories by total revenue?
How many orders were delivered late?
Which sellers have the highest average review score?
What is the average order value by payment type?
```

### RAG Route
```
What do customers complain about most?
What is the overall sentiment of customer reviews?
What are the top themes in negative reviews?
```

### Hybrid Route
```
What are the reviews saying about top-selling products?
Which product categories have the worst customer sentiment?
What do customers say about late deliveries?
```

---

## 🔄 Pipeline Details

### Query Router
A lightweight LLM classifier routes every question to one of three pipelines:
- **SQL** — structured data questions (numbers, dates, rankings)
- **RAG** — unstructured review questions (sentiment, complaints)
- **HYBRID** — questions requiring both data types

### SQL Pipeline
1. Schema-aware system prompt sent to `codellama`
2. LLM generates SQLite SQL query
3. Query executed via `pandas.read_sql_query()`
4. If SQL fails → error sent back to LLM for self-correction (retry once)
5. Result DataFrame passed to chart generator
6. LLM picks best chart type (bar / line / pie) → Plotly renders it

### RAG Pipeline
1. User query embedded using `all-MiniLM-L6-v2`
2. Top-5 most similar review chunks retrieved from FAISS index
3. Chunks passed to LLM for sentiment classification and theme extraction
4. Returns: sentiment label + top 3 complaint themes + source excerpts

### Hybrid Pipeline
1. Both SQL and RAG pipelines run on the same question
2. SQL result (DataFrame) + RAG result (sentiment dict) passed to Synthesizer
3. Synthesizer LLM merges both into a 3–5 sentence business insight

---

## 📈 Project Evaluation

| Section | Marks | Criteria |
|---------|-------|----------|
| Pipeline Correctness | 40 | SQL + Routing + RAG — core functionality |
| Output Quality | 35 | Sentiment + Hallucination + Charts + Completeness |
| System Performance | 10 | Latency + Stability |
| Code, Docs & Deployment | 15 | Structure + GitHub + Live URL + README |

---

## 🚀 Deployment

> **Live App:** _[Add Streamlit Community Cloud URL here]_

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## 📁 Deliverables

- [x] GitHub Repository with clean folder structure
- [x] `requirements.txt`
- [x] `README.md` with setup instructions
- [ ] Streamlit app deployed on Streamlit Community Cloud
- [ ] Jupyter Notebook with EDA and evaluation
- [ ] Google Drive link to dataset
- [ ] Short demo video (3–5 minutes)
- [ ] Completed project report document
