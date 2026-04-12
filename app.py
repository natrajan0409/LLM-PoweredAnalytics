import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── Download database files from Google Drive if missing ───────────────────────
def download_database():
    import gdown
    db_dir = os.path.join(BASE_DIR, "database")
    os.makedirs(db_dir, exist_ok=True)

    files = {
        "chunks.pkl":  "1ArLG_E49bubP0qqvacklJz-rkiSyC8W_",
        "faiss.index": "1Q3AWa8sHjA98J9MxA3ZThDjJ7ijehlNp",
        "olist.db":    "16p5aoPjb4LikHVVzTagpXkZAKoTFww-u",
    }

    for filename, file_id in files.items():
        dest = os.path.join(db_dir, filename)
        if not os.path.exists(dest):
            st.info(f"Downloading {filename}...")
            gdown.download(id=file_id, output=dest, quiet=False)

download_database()

from llm.router          import route_query
from sql.nl_to_sql       import nl_to_sql, run_sql
from rag.retriever       import FaissRetriever
from llm.sentiment       import analyse_sentiment
from llm.synthesizer     import synthesize
from llm.chart_generator import build_chart

INDEX_PATH  = os.path.join(BASE_DIR, "database", "faiss.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "database", "chunks.pkl")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Analytics Assistant",
    page_icon="🛒",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #1a1d27; }

/* Cards */
.card {
    background: #1e2130;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #4f8ef7;
}
.card-green  { border-left-color: #2ecc71; }
.card-red    { border-left-color: #e74c3c; }
.card-orange { border-left-color: #f39c12; }

/* Route badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-sql    { background:#1a3a5c; color:#4f8ef7; }
.badge-rag    { background:#1a3d2b; color:#2ecc71; }
.badge-hybrid { background:#3d2a1a; color:#f39c12; }

/* Input box */
.stTextInput input {
    background-color: #1e2130 !important;
    color: #e0e0e0 !important;
    border: 1px solid #3a3f55 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Divider */
hr { border-color: #2a2d3e; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("## 🛒")
with col2:
    st.markdown("## Olist Analytics Assistant")
    st.caption("Powered by LLM + RAG · Ask questions in plain English · Olist Brazilian E-Commerce Dataset")

st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 How It Works")
    st.markdown("""
    <div class='card'>
    <b>🗃️ SQL Route</b><br>
    Questions about numbers, totals, rankings — queries the SQLite database directly.
    </div>
    <div class='card card-green'>
    <b>💬 RAG Route</b><br>
    Questions about reviews, sentiment, complaints — searches 40k customer review embeddings.
    </div>
    <div class='card card-orange'>
    <b>🔀 Hybrid Route</b><br>
    Questions needing both structured data and reviews — runs both pipelines and synthesizes.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💡 Sample Questions")
    samples = [
        "Top 5 product categories by revenue?",
        "What do customers complain about most?",
        "How many orders were delivered late?",
        "What is the sentiment around electronics?",
        "Which sellers have the best reviews?",
    ]
    for s in samples:
        st.markdown(f"• _{s}_")

    st.divider()
    st.markdown("### 📊 Dataset")
    st.markdown("Brazilian E-Commerce · Olist · 2016–2018  \n~100k orders · 8 tables · ~40k reviews")


# ── Load retriever once (cached) ───────────────────────────────────────────────
@st.cache_resource
def load_retriever():
    return FaissRetriever(INDEX_PATH, CHUNKS_PATH)

retriever = load_retriever()


# ── Route badge helper ─────────────────────────────────────────────────────────
def show_route_badge(route: str):
    cls  = {"SQL": "badge-sql", "RAG": "badge-rag", "HYBRID": "badge-hybrid"}
    icon = {"SQL": "🗃️", "RAG": "💬", "HYBRID": "🔀"}
    st.markdown(
        f"<span class='badge {cls.get(route, '')}'>{icon.get(route,'')} {route} Pipeline</span>",
        unsafe_allow_html=True
    )
    st.markdown("")


# ── SQL pipeline ───────────────────────────────────────────────────────────────
def run_sql_pipeline(question: str):
    show_route_badge("SQL")

    with st.status("Generating SQL query...", expanded=True) as status:
        sql = nl_to_sql(question)
        st.code(sql, language="sql")
        df  = run_sql(sql)

        if "error" in df.columns:
            st.warning("SQL failed — retrying with error feedback...")
            sql = nl_to_sql(question, retry_error=df["error"][0])
            st.code(sql, language="sql")
            df  = run_sql(sql)

        if "error" in df.columns:
            status.update(label="Query failed", state="error")
            st.error(f"Could not execute SQL: {df['error'][0]}")
            return

        status.update(label="Query complete", state="complete")

    st.markdown("#### 📋 Results")
    st.dataframe(df, use_container_width=True)

    fig = build_chart(df, question)
    if fig:
        st.markdown("#### 📊 Chart")
        fig.update_layout(
            paper_bgcolor="#1e2130",
            plot_bgcolor="#1e2130",
            font_color="#e0e0e0",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── RAG pipeline ───────────────────────────────────────────────────────────────
def run_rag_pipeline(question: str):
    show_route_badge("RAG")

    with st.status("Searching customer reviews...", expanded=True) as status:
        chunks = retriever.retrieve(question, top_k=5)
        result = analyse_sentiment(chunks)
        status.update(label="Analysis complete", state="complete")

    sentiment = result["sentiment"]
    color_map = {"positive": "card-green", "negative": "card-red", "mixed": "card-orange"}
    icon_map  = {"positive": "😊", "negative": "😟", "mixed": "😐"}
    card_cls  = color_map.get(sentiment, "card")
    icon      = icon_map.get(sentiment, "💬")

    st.markdown(f"""
    <div class='card {card_cls}'>
        <h4>{icon} Overall Sentiment: {sentiment.upper()}</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔎 Top Complaint Themes")
    for i, theme in enumerate(result["themes"], 1):
        st.markdown(f"""
        <div class='card'>
            <b>{i}.</b> {theme}
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📄 View source review excerpts"):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**Review {i}** · order `{chunk['order_id']}`")
            st.caption(chunk["text"])
            st.divider()


# ── Hybrid pipeline ────────────────────────────────────────────────────────────
def run_hybrid_pipeline(question: str):
    show_route_badge("HYBRID")

    with st.status("Running hybrid analysis...", expanded=True) as status:
        sql     = nl_to_sql(question)
        df      = run_sql(sql)
        chunks  = retriever.retrieve(question, top_k=5)
        rag_res = analyse_sentiment(chunks)
        status.update(label="Analysis complete", state="complete")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🗃️ Structured Data")
        if "error" not in df.columns:
            st.dataframe(df, use_container_width=True)
            sql_text = df.to_string(index=False)
            fig = build_chart(df, question)
            if fig:
                fig.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", font_color="#e0e0e0")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not retrieve structured data.")
            sql_text = "No structured data available."

    with col_r:
        st.markdown("#### 💬 Review Sentiment")
        sentiment = rag_res["sentiment"]
        color_map = {"positive": "card-green", "negative": "card-red", "mixed": "card-orange"}
        icon_map  = {"positive": "😊", "negative": "😟", "mixed": "😐"}
        st.markdown(f"""
        <div class='card {color_map.get(sentiment,"card")}'>
            <h4>{icon_map.get(sentiment,"")} Sentiment: {sentiment.upper()}</h4>
        </div>
        """, unsafe_allow_html=True)
        for theme in rag_res["themes"]:
            st.markdown(f"<div class='card'>• {theme}</div>", unsafe_allow_html=True)

    st.markdown("#### 🔀 Combined Insight")
    answer = synthesize(question, sql_text, rag_res)
    st.markdown(f"""
    <div class='card card-orange'>
        {answer}
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_flow = st.tabs(["💬 Ask Assistant", "🗺️ App Architecture"])

# ── Tab 1: Chat ────────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("### 💬 Ask a Question")
    question = st.text_input(
        label="question",
        label_visibility="collapsed",
        placeholder="e.g. What are the top 5 product categories by revenue?",
    )

    if question:
        st.divider()
        with st.spinner("Routing your query..."):
            route = route_query(question)

        if route == "SQL":
            run_sql_pipeline(question)
        elif route == "RAG":
            run_rag_pipeline(question)
        else:
            run_hybrid_pipeline(question)

# ── Tab 2: Flowchart ───────────────────────────────────────────────────────────
with tab_flow:
    st.markdown("### 🗺️ Application Architecture")
    st.caption("End-to-end flow from user query to final response")

    st.graphviz_chart("""
    digraph LLM_Analytics {
        graph [bgcolor="#0f1117" rankdir=TB splines=ortho pad=0.5]
        node  [fontname="Arial" fontsize=12 style=filled shape=roundedbox]
        edge  [color="#4f8ef7" fontname="Arial" fontsize=10]

        // Input
        User [label="👤 User Question" fillcolor="#1e2130" fontcolor="#e0e0e0" color="#4f8ef7"]

        // Router
        Router [label="🔀 Query Router\n(LLM Classifier)" fillcolor="#2a1f3d" fontcolor="#c084fc" color="#c084fc" shape=diamond]

        // SQL Path
        NL2SQL  [label="🧠 NL-to-SQL\n(codellama)" fillcolor="#1a3a5c" fontcolor="#4f8ef7" color="#4f8ef7"]
        RunSQL  [label="🗃️ Execute SQL\n(SQLite)" fillcolor="#1a3a5c" fontcolor="#4f8ef7" color="#4f8ef7"]
        Retry   [label="🔁 Retry with\nError Feedback" fillcolor="#1a2a3a" fontcolor="#7ab3f5" color="#4f8ef7"]
        Chart   [label="📊 Auto Chart\n(Plotly)" fillcolor="#1a3a5c" fontcolor="#4f8ef7" color="#4f8ef7"]

        // RAG Path
        Embed   [label="🔎 Embed Query\n(all-MiniLM-L6-v2)" fillcolor="#1a3d2b" fontcolor="#2ecc71" color="#2ecc71"]
        FAISS   [label="📦 FAISS Search\n(Top-5 chunks)" fillcolor="#1a3d2b" fontcolor="#2ecc71" color="#2ecc71"]
        Senti   [label="💬 Sentiment +\nTheme Analysis" fillcolor="#1a3d2b" fontcolor="#2ecc71" color="#2ecc71"]

        // Hybrid
        Synth   [label="🔀 Synthesizer\n(LLM combiner)" fillcolor="#3d2a1a" fontcolor="#f39c12" color="#f39c12"]

        // Output
        Out_SQL    [label="✅ DataFrame\n+ Chart" fillcolor="#1e2130" fontcolor="#e0e0e0" color="#4f8ef7"]
        Out_RAG    [label="✅ Sentiment\n+ Themes" fillcolor="#1e2130" fontcolor="#e0e0e0" color="#2ecc71"]
        Out_HYB    [label="✅ Combined\nInsight" fillcolor="#1e2130" fontcolor="#e0e0e0" color="#f39c12"]

        // Data stores
        SQLite  [label="🗄️ SQLite DB\n(8 tables)" fillcolor="#2a2a1a" fontcolor="#f1c40f" color="#f1c40f" shape=cylinder]
        VecDB   [label="🧲 FAISS Index\n(40k vectors)" fillcolor="#2a2a1a" fontcolor="#f1c40f" color="#f1c40f" shape=cylinder]

        // Edges
        User   -> Router

        Router -> NL2SQL [label="SQL"]
        Router -> Embed  [label="RAG"]
        Router -> NL2SQL [label="HYBRID"]
        Router -> Embed  [label="HYBRID"]

        NL2SQL -> RunSQL
        RunSQL -> Retry  [label="on error"]
        Retry  -> RunSQL
        RunSQL -> Chart
        RunSQL -> SQLite [style=dashed color="#f1c40f"]
        Chart  -> Out_SQL

        Embed  -> FAISS
        FAISS  -> Senti
        FAISS  -> VecDB  [style=dashed color="#f1c40f"]
        Senti  -> Out_RAG

        RunSQL -> Synth  [label="HYBRID" color="#f39c12"]
        Senti  -> Synth  [label="HYBRID" color="#f39c12"]
        Synth  -> Out_HYB
    }
    """, use_container_width=True)
