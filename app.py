import os
import streamlit as st
from dotenv import load_dotenv

from llm.router          import route_query
from sql.nl_to_sql       import nl_to_sql, run_sql
from rag.retriever       import FaissRetriever
from llm.sentiment       import analyse_sentiment
from llm.synthesizer     import synthesize
from llm.chart_generator import build_chart

load_dotenv()

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH  = os.path.join(BASE_DIR, "database", "faiss.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "database", "chunks.pkl")

st.set_page_config(page_title="LLM Analytics Assistant", layout="wide")
st.title("LLM-Powered Analytics Assistant")
st.caption("Ask anything about Olist e-commerce data")

#Load retriever once (cached)
@st.cache_resource
def load_retriever():
    return FaissRetriever(INDEX_PATH,CHUNKS_PATH)
retriever = load_retriever() 

#sql pipline 
def run_sql_pipline(question:str):
    st.info("Route: SQL — querying structured database...")

    sql =nl_to_sql(question)
    st.code(sql,language="sql")

    df =run_sql(sql)

    if "error" in df.columns:
        st.warning("sql failed ,retrying")
        sql=nl_to_sql(question,return_error=df["error"][0])
        df=run_sql(sql)

    if "error" in df.columns:
        st.error(f"sql errpr : {df['error'][0]}")
        return
    
    st.dataframe(df)

    fig =build_chart(df,question)
    if fig:
        st.plotly_chart(fig,use_container_width=True)

# RAG pipline function 
