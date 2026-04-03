LLM-Powered Analytics Assistant with RAG

An intelligent analytics assistant for e-commerce data that enables business users to query a structured database and analyse customer reviews using plain English — no SQL knowledge required.

Built on the Brazilian Olist e-commerce dataset (~100K orders, 40K reviews), the system integrates three core pipelines:

SQL Path — Translates natural language questions into SQLite queries using GPT-3.5, executes them, and summarises results in plain English with auto-generated Plotly charts
RAG Path — Embeds customer reviews using all-MiniLM-L6-v2, stores them in a FAISS vector index, and retrieves the most relevant chunks to perform sentiment analysis and extract top complaint themes
Hybrid Path — Combines both pipelines for queries that require structured data and review text together
A lightweight LLM-based query router automatically classifies every incoming question and directs it to the right pipeline.

Tech Stack: Python · OpenAI GPT-3.5 · LangChain · FAISS · Sentence Transformers · SQLite · Pandas · Plotly · Streamlit
