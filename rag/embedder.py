import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()
embedding_model = os.getenv("embedding_model")
chunk_size = int(os.getenv("chunk_size", 1000))
chunk_overlap = int(os.getenv("chunk_overlap", 200))


def split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
    splitter =RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def  get_embedding(text):
    embedder =HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": "cpu"})
    return embedder.embed_query(text)
