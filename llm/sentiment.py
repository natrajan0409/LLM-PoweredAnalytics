import ollama
import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()
Main_Model = os.getenv("model", "codellama")
api_key = os.getenv("mistral_API_key")
client = Mistral(api_key=api_key)

def analyse_sentiment(chunks: list[dict]) -> dict:
        """Analyse review chunks, return sentiment + top complaint themes."""


        #join chunk text into block
        reviews_text = "\n---\n".join([c["text"] for c in chunks]) 
        system_prompt = """You are a customer feedback analyst.
    Given the customer reviews below, provide:
    1. Overall sentiment: positive, negative, or mixed
    2. Top 3 complaint themes (short phrases)

    Reply in this exact format:
    Sentiment: <positive/negative/mixed>
    Themes:
    - <theme 1>
    - <theme 2>
    - <theme 3>"""
        
      
        response = client.chat.complete(
            model=Main_Model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Reviews:\n{reviews_text}"}
            ]
    )  
        
        raw =response.choices[0].message.content.strip().lower()
        return parse_sentiment(raw)

def parse_sentiment(raw :str) ->dict:
        """Parse the LLM response into a structured dict."""
        result ={"sentiment":"mixed","themes":[]}

        for line in raw.splitlines():
                line=line.strip()
                if line.lower().startswith("sentiment"):
                    result["sentiment"]=line.split(":",1)[1].strip().lower()
                elif line.startswith("-"):
                    result["themes"].append(line.lstrip("-").strip())

        return result

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag.retriever import FaissRetriever

    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INDEX_PATH  = os.path.join(BASE_DIR, "database", "faiss.index")
    CHUNKS_PATH = os.path.join(BASE_DIR, "database", "chunks.pkl")

    retriever = FaissRetriever(INDEX_PATH, CHUNKS_PATH)
    chunks    = retriever.retrieve("late delivery damaged product", top_k=5)

    print("Retrieved chunks:", len(chunks))
    result = analyse_sentiment(chunks)
    print("Sentiment:", result["sentiment"])
    print("Themes:")
    for t in result["themes"]:
        print(" -", t)