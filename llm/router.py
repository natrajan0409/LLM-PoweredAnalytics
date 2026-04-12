import ollama
import os 
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()
Main_Model = os.getenv("model", "codellama")
api_key = os.getenv("mistral_API_key")
client = Mistral(api_key=api_key)

 


def route_query(question :str) -> str:
    """Classify question as SQL, RAG, or HYBRID."""
    system_prompt = """You are a query classifier for an e-commerce analytics system.
Classify the user question into exactly one of these three categories:

SQL    - questions about numbers, counts, totals, averages, rankings, dates
RAG    - questions about customer reviews, opinions, complaints, sentiment, feedback  
HYBRID - questions that need BOTH structured data AND customer reviews

Reply with ONLY one word: SQL, RAG, or HYBRID. Nothing else."""

    response =client.chat.complete(
            model=Main_Model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question}
            ]
        )
    
    result = response.choices[0].message.content.strip().lower()

# safety fallback — extract first word if model adds extra text
    for word in result.split():
            if word in ("SQL", "RAG", "HYBRID"):
                return word
    return "SQL"  # default fallback


if __name__ == "__main__":
    tests = [
        "What are the top 5 product categories by revenue?",
        "What do customers complain about most?",
        "What are the reviews saying about top selling products?",
        "How many orders were delivered late?",
        "What is the sentiment around electronics products?",
    ]
    for q in tests:
        print(f"{route_query(q):8} ← {q}")