import ollama
import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()
Main_Model = os.getenv("model", "codellama")
api_key = os.getenv("mistral_API_key")
client = Mistral(api_key=api_key)


def synthesize(question: str, sql_result: str, rag_result: dict) -> str:
    """Combine SQL data + RAG sentiment into one coherent answer."""

    themes = "\n".join([f"- {t}" for t in rag_result.get("themes", [])])
    sentiment = rag_result.get("sentiment", "mixed")

    context = f"""
Structured Data Result:
{sql_result}

Customer Review Analysis:
Sentiment: {sentiment}
Top Complaint Themes:
{themes}
"""

    system_prompt = """You are a business analyst assistant.
Given structured sales data and customer review analysis, write a short coherent summary.
Be concise — 3 to 5 sentences max. Focus on actionable insights."""

    response = client.chat.complete(
        model=Main_Model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Question: {question}\n\nData:\n{context}"}
        ]
    )
    return response.choices[0].message.content.strip().lower()


if __name__ == "__main__":
    # simulate what a hybrid pipeline would pass in
    sql_result = """
product_category_name  | total_revenue
-----------------------+--------------
beleza_saude           | 1258681.34
relogios_presentes     | 1205005.68
cama_mesa_banho        | 1036988.68
"""
    rag_result = {
        "sentiment": "mixed",
        "themes": ["late delivery", "wrong product received", "poor packaging"]
    }

    question = "What are the top selling categories and what do customers say about them?"
    answer = synthesize(question, sql_result, rag_result)
    print(answer)
