import os
import sqlite3
import pandas as pd
import ollama
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(BASE_DIR, "database", "olist.db")
Main_Model = os.getenv("model", "llama3.2")

schema = """
Tables in the olist sqlite database:

orders (order_id, customer_id, order_status, order_purchase_timestamp,
  order_approved_at, order_delivered_carrier_date,
  order_delivered_customer_date, order_estimated_delivery_date)

order_items (order_id, order_item_id, product_id, seller_id,
  shipping_limit_date, price, freight_value)

order_items (order_id, order_item_id, product_id, seller_id,
  shipping_limit_date, price, freight_value)
-- NOTE: there is NO quantity column. Each row = 1 item. Revenue = SUM(price)

order_payments (order_id, payment_sequential, payment_type,
  payment_installments, payment_value)

customers (customer_id, customer_unique_id, customer_zip_code_prefix,
  customer_city, customer_state)

product_category (product_category_name, product_category_name_english)

products (product_id, product_category_name, product_name_lenght,
  product_description_lenght, product_photos_qty, product_weight_g,
  product_length_cm, product_height_cm, product_width_cm)

sellers (seller_id, seller_zip_code_prefix, seller_city, seller_state)
"""


def nl_to_sql(question: str, retry_error: str = None) -> str:
    """Send natural language question to LLM, get back a SQL query."""

    user_content = question
    if retry_error:
        user_content = (
            f"Question: {question}\n\n"
            f"Your previous SQL failed with this error:\n{retry_error}\n\n"
            f"Please fix and return only the corrected SQL."
        )

    system_prompt = f"""You are a SQLite expert.
Given the schema below, write a single valid SQLite SQL query.
Rules:
- Return ONLY raw SQL — no markdown, no code fences, no explanation
- Use ONLY columns listed in the schema — do NOT invent columns
- There is NO quantity column anywhere — each order_items row is 1 unit
- To calculate revenue: SUM(order_items.price)
- payment_value is in order_payments table, price is in order_items table
{schema}"""

    api_key = os.getenv("mistral_API_key")

    client = Mistral(api_key=api_key)
    response =client.chat.complete(
        model=Main_Model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ]
    )
    sql = response.choices[0].message.content.strip()
    sql = clean_sql(sql)
    return sql


def clean_sql(text: str) -> str:
    """Strip markdown code fences and extra text from LLM response."""
    # remove ```sql ... ``` or ``` ... ``` blocks
    if "```" in text:
        lines = text.split("\n")
        sql_lines = []
        inside = False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                sql_lines.append(line)
        text = "\n".join(sql_lines).strip()
    return text.strip()


def run_sql(sql: str) -> pd.DataFrame:
    """Execute a SQL query against the Olist database, return DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)], "sql": [sql]})
    finally:
        conn.close()


if __name__ == "__main__":
    question = "What are the top 5 product categories by total revenue?"
    print("Question:", question)

    sql = nl_to_sql(question)
    print("Generated SQL:\n", sql)

    df = run_sql(sql)

    # if SQL failed, retry once sending the error back to the LLM
    if "error" in df.columns:
        print("\nSQL failed, retrying with error feedback...")
        sql = nl_to_sql(question, retry_error=df["error"][0])
        print("Corrected SQL:\n", sql)
        df = run_sql(sql)

    print("\nResult:\n", df)
