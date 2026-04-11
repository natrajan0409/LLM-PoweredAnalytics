import os
import ollama
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()
Main_Model = os.getenv("model", "codellama")


def pick_chart_type(df:pd.DataFrame) -> str:
    """ ASk LLM to choose the best chart type for this data frame"""
    columns =list(df.columns)
    sample =df.head(3).to_string(index=False)
    prompt = f"""Given a DataFrame with columns {columns} and sample data:
    {sample}
    Which single chart type best visualises this data?
    Choose ONLY one word from: bar, line, pie
    Reply with ONE word only."""

    respose = ollama.chat(
        model=Main_Model,
        messages=[{"role":"user","content":prompt}]
    )

    result =respose["message"]["content"].strip().lower()
    # extract first valid word from response
    for word in result.split():
        if word in ("bar", "line", "pie"):
            return word
    return "bar"   # default fallback

def build_chart(df:pd.DataFrame,question: str= "") -> px.bar:
    """Generate a Plotly chart from a DataFrame."""
    chart_type =pick_chart_type(df)
   # identify x (label column) and y (numeric column)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols    = df.select_dtypes(exclude="number").columns.tolist()
    if not numeric_cols:
            return None   # nothing to plot

    x_col = text_cols[0]  if text_cols  else df.columns[0]
    y_col = numeric_cols[0]

    title = question if question else f"{y_col} by {x_col}"

    if chart_type == "pie":
        fig = px.pie(df, names=x_col, values=y_col, title=title)
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title)

    return fig


if __name__ == "__main__":
    # simulate a SQL result DataFrame
    data = {
        "category":      ["beleza_saude", "relogios_presentes", "cama_mesa_banho"],
        "total_revenue": [1258681, 1205005, 1036988]
    }
    df  = pd.DataFrame(data)
    fig = build_chart(df, "Top product categories by revenue")

    print("Chart type chosen:", pick_chart_type(df))
    fig.show()   # opens browser preview