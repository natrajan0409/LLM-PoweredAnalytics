import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATA")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "olist.db")

TABLE_FILES = {
    "customers":         "olist_customers_dataset.csv",
    "geolocation":       "olist_geolocation_dataset.csv",
    "order_items":       "olist_order_items_dataset.csv",
    "order_payments":    "olist_order_payments_dataset.csv",
    "order_reviews":     "olist_order_reviews_dataset.csv",
    "orders":            "olist_orders_dataset.csv",
    "products":          "olist_products_dataset.csv",
    "sellers":           "olist_sellers_dataset.csv",
    "product_category":  "product_category_name_translation.csv",
}


class OlistLoader:
    """Loads Olist CSV files into a local SQLite database."""

    def __init__(self, db_path=DB_PATH):
        os.makedirs(DB_DIR, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def ingest_table(self, table_name, df):
        """Write a single DataFrame into SQLite, replacing if it exists."""
        df.to_sql(table_name, self.conn, if_exists="replace", index=False)
        print(f"  [{table_name}] {len(df):,} rows loaded")

    def create_tables(self):
        """Read all CSVs and write them to the database."""
        for table_name, filename in TABLE_FILES.items():
            filepath = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(filepath)
            self.ingest_table(table_name, df)
        self.conn.commit()
        print("All tables committed.")

    def load_table(self, table_name):
        """Return a table as a DataFrame."""
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)

    def load_all(self):
        """Return all tables as a dict of DataFrames."""
        return {name: self.load_table(name) for name in TABLE_FILES}

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    print(f"Loading Olist data into: {DB_PATH}")
    loader = OlistLoader()
    loader.create_tables()
    loader.close()
    print("Done.")
