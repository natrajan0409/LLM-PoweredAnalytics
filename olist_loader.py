import os 
import sqlite3
import pandas as pd
# Load the datasets
path = 'D:/workspace/LLM-PoweredAnalytics/DATA'


mydict ={"customers":pd.read_csv(os.path.join(path,"olist_customers_dataset.csv")),
"geolocation":pd.read_csv(os.path.join(path,"olist_geolocation_dataset.csv")),
"order_items":pd.read_csv(os.path.join(path,"olist_order_items_dataset.csv")),
"order_payments":pd.read_csv(os.path.join(path,"olist_order_payments_dataset.csv")),
"order_reviews":pd.read_csv(os.path.join(path,"olist_order_reviews_dataset.csv")),
"orders":pd.read_csv(os.path.join(path,"olist_orders_dataset.csv")),
"products":pd.read_csv(os.path.join(path,"olist_products_dataset.csv")),
"sellers":pd.read_csv(os.path.join(path,"olist_sellers_dataset.csv")),
"product_category_name_translation":pd.read_csv(os.path.join(path,"olist_product_category_name_translation.csv"))}




class OlistLoader:
    def __init__(self):
        os.makedirs('D:/workspace/LLM-PoweredAnalytics/database', exist_ok=True)
        self.db_path = "D:/workspace/LLM-PoweredAnalytics/database/olist.db"
        self.conn = sqlite3.connect(self.db_path)

    def ingest_data(self, table_name, df):
        df.to_sql(table_name,self.conn, if_exists='replace', index=False)    
    
    def create_tables(self):
        for table_name, df in mydict.items():
            self.ingest_data(table_name, df)

    def load_data(self, table_name):
        query = f"select * From {table_name}"
        df=pd.read_sql_query(query, self.conn)
        return df
    
    def load_all_data(self):
         return {table_name: self.load_data(table_name) for table_name in mydict}
    
    def close_connection(self):
        self.conn.close()


