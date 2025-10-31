import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_DIR = os.path.join(os.path.dirname(__file__), "db")
DB_PATH = os.path.join(DB_DIR, "reference_data.db")
os.makedirs(DB_DIR, exist_ok=True)

def save_reference_data(df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("reference_data", conn, if_exists="replace", index=False)
    conn.close()
    