import sqlite3
import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "db/inference_logs.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_data TEXT,
            prediction TEXT,
            confidence REAL       
        );
    """)
    conn.commit()
    conn.close()

def log_prediction(input_data: str, prediction: str, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO prediction_logs (timestamp, input_data, prediction, confidence) VALUES (?, ?, ?, ?)", 
        (timestamp, input_data, prediction, confidence)             
    )
    conn.commit()
    conn.close()


