import sqlite3
import datetime
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "db" 
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "inference_logs.db"

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
