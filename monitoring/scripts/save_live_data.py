import sqlite3
import pandas as pd
from datetime import datetime
import os

LIVE_DIR = os.path.join(os.path.dirname(__file__), "db")
LIVE_DB_PATH = os.path.join(LIVE_DIR, "live_data.db")
os.makedirs(LIVE_DIR, exist_ok=True)

def init_live_db():
    """Initialize live_data.db with the same schema as reference_data (minus class_name)."""
    conn = sqlite3.connect(LIVE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_data (
            animal_name TEXT,
            hair INTEGER,
            eggs INTEGER,
            milk INTEGER,
            predator INTEGER,
            toothed INTEGER,
            backbone INTEGER,
            breathes INTEGER,
            venomous INTEGER,
            legs INTEGER,
            tail INTEGER,
            can_fly INTEGER,
            can_swim INTEGER,
            is_domestic_pet INTEGER,
            prediction TEXT,
            true_label TEXT,
            event_timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_live_data(input_data: str, feature_row: dict, prediction: str):
    """Insert one live prediction record into live_data.db."""
    conn = sqlite3.connect(LIVE_DB_PATH)
    input_row = {k: v for k, v in feature_row.items() if k in [
        "hair", "eggs", "milk", "predator", "toothed", "backbone",
        "breathes", "venomous", "legs", "tail", "can_fly",
        "can_swim", "is_domestic_pet"
    ]}
    input_row["animal_name"] = input_data
    input_row["true_label"] = feature_row['class_name']
    input_row["prediction"] = prediction
    input_row["event_timestamp"] = datetime.now().isoformat()

    df = pd.DataFrame([input_row])
    df.to_sql("live_data", conn, if_exists="append", index=False)
    conn.close()
