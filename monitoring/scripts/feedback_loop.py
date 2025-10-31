import pandas as pd
from datetime import datetime
import os
import sqlite3

LIVE_DB_PATH = os.path.join(os.path.dirname(__file__), "db/live_data.db")

conn = sqlite3.connect(LIVE_DB_PATH)
cursor = conn.cursor()

cursor.execute(
    """
        UPDATE live_data
        SET true_label = ?
        WHERE animal_name = ?
    """, 
    ('Invertebrate', 'snail')
)
conn.commit()
conn.close()