from pathlib import Path
import sqlite3

LIVE_DIR = Path(__file__).resolve().parent.parent / "db" 
LIVE_DB_PATH = LIVE_DIR / "live_data.db"

conn = sqlite3.connect(LIVE_DB_PATH)
cursor = conn.cursor()

cursor.execute(
    """
        UPDATE live_data
        SET true_label = ?
        WHERE animal_name = ?
    """, 
    ('Mammal', 'rat')
)
conn.commit()
conn.close()