import sqlite3
from pathlib import Path

source_db = Path(__file__).parent.parent / "db_logs" / "live_data.db"
dest_db = Path(__file__).parent.parent / "db" / "live_data.db"

src_conn = sqlite3.connect(source_db)
dest_conn = sqlite3.connect(dest_db)

src_cursor = src_conn.cursor()
dest_cursor = dest_conn.cursor()

src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = src_cursor.fetchall()

for (table_name,) in tables:
    print(f"Copying table: {table_name}")
    # Get data from source table
    src_cursor.execute(f"SELECT * FROM {table_name}")
    rows = src_cursor.fetchall()

    # Get column info
    src_cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in src_cursor.fetchall()]
    columns_str = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))

    # Create table in target if not exists
    src_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    create_table_sql = src_cursor.fetchone()[0]
    create_table_sql = create_table_sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
    dest_cursor.execute(create_table_sql)

    # Insert data into target table
    dest_cursor.executemany(
        f"INSERT OR IGNORE INTO {table_name} ({columns_str}) VALUES ({placeholders})",
        rows
    )

# Commit and close connections
dest_conn.commit()
src_conn.close()
dest_conn.close()

print("✅ Data transferred successfully!")