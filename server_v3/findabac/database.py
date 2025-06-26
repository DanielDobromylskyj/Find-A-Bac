import os
import sqlite3

def setup_database(path):
    if os.path.exists(path):
        os.remove(path)

    db = sqlite3.connect(path)
    cursor = db.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        pswd_hash TEXT,
        auth_level INTEGER DEFAULT 0
    )
    """)

    db.commit()

    cursor.close()
    db.close()
