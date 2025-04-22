import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "model_registry.db")
def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS model_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid INTEGER,
            model_name TEXT,
            model_version TEXT,
            model_url TEXT,
            submitted_by TEXT,
            submission_time INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def is_model_name_taken(model_name, exclude_uid=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if exclude_uid is not None:
        c.execute("""
            SELECT 1 FROM model_submissions 
            WHERE model_name = ? AND uid != ? 
            LIMIT 1
        """, (model_name, exclude_uid))
    else:
        c.execute("""
            SELECT 1 FROM model_submissions 
            WHERE model_name = ? 
            LIMIT 1
        """, (model_name,))
    result = c.fetchone()
    conn.close()
    return result is not None

def insert_submission(uid, model_name, model_version, model_url, submitted_by, submission_time):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO model_submissions (uid, model_name, model_version, model_url, submitted_by, submission_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (uid, model_name, model_version, model_url, submitted_by, submission_time))
    conn.commit()
    conn.close()

def get_latest_submission(uid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT model_name, model_version FROM model_submissions 
        WHERE uid = ? 
        ORDER BY created_at DESC 
        LIMIT 1
    """, (uid,))
    result = c.fetchone()
    conn.close()
    return result