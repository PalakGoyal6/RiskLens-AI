import os
import pickle
import sqlite3
import pandas as pd
import numpy as np
import hashlib
import secrets
from faker import Faker

DB_PATH = "credit_risk.db"

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"{salt}${dk.hex()}"

def init_database():
    print("Initializing SQLite database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop tables if they exist to start fresh
    cursor.execute("DROP TABLE IF EXISTS decisions")
    cursor.execute("DROP TABLE IF EXISTS predictions")
    cursor.execute("DROP TABLE IF EXISTS sessions")
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("DROP TABLE IF EXISTS applicants")

    # Create users table
    cursor.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create sessions table
    cursor.execute("""
    CREATE TABLE sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # Create predictions table
    cursor.execute("""
    CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_id INTEGER NOT NULL,
        officer_id INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        predicted_probability REAL NOT NULL,
        risk_label TEXT NOT NULL,
        shap_values TEXT, -- stored as JSON
        narration TEXT,
        FOREIGN KEY (applicant_id) REFERENCES applicants(applicant_id),
        FOREIGN KEY (officer_id) REFERENCES users(id)
    )
    """)

    # Create decisions table
    cursor.execute("""
    CREATE TABLE decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL,
        officer_id INTEGER NOT NULL,
        decision TEXT NOT NULL, -- approved, declined, escalated
        notes TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (prediction_id) REFERENCES predictions(id),
        FOREIGN KEY (officer_id) REFERENCES users(id)
    )
    """)

    # Insert seed users
    users = [
        ("admin", "admin123", "admin"),
        ("officer1", "password123", "loan_officer"),
        ("officer2", "password123", "loan_officer")
    ]
    for u, p, r in users:
        hashed = hash_password(p)
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", (u, hashed, r))
    
    print("Users table seeded.")

    # Load applicant datasets
    print("Loading pickle datasets (this might take a moment)...")
    with open("models/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("models/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    print(f"Loaded X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Align/Join them
    df_db = X_test.copy()
    df_db['target'] = y_test
    df_db = df_db.reset_index()
    
    # Rename the index column (usually SK_ID_CURR or index) to applicant_id
    orig_index_col = df_db.columns[0]
    df_db.rename(columns={orig_index_col: "applicant_id"}, inplace=True)

    # Generate fake names using Faker
    print("Generating fake names using Faker (this might take ~5 seconds)...")
    fake = Faker()
    names = [fake.name() for _ in range(len(df_db))]
    df_db['name'] = names

    print(f"Seeding {len(df_db)} applicant records into 'applicants' table...")

    # Create applicants table schema dynamically to fit all features
    columns_defs = ["`name` TEXT NOT NULL"]
    for col, dtype in df_db.dtypes.items():
        if col in ['applicant_id', 'name']:
            continue
        if np.issubdtype(dtype, np.integer):
            sql_type = "INTEGER"
        elif np.issubdtype(dtype, np.floating):
            sql_type = "REAL"
        else:
            sql_type = "TEXT"
        columns_defs.append(f"`{col}` {sql_type}")

    create_applicants_query = f"""
    CREATE TABLE applicants (
        applicant_id INTEGER PRIMARY KEY,
        {", ".join(columns_defs)}
    )
    """
    cursor.execute(create_applicants_query)
    conn.commit()

    # Write rows to applicants table
    df_db.to_sql("applicants", conn, if_exists="append", index=False)
    print("Applicants table seeded successfully!")

    conn.commit()
    conn.close()
    print("Database initialization complete.")

if __name__ == "__main__":
    init_database()
