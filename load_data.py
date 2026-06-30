import sqlite3
import pandas as pd

def load_data():
    conn = sqlite3.connect(
        r"C:\Users\admin\Downloads\Regression.db"
    )

    data = pd.read_sql_query(
        "SELECT * FROM air_quality_index",
        conn
    )

    conn.close()

    return data