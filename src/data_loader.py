import pandas as pd
from sqlalchemy import create_engine, text

def get_connection():
    """
    Connect to local PostgreSQL crypto database.
    """
    engine = create_engine(
        "postgresql+psycopg2://postgres:yourpassword@localhost:5432/crypto_data"
    )
    return engine

def load_crypto_data(symbol="BTCUSDT", start="2025-09-08", end="2025-11-13"):
    """
    Load historical cryptocurrency data from PostgreSQL.
    """
    engine = get_connection()
    query = text("""
        SELECT *
        FROM public.coins_min_sync
        WHERE symbol = :symbol
          AND open_time >= :start
          AND open_time <= :end
        ORDER BY open_time ASC;
    """)
    df = pd.read_sql(query, engine, params={"symbol": symbol, "start": start, "end": end})
    print(f"[INFO] Loaded {len(df)} rows for {symbol}")
    return df
