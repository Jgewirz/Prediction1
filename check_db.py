"""Check the PostgreSQL database status."""

import psycopg2

DB_CONFIG = {
    "host": "ep-flat-silence-ahtaz4uz-pooler.c-3.us-east-1.aws.neon.tech",
    "database": "neondb",
    "user": "neondb_owner",
    "password": "npg_hjmprEz3wJ0B",
    "sslmode": "require",
}


def check_database():
    print("Connecting to PostgreSQL database...")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("\nTable row counts:")
    tables = [
        "betting_decisions",
        "calibration_records",
        "events_cache",
        "market_snapshots",
        "markets_cache",
        "performance_daily",
        "run_history",
    ]

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")

    # Check run_history
    print("\nRecent runs:")
    cursor.execute(
        """
        SELECT run_id, started_at, status, events_analyzed, decisions_made
        FROM run_history
        ORDER BY started_at DESC
        LIMIT 5
    """
    )
    for row in cursor.fetchall():
        print(f"  {row}")

    # Check betting_decisions
    print("\nRecent betting decisions:")
    cursor.execute(
        """
        SELECT decision_id, market_ticker, action, bet_amount, status
        FROM betting_decisions
        ORDER BY timestamp DESC
        LIMIT 5
    """
    )
    for row in cursor.fetchall():
        print(f"  {row}")

    cursor.close()
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    check_database()
