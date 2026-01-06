"""
Run database migration for position tracking.
"""

import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


async def run_migration():
    """Run the position tracking migration."""

    # Get database config from environment
    host = os.getenv("PGHOST", "")
    database = os.getenv("PGDATABASE", "neondb")
    user = os.getenv("PGUSER", "neondb_owner")
    password = os.getenv("PGPASSWORD", "")
    port = int(os.getenv("PGPORT", "5432"))
    ssl = os.getenv("PGSSLMODE", "require")

    print(f"Connecting to PostgreSQL at {host}...")

    try:
        # Connect to database
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            ssl=ssl,
        )

        print("Connected successfully!")

        # Read migration file
        migration_path = (
            Path(__file__).parent / "db" / "migrations" / "002_position_tracking.sql"
        )

        if not migration_path.exists():
            print(f"Migration file not found: {migration_path}")
            return False

        migration_sql = migration_path.read_text()

        print(f"Running migration from {migration_path}...")
        print("-" * 50)

        # Split into statements and run each one
        # Skip comments and empty lines
        statements = []
        current_statement = []
        in_function = False

        for line in migration_sql.split("\n"):
            stripped = line.strip()

            # Track if we're inside a function definition
            if (
                "CREATE OR REPLACE FUNCTION" in line.upper()
                or "CREATE FUNCTION" in line.upper()
            ):
                in_function = True

            if stripped.startswith("--") or not stripped:
                continue

            current_statement.append(line)

            # Check if statement is complete
            if in_function:
                if "$$ LANGUAGE" in line:
                    in_function = False
                    statements.append("\n".join(current_statement))
                    current_statement = []
            elif stripped.endswith(";") and not in_function:
                statements.append("\n".join(current_statement))
                current_statement = []

        # Add any remaining statement
        if current_statement:
            statements.append("\n".join(current_statement))

        success_count = 0
        error_count = 0

        for i, stmt in enumerate(statements):
            stmt = stmt.strip()
            if not stmt or stmt.startswith("--"):
                continue

            try:
                # Get first 60 chars for logging
                preview = (
                    stmt[:60].replace("\n", " ") + "..."
                    if len(stmt) > 60
                    else stmt.replace("\n", " ")
                )
                print(f"[{i+1}/{len(statements)}] {preview}")

                await conn.execute(stmt)
                success_count += 1

            except asyncpg.exceptions.DuplicateColumnError as e:
                print(f"  -> Column already exists (skipping): {e}")
                success_count += 1  # This is expected for idempotent migrations

            except asyncpg.exceptions.DuplicateTableError as e:
                print(f"  -> Table already exists (skipping): {e}")
                success_count += 1

            except asyncpg.exceptions.DuplicateObjectError as e:
                print(f"  -> Object already exists (skipping): {e}")
                success_count += 1

            except Exception as e:
                print(f"  -> ERROR: {e}")
                error_count += 1

        print("-" * 50)
        print(
            f"Migration complete: {success_count} statements succeeded, {error_count} errors"
        )

        # Verify the new columns exist
        print("\nVerifying new columns in betting_decisions...")

        columns_query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'betting_decisions'
            AND column_name IN (
                'order_id', 'filled_price_cents', 'filled_contracts',
                'stop_loss_pct', 'take_profit_pct', 'exit_order_id',
                'exit_reason', 'current_price_cents', 'unrealized_pnl_dollars'
            )
            ORDER BY column_name
        """

        rows = await conn.fetch(columns_query)

        if rows:
            print(f"Found {len(rows)} new columns:")
            for row in rows:
                print(f"  - {row['column_name']}: {row['data_type']}")
        else:
            print("No new columns found (may need to check table exists)")

        # Check if position_snapshots table exists
        print("\nChecking for position_snapshots table...")
        table_check = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'position_snapshots'
            )
        """
        )
        print(f"position_snapshots table exists: {table_check}")

        # Check if exit_events table exists
        table_check = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'exit_events'
            )
        """
        )
        print(f"exit_events table exists: {table_check}")

        await conn.close()
        print("\nMigration verification complete!")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(run_migration())
