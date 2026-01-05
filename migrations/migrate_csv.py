"""
Migrate existing CSV betting decisions to SQLite database.
"""

import asyncio
import csv
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from db import get_database, Database


async def migrate_csv_files(
    db: Database,
    csv_dir: str = "betting_decisions",
    archive: bool = False
) -> Dict[str, Any]:
    """
    Migrate all existing CSV files to SQLite.

    Args:
        db: Database connection
        csv_dir: Directory containing CSV files
        archive: If True, move migrated files to archive/ subdirectory

    Returns:
        Dict with migration statistics
    """
    console = Console()
    csv_path = Path(csv_dir)

    if not csv_path.exists():
        console.print(f"[yellow]CSV directory not found: {csv_dir}[/yellow]")
        return {"files": 0, "records": 0, "errors": 0}

    # Find all CSV files
    csv_files = list(csv_path.glob("betting_decisions_*.csv"))

    if not csv_files:
        console.print("[yellow]No CSV files found to migrate[/yellow]")
        return {"files": 0, "records": 0, "errors": 0}

    console.print(f"[blue]Found {len(csv_files)} CSV files to migrate[/blue]")

    total_records = 0
    total_errors = 0
    migrated_files = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Migrating CSV files...", total=len(csv_files))

        for csv_file in csv_files:
            progress.update(task, description=f"Processing {csv_file.name}...")

            try:
                records, errors = await _migrate_single_csv(db, csv_file)
                total_records += records
                total_errors += errors
                migrated_files.append(csv_file)
                logger.info(f"Migrated {csv_file.name}: {records} records, {errors} errors")
            except Exception as e:
                logger.error(f"Failed to migrate {csv_file.name}: {e}")
                total_errors += 1

            progress.advance(task)

    # Archive migrated files if requested
    if archive and migrated_files:
        archive_dir = csv_path / "archive"
        archive_dir.mkdir(exist_ok=True)

        for f in migrated_files:
            try:
                f.rename(archive_dir / f.name)
            except Exception as e:
                logger.warning(f"Could not archive {f.name}: {e}")

        console.print(f"[green]Archived {len(migrated_files)} files to {archive_dir}[/green]")

    summary = {
        "files": len(csv_files),
        "records": total_records,
        "errors": total_errors
    }

    console.print(f"\n[bold green]Migration complete:[/bold green]")
    console.print(f"  Files processed: {summary['files']}")
    console.print(f"  Records migrated: {summary['records']}")
    console.print(f"  Errors: {summary['errors']}")

    return summary


async def _migrate_single_csv(db: Database, csv_file: Path) -> tuple[int, int]:
    """
    Migrate a single CSV file to the database.

    Returns:
        Tuple of (records_migrated, errors)
    """
    records = 0
    errors = 0

    # Extract timestamp from filename for deduplication
    file_timestamp = csv_file.stem.replace("betting_decisions_", "")

    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        decisions_batch = []

        for row in reader:
            try:
                # Generate unique decision ID
                ticker = row.get('ticker', row.get('market_ticker', ''))
                decision_id = f"{ticker}_{file_timestamp}_{uuid.uuid4().hex[:8]}"

                # Check if already exists
                exists = await db.fetchone(
                    "SELECT 1 FROM betting_decisions WHERE decision_id = ?",
                    (decision_id,)
                )
                if exists:
                    continue

                # Map CSV columns to database columns
                decision = _map_csv_to_db(row, decision_id, file_timestamp)
                decisions_batch.append(decision)

                # Batch insert every 100 records
                if len(decisions_batch) >= 100:
                    await db.insert_decisions_batch(decisions_batch)
                    records += len(decisions_batch)
                    decisions_batch = []

            except Exception as e:
                logger.debug(f"Error processing row: {e}")
                errors += 1

        # Insert remaining records
        if decisions_batch:
            await db.insert_decisions_batch(decisions_batch)
            records += len(decisions_batch)

    return records, errors


def _map_csv_to_db(row: Dict[str, str], decision_id: str, file_timestamp: str) -> Dict[str, Any]:
    """Map CSV row to database record format."""

    def safe_float(val: str, default: Optional[float] = None) -> Optional[float]:
        if not val or val == '' or val == 'None':
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def safe_int(val: str, default: Optional[int] = None) -> Optional[int]:
        if not val or val == '' or val == 'None':
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    # Parse timestamp from row or use file timestamp
    timestamp = row.get('timestamp', '')
    if not timestamp:
        try:
            # Parse from file timestamp (format: YYYYMMDD_HHMMSS)
            timestamp = datetime.strptime(file_timestamp, "%Y%m%d_%H%M%S").isoformat()
        except:
            timestamp = datetime.now().isoformat()

    return {
        'decision_id': decision_id,
        'timestamp': timestamp,
        'event_ticker': row.get('event_ticker', ''),
        'event_title': row.get('event_title', row.get('event_name', '')),
        'market_ticker': row.get('ticker', row.get('market_ticker', '')),
        'market_title': row.get('market_title', row.get('market_name', '')),
        'action': row.get('action', ''),
        'bet_amount': safe_float(row.get('amount', row.get('bet_amount', '0')), 0),
        'confidence': safe_float(row.get('confidence', '')),
        'reasoning': row.get('reasoning', ''),
        'research_probability': safe_float(row.get('research_probability', '')),
        'research_reasoning': row.get('research_reasoning', ''),
        'research_summary': row.get('research_summary', ''),
        'raw_research': row.get('raw_research', '')[:10000] if row.get('raw_research') else '',
        'market_yes_price': safe_float(row.get('market_yes_price', row.get('yes_ask', ''))),
        'market_no_price': safe_float(row.get('market_no_price', row.get('no_ask', ''))),
        'market_yes_mid': safe_float(row.get('market_yes_mid', '')),
        'market_no_mid': safe_float(row.get('market_no_mid', '')),
        'expected_return': safe_float(row.get('expected_return', '')),
        'r_score': safe_float(row.get('r_score', '')),
        'kelly_fraction': safe_float(row.get('kelly_fraction', '')),
        'calc_market_prob': safe_float(row.get('calc_market_prob', row.get('market_price', ''))),
        'calc_research_prob': safe_float(row.get('calc_research_prob', '')),
        'is_hedge': 1 if row.get('is_hedge', '').lower() == 'true' else 0,
        'hedge_for': row.get('hedge_for', ''),
        'market_yes_bid': safe_float(row.get('market_yes_bid', row.get('yes_bid', ''))),
        'market_yes_ask': safe_float(row.get('market_yes_ask', row.get('yes_ask', ''))),
        'market_no_bid': safe_float(row.get('market_no_bid', row.get('no_bid', ''))),
        'market_no_ask': safe_float(row.get('market_no_ask', row.get('no_ask', ''))),
        'market_volume': safe_float(row.get('market_volume', row.get('volume', ''))),
        'market_status': row.get('market_status', row.get('status', '')),
        'market_close_time': row.get('market_close_time', row.get('close_time', '')),
        'status': 'pending' if row.get('action', '') != 'skip' else 'skipped'
    }


async def run_csv_migration(csv_dir: str = "betting_decisions", archive: bool = False):
    """Run CSV migration as a standalone script."""
    from config import load_config

    config = load_config()
    console = Console()

    try:
        db = await get_database(config.database.db_path)
        await migrate_csv_files(db, csv_dir, archive)
    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        logger.exception("CSV migration failed")


if __name__ == "__main__":
    asyncio.run(run_csv_migration())
