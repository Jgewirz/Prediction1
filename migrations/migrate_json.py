"""
Migrate existing calibration JSON data to SQLite database.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger
from rich.console import Console

from db import get_database, Database


async def migrate_calibration_json(
    db: Database,
    json_path: str = "calibration_data.json",
    backup: bool = True
) -> Dict[str, Any]:
    """
    Migrate calibration_data.json to SQLite calibration_records table.

    Args:
        db: Database connection
        json_path: Path to calibration JSON file
        backup: If True, rename original file to .backup

    Returns:
        Dict with migration statistics
    """
    console = Console()
    json_file = Path(json_path)

    if not json_file.exists():
        console.print(f"[yellow]Calibration file not found: {json_path}[/yellow]")
        return {"records": 0, "errors": 0}

    console.print(f"[blue]Migrating calibration data from {json_path}[/blue]")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in {json_path}: {e}[/red]")
        return {"records": 0, "errors": 1}

    # Handle different JSON structures
    predictions = []
    if isinstance(data, dict):
        # Could be {"predictions": [...]} or direct dict of prediction_id -> prediction
        if "predictions" in data:
            predictions = data["predictions"]
        else:
            # Assume dict of prediction_id -> prediction data
            for pred_id, pred_data in data.items():
                if isinstance(pred_data, dict):
                    pred_data["prediction_id"] = pred_id
                    predictions.append(pred_data)
    elif isinstance(data, list):
        predictions = data

    if not predictions:
        console.print("[yellow]No predictions found in calibration file[/yellow]")
        return {"records": 0, "errors": 0}

    console.print(f"Found {len(predictions)} predictions to migrate")

    records = 0
    errors = 0

    for pred in predictions:
        try:
            record = _map_prediction_to_db(pred)

            # Check if already exists
            exists = await db.fetchone(
                "SELECT 1 FROM calibration_records WHERE prediction_id = ?",
                (record['prediction_id'],)
            )
            if exists:
                continue

            await db.insert_calibration_record(record)
            records += 1

        except Exception as e:
            logger.debug(f"Error migrating prediction: {e}")
            errors += 1

    # Backup original file
    if backup and records > 0:
        backup_path = json_file.with_suffix('.json.backup')
        try:
            json_file.rename(backup_path)
            console.print(f"[green]Backed up original file to {backup_path}[/green]")
        except Exception as e:
            logger.warning(f"Could not backup calibration file: {e}")

    summary = {
        "records": records,
        "errors": errors
    }

    console.print(f"\n[bold green]Calibration migration complete:[/bold green]")
    console.print(f"  Records migrated: {summary['records']}")
    console.print(f"  Errors: {summary['errors']}")

    return summary


def _map_prediction_to_db(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Map calibration prediction to database record format."""

    def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
        if val is None or val == '':
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Generate prediction_id if not present
    prediction_id = pred.get('prediction_id', '')
    if not prediction_id:
        ticker = pred.get('ticker', 'UNKNOWN')
        timestamp = pred.get('timestamp', datetime.now().isoformat())
        prediction_id = f"{ticker}_{timestamp}"

    # Parse timestamp
    timestamp = pred.get('timestamp', '')
    if not timestamp:
        timestamp = datetime.now().isoformat()

    return {
        'prediction_id': prediction_id,
        'ticker': pred.get('ticker', pred.get('market_ticker', '')),
        'event_ticker': pred.get('event_ticker', ''),
        'predicted_prob': safe_float(pred.get('predicted_prob', pred.get('research_probability', ''))),
        'market_price': safe_float(pred.get('market_price', '')),
        'confidence': safe_float(pred.get('confidence', '')),
        'r_score': safe_float(pred.get('r_score', '')),
        'action': pred.get('action', ''),
        'reasoning': pred.get('reasoning', ''),
        'timestamp': timestamp,
        'outcome': safe_float(pred.get('outcome', '')),
        'resolved_timestamp': pred.get('resolved_timestamp', ''),
        'actual_payout': safe_float(pred.get('actual_payout', '')),
    }


async def run_json_migration(json_path: str = "calibration_data.json", backup: bool = True):
    """Run JSON migration as a standalone script."""
    from config import load_config

    config = load_config()
    console = Console()

    try:
        db = await get_database(config.database.db_path)
        await migrate_calibration_json(db, json_path, backup)
    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        logger.exception("JSON migration failed")


if __name__ == "__main__":
    asyncio.run(run_json_migration())
