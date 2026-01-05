#!/usr/bin/env python3
"""
Performance analysis dashboard for Kalshi Deep Trading Bot.

Analyzes betting decisions from CSV files or SQLite database to calculate
ROI, win rate, calibration, and other performance metrics.

Usage:
    uv run analyze_performance.py              # Load from CSV files
    uv run analyze_performance.py --db         # Load from SQLite database
    uv run analyze_performance.py --detailed
    uv run analyze_performance.py --pnl        # Show P&L from settled bets
"""
import argparse
import asyncio
import glob
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from calibration_tracker import CalibrationTracker


console = Console()


async def load_from_database(db_path: str = "trading_bot.db") -> Optional[pd.DataFrame]:
    """Load betting decisions from SQLite database."""
    try:
        from db import get_database

        db = await get_database(db_path)

        # Fetch all decisions
        query = "SELECT * FROM betting_decisions ORDER BY timestamp DESC"
        rows = await db.fetchall(query)

        if not rows:
            console.print("[yellow]No decisions found in database[/yellow]")
            return None

        df = pd.DataFrame(rows)
        console.print(f"[green]Loaded {len(df)} decisions from SQLite database[/green]")

        return df

    except Exception as e:
        console.print(f"[red]Error loading from database: {e}[/red]")
        return None


async def get_pnl_summary(db_path: str = "trading_bot.db") -> Dict[str, Any]:
    """Get P&L summary from settled decisions."""
    try:
        from db import get_database
        from db.queries import Queries

        db = await get_database(db_path)

        # Get overall P&L
        pnl_row = await db.fetchone(Queries.GET_TOTAL_PNL)

        # Get daily P&L
        daily_rows = await db.fetchall(Queries.GET_DAILY_PNL)

        # Get R-score effectiveness
        rscore_rows = await db.fetchall(Queries.GET_R_SCORE_EFFECTIVENESS)

        return {
            'overall': dict(pnl_row) if pnl_row else {},
            'daily': [dict(r) for r in daily_rows],
            'r_score': [dict(r) for r in rscore_rows]
        }

    except Exception as e:
        console.print(f"[red]Error getting P&L: {e}[/red]")
        return {}


def print_pnl_summary(pnl_data: Dict[str, Any]):
    """Print P&L summary tables."""
    console.print("\n")
    console.print(Panel.fit("[bold]PROFIT & LOSS SUMMARY[/bold]", style="green"))

    overall = pnl_data.get('overall', {})

    if overall:
        table = Table(title="Overall Performance", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        total_pnl = overall.get('total_pnl', 0) or 0
        total_wagered = overall.get('total_wagered', 0) or 0
        total_bets = overall.get('total_bets', 0)
        winning = overall.get('winning_bets', 0)
        losing = overall.get('losing_bets', 0)

        pnl_color = "green" if total_pnl >= 0 else "red"
        roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = winning / total_bets if total_bets > 0 else 0

        table.add_row("Settled Bets", str(total_bets))
        table.add_row("Winning Bets", str(winning))
        table.add_row("Losing Bets", str(losing))
        table.add_row("Win Rate", f"{win_rate:.1%}")
        table.add_row("Total Wagered", f"${total_wagered:,.2f}")
        table.add_row("Net P&L", f"[{pnl_color}]${total_pnl:,.2f}[/{pnl_color}]")
        table.add_row("ROI", f"[{pnl_color}]{roi:,.1f}%[/{pnl_color}]")

        console.print(table)

    # Daily P&L
    daily = pnl_data.get('daily', [])
    if daily:
        console.print("\n")
        table = Table(title="Daily P&L (Last 10 Days)", show_header=True)
        table.add_column("Date", style="cyan")
        table.add_column("Bets", style="yellow", justify="right")
        table.add_column("Wins", style="green", justify="right")
        table.add_column("Wagered", style="blue", justify="right")
        table.add_column("P&L", justify="right")

        for row in daily[:10]:
            pnl = row.get('daily_pnl', 0) or 0
            pnl_color = "green" if pnl >= 0 else "red"
            table.add_row(
                str(row.get('date', '')),
                str(row.get('total_bets', 0)),
                str(row.get('wins', 0)),
                f"${row.get('total_wagered', 0):,.2f}",
                f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]"
            )

        console.print(table)

    # R-score effectiveness
    rscore = pnl_data.get('r_score', [])
    if rscore:
        console.print("\n")
        table = Table(title="R-Score Effectiveness", show_header=True)
        table.add_column("R-Score Bucket", style="cyan")
        table.add_column("Count", style="yellow", justify="right")
        table.add_column("Win Rate", style="green", justify="right")
        table.add_column("Total P&L", justify="right")
        table.add_column("Avg P&L", justify="right")

        for row in rscore:
            total_pnl = row.get('total_pnl', 0) or 0
            avg_pnl = row.get('avg_pnl', 0) or 0
            win_rate = row.get('win_rate', 0) or 0
            pnl_color = "green" if total_pnl >= 0 else "red"

            table.add_row(
                row.get('r_score_bucket', ''),
                str(row.get('count', 0)),
                f"{win_rate:.1%}",
                f"[{pnl_color}]${total_pnl:,.2f}[/{pnl_color}]",
                f"[{pnl_color}]${avg_pnl:,.2f}[/{pnl_color}]"
            )

        console.print(table)


def load_betting_decisions(csv_dir: str = "betting_decisions") -> Optional[pd.DataFrame]:
    """Load all betting decision CSVs into a DataFrame."""
    files = glob.glob(f"{csv_dir}/betting_decisions_*.csv")

    if not files:
        console.print(f"[yellow]No betting decision files found in {csv_dir}/[/yellow]")
        return None

    console.print(f"[blue]Loading {len(files)} CSV files...[/blue]")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = Path(f).name
            dfs.append(df)
        except Exception as e:
            console.print(f"[red]Error loading {f}: {e}[/red]")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    console.print(f"[green]Loaded {len(combined)} total decisions[/green]")

    return combined


def analyze_decisions(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze betting decisions and calculate metrics."""
    # Filter to actionable decisions only
    actionable = df[df['action'] != 'skip'].copy()
    skipped = df[df['action'] == 'skip'].copy()

    # Separate main bets from hedges
    main_bets = actionable[actionable['is_hedge'] != True] if 'is_hedge' in actionable.columns else actionable
    hedges = actionable[actionable['is_hedge'] == True] if 'is_hedge' in actionable.columns else pd.DataFrame()

    metrics = {
        'total_decisions': len(df),
        'actionable_decisions': len(actionable),
        'skipped_decisions': len(skipped),
        'main_bets': len(main_bets),
        'hedge_bets': len(hedges),
    }

    if len(actionable) > 0:
        # Capital metrics
        metrics['total_capital_deployed'] = actionable['bet_amount'].sum()
        metrics['avg_bet_size'] = actionable['bet_amount'].mean()
        metrics['max_bet_size'] = actionable['bet_amount'].max()
        metrics['min_bet_size'] = actionable['bet_amount'].min()

        # Confidence metrics
        metrics['avg_confidence'] = actionable['confidence'].mean()
        metrics['high_confidence_bets'] = len(actionable[actionable['confidence'] >= 0.7])

        # R-score metrics
        if 'r_score' in actionable.columns:
            r_scores = actionable['r_score'].dropna()
            if len(r_scores) > 0:
                metrics['avg_r_score'] = r_scores.mean()
                metrics['max_r_score'] = r_scores.max()
                metrics['min_r_score'] = r_scores.min()
                metrics['r_score_std'] = r_scores.std()

        # Expected return metrics
        if 'expected_return' in actionable.columns:
            exp_returns = actionable['expected_return'].dropna()
            if len(exp_returns) > 0:
                metrics['avg_expected_return'] = exp_returns.mean()
                metrics['total_expected_value'] = (actionable['bet_amount'] * actionable['expected_return']).sum()

        # Kelly metrics
        if 'kelly_fraction' in actionable.columns:
            kelly = actionable['kelly_fraction'].dropna()
            if len(kelly) > 0:
                metrics['avg_kelly_fraction'] = kelly.mean()

        # Action distribution
        action_counts = actionable['action'].value_counts().to_dict()
        metrics['buy_yes_count'] = action_counts.get('buy_yes', 0)
        metrics['buy_no_count'] = action_counts.get('buy_no', 0)

        # Research probability vs market price spread
        if 'research_probability' in actionable.columns and 'calc_market_prob' in actionable.columns:
            actionable['edge'] = actionable['research_probability'] - actionable['calc_market_prob']
            metrics['avg_edge'] = actionable['edge'].mean()

    return metrics


def analyze_by_event(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by event."""
    actionable = df[df['action'] != 'skip'].copy()

    if len(actionable) == 0:
        return pd.DataFrame()

    # Group by event
    event_stats = actionable.groupby('event_ticker').agg({
        'bet_amount': ['count', 'sum', 'mean'],
        'confidence': 'mean',
        'r_score': 'mean',
    }).round(3)

    event_stats.columns = ['bet_count', 'total_bet', 'avg_bet', 'avg_confidence', 'avg_r_score']
    event_stats = event_stats.sort_values('total_bet', ascending=False)

    return event_stats


def analyze_by_action(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Analyze performance by action type."""
    actionable = df[df['action'] != 'skip'].copy()

    if len(actionable) == 0:
        return {}

    result = {}
    for action in ['buy_yes', 'buy_no']:
        subset = actionable[actionable['action'] == action]
        if len(subset) > 0:
            result[action] = {
                'count': len(subset),
                'total_bet': subset['bet_amount'].sum(),
                'avg_bet': subset['bet_amount'].mean(),
                'avg_confidence': subset['confidence'].mean(),
                'avg_r_score': subset['r_score'].mean() if 'r_score' in subset.columns else None,
            }

    return result


def print_summary(metrics: Dict[str, Any]):
    """Print formatted summary of metrics."""
    console.print("\n")
    console.print(Panel.fit("[bold]TRADING BOT PERFORMANCE SUMMARY[/bold]", style="blue"))

    # Decision counts table
    table1 = Table(title="Decision Summary", show_header=True)
    table1.add_column("Metric", style="cyan")
    table1.add_column("Value", style="green", justify="right")

    table1.add_row("Total Decisions", str(metrics.get('total_decisions', 0)))
    table1.add_row("Actionable Bets", str(metrics.get('actionable_decisions', 0)))
    table1.add_row("Skipped", str(metrics.get('skipped_decisions', 0)))
    table1.add_row("Main Bets", str(metrics.get('main_bets', 0)))
    table1.add_row("Hedge Bets", str(metrics.get('hedge_bets', 0)))

    console.print(table1)

    if metrics.get('actionable_decisions', 0) > 0:
        # Capital table
        table2 = Table(title="Capital Deployed", show_header=True)
        table2.add_column("Metric", style="cyan")
        table2.add_column("Value", style="green", justify="right")

        table2.add_row("Total Capital", f"${metrics.get('total_capital_deployed', 0):,.2f}")
        table2.add_row("Average Bet", f"${metrics.get('avg_bet_size', 0):,.2f}")
        table2.add_row("Max Bet", f"${metrics.get('max_bet_size', 0):,.2f}")
        table2.add_row("Min Bet", f"${metrics.get('min_bet_size', 0):,.2f}")

        console.print(table2)

        # Quality metrics table
        table3 = Table(title="Bet Quality Metrics", show_header=True)
        table3.add_column("Metric", style="cyan")
        table3.add_column("Value", style="green", justify="right")

        table3.add_row("Avg Confidence", f"{metrics.get('avg_confidence', 0):.3f}")
        table3.add_row("High Confidence Bets (≥0.7)", str(metrics.get('high_confidence_bets', 0)))

        if 'avg_r_score' in metrics:
            table3.add_row("Avg R-Score", f"{metrics.get('avg_r_score', 0):.3f}")
            table3.add_row("Max R-Score", f"{metrics.get('max_r_score', 0):.3f}")
            table3.add_row("R-Score Std Dev", f"{metrics.get('r_score_std', 0):.3f}")

        if 'avg_expected_return' in metrics:
            table3.add_row("Avg Expected Return", f"{metrics.get('avg_expected_return', 0):.1%}")
            table3.add_row("Total Expected Value", f"${metrics.get('total_expected_value', 0):,.2f}")

        if 'avg_kelly_fraction' in metrics:
            table3.add_row("Avg Kelly Fraction", f"{metrics.get('avg_kelly_fraction', 0):.3f}")

        console.print(table3)

        # Action distribution
        table4 = Table(title="Action Distribution", show_header=True)
        table4.add_column("Action", style="cyan")
        table4.add_column("Count", style="green", justify="right")
        table4.add_column("Percentage", style="yellow", justify="right")

        total = metrics.get('buy_yes_count', 0) + metrics.get('buy_no_count', 0)
        if total > 0:
            table4.add_row("BUY YES", str(metrics.get('buy_yes_count', 0)),
                          f"{metrics.get('buy_yes_count', 0)/total:.1%}")
            table4.add_row("BUY NO", str(metrics.get('buy_no_count', 0)),
                          f"{metrics.get('buy_no_count', 0)/total:.1%}")

        console.print(table4)


def print_calibration_summary():
    """Print calibration tracker summary if available."""
    try:
        tracker = CalibrationTracker()
        stats = tracker.calculate_calibration_stats()

        if stats['resolved_predictions'] > 0:
            console.print("\n")
            console.print(Panel.fit("[bold]CALIBRATION METRICS[/bold]", style="magenta"))

            table = Table(show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")

            table.add_row("Resolved Predictions", str(stats['resolved_predictions']))

            if stats['brier_score'] is not None:
                table.add_row("Brier Score", f"{stats['brier_score']:.4f}")
            if stats['accuracy'] is not None:
                table.add_row("Accuracy", f"{stats['accuracy']:.1%}")
            if stats['avg_confidence'] is not None:
                table.add_row("Avg Confidence", f"{stats['avg_confidence']:.3f}")
            if stats['confidence_calibration'] is not None:
                table.add_row("Calibration Error", f"{stats['confidence_calibration']:.3f}")

            console.print(table)

            if stats.get('bucket_accuracy'):
                bucket_table = Table(title="Accuracy by Confidence Bucket", show_header=True)
                bucket_table.add_column("Bucket", style="cyan")
                bucket_table.add_column("Count", style="yellow", justify="right")
                bucket_table.add_column("Accuracy", style="green", justify="right")

                for bucket, data in stats['bucket_accuracy'].items():
                    bucket_table.add_row(bucket, str(data['count']), f"{data['accuracy']:.1%}")

                console.print(bucket_table)
        else:
            console.print("[yellow]No resolved predictions yet for calibration analysis[/yellow]")

    except Exception as e:
        console.print(f"[yellow]Calibration data not available: {e}[/yellow]")


async def main_async(args):
    """Async main function for database operations."""
    console.print("[bold blue]Kalshi Deep Trading Bot - Performance Analysis[/bold blue]\n")

    # Handle P&L mode
    if args.pnl:
        pnl_data = await get_pnl_summary()
        if pnl_data:
            print_pnl_summary(pnl_data)
        else:
            console.print("[yellow]No P&L data available. Run --reconcile first.[/yellow]")
        return

    # Load data from database or CSV
    if args.db:
        df = await load_from_database()
    else:
        df = load_betting_decisions(args.csv_dir)

    if df is None or len(df) == 0:
        console.print("[red]No data to analyze[/red]")
        return

    # Calculate and display metrics
    metrics = analyze_decisions(df)
    print_summary(metrics)

    # Show calibration if available
    print_calibration_summary()

    # Detailed analysis
    if args.detailed:
        console.print("\n")
        console.print(Panel.fit("[bold]DETAILED EVENT ANALYSIS[/bold]", style="yellow"))

        event_stats = analyze_by_event(df)
        if len(event_stats) > 0:
            table = Table(title="Top Events by Capital Deployed", show_header=True)
            table.add_column("Event", style="cyan")
            table.add_column("Bets", style="yellow", justify="right")
            table.add_column("Total $", style="green", justify="right")
            table.add_column("Avg $", style="green", justify="right")
            table.add_column("Avg Conf", style="magenta", justify="right")
            table.add_column("Avg R-Score", style="blue", justify="right")

            for event_ticker, row in event_stats.head(15).iterrows():
                table.add_row(
                    str(event_ticker)[:30],
                    str(int(row['bet_count'])),
                    f"${row['total_bet']:,.2f}",
                    f"${row['avg_bet']:,.2f}",
                    f"{row['avg_confidence']:.3f}",
                    f"{row['avg_r_score']:.3f}" if pd.notna(row['avg_r_score']) else "N/A"
                )

            console.print(table)

        # Action breakdown
        action_stats = analyze_by_action(df)
        if action_stats:
            console.print("\n")
            table = Table(title="Performance by Action Type", show_header=True)
            table.add_column("Action", style="cyan")
            table.add_column("Count", style="yellow", justify="right")
            table.add_column("Total $", style="green", justify="right")
            table.add_column("Avg $", style="green", justify="right")
            table.add_column("Avg Conf", style="magenta", justify="right")

            for action, stats in action_stats.items():
                table.add_row(
                    action.upper(),
                    str(stats['count']),
                    f"${stats['total_bet']:,.2f}",
                    f"${stats['avg_bet']:,.2f}",
                    f"{stats['avg_confidence']:.3f}"
                )

            console.print(table)

    console.print("\n[dim]Options: --db (load from SQLite), --pnl (show P&L), --detailed (per-event)[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Analyze Kalshi trading bot performance")
    parser.add_argument("--csv-dir", default="betting_decisions", help="Directory containing CSV files")
    parser.add_argument("--db", action="store_true", help="Load data from SQLite database instead of CSV")
    parser.add_argument("--pnl", action="store_true", help="Show P&L summary from settled bets")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-event analysis")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
