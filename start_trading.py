"""
Startup orchestrator for Kalshi Trading Bot with TrendRadar integration.
Starts TrendRadar MCP server and then runs the trading bot.
"""
import subprocess
import sys
import time
import asyncio
import os
import signal
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
TRENDRADAR_DIR = SCRIPT_DIR.parent / "TrendRadar"
BOT_DIR = SCRIPT_DIR


def find_python():
    """Find the Python executable."""
    # Try common locations
    candidates = [
        sys.executable,
        "python",
        "python3",
        "py",
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return candidate
        except Exception:
            continue
    return sys.executable


def start_trendradar(python_exe: str):
    """Start TrendRadar MCP server in HTTP mode."""
    print("Starting TrendRadar MCP server...")
    print(f"TrendRadar directory: {TRENDRADAR_DIR}")

    if not TRENDRADAR_DIR.exists():
        print(f"[ERROR] TrendRadar directory not found: {TRENDRADAR_DIR}")
        return None

    # Start TrendRadar in HTTP mode
    cmd = [
        python_exe, "-m", "mcp_server.server",
        "--transport", "http",
        "--port", "3333"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(TRENDRADAR_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        return process
    except Exception as e:
        print(f"[ERROR] Failed to start TrendRadar: {e}")
        return None


async def wait_for_trendradar(url: str = "http://localhost:3333", timeout: int = 30):
    """Wait for TrendRadar server to become healthy."""
    print(f"Waiting for TrendRadar at {url}...")

    try:
        import httpx
    except ImportError:
        print("httpx not installed, skipping health check")
        await asyncio.sleep(5)  # Just wait a bit
        return True

    async with httpx.AsyncClient() as client:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = await client.get(f"{url}/", timeout=2.0)
                if response.status_code in [200, 404]:  # Server is up
                    print("[OK] TrendRadar server is ready!")
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
            print(".", end="", flush=True)

    print("\n[WARNING] TrendRadar did not become healthy within timeout")
    return False


def run_trading_bot(python_exe: str, args: list):
    """Run the trading bot with given arguments."""
    print("\nStarting Kalshi Trading Bot...")

    cmd = [python_exe, "trading_bot.py"] + args

    process = subprocess.Popen(
        cmd,
        cwd=str(BOT_DIR),
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    return process


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Start Kalshi Trading Bot with TrendRadar news intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_trading.py                     # Dry run with TrendRadar
  python start_trading.py --live              # Live trading with TrendRadar
  python start_trading.py --no-trendradar     # Run without TrendRadar
  python start_trading.py --max-expiration-hours 6 --live
"""
    )
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--no-trendradar", action="store_true", help="Skip TrendRadar startup")
    parser.add_argument("--max-expiration-hours", type=int, help="Max expiration filter in hours")
    parser.add_argument("--reconcile", action="store_true", help="Run reconciliation only")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--migrate", action="store_true", help="Run database migration only")

    args = parser.parse_args()

    python_exe = find_python()
    print(f"Using Python: {python_exe}")

    trendradar_process = None
    bot_process = None

    def cleanup(signum=None, frame=None):
        """Cleanup on exit."""
        print("\nShutting down...")
        if bot_process:
            try:
                bot_process.terminate()
                bot_process.wait(timeout=5)
            except Exception:
                pass
        if trendradar_process:
            try:
                trendradar_process.terminate()
                trendradar_process.wait(timeout=5)
            except Exception:
                pass
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Start TrendRadar if not disabled
        if not args.no_trendradar:
            trendradar_process = start_trendradar(python_exe)

            if trendradar_process:
                # Wait for it to be ready
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                healthy = loop.run_until_complete(wait_for_trendradar())
                loop.close()

                if not healthy:
                    print("[WARNING] Continuing without TrendRadar integration...")
            else:
                print("[WARNING] Could not start TrendRadar, continuing without it...")
        else:
            print("[INFO] TrendRadar integration disabled via --no-trendradar")

        # Build bot arguments
        bot_args = []
        if args.live:
            bot_args.append("--live")
        if args.max_expiration_hours:
            bot_args.extend(["--max-expiration-hours", str(args.max_expiration_hours)])
        if args.reconcile:
            bot_args.append("--reconcile")
        if args.stats:
            bot_args.append("--stats")
        if args.migrate:
            bot_args.append("--migrate")

        # Run the trading bot
        bot_process = run_trading_bot(python_exe, bot_args)

        # Wait for bot to complete
        bot_process.wait()

    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
