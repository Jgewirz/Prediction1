"""
Combined server that runs both the Dashboard API and the Trading Bot.
This is the main entry point for Render deployment.
"""
import asyncio
import os
import sys
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def run_trading_bot():
    """Run the trading bot in a separate thread."""
    import asyncio
    from continuous_bot import ContinuousTradingBot

    # Check if live trading is enabled via environment variable
    live_trading = os.getenv("LIVE_TRADING", "true").lower() == "true"

    bot = ContinuousTradingBot(live_trading=live_trading)

    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(bot.run())
    except Exception as e:
        print(f"Trading bot error: {e}")
    finally:
        loop.close()


def main():
    """Main entry point - starts dashboard API and trading bot."""
    import uvicorn

    # Start trading bot in background thread
    bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
    bot_thread.start()
    print("Trading bot started in background thread")

    # Get port from environment (Render sets PORT)
    port = int(os.getenv("PORT", "10000"))

    # Run the dashboard API (this binds to the port Render expects)
    print(f"Starting Dashboard API on port {port}")
    uvicorn.run(
        "dashboard.api:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
