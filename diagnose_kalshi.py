#!/usr/bin/env python3
"""
Kalshi API Diagnostic Script
=============================
Diagnoses issues with the Kalshi API connection.
"""

import asyncio
import base64
import time

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def sign_message(private_key_pem: str, message: str) -> str:
    """Sign a message using RSA private key."""
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(), password=None, backend=default_backend()
    )

    signature = private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )

    return base64.b64encode(signature).decode()


def get_headers(api_key: str, private_key: str, method: str, path: str) -> dict:
    """Generate headers with RSA signature."""
    timestamp = str(int(time.time() * 1000))
    message = f"{timestamp}{method}{path}"
    signature = sign_message(private_key, message)

    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json",
    }


async def diagnose_kalshi():
    """Run comprehensive Kalshi API diagnostics."""

    console.print(Panel("Kalshi API Diagnostic Tool", style="blue"))

    # Load config
    console.print("\n[cyan]Step 1: Loading configuration...[/cyan]")
    try:
        from config import BotConfig

        config = BotConfig()
        console.print(f"  [green]OK[/green] Config loaded")
        console.print(f"  Base URL: {config.kalshi.base_url}")
        console.print(f"  API Key: {config.kalshi.api_key[:8]}...")
        console.print(f"  Demo Mode: {config.kalshi.use_demo}")
    except Exception as e:
        console.print(f"  [red]FAIL[/red] Config error: {e}")
        return

    # Validate private key
    console.print("\n[cyan]Step 2: Validating private key...[/cyan]")
    try:
        private_key = serialization.load_pem_private_key(
            config.kalshi.private_key.encode(), password=None, backend=default_backend()
        )
        key_size = private_key.key_size
        console.print(f"  [green]OK[/green] Private key valid (RSA-{key_size})")
    except Exception as e:
        console.print(f"  [red]FAIL[/red] Private key error: {e}")
        return

    # Test signature generation
    console.print("\n[cyan]Step 3: Testing signature generation...[/cyan]")
    try:
        test_message = f"{int(time.time() * 1000)}GET/trade-api/v2/events"
        signature = sign_message(config.kalshi.private_key, test_message)
        console.print(
            f"  [green]OK[/green] Signature generated ({len(signature)} chars)"
        )
    except Exception as e:
        console.print(f"  [red]FAIL[/red] Signature error: {e}")
        return

    # Test API connectivity
    console.print("\n[cyan]Step 4: Testing API connectivity...[/cyan]")

    async with httpx.AsyncClient(timeout=30.0) as client:

        # Test 1: Simple GET to events endpoint
        console.print("\n  [yellow]Test 4.1: GET /trade-api/v2/events[/yellow]")
        try:
            path = "/trade-api/v2/events"
            headers = get_headers(
                config.kalshi.api_key, config.kalshi.private_key, "GET", path
            )

            url = f"{config.kalshi.base_url}{path}"
            console.print(f"    URL: {url}")

            response = await client.get(
                url, headers=headers, params={"limit": 5, "status": "open"}
            )

            console.print(f"    Status: {response.status_code}")
            console.print(f"    Headers: {dict(response.headers)[:200]}...")

            if response.status_code == 200:
                data = response.json()
                events = data.get("events", []) if data else []
                console.print(f"    [green]OK[/green] Got {len(events)} events")

                if events:
                    console.print("\n    Sample events:")
                    for i, event in enumerate(events[:3]):
                        console.print(
                            f"      {i+1}. {event.get('title', 'Unknown')[:50]}"
                        )
                        console.print(f"         Ticker: {event.get('event_ticker')}")
                        console.print(
                            f"         Markets: {len(event.get('markets', []))}"
                        )
                else:
                    console.print("    [yellow]WARNING[/yellow] No events returned")
                    console.print(f"    Response body: {response.text[:500]}")
            else:
                console.print(f"    [red]FAIL[/red] HTTP {response.status_code}")
                console.print(f"    Response: {response.text[:500]}")

        except Exception as e:
            console.print(f"    [red]FAIL[/red] Error: {e}")

        # Test 2: Exchange status (if available)
        console.print(
            "\n  [yellow]Test 4.2: GET /trade-api/v2/exchange/status[/yellow]"
        )
        try:
            path = "/trade-api/v2/exchange/status"
            headers = get_headers(
                config.kalshi.api_key, config.kalshi.private_key, "GET", path
            )

            response = await client.get(
                f"{config.kalshi.base_url}{path}", headers=headers
            )

            console.print(f"    Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                console.print(f"    [green]OK[/green] Exchange status: {data}")
            else:
                console.print(
                    f"    [yellow]INFO[/yellow] Endpoint may not exist: {response.status_code}"
                )

        except Exception as e:
            console.print(f"    [yellow]INFO[/yellow] Error: {e}")

        # Test 3: Portfolio balance
        console.print(
            "\n  [yellow]Test 4.3: GET /trade-api/v2/portfolio/balance[/yellow]"
        )
        try:
            path = "/trade-api/v2/portfolio/balance"
            headers = get_headers(
                config.kalshi.api_key, config.kalshi.private_key, "GET", path
            )

            response = await client.get(
                f"{config.kalshi.base_url}{path}", headers=headers
            )

            console.print(f"    Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                balance = data.get("balance", 0) / 100  # Cents to dollars
                console.print(f"    [green]OK[/green] Balance: ${balance:.2f}")
            else:
                console.print(f"    [red]FAIL[/red] Response: {response.text[:200]}")

        except Exception as e:
            console.print(f"    [red]FAIL[/red] Error: {e}")

        # Test 4: Portfolio positions
        console.print(
            "\n  [yellow]Test 4.4: GET /trade-api/v2/portfolio/positions[/yellow]"
        )
        try:
            path = "/trade-api/v2/portfolio/positions"
            headers = get_headers(
                config.kalshi.api_key, config.kalshi.private_key, "GET", path
            )

            response = await client.get(
                f"{config.kalshi.base_url}{path}", headers=headers
            )

            console.print(f"    Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                positions = data.get("market_positions", []) if data else []
                console.print(f"    [green]OK[/green] Positions: {len(positions)}")
            else:
                console.print(f"    [red]FAIL[/red] Response: {response.text[:200]}")

        except Exception as e:
            console.print(f"    [red]FAIL[/red] Error: {e}")

    # Check KalshiClient initialization
    console.print("\n[cyan]Step 5: Testing KalshiClient class...[/cyan]")
    try:
        from kalshi_client import KalshiClient

        kalshi = KalshiClient(config.kalshi)

        # Check if client is None before login
        console.print(f"  Client before login: {kalshi.client}")

        # Login
        await kalshi.login()
        console.print(f"  Client after login: {type(kalshi.client)}")

        # Now try get_events
        events = await kalshi.get_events(limit=5)
        console.print(f"  [green]OK[/green] get_events() returned {len(events)} events")

        if events:
            for event in events[:3]:
                console.print(f"    - {event.get('title', 'Unknown')[:50]}")

    except Exception as e:
        console.print(f"  [red]FAIL[/red] KalshiClient error: {e}")
        import traceback

        traceback.print_exc()

    console.print("\n[cyan]Diagnostic complete.[/cyan]")


if __name__ == "__main__":
    asyncio.run(diagnose_kalshi())
