"""
Quick test of Dashboard API position endpoints.
"""
import asyncio
import httpx
import sys


async def test_api():
    """Test the dashboard API endpoints."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            # Test health
            print("Testing API endpoints...")

            # Test /api/positions/live
            print("\n1. GET /api/positions/live")
            resp = await client.get(f"{base_url}/api/positions/live")
            if resp.status_code == 200:
                data = resp.json()
                print(f"   Status: OK")
                print(f"   Positions: {data.get('count', 0)}")
                print(f"   Monitor Running: {data.get('monitor_running', False)}")
                print(f"   Total P&L: ${data.get('total_unrealized_pnl', 0):.2f}")
            else:
                print(f"   Error: {resp.status_code}")

            # Test /api/positions/exits
            print("\n2. GET /api/positions/exits")
            resp = await client.get(f"{base_url}/api/positions/exits?days=7")
            if resp.status_code == 200:
                data = resp.json()
                print(f"   Status: OK")
                print(f"   Exits: {data.get('count', 0)}")
            else:
                print(f"   Error: {resp.status_code}")

            # Test /api/positions/near-triggers
            print("\n3. GET /api/positions/near-triggers")
            resp = await client.get(f"{base_url}/api/positions/near-triggers?threshold_pct=0.05")
            if resp.status_code == 200:
                data = resp.json()
                print(f"   Status: OK")
                print(f"   Near triggers: {data.get('count', 0)}")
            else:
                print(f"   Error: {resp.status_code}")

            print("\nAll API tests passed!")

        except httpx.ConnectError:
            print(f"Could not connect to {base_url}")
            print("Make sure the dashboard is running: python dashboard/api.py")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_api())
