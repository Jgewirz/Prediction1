"""
Simple Kalshi API Client with RSA authentication
"""

import asyncio
import hashlib
import json
import time
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import httpx
from loguru import logger
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from config import KalshiConfig


class KalshiClient:
    """Simple Kalshi API client for basic trading operations."""
    
    def __init__(self, config: KalshiConfig, minimum_time_remaining_hours: float = 1.0, max_markets_per_event: int = 10, max_close_ts: Optional[int] = None):
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.private_key = config.private_key
        self.minimum_time_remaining_hours = minimum_time_remaining_hours
        self.max_markets_per_event = max_markets_per_event
        self.max_close_ts = max_close_ts
        self.client = None
        self.session_token = None
        
    async def login(self):
        """Login to Kalshi API."""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0
        )
        
        # For now, we'll assume the client handles authentication
        # In the real implementation, you'd do login here
        logger.info(f"Connected to Kalshi API at {self.base_url}")
        
    async def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get top events sorted by 24-hour volume."""
        try:
            # First, fetch ALL events from the platform using pagination
            all_events = await self._fetch_all_events()
            
            # Calculate total volume_24h for each event from its markets 
            # (API already filters for "open" status events)
            enriched_events = []
            now = datetime.now(timezone.utc)
            minimum_time_remaining = self.minimum_time_remaining_hours * 3600  # Convert hours to seconds
            filter_enabled = self.max_close_ts is not None
            markets_seen = 0
            markets_kept = 0
            events_dropped_by_expiration = 0
            
            for event in all_events:
                # Get markets and select top N by volume
                all_markets = event.get("markets", [])
                markets_seen += len(all_markets)

                # Optionally filter markets by close time if max_close_ts is provided
                if self.max_close_ts is not None and all_markets:
                    filtered_markets = []
                    for market in all_markets:
                        close_time_str = market.get("close_time", "")
                        if not close_time_str:
                            continue
                        try:
                            # Parse ISO8601 close_time
                            if close_time_str.endswith('Z'):
                                close_dt = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                            else:
                                close_dt = datetime.fromisoformat(close_time_str)
                            if close_dt.tzinfo is None:
                                close_dt = close_dt.replace(tzinfo=timezone.utc)
                            close_ts = int(close_dt.timestamp())
                            if close_ts <= self.max_close_ts:
                                filtered_markets.append(market)
                        except Exception:
                            # If parsing fails, skip this market from filtered list
                            continue
                    all_markets = filtered_markets
                
                # If no markets remain after filtering, skip this event
                if not all_markets:
                    if filter_enabled:
                        events_dropped_by_expiration += 1
                    continue

                if filter_enabled:
                    markets_kept += len(all_markets)

                # Sort markets by volume (descending) and take top N
                sorted_markets = sorted(all_markets, key=lambda m: m.get("volume", 0), reverse=True)
                top_markets = sorted_markets[:self.max_markets_per_event]
                
                if len(all_markets) > self.max_markets_per_event:
                    logger.info(f"Event {event.get('event_ticker', '')} has {len(all_markets)} markets, selecting top {len(top_markets)} by volume")
                
                # Calculate volume metrics for this event using top markets
                total_liquidity = 0
                total_volume = 0
                total_volume_24h = 0
                total_open_interest = 0
                
                for market in top_markets:
                    total_liquidity += market.get("liquidity", 0)
                    total_volume += market.get("volume", 0)
                    total_volume_24h += market.get("volume_24h", 0)
                    total_open_interest += market.get("open_interest", 0)
                
                # Calculate time remaining if strike_date exists
                time_remaining_hours = None
                strike_date_str = event.get("strike_date", "")
                
                if strike_date_str:
                    try:
                        # Parse strike date
                        if strike_date_str.endswith('Z'):
                            strike_date = datetime.fromisoformat(strike_date_str.replace('Z', '+00:00'))
                        else:
                            strike_date = datetime.fromisoformat(strike_date_str)
                        
                        # Ensure timezone awareness
                        if strike_date.tzinfo is None:
                            strike_date = strike_date.replace(tzinfo=timezone.utc)
                        
                        # Calculate time remaining
                        time_remaining = (strike_date - now).total_seconds()
                        time_remaining_hours = time_remaining / 3600
                        
                        # Optional: Skip events that are very close to striking
                        if time_remaining > 0 and time_remaining < minimum_time_remaining:
                            logger.info(f"Event {event.get('event_ticker', '')} strikes in {time_remaining/60:.1f} minutes, skipping")
                            continue
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse strike_date '{strike_date_str}' for event {event.get('event_ticker', '')}: {e}")
                        # Continue without time filtering for this event
                
                # If no top markets selected, skip event
                if not top_markets:
                    continue

                enriched_events.append({
                    "event_ticker": event.get("event_ticker", ""),
                    "title": event.get("title", ""),
                    "subtitle": event.get("sub_title", ""),
                    "volume": total_volume,
                    "volume_24h": total_volume_24h,
                    "liquidity": total_liquidity,
                    "open_interest": total_open_interest,
                    "category": event.get("category", ""),
                    "mutually_exclusive": event.get("mutually_exclusive", False),
                    "strike_date": strike_date_str,
                    "strike_period": event.get("strike_period", ""),
                    "time_remaining_hours": time_remaining_hours,
                    "markets": top_markets,  # Store the top markets with the event
                    "total_markets": len(all_markets),  # Store original market count
                })
            
            # Sort by volume_24h (descending) for true popularity ranking
            enriched_events.sort(key=lambda x: x.get("volume_24h", 0), reverse=True)
            
            # Return only the top N events as requested
            top_events = enriched_events[:limit]
            
            # Summary log for expiration filter effects
            if filter_enabled and markets_seen > 0:
                dropped = markets_seen - markets_kept
                logger.info(
                    f"Expiration filter summary: kept {markets_kept}/{markets_seen} markets; "
                    f"dropped {dropped}. Events dropped due to no remaining markets: {events_dropped_by_expiration}"
                )
            
            logger.info(f"Retrieved {len(all_events)} total events, filtered to {len(enriched_events)} active events, returning top {len(top_events)} by 24h volume")
            return top_events
            
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    async def _fetch_all_events(self) -> List[Dict[str, Any]]:
        """Fetch all events from the platform using pagination."""
        all_events = []
        cursor = None
        page = 1
        
        while True:
            try:
                headers = await self._get_headers("GET", "/trade-api/v2/events")
                params = {
                    "limit": 100,  # Maximum events per page
                    "status": "open",  # Only get open events (active/tradeable)
                    "with_nested_markets": "true"
                }
                
                if cursor:
                    params["cursor"] = cursor
                
                logger.info(f"Fetching events page {page}...")
                response = await self.client.get(
                    "/trade-api/v2/events",
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                if data is None:
                    logger.error(f"Received None response from API")
                    break
                    
                events = data.get("events", []) if isinstance(data, dict) else []
                
                if not events:
                    break
                
                all_events.extend(events)
                logger.info(f"Page {page}: {len(events)} events (total: {len(all_events)})")
                
                # Check if there's a next page
                cursor = data.get("cursor")
                if not cursor:
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"Error fetching events page {page}: {e}")
                break
        
        logger.info(f"Fetched {len(all_events)} total events from {page} pages")
        return all_events
    
    async def get_markets_for_event(self, event_ticker: str) -> List[Dict[str, Any]]:
        """Get markets for a specific event (returns pre-filtered top markets from get_events)."""
        # This method is kept for compatibility but now returns pre-filtered markets
        # The actual filtering happens in get_events() to avoid duplicate API calls
        logger.warning(f"get_markets_for_event called for {event_ticker} - markets should be pre-loaded from get_events()")
        
        # Fallback: fetch markets directly if needed
        try:
            headers = await self._get_headers("GET", "/trade-api/v2/markets")
            params = {"event_ticker": event_ticker, "status": "open"}
            # Pass through server-side filter if available
            if self.max_close_ts is not None:
                params["max_close_ts"] = self.max_close_ts
            response = await self.client.get(
                "/trade-api/v2/markets",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            all_markets = data.get("markets", [])

            # Client-side filtering as a fallback when server-side filtering is not applied
            if self.max_close_ts is not None and all_markets:
                filtered_markets = []
                for market in all_markets:
                    close_time_str = market.get("close_time", "")
                    if not close_time_str:
                        continue
                    try:
                        if close_time_str.endswith('Z'):
                            close_dt = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                        else:
                            close_dt = datetime.fromisoformat(close_time_str)
                        if close_dt.tzinfo is None:
                            close_dt = close_dt.replace(tzinfo=timezone.utc)
                        close_ts = int(close_dt.timestamp())
                        if close_ts <= self.max_close_ts:
                            filtered_markets.append(market)
                    except Exception:
                        continue
                all_markets = filtered_markets
            
            # Sort by volume and take top markets
            sorted_markets = sorted(all_markets, key=lambda m: m.get("volume", 0), reverse=True)
            top_markets = sorted_markets[:self.max_markets_per_event]
            
            # Return markets without odds for research
            simple_markets = []
            for market in top_markets:
                simple_markets.append({
                    "ticker": market.get("ticker", ""),
                    "title": market.get("title", ""),
                    "subtitle": market.get("subtitle", ""),
                    "volume": market.get("volume", 0),
                    "open_time": market.get("open_time", ""),
                    "close_time": market.get("close_time", ""),
                    # Note: NOT including yes_bid, no_bid, yes_ask, no_ask for research
                })
            
            logger.info(f"Retrieved {len(simple_markets)} markets for event {event_ticker} (top {len(top_markets)} by volume)")
            return simple_markets
            
        except Exception as e:
            logger.error(f"Error getting markets for event {event_ticker}: {e}")
            return []
    
    async def get_market_with_odds(self, ticker: str) -> Dict[str, Any]:
        """Get a specific market with current odds for trading."""
        try:
            headers = await self._get_headers("GET", f"/trade-api/v2/markets/{ticker}")
            response = await self.client.get(
                f"/trade-api/v2/markets/{ticker}",
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            market = data.get("market", {})
            
            # Get specific fields
            yes_bid = market.get("yes_bid", 0)
            no_bid = market.get("no_bid", 0)
            yes_ask = market.get("yes_ask", 0)
            no_ask = market.get("no_ask", 0)
            
            # Note: Event-level filtering is already done in get_events()
            return {
                "ticker": market.get("ticker", ""),
                "title": market.get("title", ""),
                "yes_bid": yes_bid,
                "no_bid": no_bid,
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "volume": market.get("volume", 0),
                "status": market.get("status", ""),
                "close_time": market.get("close_time", ""),
            }
            
        except Exception as e:
            logger.error(f"Error getting market {ticker}: {e}")
            return {}
    
    async def get_user_positions(self) -> List[Dict[str, Any]]:
        """Get all user positions."""
        try:
            headers = await self._get_headers("GET", "/trade-api/v2/portfolio/positions")
            response = await self.client.get(
                "/trade-api/v2/portfolio/positions",
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Debug: Log the raw API response structure
            logger.debug(f"Position API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # The API returns market_positions, not positions
            positions = data.get("market_positions", [])
            
            # Also check for event_positions (though we primarily need market_positions)
            event_positions = data.get("event_positions", [])
            
            logger.info(f"Retrieved {len(positions)} market positions and {len(event_positions)} event positions")
            logger.debug(f"Market positions: {positions[:3] if positions else 'None'}")  # Log first 3 for debugging
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting user positions: {e}")
            return []
    
    async def has_position_in_market(self, ticker: str) -> bool:
        """Check if user already has a position in the specified market."""
        try:
            positions = await self.get_user_positions()
            
            for position in positions:
                if position.get("ticker") == ticker:
                    # Check if position has any contracts
                    # In Kalshi API: positive = YES contracts, negative = NO contracts, 0 = no position
                    position_size = position.get("position", 0)
                    
                    if position_size != 0:
                        position_type = "YES" if position_size > 0 else "NO"
                        logger.info(f"Found existing position in {ticker}: {abs(position_size)} {position_type} contracts")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking position for {ticker}: {e}")
            return False  # If we can't check, assume no position to be safe

    async def place_order(self, ticker: str, side: str, amount: float) -> Dict[str, Any]:
        """Place a limit order at current market price."""
        try:
            # Generate a unique client order ID
            import uuid
            client_order_id = str(uuid.uuid4())

            # First, get the current market price
            headers = await self._get_headers("GET", f"/trade-api/v2/markets/{ticker}")
            market_response = await self.client.get(f"/trade-api/v2/markets/{ticker}", headers=headers)
            market_response.raise_for_status()
            market_data = market_response.json().get("market", {})

            # Get the ask price for the side we want to buy
            # yes_ask and no_ask are in cents (0-100)
            if side.lower() == "yes":
                price_cents = market_data.get("yes_ask", 50)
            else:
                price_cents = market_data.get("no_ask", 50)

            if price_cents is None or price_cents == 0:
                logger.error(f"No ask price available for {ticker} {side}")
                return {"success": False, "error": "No ask price available"}

            # Calculate how many contracts we can buy with our budget
            # Each contract costs price_cents cents, and pays out 100 cents if we win
            amount_cents = int(amount * 100)
            num_contracts = amount_cents // price_cents

            if num_contracts < 1:
                logger.error(f"Amount ${amount} too small for price {price_cents} cents")
                return {"success": False, "error": "Amount too small for current price"}

            # Kalshi API requires one of: yes_price, no_price (in cents 1-99)
            order_data = {
                "ticker": ticker,
                "side": side.lower(),  # "yes" or "no"
                "action": "buy",
                "type": "limit",
                "client_order_id": client_order_id,
                "count": num_contracts,
            }

            # Add the correct price field based on side
            if side.lower() == "yes":
                order_data["yes_price"] = price_cents
            else:
                order_data["no_price"] = price_cents

            logger.info(f"Placing order: {ticker} {side} {num_contracts} contracts @ {price_cents}c (${amount:.2f} budget)")

            headers = await self._get_headers("POST", "/trade-api/v2/portfolio/orders")
            response = await self.client.post(
                "/trade-api/v2/portfolio/orders",
                headers=headers,
                json=order_data
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Order placed successfully: {ticker} {side} {num_contracts} @ {price_cents}c")
            return {"success": True, "order_id": result.get("order", {}).get("order_id", ""), "client_order_id": client_order_id, "contracts": num_contracts, "price": price_cents}

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"success": False, "error": str(e)}

    async def sell_position(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: Optional[int] = None,
        order_type: str = "limit"
    ) -> Dict[str, Any]:
        """
        Sell (close) a position.

        Args:
            ticker: Market ticker
            side: "yes" or "no" - which side to sell
            contracts: Number of contracts to sell
            price_cents: Limit price in cents (None = use current bid)
            order_type: "limit" or "market"

        Returns:
            {"success": bool, "order_id": str, "contracts": int, "price_cents": int, ...}
        """
        try:
            import uuid
            client_order_id = str(uuid.uuid4())

            # Get current market price if not specified
            if price_cents is None:
                market_data = await self.get_market_with_odds(ticker)
                # Sell at bid price (what buyers are willing to pay)
                if side.lower() == "yes":
                    price_cents = market_data.get("yes_bid", 50)
                else:
                    price_cents = market_data.get("no_bid", 50)

            if price_cents is None or price_cents == 0:
                logger.error(f"No bid price available for {ticker} {side}")
                return {"success": False, "error": "No bid price available"}

            order_data = {
                "ticker": ticker,
                "side": side.lower(),
                "action": "sell",  # KEY DIFFERENCE: sell instead of buy
                "type": order_type,
                "client_order_id": client_order_id,
                "count": contracts,
            }

            # Add price for limit orders
            if order_type == "limit":
                if side.lower() == "yes":
                    order_data["yes_price"] = price_cents
                else:
                    order_data["no_price"] = price_cents

            logger.info(f"Selling position: {ticker} {side} {contracts} contracts @ {price_cents}c")

            headers = await self._get_headers("POST", "/trade-api/v2/portfolio/orders")
            response = await self.client.post(
                "/trade-api/v2/portfolio/orders",
                headers=headers,
                json=order_data
            )
            response.raise_for_status()

            result = response.json()
            order = result.get("order", {})

            logger.info(f"Sell order placed: {order.get('order_id')} - {contracts} {side} @ {price_cents}c")

            return {
                "success": True,
                "order_id": order.get("order_id", ""),
                "client_order_id": client_order_id,
                "contracts": contracts,
                "price_cents": price_cents,
                "status": order.get("status", ""),
            }

        except Exception as e:
            logger.error(f"Error selling position: {e}")
            return {"success": False, "error": str(e)}

    async def liquidate_position(self, ticker: str) -> Dict[str, Any]:
        """
        Fully liquidate a position at market.

        Gets current position and sells all contracts.

        Args:
            ticker: Market ticker to liquidate

        Returns:
            {"success": bool, "order_id": str, "contracts": int, "side": str, ...}
        """
        try:
            # Get current position
            positions = await self.get_user_positions()

            position = None
            for p in positions:
                if p.get("ticker") == ticker:
                    position = p
                    break

            if not position:
                return {"success": False, "error": f"No position found for {ticker}"}

            position_size = position.get("position", 0)

            if position_size == 0:
                return {"success": True, "message": "No position to liquidate", "contracts": 0}

            # Determine side based on position sign
            # Positive = YES contracts, Negative = NO contracts
            if position_size > 0:
                side = "yes"
                contracts = position_size
            else:
                side = "no"
                contracts = abs(position_size)

            logger.info(f"Liquidating {ticker}: {contracts} {side.upper()} contracts")

            result = await self.sell_position(ticker, side, contracts)
            result["side"] = side
            result["original_position"] = position_size

            return result

        except Exception as e:
            logger.error(f"Error liquidating position {ticker}: {e}")
            return {"success": False, "error": str(e)}

    async def liquidate_all_positions(self) -> Dict[str, Any]:
        """
        Emergency: Liquidate ALL open positions.

        Returns summary of liquidation attempts.

        Returns:
            {
                "success": bool,
                "liquidated": [{"ticker": str, "contracts": int, "order_id": str}, ...],
                "failed": [{"ticker": str, "error": str}, ...],
                "total_positions": int
            }
        """
        results = {
            "success": True,
            "liquidated": [],
            "failed": [],
            "total_positions": 0
        }

        try:
            positions = await self.get_user_positions()
            results["total_positions"] = len(positions)

            logger.warning(f"EMERGENCY LIQUIDATION: Processing {len(positions)} positions")

            for position in positions:
                ticker = position.get("ticker")
                position_size = position.get("position", 0)

                if position_size == 0:
                    continue

                result = await self.liquidate_position(ticker)

                if result.get("success"):
                    results["liquidated"].append({
                        "ticker": ticker,
                        "contracts": abs(position_size),
                        "side": "yes" if position_size > 0 else "no",
                        "order_id": result.get("order_id")
                    })
                else:
                    results["failed"].append({
                        "ticker": ticker,
                        "error": result.get("error")
                    })
                    results["success"] = False

            logger.info(
                f"Liquidation complete: {len(results['liquidated'])} succeeded, "
                f"{len(results['failed'])} failed"
            )

            return results

        except Exception as e:
            logger.error(f"Error in mass liquidation: {e}")
            return {"success": False, "error": str(e), "liquidated": [], "failed": []}

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of a specific order.

        Args:
            order_id: Kalshi order ID

        Returns:
            Order details including status, fill info, etc.
        """
        try:
            headers = await self._get_headers("GET", f"/trade-api/v2/portfolio/orders/{order_id}")
            response = await self.client.get(
                f"/trade-api/v2/portfolio/orders/{order_id}",
                headers=headers
            )
            response.raise_for_status()
            return response.json().get("order", {})
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return {"error": str(e)}

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.

        Args:
            order_id: Kalshi order ID to cancel

        Returns:
            {"success": bool, "order_id": str, ...}
        """
        try:
            headers = await self._get_headers("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")
            response = await self.client.delete(
                f"/trade-api/v2/portfolio/orders/{order_id}",
                headers=headers
            )
            response.raise_for_status()
            logger.info(f"Order {order_id} cancelled successfully")
            return {"success": True, "order_id": order_id}
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {"success": False, "order_id": order_id, "error": str(e)}

    async def get_user_fills(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent order fills for the user.

        Args:
            limit: Maximum number of fills to return

        Returns:
            List of fill records with order details and settlement amounts
        """
        try:
            path = "/trade-api/v2/portfolio/fills"
            headers = await self._get_headers("GET", path)
            response = await self.client.get(
                path,
                headers=headers,
                params={"limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("fills", [])
        except Exception as e:
            logger.error(f"Error getting user fills: {e}")
            return []

    async def _get_headers(self, method: str, path: str) -> Dict[str, str]:
        """Generate headers with RSA signature."""
        timestamp = str(int(time.time() * 1000))
        
        # Create message to sign
        message = f"{timestamp}{method}{path}"
        
        # Sign the message
        signature = self._sign_message(message)
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json"
        }
    
    def _sign_message(self, message: str) -> str:
        """Sign a message using RSA private key."""
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                self.private_key.encode(),
                password=None,
                backend=default_backend()
            )
            
            # Sign the message
            signature = private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Return base64 encoded signature
            return base64.b64encode(signature).decode()
            
        except Exception as e:
            logger.error(f"Error signing message: {e}")
            raise
    
    async def get_market_status(self, ticker: str) -> Dict[str, Any]:
        """
        Get current market status including settlement info.

        Returns:
            Dict with: ticker, status, result (if settled), settlement_value, close_time
        """
        try:
            headers = await self._get_headers("GET", f"/trade-api/v2/markets/{ticker}")
            response = await self.client.get(
                f"/trade-api/v2/markets/{ticker}",
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            market = data.get("market", {})

            return {
                "ticker": market.get("ticker", ""),
                "title": market.get("title", ""),
                "status": market.get("status", ""),  # open, closed, settled
                "result": market.get("result", ""),  # yes, no (only for settled)
                "settlement_value": market.get("settlement_value"),
                "close_time": market.get("close_time", ""),
                "expiration_time": market.get("expiration_time", ""),
                "yes_bid": market.get("yes_bid"),
                "yes_ask": market.get("yes_ask"),
                "no_bid": market.get("no_bid"),
                "no_ask": market.get("no_ask"),
            }

        except Exception as e:
            logger.error(f"Error getting market status for {ticker}: {e}")
            return {}

    async def get_settled_markets(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Batch fetch settlement status for multiple markets.

        Args:
            tickers: List of market tickers to check

        Returns:
            List of settled market info dicts
        """
        settled_markets = []
        batch_size = 20

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            tasks = [self.get_market_status(ticker) for ticker in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict) and result.get("status") == "settled":
                    settled_markets.append(result)

            # Small delay between batches
            if i + batch_size < len(tickers):
                await asyncio.sleep(0.2)

        logger.info(f"Found {len(settled_markets)} settled markets out of {len(tickers)} checked")
        return settled_markets

    async def get_user_fills(self, since: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get order fills/settlements for P&L calculation.

        Args:
            since: Only return fills after this timestamp
            limit: Maximum number of fills to return

        Returns:
            List of fill records with order details and settlement info
        """
        try:
            headers = await self._get_headers("GET", "/trade-api/v2/portfolio/fills")
            params = {"limit": limit}

            if since:
                # Convert to ISO format
                params["min_ts"] = int(since.timestamp())

            response = await self.client.get(
                "/trade-api/v2/portfolio/fills",
                headers=headers,
                params=params
            )
            response.raise_for_status()

            data = response.json()
            fills = data.get("fills", [])

            logger.info(f"Retrieved {len(fills)} fills from Kalshi API")
            return fills

        except Exception as e:
            logger.error(f"Error getting user fills: {e}")
            return []

    async def get_portfolio_settlements(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get settled positions from portfolio.

        Returns:
            List of settled position records
        """
        try:
            headers = await self._get_headers("GET", "/trade-api/v2/portfolio/settlements")
            params = {"limit": limit}

            response = await self.client.get(
                "/trade-api/v2/portfolio/settlements",
                headers=headers,
                params=params
            )
            response.raise_for_status()

            data = response.json()
            settlements = data.get("settlements", [])

            logger.info(f"Retrieved {len(settlements)} settlements from Kalshi API")
            return settlements

        except Exception as e:
            logger.error(f"Error getting portfolio settlements: {e}")
            return []

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose() 