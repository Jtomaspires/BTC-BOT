import asyncio
import json
import logging
import pickle
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import websockets
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

_BOT_DIR = Path(__file__).resolve().parent
_BOT_ROOT_DIR = _BOT_DIR.parent
_ENV_PATH = _BOT_ROOT_DIR / ".env"
_LOG_PATH = _BOT_DIR / "log.log"
_PERFORMANCE_PATH = _BOT_DIR / "performance.png"
_SAVED_DATA_PATH = _BOT_DIR / "saved_data.pkl"


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


class Bot:
    def __init__(self, seq_len: int, testnet: bool = True):
        self.logger = setup_logger("log", str(_LOG_PATH))
        self.seq_len = seq_len

        self.pos_size_usd = 1000        # notional position size in USD
        self.slippage_pct = 0.05        # slippage percentage (0.05 %)
        self.sl_points: float = 3.0     # USD from entry (matches backtest BEST_SL)
        self.tp_points: float = 50.0    # USD from entry (matches backtest BEST_TP)
        self.coin = "ETH"
        self.TIME_FRAME = "1h"
        self.asset_usd = "USDT"

        # ccxt unified symbol for Bybit linear perpetual
        self.symbol = "ETH/USDT:USDT"
        # raw symbol used in WebSocket topic
        self.symbol_ = "ETHUSDT"

        self.testnet = testnet

        self.position = "NONE"
        self.position_coin = 0.0

        # Safe defaults so that check_positions / logging never crash
        # even when update_curr_futures_prices fails on the first attempt.
        self.curr_close = 0.0
        self.curr_open = 0.0

        self.open_prices = []
        self.actions = []
        self.equities = []
        self.total_fee = 0.0

        # Persisted across restarts; set before each wait_for_candle_close.
        self.pending_order_id: str | None = None
        self.sl_order_id: str | None = None
        self.tp_order_id: str | None = None

        self.exchange = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def write_to_log(self, msg: str):
        self.logger.info(msg)
        print(msg)

    # ------------------------------------------------------------------
    # Exchange lifecycle
    # ------------------------------------------------------------------

    async def create_exchange(self):
        # Load bot-local `.env` so credentials are found regardless of cwd.
        load_dotenv(dotenv_path=_ENV_PATH)

        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        secret = os.getenv("BINANCE_API_SECRET", "").strip()
        if not api_key or not secret:
            raise RuntimeError(
                "Missing BINANCE_API_KEY / BINANCE_API_SECRET. "
                "Create a `.env` (copy from `.env.example`) in binance_futures_trading_bot/."
            )

        # Optional override via env; otherwise keep constructor default.
        env_testnet = os.getenv("BINANCE_TESTNET")
        if env_testnet is not None:
            self.testnet = env_testnet.strip().lower() in ("1", "true", "yes", "y", "on")

        for _ in range(20):
            try:
                self.exchange = ccxt_async.binance({
                    "apiKey": api_key,
                    "secret": secret,
                    "options": {"defaultType": "future"},
                })
                if self.testnet:
                    # Binance deprecated the Futures Sandbox (set_sandbox_mode) in late 2025.
                    # The replacement is Demo Trading: keys generated at https://demo.binance.com
                    self.exchange.enable_demo_trading(True)

                await self.exchange.load_markets()

                # Set isolated margin mode and 10x leverage for the traded symbol.
                # These are applied once per session at startup so that every
                # order is placed under the same, predictable risk configuration.
                try:
                    await self.exchange.set_margin_mode("isolated", self.symbol)
                    self.write_to_log(f"Margin mode set to isolated ({self.symbol}).")
                except Exception as margin_err:
                    # Binance raises if margin mode is already set to the target value.
                    self.write_to_log(f"set_margin_mode (non-fatal): {margin_err}")

                try:
                    await self.exchange.set_leverage(10, self.symbol)
                    self.write_to_log(f"Leverage set to 10x ({self.symbol}).")
                except Exception as lev_err:
                    self.write_to_log(f"set_leverage (non-fatal): {lev_err}")

                env = "testnet" if self.testnet else "mainnet"
                self.write_to_log(f"Exchange created ({env}, {self.symbol}).")
                return
            except Exception as err:
                self.write_to_log(f"Failed to create exchange: {err}")
                try:
                    await self.exchange.close()
                except Exception:
                    pass
                self.exchange = None
                await asyncio.sleep(0.4)

    async def close_exchange(self):
        if self.exchange is not None:
            try:
                await self.exchange.close()
                self.write_to_log("Exchange closed.")
            except Exception as err:
                self.write_to_log(f"Error closing exchange: {err}")
            finally:
                self.exchange = None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def fetch_candles_async(self, symbol: str, num_candles: int):
        for _ in range(20):
            try:
                klines = await self.exchange.fetch_ohlcv(
                    symbol, self.TIME_FRAME, limit=num_candles
                )
                klines = np.array(klines, dtype=object)
                open_times = pd.to_datetime(klines[:, 0].astype(np.int64), unit="ms")
                opens   = klines[:, 1].astype(float)
                highs   = klines[:, 2].astype(float)
                lows    = klines[:, 3].astype(float)
                closes  = klines[:, 4].astype(float)
                volumes = klines[:, 5].astype(float)
                return open_times, opens, highs, lows, closes, volumes
            except Exception as e:
                self.write_to_log(f"Error fetch_candles_async: {e}")
                await asyncio.sleep(0.5)

    async def update_new_data_for_model(self):
        """Fetch the last seq_len closed candles (no polling — WS already confirmed close)."""
        self.write_to_log("Fetching candle data for model ...")
        open_times, opens, highs, lows, closes, volumes = await self.fetch_candles_async(
            self.symbol, num_candles=self.seq_len + 1
        )
        # Exclude the last bar (new, still-forming candle)
        self.opens_USDT   = np.array(opens[:-1],   dtype=np.float32)
        self.highs_USDT   = np.array(highs[:-1],   dtype=np.float32)
        self.lows_USDT    = np.array(lows[:-1],    dtype=np.float32)
        self.closes_USDT  = np.array(closes[:-1],  dtype=np.float32)
        self.volumes_USDT = np.array(volumes[:-1], dtype=np.float32)
        self.write_to_log(
            f"Data updated. Last close: {self.closes_USDT[-1]:.2f}  "
            f"shape: {self.closes_USDT.shape}"
        )

    async def update_curr_futures_prices(self):
        """
        Fetch live price and the open of the currently-forming candle.

        curr_close  — last traded price from the ticker.
        curr_open   — open of the candle that just started (limit=1 gives the
                      in-progress bar whose open is already fixed).
        """
        for _ in range(20):
            try:
                ticker = await self.exchange.fetch_ticker(self.symbol)
                self.curr_close = float(ticker["last"])

                # Fetch the single currently-forming candle to get its open.
                current_kline = await self.exchange.fetch_ohlcv(
                    self.symbol, self.TIME_FRAME, limit=1
                )
                self.curr_open = float(current_kline[-1][1])

                self.write_to_log(
                    f"Ticker: last={self.curr_close:.2f}  "
                    f"curr_open={self.curr_open:.2f} ({self.coin})"
                )
                return
            except Exception as err:
                self.write_to_log(f"Error update_curr_futures_prices: {err}")
                await asyncio.sleep(0.2)

        self.write_to_log(
            "WARNING: update_curr_futures_prices exhausted all retries. "
            "curr_close/curr_open may be stale."
        )

    # ------------------------------------------------------------------
    # WebSocket — wait for candle close
    # ------------------------------------------------------------------

    async def wait_for_candle_close(self):
        """Block until the current 1h candle is confirmed closed via Binance WebSocket.

        Binance stream URL encodes the subscription directly — no subscribe
        message is needed.  The kline event field `k.x` is True when the
        candle is closed.  Binance handles WebSocket ping/pong at the protocol
        level, so no application-level keepalive is required.
        """
        stream = f"{self.symbol_.lower()}@kline_{self.TIME_FRAME}"
        # Market data WebSocket is always fstream.binance.com regardless of
        # demo/mainnet — demo routing is REST-only via enable_demo_trading(True).
        ws_url = f"wss://fstream.binance.com/ws/{stream}"

        async with websockets.connect(ws_url) as ws:
            env = "testnet" if self.testnet else "mainnet"
            self.write_to_log(f"WS connected to {stream} ({env}). Waiting for candle close ...")

            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    msg = json.loads(raw)
                    kline = msg.get("k", {})
                    if kline.get("x") is True:
                        self.write_to_log(
                            f"Candle closed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        # Brief pause so REST endpoints reflect the closed candle
                        await asyncio.sleep(0.3)
                        return

                except asyncio.TimeoutError:
                    # No message for 30 s — Binance keeps the connection alive
                    # at the protocol level, so this is just a safety guard.
                    pass

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_market_order(self, side: str):
        """
        Place a market order and resolve its fill price.

        side: 'BUY' or 'SELL'
        Returns (order_id, fill_price) or (None, None) on complete failure.
        """
        dt0 = datetime.now()

        order_size = self.pos_size_usd / self.curr_close

        order_size = float(self.exchange.amount_to_precision(self.symbol, order_size))
        ccxt_side = side.lower()

        order = None
        for _ in range(20):
            try:
                order = await self.exchange.create_order(
                    self.symbol, "market", ccxt_side, order_size
                )
                break
            except Exception as e:
                self.write_to_log(f"Failed to place {side} market order: {e}")
                await asyncio.sleep(0.4)

        if order is None:
            self.write_to_log(f"FAILED to place {side} market order after all retries.")
            return None, None

        order_id = str(order["id"])

        fill_price = None
        avg = order.get("average")
        if avg is not None:
            fill_price = float(avg)
        else:
            # Binance may not populate `average` on the immediate response;
            # a short poll of fetch_order fills this in reliably.
            for _ in range(10):
                try:
                    fetched = await self.exchange.fetch_order(order_id, self.symbol)
                    avg = fetched.get("average")
                    if avg is not None:
                        fill_price = float(avg)
                        break
                    filled = float(fetched.get("filled") or 0.0)
                    if filled > 0:
                        fill_price = self.curr_close
                        break
                except Exception as err:
                    self.write_to_log(f"fetch_order after market {order_id}: {err}")
                await asyncio.sleep(0.3)

        if fill_price is None:
            fill_price = self.curr_close
            self.write_to_log(
                f"WARNING: market order {order_id} avg unknown; "
                f"using curr_close={fill_price} as fill price."
            )

        self.write_to_log(
            f"{side} market order filled. "
            f"avg={fill_price:.2f} qty={order_size} {self.coin} "
            f"duration={datetime.now() - dt0}"
        )
        return order_id, fill_price

    @staticmethod
    def _is_unknown_order_error(err: Exception) -> bool:
        """Detect Binance `-2011 Unknown order sent` across ccxt wrappings."""
        msg = str(err)
        return "-2011" in msg or "Unknown order" in msg

    async def cancel_order(self, order_id: str):
        """
        Cancel an order, treating `-2011 Unknown order` as success
        (the order already filled, was cancelled, or expired).
        """
        for attempt in range(10):
            try:
                result = await self.exchange.cancel_order(order_id, self.symbol)
                self.write_to_log(f"Cancelled order {order_id}: {result}")
                return
            except Exception as err:
                if self._is_unknown_order_error(err):
                    self.write_to_log(
                        f"Order {order_id} not open (already filled/cancelled)."
                    )
                    return
                self.write_to_log(
                    f"Could not cancel order {order_id} (attempt {attempt + 1}/10): {err}"
                )
                await asyncio.sleep(0.5)

        self.write_to_log(
            f"WARNING: cancel_order exhausted retries for {order_id}. "
            "Order may still be open on the exchange."
        )

    async def wait_for_fill(self, order_id: str, max_attempts: int = 15, sleep_s: float = 1.0):
        """
        Poll order status until any fill is detected.
        Returns average fill price (float) or None if never filled.
        """
        for attempt in range(max_attempts):
            try:
                order = await self.exchange.fetch_order(order_id, self.symbol)
                status = str(order.get("status") or "").lower()
                filled = float(order.get("filled") or 0.0)
                avg = order.get("average")

                if filled > 0:
                    if avg is not None:
                        return float(avg)
                    price = order.get("price")
                    if price is not None:
                        return float(price)
                    # Final fallback if average/price are missing.
                    ticker = await self.exchange.fetch_ticker(self.symbol)
                    last = ticker.get("last")
                    return float(last) if last is not None else None

                if status in ("canceled", "cancelled", "expired", "rejected", "closed"):
                    self.write_to_log(
                        f"Order {order_id} status={status} with no fills. "
                        "Skipping SL/TP placement."
                    )
                    return None

            except Exception as err:
                self.write_to_log(
                    f"wait_for_fill {order_id} (attempt {attempt + 1}/{max_attempts}) err: {err}"
                )

            await asyncio.sleep(sleep_s)

        self.write_to_log(f"wait_for_fill timed out for {order_id} after {max_attempts} attempts.")
        return None

    async def place_sl_tp_orders(self, entry_price: float, side: str) -> None:
        """
        Place reduce-only protection orders using Binance futures native types:
        STOP_MARKET and TAKE_PROFIT_MARKET.
        """
        if side == "BUY":
            sl_price = entry_price - self.sl_points
            tp_price = entry_price + self.tp_points
            exit_side = "sell"
        else:
            sl_price = entry_price + self.sl_points
            tp_price = entry_price - self.tp_points
            exit_side = "buy"

        sl_price = float(self.exchange.price_to_precision(self.symbol, sl_price))
        tp_price = float(self.exchange.price_to_precision(self.symbol, tp_price))

        # Backtest uses OHLC (trade/contract price), not mark price. Using
        # CONTRACT_PRICE keeps trigger semantics closer to the backtest.
        params_sl = {
            "stopPrice": sl_price,
            "reduceOnly": True,
            "closePosition": True,
            "workingType": "CONTRACT_PRICE",
        }
        params_tp = {
            "stopPrice": tp_price,
            "reduceOnly": True,
            "closePosition": True,
            "workingType": "CONTRACT_PRICE",
        }

        sl_order = await self.exchange.create_order(
            self.symbol, "STOP_MARKET", exit_side, 0.0, None, params_sl
        )
        self.sl_order_id = str(sl_order["id"])

        tp_order = await self.exchange.create_order(
            self.symbol, "TAKE_PROFIT_MARKET", exit_side, 0.0, None, params_tp
        )
        self.tp_order_id = str(tp_order["id"])

        self.write_to_log(
            f"SL/TP placed. entry={entry_price:.2f} side={side} "
            f"sl={sl_price} (id={self.sl_order_id}) "
            f"tp={tp_price} (id={self.tp_order_id})"
        )

    async def cancel_sl_tp_orders(self) -> None:
        """
        Cancel tracked SL/TP protection orders (best-effort).

        `-2011 Unknown order` is expected whenever a protection order has
        already been triggered (closed the position) or auto-cancelled by
        the exchange — treat it as success and move on silently.
        """
        for oid_attr in ("sl_order_id", "tp_order_id"):
            oid = getattr(self, oid_attr)
            if oid is None:
                continue
            try:
                result = await self.exchange.cancel_order(oid, self.symbol)
                self.write_to_log(f"Cancelled protection order {oid}: {result}")
            except Exception as err:
                if self._is_unknown_order_error(err):
                    self.write_to_log(
                        f"Protection order {oid} not open (already triggered/cancelled)."
                    )
                else:
                    self.write_to_log(f"cancel protection order {oid} (non-fatal): {err}")
            finally:
                setattr(self, oid_attr, None)

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    async def check_positions(self):
        for _ in range(20):
            try:
                positions = await self.exchange.fetch_positions([self.symbol])
                break
            except Exception as e:
                self.write_to_log(f"Error check_positions: {e}")
                await asyncio.sleep(0.2)
        else:
            self.write_to_log("check_positions: all retries failed.")
            return

        pos = next(
            (p for p in positions
             if p["symbol"] == self.symbol and float(p.get("contracts", 0) or 0) != 0),
            None,
        )

        if pos is None:
            self.position = "NONE"
            self.position_coin = 0.0
            self.write_to_log("No open position.")
        else:
            amt = float(pos["contracts"])
            side = pos.get("side", "")
            position_usd = abs(amt * self.curr_close)

            if position_usd < self.pos_size_usd * 0.8:
                self.write_to_log(f"WARNING: position size too small: ${position_usd:.2f}")

            if side == "long":
                self.position = "LONG"
                self.position_coin = amt
            else:
                self.position = "SHORT"
                self.position_coin = -amt

            self.write_to_log(
                f"Position: {self.position}  "
                f"size={self.position_coin} {self.coin} (${position_usd:.2f})"
            )

    async def check_equity(self):
        for _ in range(20):
            try:
                balance = await self.exchange.fetch_balance()
                self.equity = round(float(balance["USDT"]["total"]), 2)
                self.write_to_log(f"Equity: {self.equity:.2f} USDT")
                return
            except Exception as e:
                self.write_to_log(f"Error check_equity: {e}")
                await asyncio.sleep(0.4)

    async def get_order_fee(self, order_id: str) -> float:
        """
        Fetch the trading fee for a given order.
        Retries up to 10 times with a 1-second sleep between attempts so that
        the matching engine has time to record the trade before we query it.
        Returns 0.0 if no trades are found after all attempts.
        """
        for attempt in range(10):
            try:
                trades = await self.exchange.fetch_my_trades(self.symbol)
                order_trades = [
                    t for t in trades if str(t.get("order")) == str(order_id)
                ]

                if not order_trades:
                    self.write_to_log(
                        f"No trades yet for order {order_id} "
                        f"(attempt {attempt + 1}/10) — retrying ..."
                    )
                    await asyncio.sleep(1.0)
                    continue

                fee = sum(
                    float(t["fee"]["cost"])
                    for t in order_trades
                    if t.get("fee") and t["fee"].get("cost") is not None
                )
                currency = order_trades[0].get("fee", {}).get("currency", self.asset_usd)
                self.write_to_log(f"Order {order_id} fee: {fee} {currency}")
                return fee

            except Exception as e:
                self.write_to_log(f"Error get_order_fee {order_id} (attempt {attempt + 1}/10): {e}")
                await asyncio.sleep(1.0)

        self.write_to_log(
            f"WARNING: get_order_fee found no trades for {order_id} after all retries. "
            "Returning 0.0."
        )
        return 0.0

    # ------------------------------------------------------------------
    # Persistence and plotting
    # ------------------------------------------------------------------

    def plot_performance(self):
        if len(self.equities) < 2:
            return

        eq_min = min(self.equities)
        eq_max = max(self.equities)
        if eq_min == eq_max:
            return

        try:
            scaler = MinMaxScaler(feature_range=(eq_min, eq_max))
            scaled_prices = scaler.fit_transform(
                np.array(self.open_prices).reshape(-1, 1)
            ).flatten()

            plt.figure(figsize=(12, 6))
            plt.title(f"Equity: ${self.equity:.2f}")
            plt.plot(scaled_prices, color="green", label=f"{self.coin} Price")
            plt.plot(self.equities, label="Equity Curve", color="blue")
            plt.xlabel("Time Step")
            plt.ylabel("Equity (USD)")
            plt.legend()
            plt.grid(True)
            plt.savefig(str(_PERFORMANCE_PATH), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Couldn't plot: {e}")

    def save(self):
        data = {
            "open_prices":       self.open_prices,
            "actions":           self.actions,
            "equities":          self.equities,
            "total_fee":         self.total_fee,
            "pending_order_id":  self.pending_order_id,
            "sl_order_id":       self.sl_order_id,
            "tp_order_id":       self.tp_order_id,
        }
        with open(_SAVED_DATA_PATH, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        try:
            with open(_SAVED_DATA_PATH, "rb") as f:
                data = pickle.load(f)
            self.open_prices      = data["open_prices"]
            self.actions          = data["actions"]
            self.equities         = data["equities"]
            self.total_fee        = data["total_fee"]
            self.pending_order_id = data.get("pending_order_id")
            self.sl_order_id      = data.get("sl_order_id")
            self.tp_order_id      = data.get("tp_order_id")
        except Exception:
            print("No saved_data to load.")

    async def reconcile_pending_order(self):
        """
        Called once at startup after the exchange is available.
        If a pending_order_id was persisted (from a previous run that was
        interrupted mid-candle), attempt to cancel it and clear the state so
        that the new run starts clean.
        """
        if not self.pending_order_id:
            return

        self.write_to_log(
            f"Startup reconciliation: found pending order {self.pending_order_id}. "
            "Attempting to cancel ..."
        )
        try:
            await self.cancel_order(self.pending_order_id)
        except Exception as err:
            self.write_to_log(
                f"Reconciliation cancel failed: {err}. "
                "Order may already be filled/cancelled on the exchange."
            )
        finally:
            self.pending_order_id = None
            self.save()
            self.write_to_log("Pending order reconciliation complete.")

    async def _fetch_open_position(self):
        """
        Fetch the currently open position for `self.symbol`.

        Returns (entry_price, side) where side is 'long' / 'short',
        or None if there is no open position / entryPrice is missing.
        """
        for _ in range(20):
            try:
                positions = await self.exchange.fetch_positions([self.symbol])
                break
            except Exception as e:
                self.write_to_log(f"Error fetching position at startup: {e}")
                await asyncio.sleep(0.2)
        else:
            self.write_to_log("reconcile: fetch_positions exhausted retries.")
            return None

        pos = next(
            (p for p in positions
             if p["symbol"] == self.symbol and float(p.get("contracts", 0) or 0) != 0),
            None,
        )
        if pos is None:
            return None

        entry_price = pos.get("entryPrice")
        if entry_price is None:
            entry_price = pos.get("info", {}).get("entryPrice")
        try:
            entry_price = float(entry_price)
        except (TypeError, ValueError):
            return None
        if entry_price <= 0:
            return None

        side = str(pos.get("side") or "").lower()
        if side not in ("long", "short"):
            return None

        return entry_price, side

    async def reconcile_position_protection(self):
        """
        Ensure any open position has fresh SL/TP orders at startup.

        1) Cancel any tracked SL/TP ids (may be stale across restarts).
        2) If a position is open on the exchange, place new SL/TP anchored
           to its real entry price reported by Binance.

        Safe to call even when there is no open position.
        """
        await self.cancel_sl_tp_orders()

        position_info = await self._fetch_open_position()
        if position_info is None:
            self.write_to_log(
                "Startup protection: no open position; nothing to reconcile."
            )
            return

        entry_price, side = position_info
        entry_side = "BUY" if side == "long" else "SELL"

        self.write_to_log(
            f"Startup protection: {side.upper()} position detected at "
            f"entry={entry_price:.2f}. Placing fresh SL/TP ..."
        )

        try:
            await self.place_sl_tp_orders(entry_price, entry_side)
        except Exception as err:
            self.write_to_log(
                f"Failed to place SL/TP during reconciliation: {err}. "
                "Position remains without bot-placed protection."
            )
        finally:
            self.save()
