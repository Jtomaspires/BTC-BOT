import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt.async_support as ccxt_async
import websockets
from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_PATH = _SCRIPT_DIR / ".env"


@dataclass(frozen=True)
class ExchangeConfig:
    api_key: str
    api_secret: str
    testnet: bool
    default_type: str = "future"


def _bool_env(name: str) -> bool | None:
    v = os.getenv(name)
    if v is None:
        return None
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _load_exchange_config(force_testnet: bool | None, live: bool) -> ExchangeConfig:
    # Load credentials from bot-local .env, independent of process cwd.
    load_dotenv(dotenv_path=_ENV_PATH)

    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing BINANCE_API_KEY / BINANCE_API_SECRET. "
            "Create a `.env` (copy from `.env.example`) in binance_futures_trading_bot/."
        )

    env_testnet = _bool_env("BINANCE_TESTNET")
    if force_testnet is None:
        testnet = True if env_testnet is None else env_testnet
    else:
        testnet = force_testnet

    # Safety: never accidentally hit mainnet unless explicitly requested.
    if not live:
        testnet = True

    return ExchangeConfig(api_key=api_key, api_secret=api_secret, testnet=testnet)


async def _create_exchange(cfg: ExchangeConfig):
    ex = ccxt_async.binance(
        {
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "options": {"defaultType": cfg.default_type},
        }
    )
    if cfg.testnet:
        # Binance deprecated the Futures Sandbox (set_sandbox_mode) in late 2025.
        # The replacement is Demo Trading: keys generated at https://demo.binance.com
        ex.enable_demo_trading(True)
    await ex.load_markets()
    return ex


def _ws_stream_symbol(symbol_raw: str) -> str:
    # Binance WS streams require lowercase.
    return symbol_raw.lower()


def _ws_stream_for_kline(symbol_raw: str, timeframe: str) -> str:
    # e.g. ethusdt@kline_1m
    return f"{_ws_stream_symbol(symbol_raw)}@kline_{timeframe}"


async def cmd_balance(args: argparse.Namespace) -> int:
    cfg = _load_exchange_config(force_testnet=args.testnet, live=args.live)
    ex = await _create_exchange(cfg)
    try:
        bal = await ex.fetch_balance()
        asset = (args.asset or "USDT").upper()
        total = bal.get("total", {}).get(asset)
        free = bal.get("free", {}).get(asset)
        used = bal.get("used", {}).get(asset)
        print(
            json.dumps(
                {
                    "env": "testnet" if cfg.testnet else "mainnet",
                    "asset": asset,
                    "free": free,
                    "used": used,
                    "total": total,
                },
                indent=2,
            )
        )
        if args.verbose:
            # Warning: may be large.
            print(json.dumps(bal, indent=2, default=str)[:10000])
        return 0
    finally:
        await ex.close()


async def cmd_ws_candles(args: argparse.Namespace) -> int:
    stream = _ws_stream_for_kline(args.symbol_raw, args.timeframe)
    url = f"wss://fstream.binance.com/ws/{stream}"

    n = 0
    print(json.dumps({"connecting": url}, indent=2))
    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        async for msg in ws:
            data = json.loads(msg)
            k = data.get("k") or {}
            is_closed = bool(k.get("x"))
            if args.closed_only and not is_closed:
                continue

            out: dict[str, Any] = {
                "event_time": data.get("E"),
                "symbol": k.get("s"),
                "timeframe": k.get("i"),
                "start": k.get("t"),
                "close_time": k.get("T"),
                "closed": is_closed,
                "open": k.get("o"),
                "high": k.get("h"),
                "low": k.get("l"),
                "close": k.get("c"),
                "volume": k.get("v"),
                "trades": k.get("n"),
            }
            print(json.dumps(out, indent=2))
            n += 1
            if args.max_messages is not None and n >= args.max_messages:
                return 0
    return 0


async def _configure_symbol_risk(
    ex,
    symbol_unified: str,
    margin_mode: str | None,
    leverage: int | None,
) -> None:
    if margin_mode is not None:
        try:
            await ex.set_margin_mode(margin_mode, symbol_unified)
            print(json.dumps({"set_margin_mode": margin_mode, "symbol": symbol_unified}, indent=2))
        except Exception as e:
            print(json.dumps({"set_margin_mode": "non-fatal", "error": str(e)}, indent=2))
    if leverage is not None:
        try:
            await ex.set_leverage(leverage, symbol_unified)
            print(json.dumps({"set_leverage": leverage, "symbol": symbol_unified}, indent=2))
        except Exception as e:
            print(json.dumps({"set_leverage": "non-fatal", "error": str(e)}, indent=2))


async def cmd_order(args: argparse.Namespace) -> int:
    cfg = _load_exchange_config(force_testnet=args.testnet, live=args.live)
    ex = await _create_exchange(cfg)
    try:
        symbol = args.symbol  # ccxt unified symbol, e.g. ETH/USDT:USDT

        await _configure_symbol_risk(
            ex,
            symbol_unified=symbol,
            margin_mode=args.margin_mode,
            leverage=args.leverage,
        )

        ticker = await ex.fetch_ticker(symbol)
        last = ticker.get("last")
        if last is None:
            raise RuntimeError("fetch_ticker returned no last price")

        notional = float(args.notional_usd)
        amount = notional / float(last)
        amount = float(ex.amount_to_precision(symbol, amount))

        side = args.side.upper()
        order_type = args.type.lower()
        if order_type not in ("market", "limit"):
            raise ValueError("--type must be market or limit")

        price = None
        if order_type == "limit":
            if args.price is None:
                raise ValueError("--price is required for limit orders")
            price = float(ex.price_to_precision(symbol, float(args.price)))

        params: dict[str, Any] = {}
        if args.reduce_only:
            params["reduceOnly"] = True

        preview = {
            "env": "testnet" if cfg.testnet else "mainnet",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "notional_usd": notional,
            "last_price": last,
            "amount": amount,
            "price": price,
            "params": params,
            "dry_run": bool(args.dry_run),
        }
        print(json.dumps(preview, indent=2))

        if args.dry_run:
            return 0

        order = await ex.create_order(symbol, order_type, side, amount, price, params)
        print(json.dumps(order, indent=2, default=str))
        return 0
    finally:
        await ex.close()


async def cmd_positions(args: argparse.Namespace) -> int:
    cfg = _load_exchange_config(force_testnet=args.testnet, live=args.live)
    ex = await _create_exchange(cfg)
    try:
        symbol = args.symbol
        try:
            positions = await ex.fetch_positions([symbol] if symbol else None)
            print(json.dumps(positions, indent=2, default=str)[:20000])
            return 0
        except Exception as e:
            # Fallback: at least show balance if positions endpoint fails.
            bal = await ex.fetch_balance()
            print(json.dumps({"fetch_positions_error": str(e)}, indent=2))
            print(json.dumps(bal, indent=2, default=str)[:20000])
            return 0
    finally:
        await ex.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smoketest_cli.py",
        description="Binance Futures smoke tests: balance, WS candles, and real order placement (ccxt + WS).",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Allow mainnet calls. Without this flag, the script forces testnet for safety.",
    )
    p.add_argument(
        "--testnet",
        action="store_true",
        help="Force testnet (overrides BINANCE_TESTNET). Ignored if --live is not set (still testnet).",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    p_bal = sub.add_parser("balance", help="Fetch futures balance.")
    p_bal.add_argument("--asset", default="USDT")
    p_bal.add_argument("--verbose", action="store_true")
    p_bal.set_defaults(func=cmd_balance)

    p_ws = sub.add_parser("ws-candles", help="Listen to kline stream via WebSocket (default 1m).")
    p_ws.add_argument("--symbol-raw", default="ETHUSDT", help="Raw symbol for WS stream, e.g. ETHUSDT.")
    p_ws.add_argument("--timeframe", default="1m", help="Kline timeframe for WS stream, e.g. 1m, 5m, 1h.")
    p_ws.add_argument("--closed-only", action="store_true", help="Print only closed candles (k.x == true).")
    p_ws.add_argument("--max-messages", type=int, default=None, help="Stop after N printed messages.")
    p_ws.set_defaults(func=cmd_ws_candles)

    p_pos = sub.add_parser("positions", help="Fetch open positions (best-effort).")
    p_pos.add_argument("--symbol", default="ETH/USDT:USDT", help="ccxt unified symbol.")
    p_pos.set_defaults(func=cmd_positions)

    p_ord = sub.add_parser("order", help="Place a futures order using notional sizing.")
    p_ord.add_argument("--symbol", default="ETH/USDT:USDT", help="ccxt unified symbol, e.g. ETH/USDT:USDT.")
    p_ord.add_argument("--side", required=True, choices=["buy", "sell", "BUY", "SELL"])
    p_ord.add_argument("--type", default="market", help="market or limit")
    p_ord.add_argument("--price", default=None, help="Required for limit orders.")
    p_ord.add_argument("--notional-usd", type=float, default=1000.0, help="Notional size in USD.")
    p_ord.add_argument("--margin-mode", default="isolated", choices=["isolated", "cross", "ISOLATED", "CROSS"])
    p_ord.add_argument("--leverage", type=int, default=10)
    p_ord.add_argument("--reduce-only", action="store_true")
    p_ord.add_argument("--dry-run", action="store_true", help="Print order preview, do not send order.")
    p_ord.set_defaults(func=cmd_order)

    return p


async def _amain() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # normalize
    if hasattr(args, "margin_mode") and isinstance(args.margin_mode, str):
        args.margin_mode = args.margin_mode.lower()

    return await args.func(args)


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()

