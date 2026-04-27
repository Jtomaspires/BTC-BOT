"""
Script 0: Download multiple */USDT 1h datasets from Binance (spot)

Downloads ~N_ROWS_TARGET rows of historical data (most recent) and saves to:
- data/raw/<SYMBOL>-1h-data.csv
- CNN_<PAIR>/data/<SYMBOL>-1h-data.csv (when that folder exists)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import time
import ccxt
import pandas as pd

N_ROWS_TARGET = 69000  # number of 1h candles to keep (most recent)


def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    base = symbol.replace("/", "")
    return f"{base}-{timeframe}-data.csv"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_symbol_data(exchange, symbol: str, timeframe: str, end_date: datetime, out_files: list[Path]):
    print("=" * 70)
    print(f"DOWNLOADING {symbol} ({timeframe}) FROM BINANCE")
    print("=" * 70)

    # Overshoot by 10% to be safe, we will trim to N_ROWS_TARGET later.
    approx_hours = int(N_ROWS_TARGET * 1.1)
    start_date = end_date - timedelta(hours=approx_hours)

    print(f"Date range (overshoot): {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print(f"Target rows (most recent): {N_ROWS_TARGET:,}")

    all_ohlcv: list[list[float]] = []
    current_date = start_date

    while current_date < end_date:
        try:
            since = int(current_date.timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            current_date = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(hours=1)

            print(f"  +{len(ohlcv):4d} candles (up to {current_date.strftime('%Y-%m-%d %H:%M')})")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"  Error: {e}")
            print("  Retrying in 5 seconds...")
            time.sleep(5)

    if not all_ohlcv:
        print("✗ No data downloaded!\n")
        return None

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    original_len = len(df)
    if original_len >= N_ROWS_TARGET:
        df = df.iloc[-N_ROWS_TARGET:].reset_index(drop=True)
    else:
        print(f"! Warning: only {original_len:,} rows downloaded (less than target {N_ROWS_TARGET:,})")

    for out_file in out_files:
        _ensure_dir(out_file.parent)
        df.to_csv(out_file, index=False)
        print(f"✓ Saved: {out_file}")

    print(f"  Final candles: {len(df):,}")
    print(f"  Final range: {df['timestamp'].min()} -> {df['timestamp'].max()}\n")
    return df


def download_all():
    raw_dir = Path("data/raw")
    _ensure_dir(raw_dir)

    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )

    timeframe = "1h"
    end_date = datetime(2026, 3, 31, 23, 59)

    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "XRP/USDT",
        "SOL/USDT",
        "LINK/USDT",
        "PAXG/USDT",
    ]

    pair_folder_map = {
        "BTC/USDT": Path("CNN"),
        "ETH/USDT": Path("CNN_ETH"),
        "XRP/USDT": Path("CNN_XRP"),
        "SOL/USDT": Path("CNN_SOL"),
        "LINK/USDT": Path("CNN_LINK"),
        "PAXG/USDT": Path("CNN_PAXG"),
    }

    for symbol in symbols:
        filename = _symbol_to_filename(symbol, timeframe)
        out_files = [raw_dir / filename]

        pair_root = pair_folder_map.get(symbol)
        if pair_root is not None:
            out_files.append(pair_root / "data" / filename)

        download_symbol_data(exchange, symbol, timeframe, end_date, out_files)


if __name__ == "__main__":
    try:
        download_all()
        print("\n" + "=" * 70)
        print("DOWNLOAD COMPLETE")
        print("=" * 70)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        raise


