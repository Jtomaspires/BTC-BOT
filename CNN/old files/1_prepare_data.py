"""
Script 1: Prepare features from raw OHLCV data

Creates 40+ technical indicators and saves to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_features(df):
    """
    Create technical indicators and features from OHLCV data
    
    Returns DataFrame with all features
    """
    
    print("Creating features...")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # ========================================================================
    # 0. LECTURE29 SIMPLE FEATURES (8 features - matches notebook)
    # ========================================================================
    print("  - Lecture29 simple features (8 features)...")
    
    df['feature1'] = (df['close'] - df['open']) / df['open']
    df['feature2'] = (df['high'] - df['open']) / df['open']
    df['feature3'] = (df['high'] - df['close']) / df['close']
    df['feature4'] = (df['low'] - df['open']) / df['open']
    df['feature5'] = (df['low'] - df['close']) / df['close']
    df['feature6'] = (df['high'] - df['low']) / df['open']
    df['feature7'] = (df['high'] - df['low']) / df['close']
    df['feature8'] = df['volume']
    
    # ========================================================================
    # 1. PRICE BASIC FEATURES
    # ========================================================================
    print("  - Price basic features...")
    
    df['price_change'] = df['close'].pct_change()
    df['high_open_ratio'] = (df['high'] - df['open']) / df['open']
    df['high_close_ratio'] = (df['high'] - df['close']) / df['close']
    df['low_open_ratio'] = (df['low'] - df['open']) / df['open']
    df['low_close_ratio'] = (df['low'] - df['close']) / df['close']
    df['hl_open_range'] = (df['high'] - df['low']) / df['open']
    df['hl_close_range'] = (df['high'] - df['low']) / df['close']
    
    # ========================================================================
    # 2. TREND FEATURES (Moving Averages, MACD)
    # ========================================================================
    print("  - Trend features...")
    
    # Simple Moving Averages
    for period in [7, 14, 21, 50]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}'] - 1
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD  
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ========================================================================
    # 3. MOMENTUM FEATURES (RSI, Stochastic, etc.)
    # ========================================================================
    print("  - Momentum features...")
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    for period in [7, 14, 21]:
        df[f'rsi_{period}'] = calculate_rsi(df['close'], period)
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Williams %R
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
    
    # CCI (Commodity Channel Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # ROC (Rate of Change)
    for period in [10, 20]:
        df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100
    
    # MFI (Money Flow Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    df['mfi'] = mfi
    
    # ========================================================================
    # 4. VOLATILITY FEATURES (ATR, Bollinger Bands, Historical Volatility)
    # ========================================================================
    print("  - Volatility features...")
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper_20'] = sma_20 + (std_20 * 2)
    df['bb_lower_20'] = sma_20 - (std_20 * 2)
    df['bb_width_20'] = (df['bb_upper_20'] - df['bb_lower_20']) / sma_20
    df['bb_position_20'] = (df['close'] - df['bb_lower_20']) / (df['bb_upper_20'] - df['bb_lower_20'])
    
    # Historical Volatility
    returns = df['close'].pct_change()
    for period in [10, 20, 50]:
        df[f'hv_{period}'] = returns.rolling(window=period).std() * np.sqrt(252 * 96)  # Annualized for 15m
    
    # Volatility Regime (high/low volatility)
    df['vol_regime'] = (df['hv_20'] > df['hv_20'].rolling(window=50).mean()).astype(int)
    
    # ========================================================================
    # 5. VOLUME FEATURES
    # ========================================================================
    print("  - Volume features...")
    
    # Volume ratio (current vs average)
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # OBV (On-Balance Volume) with EMA
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ema_20'] = df['obv'].ewm(span=20, adjust=False).mean()
    
    # VPT (Volume Price Trend)
    df['vpt'] = (df['close'].pct_change() * df['volume']).fillna(0).cumsum()
    
    # Force Index
    df['force_index'] = df['close'].diff() * df['volume']
    df['force_index_ema'] = df['force_index'].ewm(span=13, adjust=False).mean()
    
    # ========================================================================
    # 6. PATTERN FEATURES
    # ========================================================================
    print("  - Pattern features...")
    
    # Price position in range
    df['price_position'] = (df['close'] - df['low'].rolling(window=20).min()) / \
                           (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min())
    
    # Higher high / Lower low
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) & 
                        (df['high'].shift(1) > df['high'].shift(2))).astype(int)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) & 
                      (df['low'].shift(1) < df['low'].shift(2))).astype(int)
    
    # Consecutive up/down
    df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int).groupby(
        (df['close'] > df['close'].shift(1)).ne(df['close'] > df['close'].shift(1)).cumsum()
    ).cumcount() + 1
    df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int).groupby(
        (df['close'] < df['close'].shift(1)).ne(df['close'] < df['close'].shift(1)).cumsum()
    ).cumcount() + 1
    
    # Body size (candle body relative to range)
    body = np.abs(df['close'] - df['open'])
    range_size = df['high'] - df['low']
    df['body_size'] = body / (range_size + 1e-10)
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    # Replace inf and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill missing values (for indicators that need warm-up period)
    df = df.fillna(method='ffill').fillna(0)
    
    # Remove rows with any remaining NaN (should be minimal)
    initial_len = len(df)
    df = df.dropna()
    removed = initial_len - len(df)
    
    if removed > 0:
        print(f"  - Removed {removed} rows with NaN values")
    
    return df

def prepare_data():
    """Main function to prepare data"""
    
    print("="*70)
    print("PREPARING FEATURES FROM RAW DATA")
    print("="*70)
    
    # Load raw data
    raw_file = Path("data/raw/BTCUSDT-15m-data.csv")
    
    if not raw_file.exists():
        print(f"\n✗ Error: {raw_file} not found!")
        print("  Please run: python scripts/0_download_data.py first")
        return
    
    print(f"\nLoading data from: {raw_file}")
    df = pd.read_csv(raw_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"  Loaded {len(df):,} rows")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Create features
    df_features = create_features(df)
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    output_file = output_dir / "btc_15m_features.csv"
    df_features.to_csv(output_file, index=False)
    
    print(f"\n✓ Features saved to: {output_file}")
    print(f"  Total rows: {len(df_features):,}")
    print(f"  Total columns: {len(df_features.columns)}")
    
    # List all feature columns (exclude OHLCV and timestamp)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    
    print(f"\n  Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:2d}. {col}")
    
    return df_features

if __name__ == "__main__":
    try:
        prepare_data()
        print("\n" + "="*70)
        print("FEATURE PREPARATION COMPLETE")
        print("="*70)
    except KeyboardInterrupt:
        print("\n\nPreparation interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

