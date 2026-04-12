import asyncio
import logging
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import ccxt
import ccxt.async_support as ccxt_async
from binance.client import Client, AsyncClient

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


class Bot:
    def __init__(self, seq_len):
        # Setup logger 
        self.logger = setup_logger('log', './log.log')
        self.seq_len = seq_len # Sequence length

        self.pos_size_usd = 10 # Position size in USD
        self.slippage_pct = 0.05 # Slippage percentage (0.05%)
        self.coin = 'SOL' # Coin
        self.TIME_FRAME = '15m' # Timeframe
        self.asset_usd = 'USDC' # USD asset
        self.price_decimal = 2 # Price decimal precision
        self.size_decimal = 2 # Size decimal precision

        self.symbol = f"{self.coin}/{self.asset_usd}" # The pair we trade 
        self.symbol_ = self.symbol.replace("/", "") # The pair we trade
        self.model_symbol = f'{self.coin}USDT' # The pair model was trained on
        self.position = "NONE" # Current position
        self.open_prices = [] # Open price list
        self.actions = [] # Action list
        self.equities = [] # Equity list
        self.total_fee = 0 # Total fee

        self.last_sec_wait = 20 # Wait until last 20 seconds


    def write_to_log(self, msg):
        self.logger.info(msg)
        print(msg)





    async def create_exchange(self):
        # Enter your API key
        apiKey = "YOUR_API_KEY"
        secret = "YOUR_SCRET_KEY"

        for _ in range(20):
            try:
                # Create Async Client
                self.async_client = await AsyncClient.create(api_key=apiKey, api_secret=secret)

                # Create CCXT
                self.ccxt = ccxt.binance()
                self.write_to_log(f"Exchange created.")
                break

            except Exception as err:
                time.sleep(0.4)
                self.write_to_log(f"Failed to get account with API key {str(err)}")



    async def close_exchange(self):
        for _ in range(20):
            try:
                await self.async_client.close_connection()
                await asyncio.sleep(0.1)
                break
            except Exception as err:
                await asyncio.sleep(0.5)
                self.write_to_log(f"Failed to close connection. Err: " + str(err))





    def fetch_candles(self, symbol, num_candles):
        for _ in range(20):
            try:
                # Fetch OHLCV
                klines = self.ccxt.fetch_ohlcv(symbol, self.TIME_FRAME, limit=num_candles)
                klines = np.array(klines, dtype=object) # Convert to NumPy array

                # Extract columns
                open_times = pd.to_datetime(klines[:, 0].astype(np.int64), unit='ms')
                open_prices  = klines[:, 1].astype(float)
                high_prices  = klines[:, 2].astype(float)
                low_prices   = klines[:, 3].astype(float)
                close_prices = klines[:, 4].astype(float)
                volumes      = klines[:, 5].astype(float)
                break
            
            except Exception as e:
                self.write_to_log(f"Error fetch_candles: {e}")
                time.sleep(0.5)

        return open_times, open_prices, high_prices, low_prices, close_prices, volumes







    async def update_curr_futures_prices(self):
        for _ in range(20):
            try:
                # Fetch klines
                klines = await self.async_client.futures_klines(
                    symbol=self.symbol_,
                    interval=Client.KLINE_INTERVAL_15MINUTE,
                    limit=1,
                )
                klines = np.array(klines, dtype=np.float32)
                self.curr_open = klines[-1, 1] # Update current open price
                self.curr_close = klines[-1, 4] # Update current close price
                self.write_to_log(f'Current Open: {self.curr_open:.2f}. Close: {self.curr_close:.2f} ({self.coin})')
                break

            except Exception as err: 
                self.write_to_log(f"Error in update_curr_futures_prices: {str(err)}")






    async def update_new_data_for_model(self): 
        self.write_to_log("... Getting the new candle ...")

        # Get current time
        open_times, opens, highs, lows, closes, volumes = self.fetch_candles(self.model_symbol, num_candles=1)
        curr_time = open_times[-1] # Current timestamp

        # Start loop check new candle
        while True:
            # Get new time
            open_times, opens, highs, lows, closes, volumes = self.fetch_candles(self.model_symbol, num_candles=self.seq_len+1)
            new_time = open_times[-1] # New timestamp
            
            # Compare new time with current time
            if new_time == curr_time:
                await asyncio.sleep(0.2)
                continue

            self.write_to_log(f"New candle appears at: {datetime.now()}")

            # Prepare USDT data (before preprocessing data for model's input)
            self.opens_USDT   = np.array(opens[:-1], dtype=np.float32)   # USDT open (exclude unfinished candle)
            self.highs_USDT   = np.array(highs[:-1], dtype=np.float32)   # USDT high (exclude unfinished candle)
            self.lows_USDT    = np.array(lows[:-1], dtype=np.float32)    # USDT low (exclude unfinished candle)
            self.closes_USDT  = np.array(closes[:-1], dtype=np.float32)  # USDT close (exclude unfinished candle)
            self.volumes_USDT = np.array(volumes[:-1], dtype=np.float32) # USDT volume (exclude unfinished candle)

            self.write_to_log(f"opens_USDT shape: {self.opens_USDT.shape}")
            self.write_to_log(f"opens_USDT: {self.opens_USDT[-5:]}")
            break







    async def place_limit_order(self, side, is_double_size=False):
        dt0 = datetime.now() # To keep track how long it takes for the order to appear on market

        # Calculate slipage price
        slippage_price = self.curr_close * self.slippage_pct / 100

        # Calculate order price
        if side == "BUY":
            order_price = self.curr_close - slippage_price
        elif side == "SELL":
            order_price = self.curr_close + slippage_price

        # Calculate order size
        order_size = self.pos_size_usd / self.curr_close

        if is_double_size:
            order_size = order_size * 2

        # Round order price & size
        order_price = np.round(order_price, self.price_decimal) 
        order_size = np.round(order_size, self.size_decimal)

        for _ in range(20):
            try:
                order = await self.async_client.futures_create_order(
                    symbol=self.symbol_,
                    side=side,
                    type='LIMIT',
                    quantity=order_size, 
                    timeInForce='GTC',
                    price=order_price,
                )

                order_id = order['orderId']
                self.write_to_log(f"{side} order was placed. Price: {order_price}. Quantity: {order_size} {self.coin}. Duration: {datetime.now() - dt0}")
                self.write_to_log(f"Order details: {order}")
                break

            except Exception as e:
                self.write_to_log(f"FAILED to placed Future limit {side}: {str(e)}")
                await asyncio.sleep(0.4)

        return order_id




    async def cancel_order(self, order_id):
        for _ in range(1):
            try:
                result = await self.async_client.futures_cancel_order(symbol=self.symbol_, 
                                                                    orderid=order_id)
                
                self.write_to_log(f"Canceled order: {result}")

            except Exception as err:
                self.write_to_log(f"Couldn't cancel orders: {err}")
                await asyncio.sleep(0.3)






    async def check_positions(self):
        for _ in range(20):
            try:
                # Fetch positions
                positions = await self.async_client.futures_position_information()
                break
            except Exception as e:
                self.write_to_log(f"Err check_positions: {e}")
                await asyncio.sleep(0.2)

        # Filter Active Positions
        active_positions = [p for p in positions if float(p['positionAmt']) != 0]
        
        if not active_positions: # If No Open Positions
            self.position = "NONE"
            position_usd = 0 
            self.write_to_log("No open positions found...")

        else: # If There Is an Open Position
            self.write_to_log(f"Found {len(active_positions)} open positions:")
            for pos in active_positions:
                if pos['symbol'] == self.symbol_:
                    # Calculate Position Size
                    self.position_coin = float(pos['positionAmt']) # Position size in coin
                    position_usd = np.abs(self.position_coin * self.curr_close) # Position size in USD

                    if 0 < position_usd < self.pos_size_usd*0.8:
                        self.write_to_log(f"WARNING: Position size is too small: {position_usd} {self.coin}")

                    if self.position_coin > 0:
                        self.position = "LONG" # buy
                    elif self.position_coin < 0:
                        self.position = "SHORT" # sell

                    self.write_to_log(f"Current position: {self.position}. Position size: {self.position_coin} {self.coin} (${position_usd:.2f})")
                    break





    async def check_equity(self):
        for _ in range(20):
            try:
                # Fetch balance
                futures_balance = await self.async_client.futures_account_balance()
                
                # Filter for USDC
                usdc_balance = None
                for item in futures_balance:
                    if item['asset'] == self.asset_usd:
                        usdc_balance = item
                        break

                self.equity = float(usdc_balance['balance']) if usdc_balance else 0.0
                self.equity = np.round(self.equity, 2) # Round equity
                self.write_to_log(f"Equity: {self.equity:.2f}")
                break
            
            except Exception as e:
                self.write_to_log(f"Error check_equity: {e}")





    async def get_order_fee(self, order_id):
        for _ in range(10):
            try:
                # Fetch Account Trades
                trades = await self.async_client.futures_account_trades(symbol=self.symbol_)

                # Filter Trades for This Order
                order_trades = [t for t in trades if int(t['orderId']) == int(order_id)]

                if not order_trades: # If No Trades Found
                    self.write_to_log(f"No trades found for order_id {order_id}")
                    return 0.0

                # Sum All Commissions
                fee = sum(float(t['commission']) for t in order_trades)
                commission_asset = order_trades[0]['commissionAsset'] # Identify Commission Asset (e.g USDT, USDC, ...)

                self.write_to_log(f"Order {order_id} fee: {fee} {commission_asset}")
                return fee

            except Exception as e:
                self.write_to_log(f"Error get_order_fee {order_id}: {e}")
                await asyncio.sleep(0.2)






    def plot_performance(self):
        try:
            # Scale the Open Prices
            scaler = MinMaxScaler(feature_range=(min(self.equities), max(self.equities)))
            scaled_open_prices = scaler.fit_transform(np.array(self.open_prices).reshape(-1, 1)).flatten() 

            # Create the Plot
            plt.figure(figsize=(12, 6))
            plt.title(f"Equity: ${self.equity:.2f}") # title
            plt.plot(scaled_open_prices, color='green', label='BTC Price') # price chart
            plt.plot(self.equities, label="Equity Curve", color="blue") # equity curve
            plt.xlabel("Time Step") # x-axis
            plt.ylabel("Equity (USD)") # y-axis
            plt.legend()
            plt.grid(True)
            plt.savefig('./trading_bot/performance.png', dpi=300) # save path
            plt.close()

        except Exception as e:
            print(f"Couldn't plot the graph: {e}")




    def save(self):
        self.saved_data = {
            "open_prices": self.open_prices,
            'actions': self.actions,
            'equities': self.equities,
            'total_fee': self.total_fee,
        }
        with open("./trading_bot/saved_data.pkl", "wb") as file:
            pickle.dump(self.saved_data, file)


    def load(self):
        try:
            with open("./trading_bot/saved_data.pkl", "rb") as file:
                self.saved_data = pickle.load(file)

            self.open_prices = self.saved_data["open_prices"]
            self.actions = self.saved_data['actions']
            self.equities = self.saved_data['equities']
            self.total_fee = self.saved_data['total_fee']
        except:
            print(f"No saved_data to load!")



































































































































































































