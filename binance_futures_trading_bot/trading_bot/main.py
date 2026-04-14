import asyncio

from .load_scalers import train_scalers, seq_len
from .load_models import get_action, load_model
from .bot import Bot


async def main():
    # load_model() MUST run before create_exchange() so that a model-loading
    # failure does not leave an open exchange connection behind.
    load_model()

    bot = Bot(seq_len, testnet=True)
    bot.load()

    await bot.create_exchange()

    try:
        # Reconcile any order that was left open when the previous run crashed
        # (e.g. WebSocket drop in the middle of a candle).
        await bot.reconcile_pending_order()

        # First wait: sit idle until the current 1h candle closes so that
        # the very first iteration starts cleanly at a candle boundary.
        await bot.wait_for_candle_close()

        while True:
            order_id = None

            # ---- Signal phase (start of new candle) ----

            # Fetch the last seq_len closed candles
            await bot.update_new_data_for_model()

            # Compute model signal: +1 long, -1 short, 0 neutral
            CURR_ACTION = get_action(
                bot.opens_USDT,
                bot.highs_USDT,
                bot.lows_USDT,
                bot.closes_USDT,
                bot.volumes_USDT,
                train_scalers,
            )

            # Fetch live price (used for limit order reference)
            await bot.update_curr_futures_prices()

            # Check current open position
            await bot.check_positions()

            # ---- Order placement ----
            # CURR_ACTION == 0 (neutral): no new order; existing position is
            # held as-is (aligned with backtest: only SL/TP/reversal closes).

            if CURR_ACTION == 1 and bot.position != "LONG":
                is_double_size = bot.position == "SHORT"
                order_id = await bot.place_limit_order("BUY", is_double_size)

            elif CURR_ACTION == -1 and bot.position != "SHORT":
                is_double_size = bot.position == "LONG"
                order_id = await bot.place_limit_order("SELL", is_double_size)

            bot.write_to_log(
                f"Action={CURR_ACTION}  Position={bot.position}  "
                f"order_id={order_id}  Waiting for candle close ..."
            )

            # Persist pending order so a crash during the wait can be recovered.
            bot.pending_order_id = order_id
            bot.save()

            # ---- Execution window: wait for this candle to close ----
            await bot.wait_for_candle_close()

            # ---- Post-candle cleanup ----

            if order_id is not None:
                await bot.cancel_order(order_id)
                fee = await bot.get_order_fee(order_id)
                bot.total_fee += fee

            # Clear pending order now that cleanup is done.
            bot.pending_order_id = None

            await bot.check_equity()

            # Record for performance tracking
            bot.actions.append(CURR_ACTION)
            bot.open_prices.append(bot.curr_open)
            bot.equities.append(bot.equity)

            bot.save()
            bot.plot_performance()

    finally:
        await bot.close_exchange()


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception as e:
            print(f"Error: {e}")
