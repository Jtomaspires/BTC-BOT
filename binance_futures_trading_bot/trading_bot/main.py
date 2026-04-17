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
        # (e.g. WebSocket drop mid-candle, stale pending id from older runs).
        await bot.reconcile_pending_order()

        # Ensure any pre-existing open position on the exchange has fresh
        # SL/TP orders attached to it. This covers the case where a previous
        # run filled an entry but failed to place protection before crashing,
        # and also restores protection after a plain restart.
        await bot.reconcile_position_protection()

        # First wait: sit idle until the current 1h candle closes so that
        # the very first iteration starts cleanly at a candle boundary.
        await bot.wait_for_candle_close()

        while True:
            order_id = None
            entry_side = None
            fill_price = None

            # ---- Signal phase (start of new candle) ----

            await bot.update_new_data_for_model()

            CURR_ACTION = get_action(
                bot.opens_USDT,
                bot.highs_USDT,
                bot.lows_USDT,
                bot.closes_USDT,
                bot.volumes_USDT,
                train_scalers,
            )

            await bot.update_curr_futures_prices()
            await bot.check_positions()

            # ---- Decide action ----
            # Rules (aligned with CNN_ETH backtest):
            #   * Enter only when currently flat.
            #   * While a position is open, ignore opposite/neutral signals.
            #   * Position exits are handled by SL/TP protection orders.
            if bot.position == "NONE":
                if CURR_ACTION == 1:
                    order_id, fill_price = await bot.place_market_order("BUY")
                    entry_side = "BUY"
                elif CURR_ACTION == -1:
                    order_id, fill_price = await bot.place_market_order("SELL")
                    entry_side = "SELL"

            if order_id is not None and fill_price is not None and entry_side is not None:
                # Backtest evaluates entries at candle open; anchor SL/TP to
                # current candle open to keep USD distances comparable.
                await bot.place_sl_tp_orders(bot.curr_open, entry_side)

            bot.write_to_log(
                f"Action={CURR_ACTION}  Position={bot.position}  "
                f"order_id={order_id}  Waiting for candle close ..."
            )

            # Persist so a crash during the wait can still be reconciled.
            bot.pending_order_id = order_id
            bot.save()

            # ---- Execution window: wait for this candle to close ----
            await bot.wait_for_candle_close()

            # ---- Post-candle cleanup ----
            # Market entries are already filled when we get here, so there is
            # NOTHING to cancel about `order_id`. We only need to:
            #   1) Collect the entry fee (if we placed an entry this cycle).
            #   2) If SL/TP fired mid-candle (position flat), drop any
            #      leftover protection id we are still tracking. No-op when
            #      position is still open in the same direction.
            if order_id is not None:
                fee = await bot.get_order_fee(order_id)
                bot.total_fee += fee

            await bot.check_positions()
            if bot.position == "NONE" and (
                bot.sl_order_id is not None or bot.tp_order_id is not None
            ):
                await bot.cancel_sl_tp_orders()

            bot.pending_order_id = None

            await bot.check_equity()

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
