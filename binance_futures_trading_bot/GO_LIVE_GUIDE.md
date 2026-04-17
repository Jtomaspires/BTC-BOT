# Go-live guide — Binance Futures trading bot (VPS)

This document describes how to run the bot on a VPS from **paper/demo** through to **mainnet (real funds)**. Adjust commands for your OS (examples use Linux).

---

## 1. What you are deploying

- **Runtime**: Python 3.11+ (3.12 is fine if dependencies install).
- **Entry point**: `trading_bot.main` — long-running loop: load models → connect exchange → wait for 1h candle closes → signals → limit orders.
- **Repo layout**: The bot expects the **`CNN_ETH`** project folder to sit **next to** `binance_futures_trading_bot` at the same parent level (same as your dev machine: `NN/CNN_ETH`, `NN/binance_futures_trading_bot`). Model loading and `scalers.pkl` resolve paths from that layout.
- **Secrets**: `binance_futures_trading_bot/.env` (never commit; keep permissions tight on the server).

---

## 2. Pre-flight checklist (before any real money)

- [ ] You have run **`smoketest_cli.py`** successfully against the **same** environment you will use (demo first, then mainnet dry checks). See [DEMO_TRADING_NOTES.md](DEMO_TRADING_NOTES.md).
- [ ] **Binance Futures** is enabled on the account you will use; you understand **isolated margin** and **10x leverage** applied at startup for `ETH/USDT:USDT` (see `bot.py`).
- [ ] You accept that this bot places **real limit orders** on mainnet when `BINANCE_TESTNET=false` — only proceed when intentional.
- [ ] **API key** is restricted: Futures enabled; optionally **IP whitelist** to the VPS egress IP; **no** withdrawal permission on the key if Binance allows that for your use case.

---

## 3. VPS baseline

1. **Create a non-root user** for running the bot (e.g. `trader`).
2. **Install dependencies** (example Debian/Ubuntu):
   - `sudo apt update && sudo apt install -y git python3 python3-venv`
3. **Clone the repository** so the tree matches local development (parent folder `NN` containing both `CNN_ETH` and `binance_futures_trading_bot`).
4. **Python venv** (recommended):

   ```bash
   cd /path/to/NN/binance_futures_trading_bot
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Firewall**: only SSH (and any monitoring you need); the bot needs **outbound HTTPS/WSS** to Binance (no inbound trading API port required).

---

## 4. Environment configuration

Copy [`.env.example`](.env.example) to `.env` in `binance_futures_trading_bot/`:

```env
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
BINANCE_TESTNET=true
```

| Mode | `BINANCE_TESTNET` | API keys from |
|------|-------------------|---------------|
| Demo / paper | `true` | [demo.binance.com](https://demo.binance.com) |
| Mainnet (real) | `false` | [binance.com](https://www.binance.com) API Management |

Demo keys and mainnet keys are **not** interchangeable. After editing `.env`:

```bash
chmod 600 .env
```

The bot loads `.env` from **`binance_futures_trading_bot/.env`** via path relative to the code, so you can run the process with any working directory as long as that file exists.

---

## 5. Phase A — Demo on the VPS (recommended)

Goal: prove connectivity, credentials, and process supervision **without** real funds.

1. Set `BINANCE_TESTNET=true` and use **demo** API keys.
2. From the repo parent (`NN`), with venv activated:

   ```bash
   cd /path/to/NN/binance_futures_trading_bot
   source .venv/bin/activate
   python smoketest_cli.py balance
   ```

3. Optional: `python smoketest_cli.py ws-candles --max-messages 3`
4. Run the bot manually once (short test; Ctrl+C to stop):

   ```bash
   cd /path/to/NN/binance_futures_trading_bot
   source .venv/bin/activate
   python -m trading_bot.main
   ```

5. Confirm logs under `binance_futures_trading_bot/trading_bot/log.log` and no startup errors.

---

## 6. Phase B — Mainnet go-live

1. **Create new API keys** on Binance **mainnet** with Futures permission; bind to VPS IP if you use IP restriction.
2. Update `.env`:
   - `BINANCE_API_KEY` / `BINANCE_API_SECRET` — mainnet keys  
   - `BINANCE_TESTNET=false`
3. **Smoke test mainnet read-only first** (still uses your real key; no bot loop):

   ```bash
   cd /path/to/NN/binance_futures_trading_bot
   source .venv/bin/activate
   python smoketest_cli.py --live balance
   ```

   Note: `--live` must appear **before** the subcommand (see [DEMO_TRADING_NOTES.md](DEMO_TRADING_NOTES.md)).

4. When satisfied, start the bot the same way as Phase A (`python -m trading_bot.main`). The exchange is created as **mainnet** when `BINANCE_TESTNET=false` (demo trading is not enabled).

---

## 7. Run 24/7 — systemd (recommended)

Run the bot as a service so it restarts on failure and survives SSH disconnects.

Example **`/etc/systemd/system/binance-bot.service`** (adjust paths and user):

```ini
[Unit]
Description=Binance Futures trading bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=trader
WorkingDirectory=/path/to/NN/binance_futures_trading_bot
Environment="PATH=/path/to/NN/binance_futures_trading_bot/.venv/bin:/usr/bin"
ExecStart=/path/to/NN/binance_futures_trading_bot/.venv/bin/python -m trading_bot.main
Restart=on-failure
RestartSec=10

# Hardening (optional; tune for your distro)
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable binance-bot
sudo systemctl start binance-bot
sudo systemctl status binance-bot
journalctl -u binance-bot -f
```

Logs also go to `trading_bot/log.log` as configured in code.

### Windows Server / Windows 10/11 VPS — Task Scheduler (recommended)

Use this when the VPS runs **Windows** (RDP). Goal: **one** long-running process after reboot, with **automatic restart** if Python exits.

1. **Layout on disk** (same as Linux): parent folder `NN` with `CNN_ETH` and `binance_futures_trading_bot` side by side; `.env` under `binance_futures_trading_bot/`. Install deps with `py -3.12 -m pip install -r requirements.txt`.

2. **Supervised launcher** (restart loop + log): point Task Scheduler at **`run_bot_supervised.bat`**, not `run_bot.bat`. The supervisor appends lines to `logs/supervisor.log` and restarts Python after every exit (crash, OOM, etc.), waiting **20 seconds** between attempts. To change the wait, set a Windows **user or system** environment variable `BOT_RESTART_DELAY_SEC` (e.g. `30`) and sign out/in or reboot so new tasks inherit it.

3. **Create Task** (use **Create Task…**, not Basic Wizard — you need full options):
   - **General**
     - **Run whether user is logged on or not** (survives RDP disconnect; stores password once).
     - **Run with highest privileges**: usually **off** unless you hit permission issues.
     - **Configure for**: your Windows version.
   - **Triggers**
     - **At startup** — **Advanced**: **Delay task for** `1–2 minutes` (lets network/DNS come up).
     - Do **not** add hourly triggers; the bot is already a 24/7 loop with hourly candles internally.
   - **Actions**
     - **Start a program**
     - **Program/script**: `C:\Windows\System32\cmd.exe`
     - **Add arguments**: `/c "C:\path\to\NN\binance_futures_trading_bot\run_bot_supervised.bat"` (adjust path).
     - **Start in (optional)**: `C:\path\to\NN\binance_futures_trading_bot`
   - **Conditions**
     - Uncheck **Start the task only if the computer is on AC power** (laptops).
   - **Settings**
     - **Allow task to be run on demand** — on.
     - **If the task fails, restart every** `1 minute` (backup restart; the `.bat` loop is the main recovery).
     - **Attempt to restart up to** `3` or higher.
     - **If the task is already running, then the following rule applies**: **Do not start a new instance** — prevents two bots after reboot or manual runs.

4. **Verify**: **Run** the task once from the library; confirm `trading_bot/log.log` updates and **only one** `python`/`py` process for the bot.

5. **Stop**: Task Scheduler → **End** task, then confirm no stray `python` in Task Manager. Review open positions on Binance if needed.

---

## 8. Operations

| Task | Action |
|------|--------|
| Stop | **Linux:** `sudo systemctl stop binance-bot` — **Windows:** Task Scheduler → task → **End**; confirm no extra `python` in Task Manager |
| Restart after `.env` change | **Linux:** `sudo systemctl restart binance-bot` — **Windows:** **End** task, then **Run** again (or reboot) |
| View logs | **Linux:** `journalctl -u binance-bot -f` — **all:** `trading_bot/log.log`; **Windows supervisor:** `logs/supervisor.log` |
| State file | `trading_bot/saved_data.pkl` — persisted actions/equity/pending order id |
| Plot output | `trading_bot/performance.png` |

Before upgrades: **stop the bot**, pull code, reinstall deps if `requirements.txt` changed, restart.

---

## 9. Security summary

- Never commit `.env`; rotate keys if leaked.
- Use a **dedicated** sub-account or API key for the bot if Binance supports it.
- Keep the VPS patched; SSH keys only; disable password SSH if possible.
- Monitor the account on Binance (open positions, order history).

---

## 10. Rollback / kill switch

- **Fastest**: `sudo systemctl stop binance-bot` — stops new logic; **review open positions on Binance** manually (the bot may leave positions depending on state).
- Set `BINANCE_TESTNET=true` only when using **demo** keys — do not point demo keys at mainnet settings or vice versa.
- Revoke or disable the API key on Binance if you need an immediate halt on new API activity.

---

## 11. Troubleshooting pointers

- **Missing API keys**: ensure `.env` is in `binance_futures_trading_bot/` and variables are set.
- **Model / scaler errors**: ensure `CNN_ETH` exists beside `binance_futures_trading_bot` and `CNN_ETH/artifacts/manifest.json` and checkpoints are present.
- **Invalid API key on mainnet**: you may still have demo keys in `.env` or forgot `python smoketest_cli.py --live ...` for mainnet tests.
- See [DEMO_TRADING_NOTES.md](DEMO_TRADING_NOTES.md) for ccxt demo-trading and CLI flag behaviour.

---

## 12. Optional next steps

- Alerts on process down (systemd `OnFailure=`, healthchecks.io, Uptime Kuma, etc.).
- Log rotation for `log.log` (`logrotate`).
- Separate **staging** VPS or demo-only profile for testing changes before touching mainnet.

This guide is operational documentation only — not financial advice. Trading futures involves substantial risk of loss.
