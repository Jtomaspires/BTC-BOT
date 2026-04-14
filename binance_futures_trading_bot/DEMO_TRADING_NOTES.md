# Binance Futures: Testnet/Sandbox → Demo Trading

## Contexto: depreciação do sandbox

A Binance **descontinuou o modo Futures Testnet/Sandbox** em finais de 2025.
O método `ccxt` anteriormente usado:

```python
exchange.set_sandbox_mode(True)
```

passa a lançar em versões recentes do `ccxt` (≥ 4.5.6):

```
NotSupported: binance testnet/sandbox mode is not supported for futures anymore,
please check the deprecation announcement https://t.me/ccxt_announcements/92
and consider using the demo trading instead.
```

### Substituição

```python
exchange.enable_demo_trading(True)
```

Requer **API keys geradas em [demo.binance.com](https://demo.binance.com)** (não são as mesmas do mainnet).

Aplicado em:

- `trading_bot/bot.py` — criação do exchange em `create_exchange()`
- `smoketest_cli.py` — função `_create_exchange()`

---

## Configuração de credenciais

Copia `.env.example` para `.env` e preenche:

```env
# Demo Trading → keys geradas em https://demo.binance.com
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
BINANCE_TESTNET=true
```

| `BINANCE_TESTNET` | Comportamento |
|---|---|
| `true` | Demo Trading (`demo.binance.com`) — saldo fictício, sem risco |
| `false` | Mainnet real — requer `--live` no CLI |
| omitido | Default: `true` (demo) |

**Demo keys e mainnet keys não são intercambiáveis.**

---

## Script `smoketest_cli.py`

Criado para validar o pipeline completo antes de arrancar o bot:
saldo, WebSocket de candles, configuração de risco, e envio de ordens com notional sizing.

### Subcomandos disponíveis

| Subcomando | O que faz |
|---|---|
| `balance` | Saldo da conta futures (REST via ccxt) |
| `ws-candles` | Ouvir stream de candles via WebSocket |
| `positions` | Listar posições abertas (best-effort) |
| `order` | Enviar ordem com notional sizing (ex: 1000 USD) |

### Flag global `--live`

**Importante:** `--live` é uma flag **global** e tem de vir **antes do subcomando**.

```bash
# CORRECTO
python smoketest_cli.py --live order --side buy ...

# ERRADO (argparse não reconhece flags globais depois do subcomando)
python smoketest_cli.py order --side buy ... --live
```

Sem `--live`, o script **força sempre demo/testnet**, independentemente do `.env`.

### Exemplos de uso (a partir de `c:\Users\jtoma\projects\NN`)

```bash
# Saldo demo
python binance_futures_trading_bot/smoketest_cli.py balance

# WebSocket — candles fechadas de 1m (Ctrl+C para parar)
python binance_futures_trading_bot/smoketest_cli.py ws-candles --symbol-raw ETHUSDT --timeframe 1m --closed-only --max-messages 5

# Preview de ordem sem enviar (dry-run)
python binance_futures_trading_bot/smoketest_cli.py order --side buy --notional-usd 1000 --dry-run

# Ordem market BUY em demo (envia de facto)
python binance_futures_trading_bot/smoketest_cli.py order --side buy --notional-usd 1000 --type market

# Fechar posição em demo (reduce-only)
python binance_futures_trading_bot/smoketest_cli.py order --side sell --notional-usd 1000 --type market --reduce-only

# Saldo em mainnet (requer keys reais no .env e --live ANTES do subcomando)
python binance_futures_trading_bot/smoketest_cli.py --live balance
```

### Comportamento do `order`

Antes de enviar cada ordem, o subcomando aplica automaticamente:

1. `set_margin_mode(isolated, ETH/USDT:USDT)` — configurável via `--margin-mode`
2. `set_leverage(10, ETH/USDT:USDT)` — configurável via `--leverage`
3. Calcula `amount = notional_usd / last_price`, arredondado à precisão do mercado
4. Imprime preview (incluindo `dry_run: true/false`) **antes** de enviar
5. Se `--dry-run`: termina aqui, sem enviar

---

## Erros encontrados durante os testes e respectivas causas

### 1) `set_sandbox_mode` deprecated

```
NotSupported: binance testnet/sandbox mode is not supported for futures anymore
```

**Causa:** `set_sandbox_mode(True)` descontinuado para Futures.  
**Fix:** substituído por `exchange.enable_demo_trading(True)` em `bot.py` e `smoketest_cli.py`.

---

### 2) `--live` como argumento não reconhecido

```
smoketest_cli.py: error: unrecognized arguments: --live
```

**Causa:** `--live` é uma flag do parser principal (`argparse`), não do subparser `order`.
O `argparse` não aceita flags do parser pai depois do subcomando.  
**Fix (uso):** colocar `--live` **antes** do subcomando:

```bash
python smoketest_cli.py --live order --side buy ...
```

---

### 3) `Invalid Api-Key ID` ao usar `--live` com keys de demo

```
AuthenticationError: binance {"code":-2008,"msg":"Invalid Api-Key ID."}
```

**Causa:** as keys no `.env` eram geradas em `demo.binance.com`. Com `--live`, o script
desactiva o demo trading e tenta conectar à mainnet real — as keys não são válidas aí.  
**Fix (uso):** para mainnet, usar keys geradas em `binance.com`. Para demo, não usar `--live`.

---

### 4) `Unclosed connector` (aviso de aiohttp)

```
binance requires to release all resources with an explicit call to the .close() coroutine.
Unclosed connector
```

**Causa:** quando `load_markets()` lança excepção (ex: auth error), o fluxo sai de
`_create_exchange` sem passar pelo `finally: await ex.close()` dos comandos, porque o
exchange nunca chegou a ser retornado.  
**Estado:** aviso cosmético, não bloqueia o script. A correcção envolve adicionar
`try/finally` em `_create_exchange` para garantir `close()` mesmo em caso de erro.

---

## Resultados validados em demo (14 Abr 2026)

| Teste | Resultado |
|---|---|
| `balance` — saldo demo | 5000 USDT disponível |
| `ws-candles` — stream 1m ETH/USDT | Candles fechadas recebidas correctamente |
| `order --dry-run` — preview BUY 1000 USD | amount=0.425 ETH @ 2349.27, sem envio |
| `order` BUY market 1000 USD | **FILLED** — 0.426 ETH @ avg 2344.55 (998.78 USDT) |
| `order` SELL market `--reduce-only` | **FILLED** — 0.426 ETH @ avg 2341.97 (997.68 USDT) |

Pipeline completo validado: demo trading, margin isolated, leverage 10x, notional sizing, e fecho de posição.

---

## Porque 1m no WebSocket?

Para testes, **1m** é o default ideal:

- valida conectividade e parsing do stream rapidamente
- dá feedback imediato sem esperar 1h

Para o bot em produção, o timeframe é **1h** (definido em `bot.py`).  
No `smoketest_cli.py` podes trocar com `--timeframe 5m` / `--timeframe 1h`.
