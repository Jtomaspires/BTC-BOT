# Migração Futures → CFDs e execução via MT5

Notas sobre o que muda quando o bot passar de **perpétuos / futures em exchange** (ex.: Binance) para **CFDs em contas de prop firm**, tipicamente com **MetaTrader 5** como camada de execução.

---

## CFDs vs Futures — O que muda

### Estrutura do instrumento

**Futures (cenário atual)**

- Contrato padronizado numa exchange (Binance, Bybit, etc.).
- Tu és o dono da posição **diretamente** na exchange.
- Liquidação / referência ligada ao ecossistema da exchange (ex.: mark price).
- **Funding rate** em perpétuos (ex.: a cada 8h).

**CFDs (prop firms)**

- Contrato entre ti e a prop firm (ou o broker por detrás dela).
- A firma (ou o seu stack) é a **contraparte** — não há “tua” API de exchange no mesmo sentido; muitas vezes não acedes à exchange subjacente.
- Preço vem do **broker da firma** (spread e microestrutura podem diferir do que vês em futures).
- **Sem funding rate** típico de perpétuos — mas pode haver **swap / overnight fee**.

---

## Implicações técnicas para o bot

### 1. API completamente diferente

As prop firms **não** expõem o mesmo modelo que **ccxt** sobre Binance/Bybit. Muitas rotas passam por **MetaTrader 5** ou plataformas próprias.

Exemplos (referência; confirmar sempre no site / suporte da firma):

| Firma      | Plataforma comum | API / integração típica        |
|-----------|------------------|---------------------------------|
| FTMO      | MT5 / cTrader    | MT5 via Python (`MetaTrader5`) |
| The5ers   | MT5              | MT5 via Python                  |
| Topstep   | Tradovate / NT8  | API própria da stack            |
| MyFundedFX| MT5              | MT5 via Python                  |

**Consequência:** o módulo que hoje fala com exchange via **ccxt** deixa de ser reutilizável tal como está. É preciso uma **camada de execução** para MT5 (ou outra API que a firma exija). O **ccxt não substitui** isso para contas MT5.

### 2. Execução via MetaTrader 5 em Python

Ilustração de envio de ordem “a mercado” (equivalente conceptual a `place_market_order`):

```python
import MetaTrader5 as mt5

mt5.initialize()
mt5.login(account, password, server)

# Equivalente ao place_market_order (conceito: deal imediato)
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": "ETHUSD",
    "volume": 0.1,
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick("ETHUSD").ask,
    "deviation": 20,
    "magic": 123456,
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
result = mt5.order_send(request)
```

**Cuidado:** o terminal / API MT5 corre de forma oficial e estável em **Windows**. Para VPS, planear **Windows** se a execução for 100% MT5.

### 3. Símbolo diferente

- Em futures (ccxt): exemplo `ETH/USDT:USDT`.
- Em CFDs MT5: depende do broker — `ETHUSD`, `ETHUSD.c`, `ETHUSD_micro`, etc.

**Cuidado:** validar **símbolo, digits, contract size e volume mínimo** na conta real/demo da firma antes de automatizar.

### 4. Dados OHLCV

Em vez de `fetch_ohlcv` (ccxt), obténs barras via MT5, por exemplo:

```python
rates = mt5.copy_rates_from_pos("ETHUSD", mt5.TIMEFRAME_H1, 0, 49)
# array com open, high, low, close, tick_volume (e tempo)
```

**O que não muda:** pipelines de features, scalers e **modelos** — desde que o timeframe e a definição de candle estejam alinhados com o que treinaste. **Muda a fonte** dos dados e o código que as busca.

### 5. Sem WebSocket nativo no mesmo estilo

MT5 não expõe o mesmo padrão WS que muitos bots crypto usam para “fecho de candle”.

**Cuidados:**

- **Polling** controlado: `copy_rates_*` ou deteção de mudança de barra pelo timestamp da última vela.
- Evitar spam à API: intervalos mínimos, backoff, e alinhamento com o fecho real do servidor (timezone / “broker time”).

### 6. Volume em lotes, não em USD direto

Em futures costumas fazer algo como `order_size = pos_size_usd / curr_close` (com precisão da exchange).

Em MT5 o **volume é em lotes**; o valor de 1 lote depende do **contract size** do símbolo.

Conversão conceptual:

```python
lot_size = pos_size_usd / curr_price / contract_size
lot_size = round(lot_size, 2)  # exemplo: passo mínimo frequentemente 0.01 — confirmar no símbolo
```

**Cuidado:** arredondamentos que desçam abaixo do mínimo = ordem rejeitada; lotes e **margem** têm de ser consistentes com as regras da conta (leverage da firma, não só a tua intenção).

---

## O que muda no pipeline (mapa rápido)

| Componente           | Futures (agora)              | CFDs / MT5 (funded típico)        |
|------------------------|------------------------------|-----------------------------------|
| Arranque / sessão      | `create_exchange` (ccxt)     | `mt5.initialize()` + `login`    |
| Velas                  | `exchange.fetch_ohlcv`       | `mt5.copy_rates_from_pos` (etc.)|
| Esperar fecho de vela  | WS (Binance/Bybit) ou híbrido| Polling / lógica por tempo de barra |
| Entrada a mercado      | `exchange.create_order` … `"market"` | `mt5.order_send` (`TRADE_ACTION_DEAL`) |
| Posições               | `exchange.fetch_positions`   | `mt5.positions_get`               |
| Equity                 | `exchange.fetch_balance`     | `mt5.account_info().equity`      |
| Fees / histórico fills | `fetch_my_trades` / fees     | `mt5.history_deals_get` (e regras do broker) |
| **Modelos / scalers**  | Igual                        | Igual (se dados alinhados)       |
| **Lógica de sinal** (`main.py` em alto nível) | Igual          | Igual se a abstração for limpa   |

---

## Recomendação prática: abstrair a “exchange” cedo

A abordagem mais limpa é introduzir uma **interface comum** enquanto ainda estás em futures; depois implementas um backend MT5 sem reescrever sinais e notebooks.

```python
# Interface comum (exemplo conceptual)
class ExchangeBase:
    async def place_market_order(self, side, size): ...
    async def check_positions(self): ...
    async def check_equity(self): ...
    async def fetch_candles(self, n): ...
    async def wait_for_candle_close(self): ...

class CcxtFuturesExchange(ExchangeBase):
    """Implementação atual (Binance / Bybit, etc.)."""

class MT5Exchange(ExchangeBase):
    """Implementação MT5 para contas funded / broker."""
```

Assim **`main.py`** (regras de entrada, espera por candle, SL/TP se modelados à parte) e **modelos** podem permanecer estáveis; apenas **injectas** a implementação certa.

---

## Cuidados extra (CFDs + prop)

- **Termos de serviço:** algumas firms restringem EAs / bots ou exigem declaração; ler T&C e regras do challenge.
- **Slippage e spread:** o backtest em dados “exchange-like” pode **não** reproduzir o CFD do broker — rever expectativas de edge.
- **Drawdown:** as regras do challenge (daily / total) têm de ser **enforced no bot** ou monitorização externa fiável (como já planeado para futures com circuit breakers).
- **Stops:** em MT5 stops podem ser enviados como parte do deal ou ordens pendentes; alinhar com o que a firma permite e com o risco que queres (igual ao que fizeste com `STOP_MARKET` / `TAKE_PROFIT_MARKET` em futures).

---

## Quando fazer esta migração

Não é obrigatório resolver MT5 **antes** de validar edge e execução em futures / capital próprio.

Ordem razoável:

1. Validar edge e execução ao vivo (roadmap: fases com exchange atual).
2. **Perto do challenge funded:** implementar e testar em **demo MT5** da mesma firma/broker (mesmos símbolos e regras que possível).
3. Reutilizar **modelos e lógica de sinal**; substituir só a camada de dados + ordens.

**Ordem de grandeza:** contar **cerca de 1–2 semanas** de desenvolvimento focado quando chegares a essa fase (primeira integração + edge cases + testes), dependendo de quão isolada já está a lógica de mercado.

---

## Resumo

| Tema | Takeaway |
|------|-----------|
| Instrumento | CFD ≠ perpétuo; funding vs swap; contraparte firma/broker. |
| Código | Nova API (MT5 ou outra); ccxt não cobre o mesmo. |
| Ambiente | MT5: planear **Windows** no VPS se for o caminho. |
| Dados / modelo | OHLCV de outra fonte; **modelo e features** podem manter-se. |
| Risco | Spread, símbolo, lotes, termos da prop, alinhamento backtest↔live. |
| Arquitetura | **Abstrair exchange** reduz custo da migração mais tarde. |
