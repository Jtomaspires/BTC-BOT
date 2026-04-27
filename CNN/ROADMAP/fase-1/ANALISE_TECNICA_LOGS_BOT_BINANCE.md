# Análise técnica — logs do bot Binance Futures (ETH/USDT)

Documento de fase 1: registo dos achados técnicos observados na execução prolongada de `py -3.12 -m trading_bot.main` (testnet/demo), cruzados com o código em `binance_futures_trading_bot/`.

---

## Contexto

- **Símbolo:** `ETH/USDT:USDT` (futures USDT-M).
- **Timeframe de sinal:** 1h (WebSocket `kline_1h` + REST para OHLCV e ticker).
- **Proteção:** `STOP_MARKET` / `TAKE_PROFIT_MARKET` com `closePosition: true` e `workingType: CONTRACT_PRICE`.
- **Ficheiros relevantes:** `binance_futures_trading_bot/trading_bot/bot.py`, `main.py`, `load_models.py`.

---

## 1. Stop-loss demasiado apertado (causa dominante da sangria de equity)

### Comportamento observado

- SL a ~3 USD do preço de entrada e TP a ~50 USD (distâncias fixas em “pontos” de preço).
- Com ETH na ordem dos **2300–2450 USD**, 3 USD de SL corresponde a **~0,12–0,13%** de movimento — facilmente absorvido pelo ruído intra-hora.
- Nos logs: após cada `SL/TP placed`, na vela seguinte aparecia frequentemente `Protection order … not open (already triggered/cancelled)` e equity descia de forma consistente (~1,3–2,2 USDT por ciclo, fee + perda de SL).

### Referência no código

Em `binance_futures_trading_bot/trading_bot/bot.py`:

- `self.sl_points = 3.0`
- `self.tp_points = 50.0`

O comentário indica alinhamento com `BEST_SL` / `BEST_TP` do backtest CNN_ETH; convém **validar em unidades** (USD absolutos vs. percentagem vs. outra convenção) e se o grid in-sample não optimizou um SL irrealista para live.

### Risco / impacto

Alta probabilidade de stop-out por ruído; relação nominal risco/recompensa (3 vs 50) não compensa se a taxa de toque ao SL for muito superior à de TP.

---

## 2. Sinal do modelo saturado em `Action=1` (long)

### Comportamento observado

- Em todo o extracto de log analisado, **só** apareceu `Action=1` (BUY); não surgiram `0` (neutro) nem `-1` (short).

### Referência no código

Em `binance_futures_trading_bot/trading_bot/load_models.py`:

- `THRESHOLD = 0.0007`
- `get_action`: média de `y_pred[:, 0]` sobre o ensemble; `> THRESHOLD` → 1, `< -THRESHOLD` → -1, senão 0.

### Hipóteses técnicas

- Bias positivo sistemático nos outputs do ensemble (treino / janela temporal).
- Possível **mismatch** entre features/scalers offline vs. dados em live.
- Checkpoints escolhidos por equity podem estar **correlacionados** na mesma direção (todos “long-friendly”).

### Impacto

A lógica em `main.py` (“só entrar flat; ignorar sinais opostos com posição aberta”) quase nunca vê sinal oposto; após cada fecho por SL o bot volta a estar flat e o próximo sinal continua a ser long.

---

## 3. Re-entrada imediata após saída por SL (paridade com backtest)

### Comportamento observado

Ciclo típico: entrada → SL/TP colocados → na vela seguinte posição fechada (mensagens de protecção já não aberta) → novo candle → `No open position` → novo `BUY market` porque `Action=1` de novo.

### Ligação ao roadmap

Existe o plano **defer SL/TP re-entry** (mini-lookahead no backtest quando SL/TP fecha intra-barra e ainda há entrada no mesmo `curr_open`). Em **live**, o padrão análogo é: sair por SL e, na **próxima** barra, re-entrar no open — acumulando fees e perdas se o regime continuar desfavorável.

### Impacto

Compounding de perdas pequenas + fees; ausência de “cooldown” ou fila `pending_entry` como no backtest corrigido.

---

## 4. Erro Binance `-4509` (TIF GTE / `closePosition` sem posição visível)

### Log típico

```
BUY market order filled. ...
Exchange closed.
Error: binance {"code":-4509,"msg":"Time in Force (TIF) GTE can only be used with open positions. Please ensure that positions are available."}
```

### Interpretação técnica

O market order **preencheu**, mas a chamada imediata a `place_sl_tp_orders` (ordens com `closePosition=True`) falhou porque a posição **ainda não estava disponível** do lado da API (propagação / race).

### Referência no código

Em `binance_futures_trading_bot/trading_bot/main.py`, após `place_market_order`, chama-se `await bot.place_sl_tp_orders(...)` **sem** retry nem plano B se falhar.

### Risco

Janela em que a conta fica **long/short sem SL/TP** até restart ou intervenção manual. No log seguinte, o restart mostrou “no open position” — a posição pode ter sido fechada por outro mecanismo na testnet; em produção o risco é material.

### Mitigação sugerida (para implementação futura)

- Retry com backoff curto após `-4509`.
- Se esgotar retries: **fechar posição a market** com `reduceOnly` e logar alerta crítico.

---

## 5. Erro `-2021` na reconciliação de startup (“Order would immediately trigger”)

### Log típico

```
Startup protection: LONG position detected at entry=2323.41. Placing fresh SL/TP ...
Failed to place SL/TP during reconciliation: binance {"code":-2021,"msg":"Order would immediately trigger."}
Position remains without bot-placed protection.
```

### Interpretação

Para uma posição LONG já aberta, o SL calculado ficaria **já violado** ao preço corrente (ou o trigger seria considerado imediato pela Binance com `CONTRACT_PRICE`). A exchange recusa a ordem.

### Código relacionado

`reconcile_position_protection` em `bot.py` tenta fechar a market se `tp_hit` / `sl_hit` com base no `last` do ticker; se essa lógica não cobrir todos os casos (versão antiga, ticker falhou, arredondamentos), pode ficar posição **sem** SL/TP.

### Risco

Posição herdada sem protecção até intervenção ou fecho manual.

---

## 6. Erro `-1007` ao cancelar ordens de protecção (estado ambíguo)

### Log típico

```
cancel protection order … (non-fatal): binance {"code":-1007,"msg":"Timeout waiting for response from backend server. Send status unknown; execution status unknown."}
```

### Interpretação

O cancel pode ter sido **executado ou não**; a API não confirma.

### Código relacionado

`cancel_sl_tp_orders` em `bot.py`: `-2011` / “Unknown order” tratam-se como sucesso; `-1007` cai no ramo genérico “non-fatal” e o id é limpo no `finally`.

### Risco

Ordem de protecção **ainda aberta** na exchange mas já não rastreada pelo bot → disparo inesperado ou conflito com nova posição.

---

## 7. Drag de fees

### Observação

Fees reportadas ~**0,40 USDT** por ordem de entrada de ~1000 USD nocional; dezenas de ciclos consecutivos somam dezenas de USDT só em comissões, independentemente da direcção do mercado.

### Impacto

Em estratégia de alta rotação com SL apertado, as fees podem dominar o PnL mesmo que o “edge” do modelo seja pequeno.

---

## Comparativo: código antes vs. depois

Esta secção cruza o comportamento observado nos **logs do bot** com a **lógica de backtest** nos notebooks `CNN_*` / `PORTFOLIO`, após o plano **defer SL/TP re-entry** (`.cursor/plans/defer_sl_tp_re-entry_c0ef3062.plan.md`). O objectivo é ver **o que mudou no simulador** e **o que o bot live ainda não espelha**.

### A) Backtest — `trading_backtest` (single-pair, notebooks `CNN_*/backtest/main.ipynb`)

| Aspeto | **Antes** | **Depois** |
|--------|-----------|------------|
| Ordem lógica na barra `i` | 1) Avaliar SL/TP com `high`/`low` da barra. 2) Se `position == 0` e `desired != 0`, entrar logo ao **`curr_open[i]`**. | 1) Sinal `desired`. 2) Se existir **`pending_entry`**, executar entrada ao **`curr_open[i]`** e limpar a fila. 3) SL/TP na posição aberta; se fecho por SL ou TP, marcar **`closed_by_sl_tp_this_bar = True`**. 4) Se flat e `desired != 0`: se acabou de fechar por SL/TP nesta barra → **`pending_entry = desired`**; senão → entrada imediata ao `curr_open` (comportamento antigo para sinais “normais”). |
| Re-entrada na mesma vela após SL/TP | **Sim** — permitia entrar ao open de uma vela em que o intra-bar já tinha sido usado para fechar, o que é um **mini-lookahead** temporal. | **Não** — o sinal fica em fila e só abre no **open da barra seguinte**. |
| Estado extra | Nenhum. | `pending_entry: int \| None` (por exemplo `None`, `1`, `-1`); `closed_by_sl_tp_this_bar` resetado a cada barra. |
| Fim da série | — | Se ainda houver `pending_entry` com conta flat, o plano prevê **descartar** o pending (sem inventar preço futuro). |
| Impacto no PnL reportado | Equity e heatmaps SL/TP incluíam trades “impossíveis” de calendarizar ao open após observar o intra-bar. | Equity tende a **cair** vs. versão antiga (menos viés); métricas alinham-se melhor com **live** e com regra “só no próximo open”. |

**Onde está o “depois” no repo:** função `trading_backtest` nas células de código dos notebooks (ex.: `CNN_ETH/backtest/main.ipynb`); mesma ideia replicada em `CNN`, `CNN_SOL`, `CNN_LINK`, `CNN_XRP`, `CNN_PAXG`. A célula markdown do heatmap referencia `pending_entry` e o mini-lookahead.

### B) Backtest — `trading_backtest_portfolio_compounding` (`PORTFOLIO/backtest_4pairs/main.ipynb`)

| Aspeto | **Antes** | **Depois** |
|--------|-----------|------------|
| Por perna `k` | Após SL/TP e cálculo de `slice_sz`, qualquer perna flat com `desired[k] != 0` entrava **no mesmo bar** ao open `o[k]`, mesmo que `k` tivesse acabado de sair por SL/TP nesse bar. | `pending_entry[k]` análogo ao single-pair: bloco **1b** executa pendentes no open **antes** do SL/TP; no bloco de novas entradas, se **`closed_by_sl_tp[k]`** → fila; senão → entrada imediata. |
| Estado extra | — | `pending_entry = [None] * K`; `closed_by_sl_tp = [False] * K` por barra; `E_pre` / `slice_pre` para pendentes. |

### C) Bot live — `binance_futures_trading_bot` (sem mudança nesta fase)

| Aspeto | **Comportamento actual (código)** | Relação com o backtest “depois” |
|--------|-----------------------------------|-----------------------------------|
| Re-entrada após SL | No `main.py`, em cada fecho de vela, se `position == "NONE"` e `CURR_ACTION == 1` (ou short), **`place_market_order`** corre de novo — **não** há `pending_entry` nem espera explícita por “só no open seguinte após SL intra-bar”. | O backtest **corrido** com defer já **não** assume re-entrada no mesmo preço temporal que o live; comparar PnL do notebook com o log do bot exige ter **presente este desalinhamento**. |
| SL/TP após market | `place_sl_tp_orders` imediato após fill; erros `-4509` / `-2021` tratados como no documento acima. | O simulador não modela estes erros de API; são risco **só live**. |

Em resumo: o **comparativo “antes/depois”** aplica-se sobretudo ao **backtest** (remoção de lookahead na re-entrada). O **bot** continua com a lógica antiga no que toca a fila de re-entrada; alinhar live ao backtest “depois” seria trabalho futuro (ex.: não enviar nova `BUY` na mesma hora em que o SL disparou intra-bar, ou replicar explicitamente `pending_entry` no `main.py`).

---

## 8. Resumo por prioridade (acções técnicas sugeridas)

| Prioridade | Tema | Acção sugerida |
|------------|------|----------------|
| Alta | SL/TP em USD fixos sobre ETH | Rever `sl_points` / `tp_points` (percentagem, ATR, ou alinhamento explícito com métricas do backtest). |
| Alta | Ensemble sempre long | Instrumentar `mean_signal` e outputs por modelo; rever `THRESHOLD`, scalers e paridade de features. |
| Média | Re-entrada pós-SL | Cooldown ou fila `pending_entry` alinhada ao backtest corrigido (ver plano defer re-entry). |
| Média | `-4509` pós-market | Retry + fallback fecho a market se SL/TP não colarem. |
| Média | `-2021` na reconciliação | Garantir ramo que fecha a market quando SL seria imediato; validar com ticker/mark. |
| Baixa | `-1007` no cancel | Re-fetch de ordens abertas antes de limpar ids; retry de cancel. |

---

## Referências de ficheiros no repositório

| Ficheiro | Conteúdo relevante |
|----------|---------------------|
| `binance_futures_trading_bot/trading_bot/bot.py` | `sl_points`, `tp_points`, `place_sl_tp_orders`, `cancel_sl_tp_orders`, `reconcile_position_protection`, WebSocket 1h |
| `binance_futures_trading_bot/trading_bot/main.py` | Loop principal, entrada flat, chamada a `place_sl_tp_orders` após market |
| `binance_futures_trading_bot/trading_bot/load_models.py` | `THRESHOLD`, `get_action` |
| `.cursor/plans/defer_sl_tp_re-entry_c0ef3062.plan.md` | Plano de correção de re-entrada pós SL/TP no backtest CNN_ETH |

---

*Documento gerado para a fase 1 do roadmap — baseado em análise de logs de execução e leitura do código no repositório NN.*
