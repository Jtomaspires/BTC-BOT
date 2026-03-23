# Alteracoes no `main.ipynb`

Este documento regista as alteracoes feitas no ficheiro `main.ipynb` da pasta `backtest_4k_StopLoss copy` durante esta sessao.

## 1) Captura de sinais raw do ensemble

- Foi mantida a lista de acoes binarias por modelo:
  - `all_actions = []`
- Foi adicionada a lista de sinais continuos por modelo:
  - `all_raw_signals = []`
- No loop de inferencia de cada modelo, alem de `all_actions.append(actions)`, foi adicionada:
  - `all_raw_signals.append(Y_pred_test_cpu[:, 0])`

Objetivo:
- Permitir calculo de conviccao media (`mean_signal`) antes da decisao final de trade.

## 2) Substituicao do bloco de backtest

O bloco anterior foi substituido por uma nova versao com SL/TP fixo + threshold de conviccao.

Nova cabecalho do bloco:
- `# ===================== Backtest SL/TP fixo + Threshold =====================`

Parametros adicionados:
- `TAKER_FEE = 0.00055`
- `pos_size = 1000.0`
- `SL_POINTS = 150.0`
- `TP_POINTS = 300.0`
- `THRESHOLD = 0.02`

Estado da carteira:
- `cash`, `position`, `entry_price`

Logs/metricas:
- `equities`, `fee_log`, `actions_log`
- `total_fees`, `entries`, `completed_trades`
- `sl_hits`, `tp_hits`, `num_longs`, `num_shorts`
- `trade_pnls`

## 3) Funcoes novas/ajustadas

- `taker_fee()`:
  - Retorna fee fixa por lado com base em `pos_size * TAKER_FEE`.

- `realize_to(price)`:
  - Fecha a posicao atual (long/short) ao preco passado.
  - Calcula PnL assinado conforme direcao.
  - Aplica fee taker na saida.
  - Atualiza `cash`, `total_fees`, `trade_pnls`, `completed_trades`.
  - Reseta `entry_price` e `position`.

- `mark_to_market(close_price)`:
  - Calcula equity mark-to-market durante posicao aberta.
  - Quando flat, retorna `cash`.

## 4) Nova logica de sinal e execucao

Para cada candle:

1. Calcula sinal medio do ensemble por valores continuos:
   - `raw_values = [arr[i] for arr in all_raw_signals]`
   - `mean_signal = np.mean(raw_values)`

2. Aplica threshold de conviccao:
   - `desired = 1` se `mean_signal > THRESHOLD`
   - `desired = -1` se `mean_signal < -THRESHOLD`
   - `desired = 0` caso contrario (zona morta)

3. Se houver posicao aberta, verifica SL e TP:
   - Long: `sl = entry - SL_POINTS`, `tp = entry + TP_POINTS`
   - Short: `sl = entry + SL_POINTS`, `tp = entry - TP_POINTS`
   - TP tem prioridade intrabar sobre SL (assuncao explicitada em comentario).

4. So abre nova posicao quando:
   - `position == 0` e `desired != 0`
   - Entrada cobra fee taker.

5. Atualiza mark-to-market no fecho do candle:
   - Guarda em `equities`
   - Guarda fee por barra em `fee_log`
   - Guarda estado da posicao em `actions_log`

6. No fim do loop:
   - Se ainda houver posicao aberta, fecha na ultima `close`.

## 5) Metricas finais impressas

Adicionadas:
- `Entries` e `Completed`
- `Longs` e `Shorts`
- `SL hits` e `TP hits`
- `Win rate` e `Avg PnL/trade`
- `Total fees`
- `Final equity`

## 6) Alteracoes removidas/substituidas do backtest anterior

- Saiu a logica antiga baseada apenas em:
  - voto binario por candle com entrada/saida implita no proprio candle
  - sem threshold de conviccao
  - sem TP fixo
  - sem modelacao de estado persistente equivalente a este fluxo

- O indicador de `accuracy` antigo deixou de ser o foco principal neste bloco, sendo substituido por metricas de trading mais diretas (`win_rate`, `avg_pnl`, hits de SL/TP, fees totais, equity final).

## 7) Validacao apos alteracoes

- Foi feita verificacao de lints no notebook apos as edicoes.
- Resultado: sem erros de lint reportados.

