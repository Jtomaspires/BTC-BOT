# TODO — BTC/ETH NN Bot (roadmap)

## Já feito
- Estrutura de pastas para novos pares: `CNN_PAXG/`, `CNN_SOL/`, `CNN_LINK/`, `CNN_XRP/` (data, artifacts, treinos, backtest).
- `TODO.md` com roadmap (ensemble constante, risco, portfolio, métricas).
- Notebook de **portfolio 4 pares**: `PORTFOLIO/backtest_4pairs/main.ipynb` (equity/drawdown mixed).
- **Download de dados** (Binance 1h, multi-par): `old files/0_download_data.py` → `data/raw/` + cópia em `CNN*/data/` para BTC, ETH, XRP, SOL, LINK, PAXG (após correr o script).

## Objetivo
- Ter um bot **multi-par (4 pares)** com **ensemble constante** por par (Conv1D + LSTM + Hybrid),
  com **risco consistente** entre pares e **métricas de rentabilidade** boas (equity estável / drawdown baixo).

## Pares (agora e próximos testes)
- **Já temos (dados + pipeline)**
  - BTC (`CNN/`) — dados em `data/raw/BTCUSDT-1h-data.csv` e `CNN/data/`
  - ETH (`CNN_ETH/`) — idem `ETHUSDT-1h-data.csv`
- **Novos pares (pastas + dados descarregados)**
  - PAXG — hedge / proteção
  - SOL — high beta
  - LINK — event-driven / técnico
  - XRP — event-driven / diversificação

## Ensemble (constante, sem regimes)
- **Modelos por par**
  - Conv1D
  - LSTM
  - Hybrid
- **Combinação (fixa)**
  - Escolher 1 método e manter constante:
    - (A) Majority vote (long/short/flat)
    - (B) Média ponderada de score/prob (pesos fixos)
    - (C) 2/3 concordam = trade; score médio define convicção
- **Outputs a guardar por par (para portfolio)**
  - Equity curve (time series)
  - Trades / posições (ideal) ou pelo menos equity por candle
  - Métricas: final equity, sharpe, max drawdown

## Risco consistente entre pares (regras fixas)
- Definir por par:
  - SL / TP
  - Threshold de entrada (se aplicável)
- Definir global:
  - Sizing por volatilidade (vol targeting) para risco semelhante por par
  - Cap de exposição total (somatório de risco)
  - Circuit breaker (ex.: limite de perdas/drawdown)

## Estrutura de pastas (espelhar padrão atual)
- **Feito:** `CNN_PAXG/`, `CNN_SOL/`, `CNN_LINK/`, `CNN_XRP/` com subpastas base.
- Cada uma deve ter (completar conforme fores treinando):
  - `data/` ✓ CSV
  - `artifacts/` (scalers, `split_info.json`, manifest — após fit/pipeline)
  - `CONV1D_model_training/CONV1D_model_training/`
  - `LSTM_model_training/LSTM_model_training/`
  - `hybrid_model_training/`
  - `backtest/`

## Ordem de testes (sugerida)
1. **Sanidade dos dados** — Abrir cada `*USDT-1h-data.csv` (raw + `CNN*/data/`); verificar colunas `timestamp, open, high, low, close, volume` e range de datas alinhado entre pares.
2. **Baseline BTC e ETH** — Backtest holdout nos notebooks `CNN/backtest/main.ipynb` e `CNN_ETH/backtest/main.ipynb`; registar equity, Sharpe, max DD (referência).
3. **Pipeline por par novo (repetir por PAXG → SOL → LINK → XRP, ou na ordem que preferires)**  
   - `fit_scalers` / splits / `artifacts` (igual ao fluxo já usado em BTC/ETH).  
   - Treino **Conv1D** → **LSTM** → **Hybrid** (checkpoints `.pt` nas pastas `models/`).  
   - **Backtest individual** em `<CNN_PAIR>/backtest/main.ipynb` (ou copiar lógica do BTC/ETH).
4. **Afinar risco por par** — SL/TP/threshold por ativo; opcional vol targeting para comparar risco entre pares.
5. **Portfolio 4 pares** — `PORTFOLIO/backtest_4pairs/main.ipynb`: escolher 4 pares, pesos, comparar equity/drawdown agregado vs cada par.
6. **Definition of Done** — Comparar métricas alvo (DD, Sharpe, profit factor, robustez em splits).

## A fazer / próximo
- Gerar **artifacts** (scalers + split) para PAXG, SOL, LINK, XRP.
- Treinar e guardar **checkpoints** Conv1D / LSTM / Hybrid por par.
- Backtests individuais e depois **portfolio** com 4 pares escolhidos.
- Fechar escolha de **método de ensemble** (A/B/C) e pesos fixos, se aplicável.

## Backtest portfolio (4 pares “mixed”)
- **Notebook:** `PORTFOLIO/backtest_4pairs/main.ipynb` (configurar `PAIRS` e pesos).
- Deve:
  - Carregar equity/resultados de cada par (BTC, ETH, +2 escolhidos)
  - Normalizar sizing/risk (regra fixa)
  - Construir equity combinada (ex.: returns ponderados → equity agregada)
  - Plotar / reportar: equity final, max drawdown, curva vs por par
- Saída: comparar **por par** vs **portfolio** (equity e drawdown)

## Métricas alvo (Definition of Done)
- Max Drawdown baixo/estável
- Sharpe/Sortino melhor que baseline
- Profit Factor > 1 (idealmente folga)
- Robustez em períodos diferentes (walk-forward / splits consistentes)
