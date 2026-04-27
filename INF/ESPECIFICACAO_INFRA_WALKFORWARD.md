# Especificação da infraestrutura modular de walk-forward testing

Este documento descreve o conteúdo esperado de cada módulo da pipeline planeada (config → dados → features → modelo → treino → backtest → métricas → relatório), **alinhado ao que já existe no sandbox `CNN/`** como referência de comportamento e de dados.

Os ficheiros Python da infraestrutura (`config.yaml`, `data_loader.py`, `features.py`, `models.py`, `backtest_engine.py`, `metrics.py`, testes em `tests/`) residem em **`INF/`** na raiz do repositório.

---

## 1. O que já tens no `CNN/` (base de referência)

| Conceito | Onde está hoje | Notas para modularizar |
|----------|----------------|--------------------------|
| CSV OHLCV | `CNN/data/*.csv` (ex.: `BTCUSDT-1h-data.csv`) | Colunas esperadas: `timestamp`, `open`, `high`, `low`, `close`, `volume`. |
| Divisão temporal fixa | `CNN/artifacts/split_info.json` + `ARTIFACTS_E_MANIFEST.md` | Hoje: `[0, train_end)`, `[train_end, val_end)`, `[val_end, total_rows)`. Walk-forward **substitui** isto por janelas deslizantes geradas a partir do `config.yaml`. |
| Scalers só no treino | `build_features` + `MaxAbsScaler` por feature em `CNN/backtest/main.ipynb` | Fit apenas no slice de treino da janela; validação e teste só `transform`. |
| Sequências `(X, Y)` | `preprocess_data` no mesmo notebook | `X[i] = features[i:i+seq_len]`, `Y[i] = features[i+seq_len]`; OHLC alinhados após `seq_len`. |
| Arquiteturas | `Model_1`, `Model_2`, `Model_3` no backtest (e notebooks de treino) | Conv1D, LSTM, hybrid; saída `num_features`; último timestep. |
| Ensemble + inferência | `ensemble_raw_signals` + `manifest.json` | Walk-forward pode começar com **uma** arquitetura por janela e estender depois ao padrão manifest. |
| Backtest realista | `trading_backtest` com `pending_entry`, SL/TP em pontos, `TAKER_FEE`, threshold no sinal médio | Comportamento a preservar ao extrair para `backtest_engine.py`; grelha SL/TP vem do config em vez de constantes. |
| Checkpoints por equity | Nomes `eq_*_ep_*.pt` nos dirs de treino | O `trainer` deve manter a mesma filosofia: guardar melhor por métrica de validação ou proxy de equity conforme definires. |

Objetivo da modularização: **replicar a semântica** destes blocos sem acoplar a notebooks, com uma única fonte de verdade (`config.yaml`) e pastas `outputs/` por corrida.

---

## 2. Conteúdo esperado por ficheiro

### 2.1 `config.yaml`

**Responsabilidade:** única fonte de verdade para uma corrida de walk-forward.

**Deve conter (campos sugeridos):**

- **Dados:** `data.csv_path` (absoluto ou relativo ao projeto), opcionalmente `pair`, `timeframe` (metadados / logging apenas).
- **Pré-processamento:** `seq_len` (ex.: 48, como no notebook).
- **Walk-forward:** `train_size`, `val_size`, `step_size` em **número de barras** (ou índices inequívocos); opcional `min_train_rows`, `anchor` (início em 0 vs data fixa).
- **Treino:** `epochs`, `batch_size`, `learning_rate`, `seed`, `device` (`auto` \| `cuda` \| `cpu`), `checkpoint_metric` (ex.: val loss ou equity proxy), `early_stopping` opcional.
- **Modelo:** `architecture` (`conv1d` \| `lstm` \| `hybrid` \| lista para ensemble futuro), caminhos opcionais para **warm-start** (não obrigatório na v1).
- **Backtest:** `taker_fee` (ex.: 0.00055), `position_notional` (ex.: 1000.0), `signal_threshold` (ex.: 0.0007), **grelhas** `sl_points: [50, 100, ...]`, `tp_points: [300, 500, ...]` (listas; o motor itera o produto ou pares explícitos se preferires uma secção `sl_tp_grid: [{sl, tp}, ...]`).
- **Outputs:** `output_dir` (ex.: `outputs/run_2026-04-19_123456/`) ou só `outputs/` + timestamp automático no `run_walkforward.py`.

**Não deve conter:** lógica Python; paths hardcoded no código — só aqui ou via override CLI opcional.

---

### 2.2 `data_loader.py`

**Responsabilidade:** ler o CSV e expor **iterador** de janelas walk-forward, agnóstico ao par.

**Deve conter:**

- Função tipo `load_ohlcv(path) -> pd.DataFrame` com validação mínima de colunas (`open`, `high`, `low`, `close`, `volume`; `timestamp` opcional mas recomendado).
- Classe ou generator `iter_walkforward_windows(df, train_size, val_size, step_size) -> Iterator[WalkWindow]` onde cada item expõe pelo menos:
  - índices inteiros: `train_start`, `train_end`, `val_start`, `val_end` (sem sobreposição train/val dentro da mesma janela, salvo se explicitamente quiseres overlap — documenta a escolha);
  - `window_id` para pastas e logs;
  - opcional: `test_start`, `test_end` se a janela incluir fatia “out-of-sample” após val (comum em WF: treino → val → **teste** a avançar).
- Documentação clara: `step_size` desloca o início da próxima janela; o que acontece no fim do CSV se sobrar menos que `train_size + val_size`.

**Não deve conter:** `build_features`, PyTorch, backtest, SL/TP.

**Ligação ao CNN:** substitui o conceito estático de `split_info.json` por geração dinâmica; o CSV é o mesmo formato já usado em `CNN/backtest/main.ipynb`.

---

### 2.3 `features.py`

**Responsabilidade:** `build_features` + criação / aplicação de scalers — **único sítio** para novas features (EMA, RSI, volatilidade).

**Deve conter:**

- `build_features(opens, highs, lows, closes, volumes, train_scalers=None) -> (scaled_features: np.ndarray, num_features: int, train_scalers: list)` com a mesma convenção do notebook: se `train_scalers is None`, faz `fit_transform` em todas as linhas passadas; senão, só `transform`.
- Lista de features alinhada ao que já tens (8 features com `MaxAbsScaler` por canal), para **paridade** com `CNN/fit_scalers_bootstrap.py` e o notebook.
- Função auxiliar opcional `make_scalers(num_features) -> list[MaxAbsScaler]` para claridade.

**Não deve conter:** construção de tensores `(X, Y)` — isso pode ficar num pequeno `preprocess` no `trainer` ou módulo `datasets.py` se quiseres separar; na v1 pode ser função `build_sequences(...)` aqui ou ao lado no trainer desde que features fiquem isoladas.

**Ligação ao CNN:** copiar/adaptar literalmente o bloco de `build_features` de `CNN/backtest/main.ipynb`; o bootstrap `fit_scalers_bootstrap.py` deve poder ser **substituído** por “primeira fatia de treino da janela” + mesma lógica de fit.

---

### 2.4 `models.py`

**Responsabilidade:** definições `Model_1`, `Model_2`, `Model_3` e fábrica / mapeamento de arquitetura.

**Deve conter:**

- Classes `nn.Module` equivalentes às do notebook, parametrizadas por `num_features` (hoje o notebook usa `num_features` global; em módulo, passar no `__init__` ou via factory).
- `get_model(architecture: str, num_features: int) -> nn.Module` ou `ARCH_MAP` como dict.
- `get_action(raw_signals: Sequence[float], threshold: float) -> int` (-1, 0, 1) espelhando a regra `mean_signal > THRESHOLD` / `< -THRESHOLD` do `trading_backtest`, para o backtest não duplicar regra de trading.

**Não deve conter:** loop de treino, leitura de CSV, SL/TP.

**Ligação ao CNN:** extrair as três classes do notebook; garantir que shapes `(batch, seq_len, num_features)` permanecem iguais.

---

### 2.5 `trainer.py`

**Responsabilidade:** treinar **uma** janela: recebe arrays/tensores de treino e validação, corre epochs, guarda checkpoints, devolve o melhor modelo (path ou `state_dict`).

**Deve conter:**

- Assinatura conceptual: `train_window(cfg, X_train, Y_train, X_val, Y_val, out_dir, window_id) -> TrainResult` com `best_checkpoint_path`, histórico mínimo (loss por epoch), `num_features`.
- Dataloader PyTorch, loss (ex.: MSE sobre o primeiro canal ou todos — **igual ao notebook de treino** que estiveres a replicar), optimizer (ex.: Adam), `device`.
- Política de checkpoint: salvar `eq_*_ep_*.pt` ou `best_val_loss.pt` em `out_dir / f"window_{window_id}" / checkpoints /`.
- `set_seed` reproduzível (como `set_all_seeds` no notebook).

**Não deve conter:** walk-forward loop completo (isso é `run_walkforward.py`); não deve refitar scalers em dados fora do treino da janela.

**Ligação ao CNN:** alinhar hiperparâmetros e critério de “melhor” com um dos `main.ipynb` de `CONV1D_model_training`, `LSTM_model_training`, ou `hybrid_model_training` (escolhe um como referência v1 e documenta no cabeçalho do módulo).

---

### 2.6 `backtest_engine.py`

**Responsabilidade:** simulação bar-a-bar com `pending_entry`, fees, SL/TP; para cada par (SL, TP) da grelha devolve série de equity e contadores.

**Deve conter:**

- Função principal tipo `run_backtest_grid(raw_signal_per_bar: np.ndarray, opens, highs, lows, closes, grid: SlTpGrid, fee: float, notional: float, threshold: float) -> dict[tuple[float,float], BacktestResult]` ou lista de resultados ordenada.
- Lógica equivalente a `trading_backtest` no `CNN/backtest/main.ipynb`: ordem de avaliação SL/TP no mesmo candle, `pending_entry` quando fecha por SL/TP no mesmo bar que nasce sinal, entrada na open, etc.
- Convenção intra-barra documentada: em ambiguidades de range (barra toca SL e TP), avaliar primeiro o ramo de SL (`if`) e só depois TP (`elif`) para manter paridade com notebook.
- Parâmetros configuráveis (nada de magic numbers só no meio do código sem defaults vindos do config).

**Não deve conter:** Sharpe ou CSV final — isso é `metrics.py` / `reporter.py`.

**Ligação ao CNN:** extrair e testar contra o comportamento atual do notebook para uma grelha 1×1 com os mesmos números.

---

### 2.7 `metrics.py`

**Responsabilidade:** funções puras sobre séries já simuladas.

**Deve conter:**

- `sharpe_ratio(equity_or_returns, periods_per_year=...)` (definir se input é equity curve ou retornos log/simples).
- `max_drawdown(equity: np.ndarray) -> float`.
- `win_rate(trade_pnls: list[float]) -> float`.
- `total_pnl`, `num_trades`, opcionalmente `profit_factor`, `avg_trade`.
- Função agregadora `summarize_window(results_by_sl_tp) -> pd.DataFrame` opcional para alimentar o reporter.
- Função de seleção explícita do melhor par da grelha (ex.: `select_best_sl_tp`), dirigida por `backtest.grid_select_metric` no `config.yaml`.

**Não deve conter:** PyTorch, leitura de ficheiros.

**Ligação ao CNN:** hoje parte das métricas está implícita nos `print` do `trading_backtest`; formalizar os mesmos conceitos (entries, SL/TP hits, win rate, fees).

---

### 2.8 `reporter.py`

**Responsabilidade:** I/O e visualização: gráficos, CSV, resumo no terminal.

**Deve conter:**

- Escrita de `results.csv` (por janela e por combinação SL/TP): colunas `window_id`, `sl`, `tp`, `sharpe`, `max_dd`, `win_rate`, `final_equity`, `n_trades`, …
- `plot_equity_curves(...)` por janela ou painel multi-janela.
- `print_summary(df)` para o terminal ao fim da corrida.
- Caminhos derivados de `output_dir` do config.

**Não deve conter:** treino nem lógica de grelha de backtest.

---

### 2.9 `run_walkforward.py`

**Responsabilidade:** orquestração: CLI ou `main`, carrega config, itera janelas, chama pipeline.

**Fluxo esperado:**

1. Carregar `config.yaml`.
2. `df = data_loader.load_ohlcv(...)`.
3. Para cada `window` de `iter_walkforward_windows`:
   - Extrair `df_train`, `df_val` (e opcionalmente `df_test`).
   - `build_features` **só** no treino → obter scalers; aplicar nas três fatias sem refit.
   - Construir `X_train, Y_train, X_val, Y_val` (e teste se existir).
   - `trainer.train_window(...)` → `best_ckpt`.
   - Carregar modelo; gerar `raw_signal` na fatia de **avaliação out-of-sample** (val ou teste — documentar qual é o “relatório” principal da janela).
   - `backtest_engine.run_backtest_grid(...)` para todos SL/TP.
   - `metrics` em cada resultado; acumular linhas para o reporter.
4. `reporter.save_all(...)`.

**Não deve conter:** definições de layers conv/LSTM (delegar a `models.py`).

---

### 2.10 Pasta `outputs/`

**Gerada automaticamente** pelo run (não versionar no git, exceto `.gitkeep` opcional).

**Estrutura sugerida:**

```text
outputs/<run_id>/
  config.resolved.yaml          # cópia do config efectivo (opcional)
  window_000/
    checkpoints/
    metrics.csv                 # por SL/TP nesta janela
    equity_sl{sl}_tp{tp}.png
  window_001/
    ...
  summary_all_windows.csv
  summary_terminal.txt
```

---

## 3. Ordem sugerida de criação dos ficheiros

A ordem maximiza testes incrementais e dependências mínimas entre módulos.

| Ordem | Ficheiro | Motivo |
|------:|----------|--------|
| 1 | `config.yaml` + schema mental dos campos | Fixa contratos para todos os outros módulos. |
| 2 | `data_loader.py` | Sem PyTorch; podes testar só com prints/plots das janelas sobre o CSV real. |
| 3 | `features.py` | Copia fiável do CNN; testes unitários: paridade de shape e scaler fit/transform num slice fixo vs notebook. |
| 4 | `models.py` | Forward pass com tensor aleatório; `num_features` coerente com `features.py`. |
| 5 | `metrics.py` | Funções puras; testes com equity sintética. |
| 6 | `backtest_engine.py` | Depende de sinais + OHLC; **não** precisa de modelo treinado se alimentares sinal fixo ou aleatório; valida contra uma cópia do `trading_backtest` 1×1. |
| 7 | `trainer.py` | Integra `models` + dados sintéticos ou primeira janela pequena; valida loss a descer e checkpoint escrito. |
| 8 | `reporter.py` | Dados fictícios em DataFrame; gráficos e CSV. |
| 9 | `run_walkforward.py` | Cola tudo; começa com `train_size`/`val_size` grandes e `step_size` grande para **uma** janela, depois expande. |
| 10 | `outputs/` + `.gitignore` | Quando o run estiver estável, ignorar `outputs/*` no git. |

---

## 4. Extensões futuras (fora da v1 mínima)

- **Ensemble** como no `manifest.json`: secção no config com lista de arquiteturas + N checkpoints por arch; função de inferência partilhada entre `trainer` (val) e `backtest_engine`.
- **Warm-start** por janela: carregar pesos da janela anterior.
- **Teste walk-forward com “embargo”** entre treino e val para reduzir leakage temporal.
- **CLI** (`tyro` / `argparse`) para `--config` e overrides pontuais.

---

## 5. Resumo

Cada ficheiro tem **uma** razão para mudar: features → `features.py`; arquitetura → `models.py`; janelas temporais → `data_loader.py`; hiperparâmetros e grelha SL/TP → `config.yaml`. A base `CNN/` já implementa quase tudo em forma de notebook; esta infraestrutura é sobretudo **extração, contratos explícitos e orquestração** com walk-forward em vez de `split_info.json` fixo.
