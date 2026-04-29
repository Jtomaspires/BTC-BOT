---
name: Streamlit dashboard (consolidated)
overview: Multi-page Streamlit app for interactive exploration of walk-forward results, with fast SL/TP grids via persisted per-window signals.csv, forensic window analysis, correlations, timeline, and experiment comparison—aligned with INF/outputs layout.
todos:
  - id: prereq-signals-csv
    content: Extend pipeline to persist per-window signals CSV (raw_signal, desired, OHLCV) for dashboard heatmaps without model reload
    status: completed
  - id: dashboard-scaffold
    content: Add dashboard/ app + utils (loader, indicators, windows, backtest_lite) + requirements
    status: completed
  - id: pages-1-6
    content: Implement Streamlit pages 1–6 per consolidated spec with Plotly and st.cache_data
    status: completed
  - id: session-state-navigation
    content: Implement shared st.session_state keys (selected_run_id/experiment/window) for click-through defaults across pages
    status: completed
isProject: false
---

# Streamlit dashboard — especificação consolidada

## Filosofia

Separar **descoberta de features / diagnóstico** (dashboard interativo, rápido) da **produção de treino** (pipeline pesado). O dashboard opera só sobre artefactos já gerados.

## Pré-requisito crítico (habilita a página 3)

Sem isto, o heatmap SL/TP fica lento ou obriga a inferência do modelo.

**Persistir sinais por janela antes de aplicar SL/TP**, num CSV consumível pelo dashboard:

- Caminho sugerido (alinhado ao layout actual):

`INF/outputs/run_<run_id>/experiments/<experiment_slug>/window_<NNN>/signals.csv`

- Colunas mínimas (compatível com semântica em [`INF/backtest_engine.py`](INF/backtest_engine.py)):

  - `timestamp` (se existir no OHLCV; senão índice inteiro `bar_i`)
  - `signal` — valor contínuo raw (`raw_signal_per_bar[i]`), comprimento `len(opens)-1`
  - `desired` — `-1/0/1` derivado do threshold (opcional mas útil para debug)
  - `open`, `high`, `low`, `close`, `volume` — alinhados ao mesmo slicing da janela/test split usado no backtest final

Integração técnica (referência): o motor já calcula `desired` dentro de [`run_single_backtest`](INF/backtest_engine.py) via threshold; o mais robusto é gravar **antes** da simulação SL/TP os arrays já alinhados ao mesmo `n = len(opens)-1`.

## Fontes de dados (repo real)

- Registry global: [`INF/outputs/runs_summary.csv`](INF/outputs/runs_summary.csv)

  - Métricas agregadas + colunas por janela no formato normalizado `NNN_roi` / `NNN_dd` (legacy `window_NNN_*` deve ser ignorado ou migrado no loader).

- Por corrida:

  - `INF/outputs/run_<id>/config.resolved.yaml` — walkforward + csv_path + backtest cfg
  - `INF/outputs/run_<id>/experiments/<slug>/window_<NNN>/metrics.csv` — métricas por split (útil para trades/win_rate quando disponível)
  - `INF/outputs/run_<id>/experiments/<slug>/summary_all_windows.csv` — agregação por janela (fallback forte)
  - `INF/outputs/run_<id>/experiments/<slug>/window_<NNN>/signals.csv` — **novo artefacto** (pré-requisito página 3)

- Mercado:

  - `data.csv_path` do YAML resolvido relativamente à raiz do repo (mesma convenção que [`INF/data_loader.py`](INF/data_loader.py)).

- `split_info.json`:

  - Não assumir como obrigatório. Preferir derivar `(start,end)` por índices de linha a partir do walk-forward + `recent_rows`, ou ler timestamps se existirem no CSV.

## Estrutura de ficheiros (dashboard)

```
dashboard/
  app.py                 # tema escuro + navegação + sidebar partilhada
  pages/
    1_overview.py
    2_window_analysis.py
    3_sl_tp_heatmap.py
    4_correlations.py
    5_timeline.py
    6_compare.py
  utils/
    loader.py            # runs_summary, runs discovery, YAML, signals.csv
    indicators.py        # RSI/ATR/MACD/Vol..., só pandas/numpy
    windows.py           # map window_id → iloc slices (train/val/test)
    backtest_lite.py     # grid SL/TP só com signals.csv + regras do engine
```

Dependências (mínimo): `streamlit`, `plotly`, `pandas`, `numpy`, `pyyaml` (`seaborn` opcional).

Usar `st.cache_data` para IO pesado e computação repetida.

## Persistência de navegação (`st.session_state`)

Para permitir click-through sem voltar a seleccionar manualmente (ex.: página 5 → página 2), define chaves globais **partilhadas entre todas as páginas**:

- `selected_run_id: str`
- `selected_experiment: str`
- `selected_window: int`

### Contrato de escrita/leitura

- **Escrevem** estas chaves (quando o utilizador faz uma escolha “principal”):
  - página **1** (overview): ao clicar numa linha/tabela ou botão “Open window…”
  - página **5** (timeline): ao clicar num segmento de janela
  - página **6** (compare): ao seleccionar uma janela “interessante” (opcional), ou ao usar botões “Inspect A/B window…”

- **Lêem** estas chaves como **defaults** dos selectors (se válidos relativamente aos filtros actuais):
  - página **2** (forensics)
  - página **3** (heatmap — desde que exista `signals.csv` para esse `(run_id, experiment, window)`)
  - página **4** (correlations)

### Comportamento esperado na página 2

Ao montar os widgets:

1. Calcula candidatos válidos `(run_id, experiment)` depois dos filtros globais.
2. Se `selected_run_id`/`selected_experiment` existirem **e** estiverem nos candidatos → pré-seleccionar.
3. Para `selected_window`: se existir e pertencer ao conjunto de janelas disponíveis para esse experimento/run → pré-seleccionar e aplicar ao gráfico.

### Timestamps opcionais

Se não existirem chaves válidas no session state (primeira visita), usa defaults sensatos (último `run_id`, primeiro experiment da lista filtrada).

## Sidebar partilhada (todas as páginas)

- Selectors: `pair`, `timeframe`
- Filtro de datas (quando `timestamp` existir no OHLCV)
- Filtros de resultados: `config_name`, `experiment_name`, `run_id` (multi-select onde fizer sentido)

Nota: os filtros globais podem invalidar `selected_*`; quando isso acontecer, limpa apenas as chaves que deixarem de ser válidas e mantém as que continuarem compatíveis.

## Página 1 — Overview de experiments

- Tabela ordenável:

  - `experiment_name`, `config_name`, `avg_return`, `avg_drawdown`, `avg_sharpe`
  - `pct_windows_positive` (% de `NNN_roi > 0` entre janelas presentes)
  - `worst_window`, `best_window` (por ROI)

- Plotly:

  - barras: ROI médio por experimento (verde/vermelho)
  - box plot: distribuição de ROI por janela por experimento

- Destaque da linha “melhor equilíbrio”: melhor `avg_return` **e** melhor `pct_windows_positive` (regra simples documentada na UI).

## Página 2 — Análise forense de janelas

- Select: experimento (+ opcionalmente `run_id` se vários)

- Dropdown de janelas **ordenado por ROI ascendente** (piores primeiro)

- Chart Plotly candlestick no período da janela (preferência)

- Overlay multiselect de indicadores (eixo Y2): RSI/ATR/MACD/volume/volatilidade

- Metric cards: ROI, DD; acrescentar `n_trades`, `win_rate`, `sharpe` quando existirem em `metrics.csv`/`summary_all_windows.csv`

- Tabela das piores janelas (todas as linhas do experimento seleccionado, ordenadas por ROI)

## Página 3 — Heatmap SL/TP (rápido)

**Input obrigatório:** `signals.csv` da janela.

- Controls:

  - select `run_id`, `experiment`, `window`
  - sliders: SL min/max/step, TP min/max/step
  - thresholds e trailing stops como listas (inputs texto comma-separated), alinhado ao grid existente quando aplicável

- Botão **Run grid** → [`dashboard/utils/backtest_lite.py`](dashboard/utils/backtest_lite.py):

  - reutiliza a lógica de saídas de [`INF/backtest_engine.py`](INF/backtest_engine.py) (intrabar high/low, fees, notional), sem modelo

  - outputs por célula: `equity`, `sharpe`, `max_dd`, `win_rate` (quando calculável)

  - **robustez**: score por vizinhança (média das 8 células vizinhas na grelha de equity ou sharpe — definir uma única métrica na UI)

- UI:

  - tabs Plotly: heatmap equity vs heatmap sharpe
  - destacar célula óptima por sharpe (primário) e mostrar também melhor equity
  - export YAML snippet (`BEST_SL`, `BEST_TP`, …) para clipboard

## Página 4 — Correlação de indicadores

- Para cada janela (test split): médias/indicadores agregados no período

- Heatmap: indicadores × experimentos → correlação de Pearson com ROI por janela

- Scatter com regressão + R² para um indicador seleccionado; pontos por janela; cor por experimento

- Ranking de indicadores por `abs(correlation)` média entre experimentos

  - flag automática “candidato forte” se `abs(mean_corr) > 0.4`

- Extra (opcional mas especificado):

  - separação **long vs short**: quando métricas existirem por lado em artefactos, calcular correlações separadas; caso não existam colunas, mostrar aviso e esconder o modo.

## Página 5 — Timeline walk-forward

- Faixa horizontal por janela com cor/intensidade por ROI

- Clique segmento → actualiza **todas** as chaves (`selected_run_id`, `selected_experiment`, `selected_window`) e navega para a página 2 (`st.switch_page`), garantindo que a página 2 abre já focada na mesma janela.

- Série de `close` global no mesmo intervalo filtrado + linhas verticais nos boundaries das janelas

- Overlay “regime”:

  - primeira versão: regime simples derivado de indicadores (ex.: MA distance / vol ratio), já consistente com análises existentes.
  - Se mais tarde existir um módulo “PZ regime”, encapsular como função única para não misturar heurísticas pela UI.

## Página 6 — Comparação de configs

- Escolher experimento A vs B (idealmente mesmo `run_id` ou mesmo `config_name` para comparabilidade)

- Gráficos:

  - barras lado-a-lado ROI por janela

  - diferença `ROI_A - ROI_B` por janela com cores verde/vermelho

- Resumo: contagens “A ganha X janelas / B ganha Y”

- Tabela apenas diferenças “grandes” (threshold configurável, default ex.: `abs(delta_roi) > 0.05`)

## Critérios de aceitação

- `streamlit run dashboard/app.py` funciona localmente

- Página 1 renderiza sem comandos manuais

- Página 3 só aparece como “available” quando `signals.csv` existe (mensagem clara caso contrário)

- Páginas 2/4 funcionam mesmo sem `signals.csv` (fallback OHLCV + índices)

## Notas de implementação

- Preferir Plotly para zoom/hover em todos os gráficos interactivos.

- Dark theme via `st.set_page_config` + CSS mínimo.

- Resolver paths relativos à raiz do repo (mesmo raciocínio que `default_project_root()` em [`INF/data_loader.py`](INF/data_loader.py)).
