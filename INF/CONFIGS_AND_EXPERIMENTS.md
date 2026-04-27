Neste momento estou em **Ask mode**, por isso **não posso criar/editar ficheiros** no teu repo. Mas deixo-te já o conteúdo completo de um `.md` para copiares e guardares, por exemplo como `INF/CONFIGS_AND_EXPERIMENTS.md`.

```markdown
# Configs nomeados e Experiments (como correr e comparar)

Este projeto suporta dois estilos de execução para facilitar iteração e comparação:

- **Vários experiments no mesmo ficheiro YAML** (recomendado)
- **Vários ficheiros YAML** (um por suite / contexto)

A comparação vai sendo acumulada em:

- `INF/outputs/runs_summary.csv` (1 linha por experiment/run)
- `INF/outputs/comparison_by_config.csv` (agregado por `config_name` + `experiment_name`)

---

## Conceitos

### `config_name`
Nome da “suite”/contexto (ex.: `eth_1h_baseline`, `btc_1h_features_v2`).
Serve para distinguir runs no `runs_summary.csv` e nos outputs.

### `experiments[].name`
Nome do experiment dentro da suite (ex.: `base`, `thr_0_004`, `epochs_50`).
Cada experiment é um override do config base.

### Onde os outputs vão parar

Quando usas `experiments:`:

- `INF/outputs/run_YYYYMMDD_HHMMSS/experiments/<experiment_name>/...`

Cada experiment gera os seus ficheiros:
- `summary_all_windows.csv`
- `window_XXX/metrics.csv`
- `run_summary.csv`

Além disso, o projeto mantém um registo cumulativo:
- `INF/outputs/runs_summary.csv` (append)
- `INF/outputs/comparison_by_config.csv` (recalculado)

---

## Opção A (recomendada): 1 config com vários `experiments`

### Quando usar
- Queres comparar variações pequenas (thresholds, epochs, SL/TP grids, etc.)
- Queres correr tudo seguido e ficar com um `batch_id` comum

### Como fazer

1) No topo do teu `INF/config.yaml`, define:

```yaml
config_name: eth_1h_feat_suite
```

2) Adiciona `experiments:` no fim (ou logo após `outputs:`), por exemplo:

```yaml
experiments:
  - name: base

  - name: epochs_50
    training:
      epochs: 50

  - name: thr_0_004
    backtest:
      signal_threshold: 0.004
```

3) Corre:

```bash
python INF/run_walkforward.py --config INF/config.yaml
```

### O que acontece
- O projeto corre `base`, depois `epochs_50`, depois `thr_0_004`
- Guarda outputs separados por experiment
- Adiciona 1 linha por experiment em `INF/outputs/runs_summary.csv` com:
  - `config_name`
  - `experiment_name`
  - `batch_id` (igual ao `run_id` do batch)

---

## Opção B: vários ficheiros `config_*.yaml`

### Quando usar
- Queres separar suites grandes (ex.: `ETH 1h` vs `BTC 1h`)
- Queres manter histórico/controlo por ficheiro
- Queres ter configs “congelados” para reproduzir resultados

### Como fazer

1) Cria um novo ficheiro, por exemplo:
- `INF/config_eth_suite.yaml`
- `INF/config_btc_suite.yaml`

2) Em cada ficheiro define um `config_name` diferente:

```yaml
config_name: btc_1h_baseline
```

3) Corre cada um:

```bash
python INF/run_walkforward.py --config INF/config_eth_suite.yaml
python INF/run_walkforward.py --config INF/config_btc_suite.yaml
```

### O que acontece
- Cada execução cria um `run_YYYYMMDD_HHMMSS` novo
- O `runs_summary.csv` vai acumulando (bom para histórico)
- `comparison_by_config.csv` passa a comparar suites diferentes também

---

## Regra simples

- **Variações da mesma suite** → usa `experiments:` dentro do mesmo YAML
- **Suites diferentes (par/timeframe/dataset/objetivo)** → cria outro `config_*.yaml` e muda `config_name`

---

## Checklist rápida (para não contaminar comparação)

- Mantém o mesmo `seed` quando estás a comparar experiments
- Mantém os mesmos `walkforward.*` (train/val/test/step) quando estás a comparar
- Mantém o mesmo dataset (`data.csv_path`, timeframe)
- Só muda uma coisa por experiment (quando possível)

---

## Onde comparar resultados

- Histórico completo:
  - `INF/outputs/runs_summary.csv`

- Comparação agregada por suite/experiment:
  - `INF/outputs/comparison_by_config.csv`
```

Se quiseres, diz-me que 3–6 experiments queres criar já (ex.: 3 thresholds + 2 epochs) e eu devolvo-te um bloco `experiments:` pronto a colar no teu `INF/config.yaml`.