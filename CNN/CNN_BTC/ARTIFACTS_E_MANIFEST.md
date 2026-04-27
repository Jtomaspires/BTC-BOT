# Artefactos do sandbox `CNN/` — `split_info`, `scalers` e `manifest`

Este guia explica os ficheiros em `CNN/artifacts/` e como apontar manualmente os checkpoints para o backtest quando não corres a célula final dos notebooks de treino.

---

## 1. `artifacts/split_info.json`

Define os índices de linha do mesmo CSV (`CNN/data/BTCUSDT-1h-data.csv`) para:

- treino: `[0, train_end)`
- fatia de seleção (test): `[train_end, val_end)`
- holdout (só no backtest): `[val_end, total_rows)`

Exemplo:

```json
{
  "train_end": 54000,
  "val_end": 63000,
  "total_rows": 69084
}
```

- `total_rows` deve coincidir com `len(df)` do CSV que usas nos treinos e no backtest.
- Se alterares o CSV (mais/menos linhas), atualiza `total_rows` (e eventualmente `train_end` / `val_end` se mudares o desenho).

---

## 2. `artifacts/scalers.pkl`

Lista de `MaxAbsScaler` fitados **apenas** nas linhas `[0, train_end)` com as mesmas 8 features dos notebooks.

- Gerado automaticamente pela **célula de artefactos** no fim de cada `main.ipynb` de treino (sobrescreve o ficheiro).
- Alternativa sem treino completo: `CNN/fit_scalers_bootstrap.py` gera um `scalers.pkl` coerente com `split_info.json` e o CSV em `CNN/data/`.

O backtest **nunca** refita scalers: só carrega este ficheiro e faz `transform` nas fatias pós-treino.

---

## 3. `artifacts/manifest.json` — ensemble no backtest

O notebook `CNN/backtest/main.ipynb` lê **só** este ficheiro para saber que ficheiros `.pt` carregar. **Não** é preciso editar paths no código do backtest.

### Formato

- Chaves fixas (nomes exatos): `conv1d`, `lstm`, `hybrid`.
- Cada chave é uma **lista** de strings: cada string é um caminho **relativo à pasta `CNN/`**, usando **`/`** (forward slash).

| Chave     | Classe no backtest | Onde costumam estar os `.pt` |
|-----------|--------------------|--------------------------------|
| `conv1d`  | `Model_1`          | `CONV1D_model_training/CONV1D_model_training/models/` |
| `lstm`    | `Model_2`          | `LSTM_model_training/LSTM_model_training/models/` |
| `hybrid`  | `Model_3`          | `hybrid_model_training/models/` |

### Exemplo completo

```json
{
  "conv1d": [
    "CONV1D_model_training/CONV1D_model_training/models/eq_2000_ep_15.pt"
  ],
  "lstm": [
    "LSTM_model_training/LSTM_model_training/models/eq_1800_ep_20.pt",
    "LSTM_model_training/LSTM_model_training/models/eq_1750_ep_18.pt"
  ],
  "hybrid": [
    "hybrid_model_training/models/eq_1900_ep_10.pt"
  ]
}
```

### Regras práticas

1. **Não** repitas o prefixo `CNN/` no path — o código faz `CNN_ROOT / teu_path`.
2. Cada `.pt` tem de existir nesse caminho (relativo a `CNN/`).
3. Cada peso tem de corresponder à **arquitetura** da chave (Conv1D vs LSTM vs hybrid); caso contrário `load_state_dict` falha.
4. Se só treinaste uma arquitetura, **podes omitir** as outras chaves (o backtest ignora chaves em falta).
5. Podes listar **vários** `.pt` na mesma chave; cada um entra no ensemble (média dos sinais no backtest).

### Criar o ficheiro à mão

1. Copia o exemplo acima para `CNN/artifacts/manifest.json`.
2. Substitui pelos nomes reais dos teus `eq_*_ep_*.pt`.
3. Confirma no Explorador de ficheiros o caminho desde `CNN\` até ao ficheiro e converte `\` em `/`.

Há também um modelo em `CNN/artifacts/manifest.example.json` (podes duplicar e renomear para `manifest.json`).

---

## 4. Ordem sugerida

1. Garantir `split_info.json` e `scalers.pkl` alinhados com o CSV.
2. Preencher `manifest.json` com os teus checkpoints.
3. Correr `CNN/backtest/main.ipynb` (idealmente com working directory que permita encontrar `CNN/data/` — a primeira célula resolve `CNN_ROOT` subindo diretórios até encontrar `data/BTCUSDT-1h-data.csv` ou `CNN/data/...`).
