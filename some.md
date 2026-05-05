## INF — Mudança para Triple Barrier (TB): resumo e implicações

Mudámos o pipeline `INF` de um **problema de regressão** (prever um valor/feature futura, estilo *candle-body prediction*) para um **problema de classificação 3‑classes** usando **Triple Barrier (TB)**, mantendo o backtest/reporting compatíveis através de um **sinal contínuo** derivado das probabilidades do classificador.

### O que mudou (alto nível)

- **Novo modo por YAML**: quando o config tem `target.type: triple_barrier`, o pipeline entra em modo TB; configs antigos continuam a funcionar (modo regressão).

Exemplo de estrutura TB:

```yaml
target:
  type: triple_barrier
  tp_pct: 0.015
  sl_pct: 0.010
  horizon: 8
training:
  loss: cross_entropy
  class_balance: weighted
model:
  output_dim: 3
```

### Labels TB (`INF/labels.py`)

Foi adicionado `build_triple_barrier_labels(closes, highs, lows, tp_pct, sl_pct, horizon)`:

- **Classe 0**: *timeout* (nenhuma barreira tocada dentro do horizonte)
- **Classe 1**: TP tocado primeiro (**long**)
- **Classe 2**: SL tocado primeiro (**short**)
- Os últimos `horizon` pontos são marcados como **-1 (inválidos)**.

### Trainer: 2 modos (legacy vs TB) (`INF/trainer.py`)

O `train_window` passou a suportar dois regimes:

- **Legacy (regressão)**:
  - `training.loss: mse`
  - `Y_*` é **2D float**
  - sinal = `preds[:, 0]`

- **Triple Barrier (classificação)**:
  - `training.loss: cross_entropy`
  - `Y_*` é **1D int64** (0/1/2)
  - modelo produz **logits** com `output_dim=3`
  - `val_equity` (métrica de checkpoint) continua a existir, mas passa a usar o sinal contínuo derivado dos logits.

### Output dos modelos (`INF/models.py`)

As arquiteturas (`conv1d`, `lstm`, `hybrid`) passaram a aceitar `output_dim`.

- **Legacy**: `output_dim` omitido ⇒ mantém compatibilidade (saída ≈ `num_features`).
- **TB**: `output_dim: 3` ⇒ logits 3 classes.

### Sinal contínuo para manter o backtest compatível

Para não alterar `backtest_engine.py`, convertemos logits em sinal contínuo:

- `probs = softmax(logits[:3])`
- **signal = p_long − p_short** (classes 1 e 2)
- `signal ∈ [-1, 1]`

Isto permite manter a regra existente:
- long se `signal > threshold`
- short se `signal < -threshold`
- no-trade caso contrário

### Walk-forward sem contaminar splits (`INF/run_walkforward.py`)

No modo TB:

- As labels TB são calculadas **por split** (ex.: `train_df` gera `train_labels`; `val_df` gera `val_labels`).
- Os pontos `-1` são filtrados depois (`valid_*` masks).
- `x_test` continua a ser criado apenas com features; não se geram labels para `test_df` no pipeline.
- `val_opens/val_closes` são alinhados com o mask de labels válidos.

### Implicações práticas

#### 1) `signal_threshold` mudou de significado
Agora o threshold atua sobre \(p_{long} - p_{short}\) (intervalo \([-1,1]\)).
Thresholds altos (ex.: `0.20`, `0.25`) exigem muita convicção do modelo — podem filtrar demasiado ou deixar passar só sinais enviesados.

#### 2) Class imbalance passa a ser crítico
Com TB é comum haver classes dominantes (muitos timeouts ou mais SL do que TP).
Por isso `training.class_balance: weighted` ajuda a evitar que o treino colapse para uma classe.

#### 3) “Perda” normal de amostras por `horizon`
Os últimos `horizon` pontos em cada split tornam-se inválidos (`-1`) e são removidos.
Isto reduz amostras, mas **não deveria matar janelas inteiras**; se acontecer, normalmente é porque `train/val` ficaram curtos vs `seq_len + horizon`.

#### 4) Checkpoints TB não são comparáveis com legacy
Os checkpoints TB carregam metadata (`loss`, `output_dim`, `target_type`) e não devem ser reutilizados no modo antigo.

#### 5) Features “de estado” tornam-se mais importantes
TB mede eventos num horizonte; muitas vezes exige features com contexto (volatilidade, momentum, regime), não só informação do candle isolado.

### Checklist rápida de sanidade TB

- Ver distribuição de labels (0/1/2) e confirmar que existe sinal em 1 e 2.
- Ativar `class_balance: weighted` no início.
- Começar com barreiras “fáceis” (ex.: `tp_pct == sl_pct`, `horizon` maior) para aumentar exemplos.
- Testar thresholds mais baixos (ex.: 0.05 / 0.10 / 0.15) antes de 0.20+.