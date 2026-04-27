# Heatmaps SL/TP no `main.ipynb`

Este documento descreve o que os heatmaps fazem, o que significam os eixos, e as alterações feitas à célula da grelha SL/TP (heatmap) para torná-la legível e alinhada com valores pequenos de stop/take (ex.: `BEST_SL = 0.02`, `BEST_TP = 0.4`).

---

## O que o heatmap mede

- Corre o `trading_backtest` **apenas na janela de seleção** (entre `train_end` e `val_end`), com `verbose=False`, para cada par **(SL, TP)** da grelha.
- Cada célula guarda a **equity final** dessa corrida (último valor da série `equities`).
- O mapa mostra **onde a equity final é mais alta** na grelha — é uma ferramenta de **exploração**, não substitui validação no holdout.

**Importante:** a grelha serve para **escolher** SL/TP; o holdout deve correr **depois**, com os valores fixados (`BEST_SL` / `BEST_TP`), sem “tunar” no holdout.

---

## O que são `SL_POINTS` e `TP_POINTS`

No `trading_backtest`, **são distâncias em USD no preço** desde a entrada, nas **mesmas unidades** que `open`, `high`, `low`, `close` no CSV.

- Não são percentagens.
- Não são “pips” ou pontos de outro mercado.

Se o preço do ativo é da ordem de **unidades ou dezenas de USD**, stops e takes “realistas” podem ser **frações de dólar**. Uma grelha que só começa em SL=3 e TP=10 pode **nunca** testar a região que na prática usas à mão — o problema é a **escala da grelha**, não necessariamente o backtest.

---

## Por que o mapa antigo “não servia”

Dois problemas típicos:

1. **Grelha desalinhada com os melhores valores**  
   Ex.: melhor par com `TP = 0.4` mas a grelha começava em `TP = 0.5` — esse TP **não existia** no mapa.

2. **`imshow` + muitos ticks com valores “feios”**  
   Com dezenas de valores por eixo, os rótulos sobrepõem-se e a maior parte da área útil fica numa “mancha” sem leitura clara da zona onde a performance é boa.

---

## Alterações feitas (última versão da célula)

### 1. Grelha numérica

- **SL:** `geomspace` de **0.02 → 5.0** (22 pontos), mais âncoras explícitas (incl. **0.01**, **0.015**, valores intermédios pequenos, até 3.0).
- **TP:** `geomspace` de **0.1 → 30.0** (22 pontos), mais âncoras explícitas, incluindo **`0.2`, `0.3`, `0.4`, `0.5`, …** até 30.

Objetivo: cobrir **regimes apertados** (ativos baratos ou stops curtos) **e** manter um intervalo útil até TP ~30 sem explodir só para BTC-style.

*(A contagem de combinações aumenta face a uma grelha mínima; o notebook fica mais lento mas o mapa torna-se significativo.)*

### 2. Visualização: `pcolormesh` + eixos log

- Em vez de `imshow` (índices 0..N-1 nos eixos), usa-se **`pcolormesh`** com coordenadas derivadas dos valores SL/TP.
- Eixos **`xscale="log"`** e **`yscale="log"`**, porque SL e TP variam em ordens de grandeza; assim a **zona baixa** (onde muitas vezes está o pico) não fica esmagada num canto ilegível.

### 3. Bordas das células (`_edges_from_centers`)

Os valores da grelha são tratados como **centros** de intervalos; constrói-se um vetor de **bordas** compatível com escala log (geometria entre vizinhos), para o `pcolormesh` alinhar cada retângulo ao par (TP, SL) correto.

### 4. Menos ticks, mais legíveis (`_nice_log_ticks`)

- Ticks do tipo **1, 2, 5** por ordem de grandeza (estilo log “limpo”).
- Rótulos do eixo X com **`rotation=45`** e **`ha="right"`** para reduzir sobreposição.

### 5. Zoom em torno de `BEST_SL` / `BEST_TP`

No fim da célula:

- Se `BEST_SL` e `BEST_TP` **já estiverem definidos** (célula seguinte do notebook), desenha-se um **segundo heatmap** só com a fatia da matriz onde:
  - SL está entre aproximadamente **`BEST_SL/5`** e **`BEST_SL*5`**
  - TP entre **`BEST_TP/5`** e **`BEST_TP*5`**
- Marca o ponto escolhido com um **`x`** branco.

**Nota:** na primeira execução, se ainda não existirem `BEST_SL`/`BEST_TP`, o `try/except NameError` ignora o zoom — corre de novo a célula depois de preencheres esses valores.

### 6. Saída textual

Mantém-se o `print` com o **máximo da grelha** (melhor célula segundo equity final), para conferência rápida com o que vês no mapa.

---

## Como usar na prática

1. Corre a célula do heatmap (demora mais que uma grelha pequena).
2. Ajusta `BEST_SL` e `BEST_TP` na célula seguinte com base no mapa (e no zoom, se aplicável).
3. Volta a correr **só** a célula do heatmap se quiseres o **zoom** com o `x` no teu par.
4. Corre o backtest final (seleção + holdout) com esses valores fixos.

---

## Resumo

| Aspeto | Antes (problemático) | Depois |
|--------|----------------------|--------|
| TP muito baixo (ex. 0.4) | Podia estar **fora** da grelha | **Incluído** nas âncoras |
| Eixos | Índices + muitos rótulos | **Log** + ticks esparsos |
| Plot | `imshow` | **`pcolormesh`** com bordas coerentes |
| Zona “boa” comprimida | Difícil de ler | **Segundo painel zoom** + marca no `BEST_*` |

Este ficheiro descreve a implementação na célula de código do heatmap em `main.ipynb` (grelha SL/TP, `pcolormesh` com eixos log, e segundo gráfico de zoom quando `BEST_SL` / `BEST_TP` já estão definidos noutra célula).
