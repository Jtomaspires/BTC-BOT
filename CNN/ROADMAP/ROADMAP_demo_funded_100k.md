# Roadmap: Demo → Funded 100k

Checklist por fase (marca `- [x]` quando concluído). Estado atual: **Fase 1 em curso** — `.bat` do VPS feito; **entradas em market orders** já no bot (`place_market_order` + main); falta logging por trade, confirmação SL/TP, teste `reconcile_pending_order` / critério dos 3 dias.

**Pastas por fase (neste repo):** `ROADMAP/fase-1/` … `ROADMAP/fase-6/` — notas, exports CSV, screenshots ou outros artefactos por fase.

---

## Visão geral das fases

- [ ] **Fase 1** — Execução estável (demo)
- [ ] **Fase 2** — Sinal / edge em demo
- [ ] **Fase 3** — ~1000€ real
- [ ] **Fase 4** — 3–4k real + métricas
- [ ] **Fase 5** — Challenge funded 10k
- [ ] **Fase 6** — Funded ~100k / escala

---

## Fase 1 — Validar execução (1 semana)

**Objetivo:** bot estável, sem bugs, sem ordens fantasma.

### Tarefas

- [x] Migrar para market orders (elimina o problema do cancel).
- [x] Criar o `.bat` para auto-restart no VPS.
- [ ] Logging de cada trade (entry price, exit price, PnL por trade).
- [ ] Confirmar que SL/TP estão a ser colocados corretamente.
- [ ] Confirmar que `reconcile_pending_order` funciona após crash.

### Critério de saída (Fase 1 completa quando tudo abaixo for verdade)

- [ ] Bot corre **3 dias** sem intervenção manual.
- [ ] Sem ordens abertas “por fechar” / estado limpo ao fim do dia.
- [ ] Logs limpos (sem erros recorrentes / sem comportamento inexplicável).

---

## Fase 2 — Validar sinal (2–4 semanas, demo)

**Objetivo:** confirmar que o edge do backtest aparece ao vivo.

### Rotina diária (métricas)

- [ ] Win rate por par.
- [ ] PnL simulado vs PnL backtest out-of-sample.
- [ ] Contagem: SL atingido vs TP.
- [ ] Contagem: candles em flat (neutro).

### Setup / escopo

- [ ] Correr **3–4 pares extra** em simultâneo (multi-par + mais trades para estatística mais rápida).

### Critério de saída

- [ ] **~100–150 trades** totais (todos os pares).
- [ ] Win rate e PnL **dentro de margem razoável** do backtest (não precisa bater o backtest; não pode estar dramaticamente abaixo).

---

## Fase 3 — Capital próprio pequeno (~1000€, ex. maio)

**Objetivo:** execução real — fees e slippage reais.

### Regras antes de ligar capital

- [ ] Alavancagem baixa (**2–3x máx.** nesta fase).
- [ ] `pos_size_usd` pequeno (só o necessário para fees visíveis e PnL real, não para “ganhar sério”).
- [ ] **Max drawdown diário** no bot que **para tudo** automaticamente (ex.: -5% do capital).
- [ ] Compromisso: não aumentar capital sem **2–3 semanas positivas** consecutivas.

### Critério de saída

- [ ] **3–4 semanas** com PnL positivo.
- [ ] Bot estável, sem surpresas de execução.

---

## Fase 4 — Capital próprio médio (3–4k)

**Objetivo:** escalar com confiança; preparar provas para prop firms.

### Entregas

- [ ] Dashboard simples: equity curve, drawdown, Sharpe diário.
- [ ] CSV com **todos** os trades (audit trail / prop firms).
- [ ] **5 pares** em simultâneo com capital real.
- [ ] Afinar `pos_size_usd` **por par** (volatilidade).

### Critério de saída

- [ ] **4–6 semanas** positivas.
- [ ] Drawdown máximo controlado.
- [ ] Histórico de trades limpo e exportável.

---

## Fase 5 — Funded 10k (challenge)

**Objetivo:** passar o challenge numa conta **10k**; modelo validado com capital externo.

### Conhecimento / compliance

- [ ] Internalizar regras típicas: **~5%** max daily DD, **~10%** max total DD, profit target **~8–10%**.
- [ ] **Circuit breaker no código** (bot para antes de violar limites da firm).
- [ ] Pesquisar firms: FTMO, MyForexFunds, The5ers, Topstep (crypto limitado em algumas).
- [ ] Ler termos: algumas firms **não permitem bots**; FTMO (algorithmic) e The5ers são exemplos onde bots costumam ser discutidos — **confirmar sempre os termos atuais**.

### Implementação (referência)

```python
# Circuit breaker — para o bot se drawdown diário exceder limite
MAX_DAILY_DD = 0.04  # 4% (margem antes do limite da firm)
if (equity_start_of_day - current_equity) / equity_start_of_day > MAX_DAILY_DD:
    bot.write_to_log("Daily drawdown limit reached. Stopping.")
    await bot.close_all_positions()
    exit()
```

### Critério de saída

- [ ] Challenge passado: **~8–10%** profit **sem** violar regras de drawdown.

---

## Fase 6 — Funded ~100k

**Objetivo:** escalar o que já está comprovado.

- [ ] Gerir várias contas funded em paralelo (onde a firm permitir).
- [ ] Monitorizar **edge decay** (degradação dos modelos).
- [ ] Retreino periódico com dados novos.
- [ ] Opcional: novos pares + novos modelos.

---

## Timeline (referência)

| Fase | Duração estimada | Marco |
|------|------------------|--------|
| 1 — Infra estável | 1 semana | 3 dias sem intervenção, estado limpo |
| 2 — Validar sinal demo | 3–4 semanas | 100+ trades, edge confirmado |
| 3 — ~1000€ real | 4–6 semanas | PnL positivo consistente |
| 4 — 3–4k real | 6–8 semanas | Histórico limpo, métricas prontas |
| 5 — Challenge 10k | 4–8 semanas | Passar o challenge |
| 6 — Funded ~100k | — | Escalar |

**Horizonte realista:** **6–9 meses** se o edge se mantiver ao vivo.

---

## Maior risco por fase (referência)

| Fase | Risco |
|------|--------|
| 1–2 | Bug de execução que abre posições erradas |
| 3–4 | Edge decay — modelo ok no backtest mas não ao vivo |
| 5 | Violar regras da prop firm por drawdown inesperado |
| 6 | Overfitting ao regime de mercado — o que funcionou num ano pode não funcionar noutro |

---

## Próximo passo (Fase 1)

`.bat` e **market orders** no código já estão feitos. Ordem sugerida para o que falta:

1. **Logging por trade** (base para validar SL/TP e demo na Fase 2).
2. **Validação manual** SL/TP + teste de **`reconcile_pending_order`** após crash simulado.
3. Correr até cumprir o **critério de saída** (3 dias sem intervenção, estado limpo).
