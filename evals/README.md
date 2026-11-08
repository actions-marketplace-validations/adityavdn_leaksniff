# Agent experiment

Same prompt every time: a daily-sales dataset (date, store_id, sales, promo, temperature), build lag and
rolling features, run 5-fold cross-validation, "use whatever gives the best CV score". Model: Claude
Haiku 5.5, one fresh agent per run, no tools.

| | runs | leaky split | baseline model |
|---|---|---|---|
| `no_skill/` | 4 | 4 (shuffled or blocked KFold on dated rows) | 0 |
| `with_skill/` (read `skills/leaksniff/SKILL.md` first) | 3 | 0 (TimeSeriesSplit) | 3 |

The files keep each run's split and evaluation code exactly and trim the feature engineering, which was
correct (backward-looking) in every run. A larger model (Claude Opus 5.5) got this right without the skill
in 3 of 3 runs, so the skill matters most for smaller, cheaper agents. Small sample; treat it as a smoke
test, not a benchmark.

`pytest` checks that leaksniff flags every `no_skill` run and passes every `with_skill` run.
