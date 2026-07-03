# Open Trajectory Cost Atlas

Strong-model (Claude Opus 4.8) token/cost breakdown across three public
coding-agent trajectory datasets (Pi, KernelBook, OpenHands).

All USD figures use one unified Opus 4.8 list price
(`$5` fresh / `$6.25` cache-write / `$0.50` cache-read / `$25` decode per MTok)
and are reproducible from raw token counts via [`recompute_cost.py`](./recompute_cost.py)
(output: [`recomputed_cost.json`](./recomputed_cost.json)).

Note: Pi's *recorded* cost ($2.06) uses a discounted rate ~1/7 of list and is
shown only as context; the dashboard's $12.46 is the list-price recompute.

Live: open `index.html`.
