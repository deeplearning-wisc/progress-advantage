# Data

The artifacts below are **not committed** to the repo. They are pulled
on demand from a Hugging Face dataset whose tree mirrors this folder:

```bash
python -m pa.data --scenario all     # or: uq | tts | fa
```

(the reproduction scripts call this automatically when `data/` is
missing). Override the source with `--repo-id <user>/<repo>` or
`PA_DATA_REPO`.

```
data/
├── tau2/
│   └── greedy/
│       ├── gemma4-4b/{airline,retail}.json     # UQ
│       └── qwen3.5-9b/{airline,retail}.json    # UQ
├── webshop/
│   └── bon8/
│       ├── gemma4-4b/trial_{0..7}/webshop.json    # TTS
│       └── qwen3.5-9b/trial_{0..7}/webshop.json   # TTS
└── who_and_when/
    ├── Hand-Crafted/*.json          # FA (full dataset)
    ├── Algorithm-Generated/*.json   # FA (full dataset)
    └── LICENSE
```

All three scenarios reproduce out of the box for the `Gemma4-4B` and
`Qwen3.5-9B` backbones. Each `{...}.json` follows tau2-bench's schema:

```json
{"simulations": [
  {"id": "...", "task_id": "...",
   "messages": [{"role": "...", "content": "...", "tool_calls": [...]}, ...],
   "reward_info": {"reward": 0.0}},
  ...
]}
```

## UQ — `tau2/greedy/`

Greedy-decoding $\tau^2$-bench trajectories for the two backbones,
enough to reproduce their `Gemma4-4B` / `Qwen3.5-9B` columns in
the main UQ table (~29 MB combined).

| Backbone   | Domain  | N   | Greedy success |
| ---------- | ------- | --- | -------------- |
| Gemma4-4B  | Airline | 50  | 34.0 %         |
| Gemma4-4B  | Retail  | 114 | 45.6 %         |
| Qwen3.5-9B | Airline | 50  | 60.0 %         |
| Qwen3.5-9B | Retail  | 114 | 64.9 %         |

Other backbones: drop tau2-bench rollouts under
`tau2/greedy/<pair>/{airline,retail}.json` (generate with the
tau2-bench CLI, https://github.com/sierra-research/tau2-bench).

## TTS — `webshop/bon8/`

Eight independent WebShop rollouts per task (`k=8`, sampled at
`T=0.7`) for `Gemma4-4B` and `Qwen3.5-9B`, 100 tasks each (~32 MB
combined). `trial_<i>/webshop.json` holds rollout `i` of every task;
`runners/tau2_bon.py` groups by `task_id` across the eight trials,
selects the best by progress advantage, and reports the success rate
of the selection (the WebShop best-of-8 columns).

`reward_info.reward` is the WebShop match score in `[0, 1]`; a task is
a success when `reward >= 1.0`. As a model-free check, the oracle
Pass@8 over these files is **53.0 % (Gemma4-4B)** and **49.0 %
(Qwen3.5-9B)**, matching the `Pass@N (oracle)` / WebShop row.

The runner is benchmark-agnostic: BFCLv4-MT, AgentDojo, and
$\tau^2$-Airline rollouts reproduce the rest of the best-of-8 table once
converted to the same `trial_<i>/<domain>.json` schema. Those cached
rollouts are not hosted yet.

## FA — `who_and_when/` (full dataset)

The complete Who & When failure-attribution dataset — `Hand-Crafted`
(58 trajectories) and `Algorithm-Generated` (126) — redistributed from
https://github.com/ag2ai/Agents_Failure_Attribution under its MIT
license (see `who_and_when/LICENSE`). Each `*.json` carries a `history`
(list of `{role, content}`), the ground-truth `mistake_step`, and
`mistake_agent`. `runners/fa.py` scores every step and predicts the
decisive error step; any backbone in `pa/models.py`
works as the scorer.
