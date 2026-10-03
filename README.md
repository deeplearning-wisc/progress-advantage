# Progress Advantage for LLM Agents

Official codebase of the paper "_Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents_", 
> By [Changdae Oh](https://changdaeoh.github.io/)<sup>1</sup>, [Wendi Li](https://wendili.github.io/)<sup>1</sup>, [Seongheon Park](https://seongheon-96.github.io/)<sup>1</sup>, [Samuel Yeh](https://mhyeh.github.io/)<sup>1</sup>, 
> <br/>
[Tanwi Mallick](https://tanwimallick.github.io/)<sup>2</sup>, and [Sharon Li](https://pages.cs.wisc.edu/~sharonli/)<sup>1</sup>.
> <br/>
> <br/>
> <sup>1</sup>University of Wisconsin--Madison, <sup>2</sup>Argonne National Laboratory

[![Paper](https://img.shields.io/badge/ArXiv-2606.26080-orange)](https://arxiv.org/pdf/2606.26080)
[![Project Page](https://img.shields.io/badge/ProjectPage-ProgressAdvantage-teal)](https://changdaeoh.github.io/progress-advantage/)


## News
- [Sep 24, 2026] The paper got accepted by NeurIPS 2026🎉
- [Jul 2, 2026] Progress Advantage won the best paper award🏆 at [RLxF@ICML2026](https://sites.google.com/view/rlxf-icml2026/accepted-papers)
- [Jun 24, 2026] The paper is now alive at arXiv [🔗](https://arxiv.org/abs/2606.26080); we release the initial codebase.
- [Jun 1, 2026] Progress Advantage got accepted by a workshop at ICML 2026, [RLxF: Reinforcement Learning from World Feedback](https://sites.google.com/view/rlxf-icml2026)🎉


## 0. Overview

Progress advantage,

```
A_t = β · log [ π_θ(a_t | s_t) / π_ref(a_t | s_t) ],
```

is a training-free trajectory scorer for LLM agents that can be built from the pairs of RL-trained policy ``π_θ`` and its (base) reference policy ``π_ref``. This repository provides essential code to reproduce three application scenarios from the paper: best-of-N test-time scaling (TTS), trajectory-level uncertainty quantification (UQ), and step-level failure attribution (FA).



Progress Advantage and the log-prob / certainty methods share one forward pass per model (`runners/{tau2_uq,tau2_bon,fa}.py`); the off-the-shelf RM baselines live in `runners/baselines.py`.

## 1. Layout

```
pa/
├── aggregations.py    # token/step aggregations
├── baselines.py       
├── data.py            # on-demand artifact download
├── merging.py         # weight-space merged reference policies
├── models.py          # MODEL_PAIRS - policy/reference checkpoints
├── scoring.py         # one-pass log-prob extraction
└── trajectory.py      # tau2/Who&When message -> action span rendering

runners/
├── tau2_uq.py         # UQ: trajectory-level success prediction
├── tau2_bon.py        # TTS: best-of-N selection
├── fa.py              # FA: step-level error prediction
└── baselines.py       # WildReward / ThinkPRM for the above

scripts/               # bash launchers
data/                  # fetched on demand via `python -m pa.data`; see data/README.md
```

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers numpy scipy scikit-learn huggingface_hub
# optional, only needed for the ThinkPRM and any vLLM-based rollout work:
pip install vllm
```

The library was mainly built with PyTorch + Hugging Face Transformers; ``vllm`` is
required only for ``pa.baselines.ThinkPRMScorer``.

## 3. Data

The trajectory artifacts (~67 MB) are not committed; they are pulled on
demand from a Hugging Face dataset whose tree mirrors `data/`:

```bash
python -m pa.data --scenario all          # uq, tts, fa
```

The reproduction scripts below call this automatically when `data/` is
missing. Point it at your own mirror with `--repo-id <user>/<repo>` or
`PA_DATA_REPO=<user>/<repo>`. See `data/README.md` for the layout.

## 4. Quick Start

One progress-advantage run per scenario (auto-fetches its data; `PAIR ∈
{gemma4-4b, qwen3.5-9b}`):

```bash
PAIR=qwen3.5-9b bash scripts/tau2_uq_onpolicy.sh   # UQ  -> results/uq_onpolicy/
PAIR=qwen3.5-9b bash scripts/webshop_bon8.sh       # TTS -> results/bon8/
PAIR=qwen3.5-9b bash scripts/fa.sh                 # FA  -> results/fa/
```

## 5. How to Reproduce?

All three scenarios reproduce out of the box for the `Gemma4-4B` and
`Qwen3.5-9B` backbones. Each loads `π_θ` and `π_ref` once; on a single
GPU, pair evaluation (base, post-trained) takes a few minutes.

### 5.1. UQ — trajectory-level outcome prediction

The scripts fetch the four greedy-decoding tau2-bench trajectory files
corresponding to the `Gemma4-4B` and `Qwen3.5-9B` columns of the
greedy-decoding results table (~29 MB total), then run on-policy UQ:

```bash
PAIR=qwen3.5-9b bash scripts/tau2_uq_onpolicy.sh
PAIR=gemma4-4b  bash scripts/tau2_uq_onpolicy.sh
```

Each `PAIR` run loads `π_θ` and `π_ref` and scores all methods on both
Airline and Retail, writing AUROC and Spearman ρ to
`results/uq_onpolicy/<pair>_<domain>.json`. The `progress_advantage`
AUROC should land close to the matching `Gemma4-4B` / `Qwen3.5-9B` cells
of the corresponding table in our manuscript. 

### 5.2. TTS — best-of-8 test-time scaling

Eight WebShop rollouts per task (100 tasks, `k=8`) are fetched for both
backbones under `data/webshop/bon8/`:

```bash
PAIR=qwen3.5-9b bash scripts/webshop_bon8.sh
PAIR=gemma4-4b  bash scripts/webshop_bon8.sh
```

Each run scores the 8 rollouts of every task with progress advantage —
using each backbone's tuned token/step aggregation (`max/min` for
Gemma4-4B, `min/last` for Qwen3.5-9B; override with `TOKEN_AGG`/
`STEP_AGG`) — keeps the best, and writes the selection's success rate to
`results/bon8/webshop_<pair>_progress_advantage.json`. 

### 5.3. FA — step-level failure attribution

The full Who & When dataset is fetched to `data/who_and_when/`:

```bash
PAIR=qwen3.5-9b bash scripts/fa.sh
PAIR=gemma4-4b  bash scripts/fa.sh
```

Each run scores every step of both splits and predicts the decisive
error step, writing step-level accuracy / MAE to `results/fa/<pair>.json`.

### 5.4. Pre-trained reward model baselines

`runners/baselines.py` reproduces the WildReward / ThinkPRM rows of the
same tables. `TASK ∈ {uq, bon, fa}`, `MODEL ∈ {thinkprm, wildreward}`.
ThinkPRM (process RM) covers all three; WildReward (outcome RM) covers
`uq`/`bon` only — it has no per-step signal for `fa`:

```bash
TASK=uq  MODEL=wildreward PAIR=qwen3.5-9b              bash scripts/baselines.sh
TASK=bon MODEL=wildreward PAIR=gemma4-4b               bash scripts/baselines.sh
TASK=fa  MODEL=thinkprm   MODEL_ID=launch/ThinkPRM-14B bash scripts/baselines.sh
```


## 6. How It Works (Pseudocode)

### 6.1. Progress advantage of one trajectory

```python
# one forward pass per model gives per-token log p(a_t | s_t)
for t, action_step in enumerate(trajectory):          # action tokens only
    lp_policy = aggregate_tokens(logp_θ[action_step],   token_agg) 
    lp_ref    = aggregate_tokens(logp_ref[action_step], token_agg)
    A[t] = beta * (lp_policy - lp_ref)                 # progress advantage

score = aggregate_steps(A, step_agg)                   # trajectory scalar
```



### 6.2. Reference policy merging

Instead of the pure base checkpoint reference policy, the denominator of the progress advantage can be a weight-space merge `π_α = π_ref + α · τ` of the reference and policy checkpoints (`α = 0` recovers `π_ref`, `α = 1` the policy). `MergedReference` is a drop-in for `LogprobScorer`, and `set_alpha` rewrites the resident model, so an α sweep loads the checkpoints once.

```python
from pa import MODEL_PAIRS, MergedReference

pair = MODEL_PAIRS["qwen3.5-9b"]
ref = MergedReference(pair["policy"], pair["reference"], method="ties", density=0.2)
for alpha in [0.0, 0.2, 0.4, 0.6, 0.8]:
    ref.set_alpha(alpha)
    cache = ref.score(message_texts, is_action)   # then score as usual
```

`method` is one of `linear`, `ties`, `dare_linear`, `dare_ties` (sparsification and sign consensus follow [mergekit](https://github.com/arcee-ai/mergekit)) or `emr` ([EMR-Merging](https://github.com/harveyhuang18/EMR_Merging)). 

### Per-scenario reduction

```python
# UQ  — does the score rank successes above failures?
auroc = roc_auc_score(labels, [score(traj) for traj in trajectories])

# TTS — keep the highest-scoring of N rollouts per task
best = max(rollouts, key=score)

# FA  — flag the step with the lowest running reward
err_step = argmin(cumsum(A))        # or sharpest drop / earliest below threshold
```

## License
* We release our codebase and artifacts under MIT license. The datasets and models we used for this project have their own license, noted as below.
* If you use our progress advantage method with other model families and datasets, you should check their own licenses.

### Dataset
| Dataset Name  | License    |
| ---------- | ---------- |
| BFCLv4 | Apache-2.0 |
| WebShop         | MIT |
| AgentDojo          | MIT |
| $\tau^{2}$-bench        | MIT |
| Who & When | MIT |

### Model
| Organization | Model Pair                                     | License    |
| ------------ | ---------------------------------------------- | ---------- |
| Qwen         | (`Qwen3.5-9B`, `Qwen3.5-9B-Base`)                | Apache-2.0 |
| Qwen         | (`Qwen3-14B`, `Qwen3-14B-Base`)                  | Apache-2.0 |
| Google       | (`Gemma-4-E4B-it`, `Gemma-4-E4B`)                | Apache-2.0 |
| AI2          | (`Olmo-3-7B-Instruct`, `Olmo-3-7B-Instruct-DPO`) | Apache-2.0 |


## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{oh2026neglected,
  title     = {Neglected Free Lunch from Post-training: Progress Advantage for {LLM} Agents},
  author    = {Oh, Changdae and Li, Wendi and Park, Seongheon and Yeh, Samuel and Mallick, Tanwi and Li, Sharon},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
