"""Off-the-shelf reward-model baselines used in the paper."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple


WILDREWARD_ID = "THU-KEG/WildReward-8B"
THINKPRM_7B_ID = "launch/ThinkPRM-7B"
THINKPRM_14B_ID = "launch/ThinkPRM-14B"


WILDREWARD_TEMPLATE = """# Task Description
You are an expert conversation evaluator. Your task is to judge the **User's Satisfaction** with the Assistant's response based on the conversation context.
Please rate the response on a scale of 1 to 5 integers.

# Scoring Criteria
[1] CLEARLY NEGATIVE / REJECTION
[2] CORRECTION / ERROR POINTER (Negative)
[3] NEUTRAL
[4] POSITIVE ENGAGEMENT
[5] CLEAR SATISFACTION

# Input Data
## Context (History)
{history}

## User Query
{query}

## Assistant Response
{response}

# Output
Based on the criteria above, please output ONLY the integer score (1, 2, 3, 4, or 5)."""


THINKPRM_TEMPLATE_AGENT = """You are given a task and a proposed step-by-step solution:

[Task]

{task}

[Solution]

{solution}

Review and critique each step in the proposed solution to determine whether each step is correct. If the solution is incomplete, only verify the provided steps. For each step, end your critique with a single line of the form 'Step N is \\boxed{{correct}}' or 'Step N is \\boxed{{incorrect}}'.
"""


THINKPRM_TEMPLATE_FA = """You are given a multi-agent problem-solving trajectory and must verify each step.

[Task]

{task}

[Solution]

{solution}

Review and critique each step in the proposed solution to determine whether each step is correct. For each step, end your critique with a single line of the form 'Step N is \\boxed{{correct}}' or 'Step N is \\boxed{{incorrect}}'.
"""


# Message rendering for the baselines


def initial_user_query(messages: List[Dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            c = str(m.get("content", "") or "").strip()
            if c:
                return c
    return ""


def _render_action(msg: Dict) -> str:
    content = str(msg.get("content", "") or "").strip()
    parts: List[str] = []
    for tc in msg.get("tool_calls") or []:
        if not tc:
            continue
        fn = tc.get("function") if isinstance(tc, dict) and "function" in tc else tc
        name = fn.get("name", "") if isinstance(fn, dict) else ""
        args = fn.get("arguments", {}) if isinstance(fn, dict) else {}
        parts.append(f"tool_call: {name}({args})")
    out = "\n".join(p for p in [content, "\n".join(parts)] if p)
    return out or "[empty]"


def messages_to_flat_text(messages: List[Dict], max_chars: int = 24000) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "?")
        if role == "assistant":
            lines.append(f"[assistant]: {_render_action(m)}")
        else:
            content = str(m.get("content", "") or "").strip()
            lines.append(f"[{role}]: {content}" if content else f"[{role}]: (empty)")
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


def messages_to_numbered_steps(
    messages: List[Dict], max_chars: int = 22000
) -> Tuple[str, int]:
    blocks: List[str] = []
    pending: List[str] = []
    n = 0
    for m in messages:
        role = m.get("role", "?")
        if role == "assistant":
            if pending and blocks:
                blocks[-1] += "\nObservation:\n" + "\n".join(pending)
                pending = []
            elif pending:
                pending = []
            n += 1
            blocks.append(f"Step {n}:\n{_render_action(m)}")
        else:
            content = str(m.get("content", "") or "").strip()
            if content:
                pending.append(f"[{role}]: {content}")
    if pending and blocks:
        blocks[-1] += "\nObservation:\n" + "\n".join(pending)
    solution = "\n\n".join(blocks)
    if len(solution) > max_chars:
        solution = solution[len(solution) - max_chars:]
    return solution, n


# WildReward scorer


class WildRewardScorer:
    def __init__(
        self,
        model_id: str = WILDREWARD_ID,
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        max_length: int = 4096,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_length = max_length

    def score_messages(self, messages: List[Dict], max_chars: int = 24000) -> float:
        import torch

        query = initial_user_query(messages)
        response = messages_to_flat_text(messages, max_chars=max_chars)
        text = WILDREWARD_TEMPLATE.format(history="", query=query, response=response)
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0].float()
        return 1.0 + float(torch.sigmoid(logits).sum().item())


# ThinkPRM scorer (vLLM batched)


_BOXED_RE = re.compile(
    r"Step\s*(\d+)[^{]*?\\boxed\{\s*(correct|incorrect)\s*\}", re.IGNORECASE
)


def _p_correct(logprob_entry) -> Optional[float]:
    if logprob_entry is None:
        return None
    lp_c, lp_i = None, None
    for _tid, lp in logprob_entry.items():
        tok = (getattr(lp, "decoded_token", "") or "").strip().lower()
        if not tok:
            continue
        if "incorrect" in tok and lp_i is None:
            lp_i = float(lp.logprob)
        elif tok.startswith("correct") and lp_c is None:
            lp_c = float(lp.logprob)
    if lp_c is None and lp_i is None:
        return None
    if lp_c is None:
        lp_c = lp_i - 20.0
    if lp_i is None:
        lp_i = lp_c - 20.0
    m = max(lp_c, lp_i)
    return math.exp(lp_c - m) / (math.exp(lp_c - m) + math.exp(lp_i - m))


def _extract_step_scores(out, tokenizer, n_expected: int) -> List[float]:
    token_ids = list(out.token_ids or [])
    logprobs = list(out.logprobs or [])
    text = out.text or ""

    graded: Dict[int, float] = {}
    seen = ""
    for i, tid in enumerate(token_ids):
        seen += tokenizer.decode([tid], skip_special_tokens=True)
        if seen.rstrip().endswith("\\boxed{") or seen.rstrip().endswith("boxed{"):
            nxt = i + 1
            if nxt < len(logprobs):
                p = _p_correct(logprobs[nxt])
                if p is not None:
                    graded[len(graded) + 1] = p

    if graded:
        return [graded[k] for k in sorted(graded.keys())][:n_expected]

    binary: Dict[int, float] = {}
    for m in _BOXED_RE.finditer(text):
        binary[int(m.group(1))] = 1.0 if m.group(2).lower() == "correct" else 0.0
    if binary:
        return [binary[k] for k in sorted(binary.keys())][:n_expected]

    fallback: List[float] = []
    for m in re.finditer(r"\\boxed\{\s*(correct|incorrect)\s*\}", text, re.IGNORECASE):
        fallback.append(1.0 if m.group(1).lower() == "correct" else 0.0)
    return fallback[:n_expected]


class ThinkPRMScorer:
    def __init__(
        self,
        model_id: str = THINKPRM_7B_ID,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.55,
        max_new_tokens: int = 4096,
        dtype: str = "bfloat16",
        template: str = "agent",
    ) -> None:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.llm = LLM(
            model=model_id, dtype=dtype, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=20)
        self._template = {
            "agent": THINKPRM_TEMPLATE_AGENT,
            "fa": THINKPRM_TEMPLATE_FA,
        }[template]

    def build_prompt(self, task: str, solution: str) -> str:
        body = self._template.format(task=task, solution=solution)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True,
        ) + "\nLet's verify step by step:"

    def score_prompts(self, prompts: List[str], n_expected: List[int]) -> List[List[float]]:
        """Batch-verify pre-built prompts; returns per-step P(correct)."""
        if not prompts:
            return []
        outs = self.llm.generate(prompts, self.sampling)
        return [_extract_step_scores(o.outputs[0], self.tokenizer, n_expected[i])
                for i, o in enumerate(outs)]

    def score_trajectories(
        self, pairs: List[Tuple[str, List[Dict]]], max_sol_chars: int = 22000,
    ) -> List[List[float]]:
        """Score (task, messages) pairs with the agent-style step rendering."""
        prompts, n_expected = [], []
        for task, messages in pairs:
            solution, n = messages_to_numbered_steps(messages, max_chars=max_sol_chars)
            prompts.append(self.build_prompt(task, solution))
            n_expected.append(n)
        return self.score_prompts(prompts, n_expected)
