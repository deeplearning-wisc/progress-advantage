"""Render messages into per-message text spans for one-pass log-prob scoring."""

from __future__ import annotations

from typing import Dict, List, Tuple


def render_tau2_messages(messages: List[Dict]) -> Tuple[List[str], List[bool]]:
    texts: List[str] = []
    is_action: List[bool] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = str(msg.get("content", "") or "")
        tool_calls = msg.get("tool_calls")
        sep = "\n" if i > 0 else ""
        if role == "assistant":
            tc_str = ""
            if tool_calls:
                parts = []
                for tc in tool_calls:
                    if tc:
                        name = tc.get("name", "")
                        args = tc.get("arguments", {})
                        parts.append(f"tool_call({name}, {args})")
                if parts:
                    tc_str = "\n" + "\n".join(parts)
            action_text = (content + tc_str).strip() or "[assistant action]"
            texts.append(f"{sep}[assistant]: {action_text}")
            is_action.append(True)
        else:
            texts.append(f"{sep}[{role}]: {content}")
            is_action.append(False)
    return texts, is_action


def render_fa_messages(
    history: List[Dict], split: str
) -> Tuple[List[str], List[bool]]:
    texts: List[str] = []
    is_action: List[bool] = []
    for i, msg in enumerate(history):
        role = msg.get("role", "")
        content = str(msg.get("content", "") or "")
        sep = "\n" if i > 0 else ""
        texts.append(f"{sep}[{role}]: {content}")
        if split == "Hand-Crafted" and i == 0 and role.lower() == "human":
            is_action.append(False)
        else:
            is_action.append(bool(content.strip()))
    return texts, is_action
