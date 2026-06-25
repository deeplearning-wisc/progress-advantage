"""Policy / reference checkpoint pairs used by progress advantage."""

MODEL_PAIRS = {
    "olmo3-7b-instruct": {
        "policy": "allenai/Olmo-3-7B-Instruct",
        "reference": "allenai/Olmo-3-7B-Instruct-DPO",
    },
    "qwen3-14b": {
        "policy": "Qwen/Qwen3-14B",
        "reference": "Qwen/Qwen3-14B-Base",
    },
    "qwen3.5-9b": {
        "policy": "Qwen/Qwen3.5-9B",
        "reference": "Qwen/Qwen3.5-9B-Base",
    },
    "gemma4-4b": {
        "policy": "google/gemma-4-E4B-it",
        "reference": "google/gemma-4-E4B",
    },
}
