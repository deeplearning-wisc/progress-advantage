"""On-demand download of the UQ / TTS / FA artifacts from the Hugging Face Hub.

The trajectory artifacts are not committed to the repo; they live in a
Hugging Face *dataset* whose tree mirrors ``data/`` (``tau2/``,
``webshop/``, ``who_and_when/``). Override the source with ``--repo-id``
or the ``PA_DATA_REPO`` environment variable.
"""

from __future__ import annotations

import argparse
import os

DEFAULT_REPO = os.environ.get("PA_DATA_REPO", "changdae/progress-advantage-artifacts")

SCENARIOS = {
    "uq": ["tau2/**"],
    "tts": ["webshop/**"],
    "fa": ["who_and_when/**"],
}


def download(scenario: str = "all", repo_id: str = DEFAULT_REPO,
             data_dir: str = "data") -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required for on-demand download:\n"
            "    pip install huggingface_hub"
        ) from e

    patterns = None if scenario == "all" else SCENARIOS[scenario]
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=patterns,
    )
    return data_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="all",
                    choices=["all", *SCENARIOS])
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    out = download(args.scenario, args.repo_id, args.data_dir)
    print(f"fetched '{args.scenario}' artifacts from {args.repo_id} -> {out}/")


if __name__ == "__main__":
    main()
