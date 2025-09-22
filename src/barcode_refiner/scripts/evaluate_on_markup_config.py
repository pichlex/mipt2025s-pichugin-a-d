from __future__ import annotations

import argparse
import json
import os
from typing import List

import yaml

from .evaluate_on_markup import run_markup_eval
from ..logging_utils import setup_logging


def main():
    p = argparse.ArgumentParser(description="Evaluate on markup dataset via YAML config")
    p.add_argument("--config", required=True, help="YAML with images_dir, markup_dir, out, methods, noise, verbose, progress")
    args = p.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    images_dir: str = cfg["images_dir"]
    markup_dir: str = cfg["markup_dir"]
    out: str = cfg.get("out", "results/markup_eval")
    methods: List[str] = cfg.get("methods", ["noop", "subpix", "edge"])  # default
    noise: float = float(cfg.get("noise", 3.0))
    verbose: int = int(cfg.get("verbose", 1))
    progress: bool = bool(cfg.get("progress", True))

    setup_logging(verbose)
    res = run_markup_eval(images_dir, markup_dir, out, methods, noise, verbose=verbose, progress=progress)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
