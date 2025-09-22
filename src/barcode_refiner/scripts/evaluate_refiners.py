from __future__ import annotations

import argparse
import json
import os
from typing import List

import yaml

from ..eval import evaluate_on_dir


def main():
    p = argparse.ArgumentParser(description="Evaluate corner refiners on a directory of images + annotations")
    p.add_argument("--config", required=True, help="YAML config file")
    args = p.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    images_dir = cfg["images_dir"]
    methods: List[str] = cfg.get("methods", ["noop", "subpix", "edge"])  # default set
    init_suffix = cfg.get("init_suffix", ".init.json")
    gt_suffix = cfg.get("gt_suffix", ".gt.json")
    out_dir = cfg.get("out_dir", "results/eval")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for m in methods:
        run_dir = os.path.join(out_dir, m)
        os.makedirs(run_dir, exist_ok=True)
        summary = evaluate_on_dir(images_dir, init_suffix=init_suffix, gt_suffix=gt_suffix, method=m, out_dir=run_dir)
        all_results[m] = summary

    with open(os.path.join(out_dir, "all_results.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

