from __future__ import annotations

import argparse
import json
import os
from typing import List

import yaml

from .detector_eval import main as detector_eval_main


def main():
    # We re-parse via YAML, then construct argv for detector_eval.main
    p = argparse.ArgumentParser(description="Evaluate detector + refiner via YAML config")
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    images_dir = cfg["images_dir"]
    markup_dir = cfg["markup_dir"]
    out = cfg.get("out", "results/detector_eval")
    detector = cfg.get("detector", "yolo")
    methods: List[str] = cfg.get("methods", ["noop", "subpix", "edge"])  # default
    verbose = str(cfg.get("verbose", 1))
    progress = cfg.get("progress", True)
    save_preds = cfg.get("save_preds", True)
    save_segm = cfg.get("save_segm", False)
    decode = cfg.get("decode", "qr")
    collage_all = cfg.get("collage_all", False)
    weights = cfg.get("weights", "models/detector_yolo.pt")
    device = cfg.get("device", "auto")
    imgsz = str(cfg.get("imgsz", 640))
    conf = str(cfg.get("conf", 0.25))
    iou = str(cfg.get("iou", 0.45))
    max_det = str(cfg.get("max_det", 300))
    tta_rot = cfg.get("tta_rot", True)
    edge_radius = str(cfg.get("edge_radius", 4))
    edge_canny_low = str(cfg.get("canny_low", 50))
    edge_canny_high = str(cfg.get("canny_high", 150))
    sam2_checkpoint = cfg.get("sam2_checkpoint", "models/sam2.1_hiera_large.pt")
    sam2_config = cfg.get("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml")

    # Build sys.argv for detector_eval
    import sys
    argv = [
        sys.argv[0],
        "--images_dir", images_dir,
        "--markup_dir", markup_dir,
        "--detector", detector,
        "--out", out,
        "--methods", *methods,
        "--verbose", verbose,
        "--edge-radius", edge_radius,
        "--edge-canny-low", edge_canny_low,
        "--edge-canny-high", edge_canny_high,
    ]
    if not progress:
        argv.append("--no-progress")
    if save_preds:
        argv.append("--save-preds")
    if save_segm:
        argv.append("--save-segm")
    if decode:
        argv += ["--decode", str(decode)]
    if collage_all:
        argv.append("--collage-all")
    if detector == "yolo":
        if weights:
            argv += ["--weights", str(weights)]
        argv += ["--device", str(device), "--imgsz", imgsz, "--conf", conf, "--iou", iou, "--max-det", max_det]
        if tta_rot:
            argv.append("--tta-rot")
    if detector == "barseg":
        if weights:
            argv += ["--weights", str(weights)]
        argv += [
            "--device", str(device),
            "--imgsz", imgsz,
            "--conf", conf,
            "--iou", iou,
            "--max-det", max_det,
            "--sam2-checkpoint", str(sam2_checkpoint),
            "--sam2-config", str(sam2_config),
        ]
    sys.argv = argv
    detector_eval_main()


if __name__ == "__main__":
    main()
