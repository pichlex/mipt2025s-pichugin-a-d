from __future__ import annotations

import argparse
import json
import os
import time
import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..detectors import get_detector
from ..refine import refine_corners, CornerRefinerConfig
from ..geometry import quad_iou, draw_quad, draw_legend, warp_quad
from ..logging_utils import setup_logging, get_progress


def _load_markup(path: str) -> List[np.ndarray]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    quads = []
    for o in data.get('objects', []):
        if o.get('type') != 'quad':
            continue
        pts = np.asarray(o.get('data'), dtype=np.float32)
        if pts.shape == (4, 2):
            quads.append(pts)
    return quads


def _match_greedy(preds: List[np.ndarray], gts: List[np.ndarray]) -> List[Tuple[int, int, float]]:
    # returns list of (pi, gi, iou)
    matches: List[Tuple[int, int, float]] = []
    used_p = set()
    used_g = set()
    # build all pairs
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou = quad_iou(p, g)
            pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        if iou <= 0:
            continue
        matches.append((i, j, iou))
        used_p.add(i)
        used_g.add(j)
    return matches


def _decode_success(img: np.ndarray, decoder: str = "qr") -> bool:
    d = (decoder or "qr").lower()
    if d == "none":
        return False
    if d == "pyzbar":
        try:
            from pyzbar.pyzbar import decode as zbar_decode  # type: ignore

            res = zbar_decode(img)
            return bool(res)
        except Exception:
            return False
    # default: OpenCV QR
    try:
        qr = cv2.QRCodeDetector()
        val, _, _ = qr.detectAndDecode(img)
        return bool(val)
    except Exception:
        return False


def _save_masks_overlay(img: np.ndarray, masks: List[np.ndarray], out_path: str) -> None:
    if not masks:
        return
    ov = img.copy()
    color = (0, 255, 255)
    alpha = 0.5
    for m in masks:
        m8 = (m.astype(np.uint8) * 255)
        col = np.zeros_like(ov)
        col[:, :] = color
        ov = np.where(m8[..., None] > 0, (alpha * col + (1 - alpha) * ov).astype(np.uint8), ov)
        # draw contour edge for clarity
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 10:
                continue
            cv2.drawContours(ov, [c], -1, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, ov)


def _put_label(img: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Draws a small opaque label at top-left of the image."""
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    x1, y1 = 8, 8
    x2, y2 = x1 + tw + 2 * pad, y1 + th + baseline + 2 * pad
    # background box for readability
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    # label text
    cv2.putText(vis, text, (x1 + pad, y2 - baseline - pad), font, scale, color, thickness, cv2.LINE_AA)
    return vis


def _collage_quads(img: np.ndarray, gt: np.ndarray | None, init: np.ndarray | None, refined: np.ndarray | None) -> np.ndarray:
    """Build a horizontal collage with three panels: GT, Segmented, Refined.

    Each panel shows only its respective quad with a colored label:
    - GT: blue (255, 0, 0)
    - Segmented: red (0, 0, 255)
    - Refined: green (0, 255, 0)
    """
    # Prepare separate visualizations
    gt_vis = img.copy()
    if gt is not None:
        gt_vis = draw_quad(gt_vis, gt, (255, 0, 0), 2)
        gt_vis = _put_label(gt_vis, "gt", (255, 0, 0))
    else:
        gt_vis = _put_label(gt_vis, "gt: none", (255, 0, 0))

    init_vis = img.copy()
    if init is not None:
        init_vis = draw_quad(init_vis, init, (0, 0, 255), 2)
        init_vis = _put_label(init_vis, "segmented", (0, 0, 255))
    else:
        init_vis = _put_label(init_vis, "segmented: none", (0, 0, 255))

    ref_vis = img.copy()
    if refined is not None:
        ref_vis = draw_quad(ref_vis, refined, (0, 255, 0), 2)
        ref_vis = _put_label(ref_vis, "refined", (0, 255, 0))
    else:
        ref_vis = _put_label(ref_vis, "refined: none", (0, 255, 0))

    # Add small padding between panels for readability
    pad = 8
    h = img.shape[0]
    sep = np.full((h, pad, 3), 32, dtype=np.uint8)
    collage = np.hstack([gt_vis, sep, init_vis, sep, ref_vis])
    return collage


def main():
    ap = argparse.ArgumentParser(description="Evaluate detector -> (optional) refiner against GT markup")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--markup_dir", required=True)
    ap.add_argument("--detector", default="yolo", choices=["yolo", "auto", "barseg"], help="Detector backend")
    ap.add_argument("--weights", default="models/detector_yolo.pt", help="Path to YOLO weights (for detector=yolo)")
    ap.add_argument("--device", default="auto", help="Device: auto|mps|cuda|cpu")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO imgsz")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (YOLO)")
    ap.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    ap.add_argument("--tta-rot", action="store_true", help="Enable rotation TTA (0/90/180/270)")
    ap.add_argument("--methods", nargs="*", default=["noop", "subpix", "edge"], help="Refine methods to compare")
    ap.add_argument("--edge-radius", type=int, default=4, help="Edge refiner: search radius (band half-width)")
    ap.add_argument("--edge-canny-low", type=int, default=50, help="Edge refiner: Canny low threshold")
    ap.add_argument("--edge-canny-high", type=int, default=150, help="Edge refiner: Canny high threshold")
    ap.add_argument("--out", default="results/detector_eval")
    ap.add_argument("--verbose", type=int, default=1, help="0=warn, 1=info, 2=debug")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    ap.add_argument("--save-preds", action="store_true", help="Save per-image detector predictions overlays")
    ap.add_argument("--save-segm", action="store_true", help="Save per-image segmentation overlay (if available)")
    ap.add_argument("--decode", default="qr", choices=["none", "qr", "pyzbar"], help="Decoder for success metric")
    # SAM2 (for barseg)
    ap.add_argument("--sam2-checkpoint", default="models/sam2.1_hiera_large.pt", help="Path to SAM2 checkpoint (barseg)")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to SAM2 config yaml (barseg)")
    ap.add_argument("--collage-all", action="store_true", help="Save at least one collage per image even without matches")
    args = ap.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    os.makedirs(args.out, exist_ok=True)
    logger.info(f"Detector: {args.detector}")
    det_kwargs = {}
    if args.detector == "yolo":
        det_kwargs = {"weights": args.weights, "device": args.device, "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "max_det": args.max_det, "tta_rot": args.tta_rot}
    elif args.detector == "barseg":
        det_kwargs = {
            "weights": args.weights,
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "sam2_checkpoint": args.sam2_checkpoint,
            "sam2_config": args.sam2_config,
        }
    det = get_detector(args.detector, **det_kwargs)
    if hasattr(det, "device"):
        logger.info(f"Detector device: {getattr(det, 'device')}")

    images = [f for f in os.listdir(args.images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    images.sort()
    logger.info(f"Found {len(images)} images in {args.images_dir}")

    # prepare output structure per method under detector name
    root = os.path.join(args.out, args.detector)
    os.makedirs(root, exist_ok=True)

    # We will compute summary per method
    all_results: Dict[str, Dict] = {}

    for method in get_progress(args.methods, desc="methods", total=len(args.methods), progress=not args.no_progress):
        run_dir = os.path.join(root, method)
        os.makedirs(run_dir, exist_ok=True)
        details = []
        t0 = time.perf_counter()
        logger.info(f"Start method='{method}' -> {run_dir}")

        for name in get_progress(images, desc=f"{method}", total=len(images), progress=not args.no_progress):
            stem, ext = os.path.splitext(name)
            img_path = os.path.join(args.images_dir, name)
            # Resolve markup path (if any). If markup is missing, proceed with empty GTs
            markup_path = os.path.join(args.markup_dir, stem + ext + ".json")
            if not os.path.exists(markup_path):
                alt = os.path.join(args.markup_dir, stem + ".json")
                if os.path.exists(alt):
                    markup_path = alt
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue
            # Load GT quads when markup is present; otherwise work with an empty list
            if os.path.exists(markup_path):
                gts = _load_markup(markup_path)
            else:
                gts = []
                logger.debug(f"No markup for image: {name}; proceeding without GT")
            preds = det.detect(img)
            logger.debug(f"{name}: preds={len(preds)} gts={len(gts)}")
            if args.save_preds:
                # save predicted quads overlay (yellow) for debugging
                ov = img.copy()
                for q in preds:
                    pts = q.astype(np.int32)
                    for i in range(4):
                        p1 = tuple(pts[i])
                        p2 = tuple(pts[(i + 1) % 4])
                        cv2.line(ov, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
                preds_dir = os.path.join(root, "preds")
                os.makedirs(preds_dir, exist_ok=True)
                # annotate count
                cv2.putText(ov, f"preds: {len(preds)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(preds_dir, f"{stem}.png"), ov)
            if args.save_segm and hasattr(det, "last_masks"):
                masks = getattr(det, "last_masks")
                if isinstance(masks, list) and len(masks) > 0:
                    segm_dir = os.path.join(root, "segm")
                    os.makedirs(segm_dir, exist_ok=True)
                    _save_masks_overlay(img, masks, os.path.join(segm_dir, f"{stem}.png"))

            saved_for_image = False
            if preds and gts:
                matches = _match_greedy(preds, gts)
                logger.debug(f"{name}: matches={len(matches)}")
                for mi, (pi, gi, iou_p) in enumerate(matches):
                    init = preds[pi].astype(np.float32)
                    gt = gts[gi].astype(np.float32)
                    cfg = CornerRefinerConfig(method=method, search_radius=args.edge_radius, canny_low=args.edge_canny_low, canny_high=args.edge_canny_high)
                    refined, meta = refine_corners(img, init, cfg)
                    iou_r = quad_iou(refined, gt)

                    # decode success improvement (optional, QR only)
                    before_patch = warp_quad(img, init)
                    after_patch = warp_quad(img, refined)
                    dec_before = _decode_success(before_patch, args.decode)
                    dec_after = _decode_success(after_patch, args.decode)

                    rec = {
                        "image": name,
                        "pair": mi,
                        "method": method,
                        "iou_before": float(iou_p),
                        "iou_after": float(iou_r),
                        "iou_gain": float(iou_r - iou_p),
                        "decoded_before": bool(dec_before),
                        "decoded_after": bool(dec_after),
                        "decoded_gain": int(bool(dec_after)) - int(bool(dec_before)),
                    }
                    details.append(rec)

                    # save collage visualization (GT | Init | Refined)
                    vis = _collage_quads(img, gt, init, refined)
                    cv2.imwrite(os.path.join(run_dir, f"{stem}_{mi:02d}.png"), vis)
                    saved_for_image = True

            # Ensure at least one collage per image if requested
            if args.collage_all and not saved_for_image:
                init_f = None
                gt_f = None
                refined_f = None
                if preds and gts:
                    # choose best IoU pair even if IoU<=0
                    best_iou = -1.0
                    best = None
                    for i, p in enumerate(preds):
                        for j, g in enumerate(gts):
                            v = quad_iou(p, g)
                            if v > best_iou:
                                best_iou = v
                                best = (i, j)
                    if best is not None:
                        pi, gi = best
                        init_f = preds[pi].astype(np.float32)
                        gt_f = gts[gi].astype(np.float32)
                elif preds:
                    init_f = preds[0].astype(np.float32)
                elif gts:
                    gt_f = gts[0].astype(np.float32)

                if init_f is not None:
                    cfg = CornerRefinerConfig(method=method, search_radius=args.edge_radius, canny_low=args.edge_canny_low, canny_high=args.edge_canny_high)
                    refined_f, _ = refine_corners(img, init_f, cfg)

                vis = _collage_quads(img, gt_f, init_f, refined_f)
                cv2.imwrite(os.path.join(run_dir, f"{stem}_00.png"), vis)

        # summary stats
        gains = [d["iou_gain"] for d in details]
        dec_gains = [d["decoded_gain"] for d in details]
        summary = {
            "count": len(details),
            "avg_iou_gain": float(np.mean(gains)) if gains else None,
            "median_iou_gain": float(np.median(gains)) if gains else None,
            "decoded_gain_total": int(np.sum(dec_gains)) if dec_gains else None,
            "decoded_improved": int(np.sum([1 for g in dec_gains if g > 0])) if dec_gains else None,
            "details": details,
        }
        with open(os.path.join(run_dir, "summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        all_results[method] = summary
        dt = time.perf_counter() - t0
        logger.info(f"Done method='{method}': {len(details)} matches in {dt:.2f}s (avg {dt / max(1, len(details)):.3f}s/item)")

    with open(os.path.join(root, "all_results.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
