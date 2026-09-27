"""Run the stock KITTI PointPillars (no fine-tuning) on a folder of scans, for comparison (inside WSL, `pcdet` env).

    python training/infer_kitti.py --dir <folder of .npy> --out preds.json [--z-shift -1.12] [--score 0.1]

KITTI's model only looks forward (x in [0, 69 m]), so each scan is run twice, as is and turned 180° about z, and
the boxes are merged back into the scan's frame. KITTI's LiDAR sat 1.73 m above the road: --z-shift moves the scan
(and back again for the boxes) so its floor lands where KITTI's was. Intensity is scaled to 0-1 as KITTI's is.
Output matches infer.py, plus "labels" (Car / Pedestrian / Cyclist) per box.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import PCDET, build_network, cfg, cfg_from_yaml_file, common_utils, load_data_to_gpu  # noqa: E402,I001

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(Path.home() / "pcdet_ckpt" / "pointpillar_7728.pth"))
    ap.add_argument("--z-shift", type=float, default=0.0)
    ap.add_argument("--score", type=float, default=0.1)
    args = ap.parse_args()

    os.chdir(PCDET / "tools")  # the model yaml's _BASE_CONFIG_ path is relative to tools/
    cfg_from_yaml_file("cfgs/kitti_models/pointpillar.yaml", cfg)
    logger = common_utils.create_logger()
    dataset = DatasetTemplate(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
                              root_path=Path(args.dir), logger=logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(args.ckpt, logger, to_cpu=True)
    model.cuda().eval()

    files = sorted(f for f in Path(args.dir).iterdir() if f.suffix == ".npy")
    results = {}
    for f in files:
        pts = np.load(f).astype(np.float32)
        if pts.shape[1] == 3:
            pts = np.column_stack([pts, np.zeros(len(pts), np.float32)])
        pts = pts[:, :4].copy()
        if pts[:, 3].max() > 1.0:
            pts[:, 3] /= 255.0
        pts[:, 2] += args.z_shift
        boxes, scores, labels = [], [], []
        for flip in (1.0, -1.0):  # front half, then the back half turned to face forward
            p = pts.copy()
            p[:, :2] *= flip
            batch = dataset.collate_batch([dataset.prepare_data({"points": p, "frame_id": f.stem})])
            load_data_to_gpu(batch)
            pred = model(batch)[0][0]
            keep = (pred["pred_scores"] >= args.score).cpu().numpy()
            b = pred["pred_boxes"].cpu().numpy()[keep]
            b[:, :2] *= flip
            if flip < 0:
                b[:, 6] += np.pi
            b[:, 2] -= args.z_shift
            boxes += b.round(4).tolist()
            scores += pred["pred_scores"].cpu().numpy()[keep].round(4).tolist()
            labels += [cfg.CLASS_NAMES[i - 1] for i in pred["pred_labels"].cpu().numpy()[keep]]
        results[f.stem] = {"boxes": boxes, "scores": scores, "labels": labels}
        print(json.dumps({"frame": f.stem, "detections": len(scores)}), flush=True)
    Path(args.out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
