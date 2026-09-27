"""Run an exported pedestrian detector on point clouds (inside WSL, `pcdet` env).

    python training/infer.py --ckpt model.pth --points a.npy b.bin ...     [--out preds.json] [--score 0.3]
    python training/infer.py --ckpt model.pth --dir <folder of .npy/.bin> --out preds.json
    python training/infer.py --ckpt model.pth --data <dataset dir> --ids r02_000010,r02_000050 --out preds.json

Points must be in the model's frame: the LiDAR's own frame leveled (roll and pitch removed, z up, origin at the
sensor), N x 4 (x, y, z, intensity) or N x 3; .npy, or .bin float32 N x 4 (KITTI style). The config the model was
trained with travels inside the checkpoint ("config"). Output: {frame: {"boxes": [[x, y, z, dx, dy, dz, yaw]],
"scores": [...]}}, printed as one JSON line per frame and optionally saved to --out.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import PCDET, CustomDataset, build_network, cfg, cfg_from_yaml_file, common_utils, load_data_to_gpu  # noqa: E402,I001

import numpy as np  # noqa: E402
import torch  # noqa: E402


def load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        pts = np.load(path).astype(np.float32)
    else:
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    if pts.shape[1] == 3:
        pts = np.column_stack([pts, np.zeros(len(pts), np.float32)])
    return pts[:, :4]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--points", nargs="*", default=[])
    ap.add_argument("--dir", help="run on every .npy / .bin in this folder, in name order")
    ap.add_argument("--data")
    ap.add_argument("--ids", default="")
    ap.add_argument("--out")
    ap.add_argument("--score", type=float, default=0.3)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(state["config"])
    cfg_from_yaml_file(f.name, cfg)
    cfg.ROOT_DIR = PCDET
    cfg.DATA_CONFIG.INFO_PATH = {"train": [], "test": []}
    logger = common_utils.create_logger()
    root = Path(args.data) if args.data else Path(tempfile.gettempdir())
    dataset = CustomDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, root_path=root,
                            training=False, logger=logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_state_dict(state["model_state"])
    model.cuda().eval()

    frames = [(Path(p).stem, Path(p)) for p in args.points]
    if args.dir:
        frames += [(f.stem, f) for f in sorted(Path(args.dir).iterdir()) if f.suffix in (".npy", ".bin")]
    frames += [(i, Path(args.data) / "points" / f"{i}.npy") for i in args.ids.split(",") if i]
    results = {}
    for name, path in frames:
        batch = dataset.collate_batch([dataset.prepare_data({"points": load_points(path), "frame_id": name})])
        load_data_to_gpu(batch)
        pred = model(batch)[0][0]
        keep = pred["pred_scores"] >= args.score
        results[name] = {"boxes": pred["pred_boxes"][keep].cpu().numpy().round(4).tolist(),
                         "scores": pred["pred_scores"][keep].cpu().numpy().round(4).tolist()}
        print(json.dumps({"frame": name, "detections": len(results[name]["scores"])}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
