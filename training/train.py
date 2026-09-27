"""PointPillars pedestrian training on an exported dataset (runs inside WSL, in the `pcdet` env from setup_wsl.sh).

    micromamba run -n pcdet python training/train.py --data <dataset dir> --out <train dir>
        [--epochs 10] [--batch 4] [--pretrained ~/pcdet_ckpt/pointpillar_7728.pth]
        [--max-train N] [--max-val N] [--noise-sigma 0.02] [--drop-prob 0.15] [--occlusion]

The dataset is OpenPCDet's custom layout (points/<id>.npy N x 4, labels/<id>.txt, ImageSets/{train,val}.txt),
as written by the pipeline's Filter & dataset stage; dataset.json next to it gives the point range and the
floor height in the training frame. Without dataset.json, pass --range-xy and --floor-z.

Writes into --out:
    metrics.jsonl   one JSON object per line: {"type": "iter", epoch, iter, iters_per_epoch, loss, lr, ...},
                    {"type": "eval", epoch, ap, recall, precision, n_gt}, {"type": "done", ...}
    ckpt/epoch_N.pth, ckpt/best.pth, pointpillar_ped.yaml (the full config)
Only the Pedestrian class trains. Pedestrian_ignore boxes (too few points) are dropped by OpenPCDet's class
filter, i.e. treated as background, and are not counted in the evaluation.
"""
import argparse
from functools import partial
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

PCDET = Path.home() / "OpenPCDet"
sys.path.insert(0, str(PCDET / "tools"))

import types  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# pcdet.datasets imports every dataset module; Argoverse 2's needs av2/kornia/pandas, which this setup doesn't
# install (only CustomDataset is used). Pre-register a stub for that one module.
for _name in ("pcdet.datasets.argo2", "pcdet.datasets.argo2.argo2_dataset"):
    sys.modules[_name] = types.ModuleType(_name)
sys.modules["pcdet.datasets.argo2.argo2_dataset"].Argo2Dataset = None
import yaml  # noqa: E402
from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor  # noqa: E402
from pcdet.datasets.custom.custom_dataset import CustomDataset  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.ops.iou3d_nms import iou3d_nms_utils  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
from train_utils.optimization import build_optimizer, build_scheduler  # noqa: E402



# Sensor-noise augmentations OpenPCDet lacks, registered on its DataAugmentor so config entries can name them
# (it looks augmentors up with getattr(self, NAME)). Training frame origin = the sensor, so rays start at 0.
def random_point_dropout(self, data_dict=None, config=None):
    """Drop each point with probability DROP_PROB (real returns that never come back)."""
    if data_dict is None:
        return partial(random_point_dropout, self, config=config)
    pts = data_dict["points"]
    data_dict["points"] = pts[np.random.rand(len(pts)) >= config["DROP_PROB"]]
    return data_dict


def random_range_noise(self, data_dict=None, config=None):
    """Gaussian range error along each ray: p *= 1 + N(0, SIGMA) / |p| (direction unchanged)."""
    if data_dict is None:
        return partial(random_range_noise, self, config=config)
    pts = data_dict["points"]
    r = np.linalg.norm(pts[:, :3], axis=1, keepdims=True)
    pts[:, :3] *= 1.0 + np.random.normal(0.0, config["SIGMA"], (len(pts), 1)) / np.maximum(r, 1e-3)
    return data_dict


if "bool" not in np.__dict__:  # OpenPCDet's pyramid augmentation still uses np.bool, removed in NumPy 1.24
    np.bool = bool
DataAugmentor.random_point_dropout = random_point_dropout
DataAugmentor.random_range_noise = random_range_noise


def noise_aug_yaml(noise_sigma, drop_prob, occlusion):
    """AUG_CONFIG_LIST entries for the sensor-noise options (applied before flip / rotation / scaling)."""
    out = ""
    if noise_sigma > 0:
        out += f"""
            - NAME: random_range_noise
              SIGMA: {noise_sigma}"""
    if drop_prob > 0:
        out += f"""
            - NAME: random_point_dropout
              DROP_PROB: {drop_prob}"""
    if occlusion:  # built-in (SE-SSD); only its per-box pyramid dropout is on
        out += """
            - NAME: random_local_pyramid_aug
              DROP_PROB: 0.25
              SPARSIFY_PROB: 0.0
              SPARSIFY_MAX_NUM: 50
              SWAP_PROB: 0.0
              SWAP_MAX_NUM: 50"""
    return out


PED = {"size": [0.6, 0.5, 1.7]}  # anchor l, w, h (IRA characters measured ~0.57 x 0.39 x 1.7-1.86)


def config_text(range_xy, floor_z, epochs, batch, noise_sigma=0.0, drop_prob=0.0, occlusion=False):
    z0 = floor_z - 0.5
    pcr = [-range_xy, -range_xy, round(z0, 3), range_xy, range_xy, round(z0 + 4.0, 3)]
    return f"""
CLASS_NAMES: ['Pedestrian']

DATA_CONFIG:
    DATASET: 'CustomDataset'
    POINT_CLOUD_RANGE: {pcr}
    MAP_CLASS_TO_KITTI: {{'Pedestrian': 'Pedestrian'}}
    DATA_SPLIT: {{'train': train, 'test': val}}
    INFO_PATH: {{'train': [custom_infos_train.pkl], 'test': [custom_infos_val.pkl]}}
    # Sim intensity does not match real sensors: train on geometry only.
    POINT_FEATURE_ENCODING: {{
        encoding_type: absolute_coordinates_encoding,
        used_feature_list: ['x', 'y', 'z'],
        src_feature_list: ['x', 'y', 'z', 'intensity'],
    }}
    DATA_AUGMENTOR:
        DISABLE_AUG_LIST: ['placeholder']
        AUG_CONFIG_LIST:{noise_aug_yaml(noise_sigma, drop_prob, occlusion)}
            - NAME: random_world_flip
              ALONG_AXIS_LIST: ['x', 'y']
            - NAME: random_world_rotation
              WORLD_ROT_ANGLE: [-3.14159265, 3.14159265]
            - NAME: random_world_scaling
              WORLD_SCALE_RANGE: [0.95, 1.05]
    DATA_PROCESSOR:
        - NAME: mask_points_and_boxes_outside_range
          REMOVE_OUTSIDE_BOXES: True
        - NAME: shuffle_points
          SHUFFLE_ENABLED: {{'train': True, 'test': False}}
        - NAME: transform_points_to_voxels
          VOXEL_SIZE: [0.16, 0.16, 4]
          MAX_POINTS_PER_VOXEL: 32
          MAX_NUMBER_OF_VOXELS: {{'train': 16000, 'test': 40000}}

MODEL:
    NAME: PointPillar
    VFE:
        NAME: PillarVFE
        WITH_DISTANCE: False
        USE_ABSLOTE_XYZ: True
        USE_NORM: True
        NUM_FILTERS: [64]
    MAP_TO_BEV:
        NAME: PointPillarScatter
        NUM_BEV_FEATURES: 64
    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [3, 5, 5]
        LAYER_STRIDES: [2, 2, 2]
        NUM_FILTERS: [64, 128, 256]
        UPSAMPLE_STRIDES: [1, 2, 4]
        NUM_UPSAMPLE_FILTERS: [128, 128, 128]
    DENSE_HEAD:
        NAME: AnchorHeadSingle
        CLASS_AGNOSTIC: False
        USE_DIRECTION_CLASSIFIER: True
        DIR_OFFSET: 0.78539
        DIR_LIMIT_OFFSET: 0.0
        NUM_DIR_BINS: 2
        ANCHOR_GENERATOR_CONFIG: [
            {{
                'class_name': 'Pedestrian',
                'anchor_sizes': [{PED['size']}],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [{round(floor_z, 3)}],
                'align_center': False,
                'feature_map_stride': 2,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }}
        ]
        TARGET_ASSIGNER_CONFIG:
            NAME: AxisAlignedTargetAssigner
            POS_FRACTION: -1.0
            SAMPLE_SIZE: 512
            NORM_BY_NUM_EXAMPLES: False
            MATCH_HEIGHT: False
            BOX_CODER: ResidualCoder
        LOSS_CONFIG:
            LOSS_WEIGHTS: {{
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'dir_weight': 0.2,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }}
    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        SCORE_THRESH: 0.1
        OUTPUT_RAW_SCORE: False
        EVAL_METRIC: kitti
        NMS_CONFIG:
            MULTI_CLASSES_NMS: False
            NMS_TYPE: nms_gpu
            NMS_THRESH: 0.01
            NMS_PRE_MAXSIZE: 4096
            NMS_POST_MAXSIZE: 500

OPTIMIZATION:
    BATCH_SIZE_PER_GPU: {batch}
    NUM_EPOCHS: {epochs}
    OPTIMIZER: adam_onecycle
    LR: 0.003
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9
    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001
    LR_WARMUP: False
    WARMUP_EPOCH: 1
    GRAD_NORM_CLIP: 10
"""


def read_ids(data: Path, split: str, cap: int | None, seed=0):
    ids = [x.strip() for x in (data / "ImageSets" / f"{split}.txt").read_text().split() if x.strip()]
    if cap and len(ids) > cap:
        ids = sorted(random.Random(seed).sample(ids, cap))
    return ids


def write_infos(data: Path, out: Path, split: str, ids, logger):
    """OpenPCDet info pickle for the chosen frames (kept in --out so the dataset folder is never modified)."""
    infos = []
    for sid in ids:
        boxes, names = [], []
        for line in (data / "labels" / f"{sid}.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) == 8:
                boxes.append([float(v) for v in parts[:7]])
                names.append(parts[7])
        infos.append({"point_cloud": {"num_features": 4, "lidar_idx": sid},
                      "annos": {"name": np.array(names), "gt_boxes_lidar": np.array(boxes, np.float32).reshape(-1, 7)}})
    path = out / f"custom_infos_{split}.pkl"
    with open(path, "wb") as f:
        pickle.dump(infos, f)
    logger.info(f"{split}: {len(infos)} frames")
    return path


def average_precision(scores, tps, n_gt):
    """KITTI-style 40-point interpolated AP."""
    if n_gt == 0:
        return float("nan"), float("nan"), float("nan")
    if not scores:
        return 0.0, 0.0, 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, np.float64)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    ap = np.mean([precision[recall >= r].max() if (recall >= r).any() else 0.0 for r in np.linspace(1 / 40, 1, 40)])
    return float(ap), float(recall[-1]), float(precision[-1])


@torch.no_grad()
def evaluate(model, loader, iou_thr=0.5, score_thr=0.3):
    model.eval()
    scores, tps, n_gt = [], [], 0
    for batch in loader:
        load_data_to_gpu(batch)
        preds, _ = model(batch)
        for b, pred in enumerate(preds):
            gt = batch["gt_boxes"][b]
            gt = gt[gt[:, 3] > 0][:, :7] if gt.shape[0] else gt[:, :7]
            n_gt += int(gt.shape[0])
            pb, ps = pred["pred_boxes"], pred["pred_scores"]
            if pb.shape[0] == 0:
                continue
            order = torch.argsort(ps, descending=True)
            pb, ps = pb[order], ps[order]
            iou = iou3d_nms_utils.boxes_iou3d_gpu(pb[:, :7], gt) if gt.shape[0] else None
            used = set()
            for i in range(pb.shape[0]):
                hit = False
                if iou is not None:
                    j = int(torch.argmax(iou[i]))
                    if float(iou[i, j]) >= iou_thr and j not in used:
                        used.add(j)
                        hit = True
                scores.append(float(ps[i]))
                tps.append(1.0 if hit else 0.0)
    model.train()
    ap, recall, precision = average_precision(scores, tps, n_gt)
    # recall / precision at the operating point score >= score_thr
    s, t = np.asarray(scores), np.asarray(tps)
    keep = s >= score_thr
    op_recall = float(t[keep].sum() / n_gt) if n_gt else float("nan")
    op_precision = float(t[keep].mean()) if keep.any() else float("nan")
    return {"ap": ap, "recall": op_recall, "precision": op_precision, "max_recall": recall, "n_gt": n_gt,
            "score_thr": score_thr, "iou_thr": iou_thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pretrained", default=str(Path.home() / "pcdet_ckpt" / "pointpillar_7728.pth"))
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=500)
    ap.add_argument("--range-xy", type=float, default=None)
    ap.add_argument("--floor-z", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--noise-sigma", type=float, default=0.0, help="range noise std (m), training only")
    ap.add_argument("--drop-prob", type=float, default=0.0, help="random point dropout probability, training only")
    ap.add_argument("--occlusion", action="store_true", help="drop parts of pedestrians (pyramid dropout)")
    args = ap.parse_args()

    data, out = Path(args.data).resolve(), Path(args.out).resolve()
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "dataset.json").read_text()) if (data / "dataset.json").exists() else {}
    range_xy = args.range_xy or meta.get("range_xy")
    floor_z = args.floor_z if args.floor_z is not None else meta.get("floor_z")
    if range_xy is None or floor_z is None:
        sys.exit("need dataset.json or --range-xy/--floor-z")
    metrics = open(out / "metrics.jsonl", "w", buffering=1)

    def emit(**kw):
        metrics.write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)

    cfg_file = out / "pointpillar_ped.yaml"
    cfg_file.write_text(config_text(float(range_xy), float(floor_z), args.epochs, args.batch,
                                   args.noise_sigma, args.drop_prob, args.occlusion))
    cfg_from_yaml_file(str(cfg_file), cfg)
    cfg.ROOT_DIR = PCDET
    logger = common_utils.create_logger(out / "train.log")

    train_ids = read_ids(data, "train", args.max_train)
    val_ids = read_ids(data, "val", args.max_val)
    cfg.DATA_CONFIG.INFO_PATH = {"train": [str(write_infos(data, out, "train", train_ids, logger))],
                                 "test": [str(write_infos(data, out, "val", val_ids, logger))]}

    common_utils.set_random_seed(0)
    train_set, train_loader, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.batch, False, root_path=data,
                                                  workers=args.workers, logger=logger, training=True,
                                                  total_epochs=args.epochs)
    val_set, val_loader, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.batch, False, root_path=data,
                                              workers=args.workers, logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.pretrained and Path(args.pretrained).exists():
        model.load_params_from_file(args.pretrained, logger, to_cpu=True)
    model.cuda().train()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    iters = len(train_loader)
    scheduler, _ = build_scheduler(optimizer, iters, args.epochs, -1, cfg.OPTIMIZATION)
    emit(type="start", augmentation={"noise_sigma": args.noise_sigma, "drop_prob": args.drop_prob,
                                     "occlusion": args.occlusion}, epochs=args.epochs, iters_per_epoch=iters, train_frames=len(train_ids), val_frames=len(val_ids),
         range_xy=range_xy, floor_z=floor_z, pretrained=bool(args.pretrained and Path(args.pretrained).exists()))

    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(args.epochs):
        for it, batch in enumerate(train_loader):
            scheduler.step(step)
            load_data_to_gpu(batch)
            optimizer.zero_grad()
            ret, tb, _ = model(batch)
            loss = ret["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            optimizer.step()
            step += 1
            if it % args.log_every == 0 or it == iters - 1:
                emit(type="iter", epoch=epoch, iter=it, iters_per_epoch=iters, step=step, loss=float(loss),
                     loss_cls=float(tb.get("rpn_loss_cls", 0)), loss_loc=float(tb.get("rpn_loss_loc", 0)),
                     lr=float(optimizer.lr if hasattr(optimizer, "lr") else optimizer.param_groups[0]["lr"]),
                     elapsed=round(time.time() - t0, 1))
            if (out / "STOP").exists():
                break
        ev = evaluate(model, val_loader)
        emit(type="eval", epoch=epoch, **ev)
        state = {"epoch": epoch + 1, "model_state": model.state_dict(), "config": cfg_file.read_text(), "eval": ev}
        torch.save(state, out / "ckpt" / f"epoch_{epoch + 1}.pth")
        if not np.isnan(ev["ap"]) and ev["ap"] >= best:
            best = ev["ap"]
            torch.save(state, out / "ckpt" / "best.pth")
        if (out / "STOP").exists():
            emit(type="stopped", epoch=epoch)
            break
    emit(type="done", best_ap=best, elapsed=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
