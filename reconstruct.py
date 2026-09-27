"""Run VGGT-Omega on a folder of images and write a GLB (point cloud + cameras), same format as the HF demo.

Usage:
    .venv/Scripts/python reconstruct.py data/hallway_photos
    .venv/Scripts/python reconstruct.py data/hallway_photos --out data/hallway.glb --conf-thres 30
    .venv/Scripts/python glb_to_mesh.py data/hallway.glb --out out/hallway      # next step

The checkpoint is gated: run `hf auth login` once with an account that has access to facebook/VGGT-Omega.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent / "vggt-omega"
sys.path.insert(0, str(REPO))

from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402
from visual_util import predictions_to_glb  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}
CHECKPOINTS = {512: "vggt_omega_1b_512.pt", 416: "vggt_omega_1b_416_reproduce.pt"}


def load_model(checkpoint, resolution, bf16_weights):
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download("facebook/VGGT-Omega", CHECKPOINTS[resolution])
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    if bf16_weights:
        # The aggregator already runs under bf16 autocast, so storing its weights in bf16 saves ~2 GB
        # of VRAM. Its outputs are cast back to fp32 because the heads run with autocast disabled.
        model.aggregator.to(torch.bfloat16)
        forward = model.aggregator.forward

        def aggregator_fp32_out(*a, **kw):
            tokens, start = forward(*a, **kw)
            return [t.float() if t is not None else None for t in tokens], start

        model.aggregator.forward = aggregator_fp32_out
    return model.to("cuda")


def unproject(depth, extrinsic, intrinsic):
    """Depth maps (S,H,W,1) + world-to-camera extrinsics (S,3,4) -> world points (S,H,W,3)."""
    depth = depth[..., 0]
    s, h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx, fy = intrinsic[:, 0, 0, None, None], intrinsic[:, 1, 1, None, None]
    cx, cy = intrinsic[:, 0, 2, None, None], intrinsic[:, 1, 2, None, None]
    cam = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    R, t = extrinsic[:, :3, :3], extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", R.transpose(0, 2, 1), cam - t[:, None, None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--out", default=None, help="output .glb (default: data/<image_dir name>.glb)")
    ap.add_argument("--checkpoint", default=None, help="local .pt (default: download from Hugging Face)")
    ap.add_argument("--resolution", type=int, default=512, choices=sorted(CHECKPOINTS))
    ap.add_argument("--mode", default="balanced", choices=["balanced", "max_size"],
                    help="max_size uses smaller inputs and less VRAM")
    ap.add_argument("--conf-thres", type=float, default=25.0, help="drop this percentile of lowest-confidence points")
    ap.add_argument("--max-points", type=int, default=4_000_000)
    ap.add_argument("--bf16-weights", choices=["auto", "on", "off"], default="auto",
                    help="store aggregator weights in bf16 (auto: on if GPU has < 12 GB)")
    ap.add_argument("--no-cams", action="store_true", help="omit camera frustums from the GLB")
    ap.add_argument("--save-npz", action="store_true", help="also save raw predictions (depth, conf, cameras)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available. Run with the project venv: .venv/Scripts/python reconstruct.py ...")

    image_dir = Path(args.image_dir)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {image_dir}")
    out = Path(args.out) if args.out else Path("data") / f"{image_dir.name}.glb"
    out.parent.mkdir(parents=True, exist_ok=True)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16 = args.bf16_weights == "on" or (args.bf16_weights == "auto" and vram_gb < 12)
    print(f"{len(images)} images | GPU {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB) | bf16 weights: {bf16}")

    t0 = time.time()
    model = load_model(args.checkpoint, args.resolution, bf16)
    print(f"model loaded in {time.time() - t0:.1f}s")

    batch = load_and_preprocess_images([str(p) for p in images], mode=args.mode,
                                       image_resolution=args.resolution).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        pred = model(batch)
    extrinsic, intrinsic = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    pred["extrinsic"], pred["intrinsic"] = extrinsic, intrinsic
    print(f"inference {time.time() - t0:.1f}s on {tuple(batch.shape)} | peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    pred_np = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().cpu().numpy()
            pred_np[k] = v[0] if v.shape[0] == 1 else v
    del model, pred, batch
    torch.cuda.empty_cache()
    pred_np["world_points_from_depth"] = unproject(pred_np["depth"], pred_np["extrinsic"], pred_np["intrinsic"])

    if args.save_npz:
        np.savez(out.with_suffix(".npz"), **pred_np)

    scene = predictions_to_glb(pred_np, conf_thres=args.conf_thres, show_cam=not args.no_cams,
                               max_points=args.max_points)
    scene.export(out)
    n_pts = sum(len(g.vertices) for g in scene.geometry.values() if g.__class__.__name__ == "PointCloud")
    print(f"wrote {out} ({n_pts:,} points)")


if __name__ == "__main__":
    main()
