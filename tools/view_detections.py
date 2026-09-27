"""Standalone viewer: real LiDAR frames + the boxes the model predicted on them (Open3D window).

    .venv\\Scripts\\python tools\\view_detections.py <folder of .pcd> <preds.json> [--score 0.3]

Boxes: green Pedestrian, orange Cyclist, blue Car (the stock KITTI model's classes; infer.py's are all Pedestrian).
Keys: Right / Left = next / previous frame, Space = play / pause, Up / Down = min score +/- 0.05,
      mouse = orbit / pan / zoom. Frame info prints in the terminal.
preds.json is infer.py's output: {frame name: {"boxes": [[x, y, z, dx, dy, dz, yaw]], "scores": [...]}}, where the
frame name is the file name without extension.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

CLASS_COLORS = {"Pedestrian": [0.2, 1.0, 0.3], "Cyclist": [1.0, 0.6, 0.1], "Car": [0.3, 0.6, 1.0]}
EDGES = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


def box_lines(boxes, colors):
    pts = []
    for b in boxes:
        c, e, yaw = np.array(b[:3]), np.array(b[3:6]) / 2, b[6]
        local = np.array([[sx * e[0], sy * e[1], sz * e[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        pts.append(local @ R.T + c)
    ls = o3d.geometry.LineSet()
    if pts:
        ls.points = o3d.utility.Vector3dVector(np.concatenate(pts))
        ls.lines = o3d.utility.Vector2iVector(np.concatenate([np.array(EDGES) + 8 * k for k in range(len(pts))]))
        ls.colors = o3d.utility.Vector3dVector(np.repeat(np.array(colors, float), 12, axis=0))
    return ls


def load_points(f: Path):
    pc = o3d.io.read_point_cloud(str(f), remove_nan_points=True, remove_infinite_points=True)
    xyz = np.asarray(pc.points)
    return xyz[np.abs(xyz).sum(1) > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("preds")
    ap.add_argument("--score", type=float, default=0.3)
    args = ap.parse_args()

    files = sorted(f for f in Path(args.folder).iterdir() if f.suffix.lower() in (".pcd", ".ply"))
    preds = json.loads(Path(args.preds).read_text())
    st = {"i": 0, "score": args.score, "play": False, "last": 0.0, "first": True}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Pedestrian detections", 1400, 900)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.1])
    opt.point_size = 2.0
    opt.line_width = 3.0
    cloud, boxes = o3d.geometry.PointCloud(), o3d.geometry.LineSet()

    def show():
        f = files[st["i"]]
        xyz = load_points(f)
        z = xyz[:, 2]
        h = np.clip((z - np.percentile(z, 1)) / max(1e-6, np.percentile(z, 99) - np.percentile(z, 1)), 0, 1)
        cloud.points = o3d.utility.Vector3dVector(xyz)
        cloud.colors = o3d.utility.Vector3dVector(np.column_stack([0.3 + 0.5 * h, 0.4 + 0.3 * h, 0.9 - 0.6 * h]))
        p = preds.get(f.stem, {"boxes": [], "scores": []})
        keep = [k for k, s in enumerate(p["scores"]) if s >= st["score"]]
        labels = p.get("labels") or ["Pedestrian"] * len(p["scores"])
        new = box_lines([p["boxes"][k] for k in keep], [CLASS_COLORS.get(labels[k], [1, 1, 1]) for k in keep])
        boxes.points, boxes.lines, boxes.colors = new.points, new.lines, new.colors
        if st["first"]:
            vis.add_geometry(cloud)
            vis.add_geometry(boxes)
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(0.5))  # the sensor (x red, y green)
            st["first"] = False
        else:
            vis.update_geometry(cloud)
            vis.update_geometry(boxes)
        found = ", ".join(f"{labels[k]} {p['scores'][k]:.2f}" for k in keep) or "-"
        print(f"frame {f.stem} ({st['i'] + 1}/{len(files)})  min score {st['score']:.2f}  "
              f"{len(keep)} box(es): {found}", flush=True)

    def step(d):
        st["i"] = (st["i"] + d) % len(files)
        show()
        return False

    def score(d):
        st["score"] = float(np.clip(st["score"] + d, 0.0, 0.95))
        show()
        return False

    def toggle(_):
        st["play"] = not st["play"]
        return False

    vis.register_key_callback(262, lambda _: step(1))      # Right
    vis.register_key_callback(263, lambda _: step(-1))     # Left
    vis.register_key_callback(265, lambda _: score(0.05))  # Up
    vis.register_key_callback(264, lambda _: score(-0.05))  # Down
    vis.register_key_callback(32, toggle)                  # Space

    show()
    print("Right/Left: frame   Space: play   Up/Down: min score", flush=True)
    while vis.poll_events():
        if st["play"] and time.time() - st["last"] > 0.1:
            st["last"] = time.time()
            step(1)
        vis.update_renderer()
    vis.destroy_window()


if __name__ == "__main__":
    main()
