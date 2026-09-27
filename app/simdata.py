"""Reading capture runs (sim/run_capture.py output) and judging each scan's labels.

The label handler and scan filter, in plain numpy, so the GUI can re-filter live as sliders move:
- every scan is analysed once (per-pedestrian point count, containment, box in the training frame) and
  cached in <run>/analysis.json;
- filtering is then cheap: evaluate(analysis, thresholds).

Frames: points and boxes are written in the lidar's sensor frame. The TRAINING frame is the sensor frame
leveled: sensor at the origin, z up, x along the sensor's heading (its forward axis projected onto the floor).
Only roll and pitch are removed, not the sensor's yaw in the reconstructed world, because on a real deployment
the mount's tilt is known but its yaw in our reconstruction is not. All 4x4s use USD's row-vector
convention (p_out = [p_in, 1] @ M).
"""
import json
import math
from pathlib import Path

import numpy as np

PEDESTRIAN_CLASS = 30
CONTAINMENT_TOL = 0.02  # metres

DEFAULTS = {"overlap": 0.15, "min_containment": 0.90, "min_points": 10, "min_label_points": 10}


def list_runs(project) -> list[Path]:
    root = project.path / "sim" / "runs"
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if (p / "run_info.json").exists() and (p / "velodyne").exists())


def num_scans(run_dir: Path) -> int:
    return len(list((run_dir / "velodyne").glob("*.bin")))


def run_info(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_info.json").read_text())


def sensor_to_world(run_dir: Path) -> np.ndarray:
    """Row-vector sensor -> world 4x4. run_info.json is written when a run ends; while it is still capturing,
    use the sensor.json the Simulate stage copied into the run folder (same pose Isaac was given)."""
    if (run_dir / "run_info.json").exists():
        return np.array(run_info(run_dir)["lidar_local_to_world_row_vector"], dtype=np.float64)
    return pose_matrix(json.loads((run_dir / "sensor.json").read_text()))


def pose_matrix(sensor: dict) -> np.ndarray:
    """Row-vector sensor -> world 4x4 from a sensor.json dict (position + wxyz), as Isaac builds it."""
    import viser.transforms as vtf
    M = np.eye(4)
    M[:3, :3] = vtf.SO3(np.asarray(sensor["wxyz"])).as_matrix().T
    M[3, :3] = sensor["position"]
    return M


def same_pose(a, b, pos_tol=0.01, deg_tol=0.5) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if np.linalg.norm(a[3, :3] - b[3, :3]) > pos_tol:
        return False
    cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos))) <= deg_tol


def training_rotation(sensor_to_world_row) -> np.ndarray:
    """4x4 (row-vector) sensor -> training frame: level, keeping the sensor's heading."""
    Rw = np.asarray(sensor_to_world_row, np.float64)[:3, :3]  # rows = sensor axes in world
    fwd = Rw[0].copy()
    if np.hypot(fwd[0], fwd[1]) < 0.3:  # forward axis near vertical (sideways mount): use the scan axis
        fwd = Rw[2].copy()
    yaw = np.arctan2(fwd[1], fwd[0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])  # row-vector rotation by -yaw
    M = np.eye(4)
    M[:3, :3] = Rw @ Rz
    return M


def train_rotation(run_dir: Path) -> np.ndarray:
    return training_rotation(sensor_to_world(run_dir))


def load_scan(run_dir: Path, i: int):
    """(points N x 4 sensor frame, labels N uint32, raw box list)."""
    stem = f"{i:06d}"
    pts = np.fromfile(run_dir / "velodyne" / f"{stem}.bin", dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(run_dir / "labels" / f"{stem}.label", dtype=np.uint32)
    bf = run_dir / "boxes" / f"{stem}.json"
    boxes = json.loads(bf.read_text())["boxes"] if bf.exists() else []
    if len(pts) != len(labels):
        raise ValueError(f"{run_dir.name} scan {stem}: {len(pts)} points but {len(labels)} labels")
    return pts, labels, boxes


def transform(points, M):
    p = np.column_stack([np.asarray(points, np.float64)[:, :3], np.ones(len(points))])
    return (p @ M)[:, :3]


def analyse_scan(pts, labels, boxes, rotation):
    """Per box: instance id, n_points, containment, and the box (center, extent, yaw) in the training frame.
    Plus pedestrians that have points but no box (an Isaac glitch)."""
    xyz1 = np.column_stack([pts[:, :3].astype(np.float64), np.ones(len(pts))])
    inst = labels >> 16
    is_ped = (labels & 0xFFFF) == PEDESTRIAN_CLASS
    out = []
    for box in boxes:
        iid = int(box["instance_id"])
        lo, hi = np.array(box["local_min"]), np.array(box["local_max"])
        b2s = np.array(box["box_to_sensor"], dtype=np.float64)
        own = xyz1[is_ped & (inst == iid)]
        if len(own):
            local = (own @ np.linalg.inv(b2s))[:, :3]
            containment = float(np.all((local >= lo - CONTAINMENT_TOL) & (local <= hi + CONTAINMENT_TOL), 1).mean())
        else:
            containment = None
        b2t = b2s @ rotation
        axes = b2t[:3, :3]
        center = (np.append((lo + hi) / 2, 1.0) @ b2t)[:3]
        out.append({"instance_id": iid, "n_points": int(len(own)), "containment": containment,
                    "center": center.tolist(), "extent": (hi - lo).tolist(),
                    "yaw": math.atan2(axes[0, 1], axes[0, 0]),
                    "upright": bool(abs(axes[2, 2] - 1) < 1e-3)})
    boxed = {b["instance_id"] for b in out}
    ids, counts = np.unique(inst[is_ped], return_counts=True)
    missing = [{"instance_id": int(a), "n_points": int(c)} for a, c in zip(ids, counts) if int(a) not in boxed]
    return {"boxes": out, "missing": missing, "n_points": int(len(pts))}


def analyse_run(run_dir: Path, force=False, progress=None) -> list[dict]:
    cache = run_dir / "analysis.json"
    n = num_scans(run_dir)
    if cache.exists() and not force:
        data = json.loads(cache.read_text())
        if len(data) == n:
            return data
    R = train_rotation(run_dir)
    data = []
    for i in range(n):
        data.append(analyse_scan(*load_scan(run_dir, i), R))
        if progress:
            progress(i + 1, n)
    cache.write_text(json.dumps(data))
    return data


# ---- filtering (export_filtered_pcdet.py semantics) --------------------------------------------------------
def _footprint(center, extent, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    half = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * (np.asarray(extent[:2]) / 2.0)
    return half @ np.array([[c, s], [-s, c]]) + np.asarray(center[:2])


def _clip(subject, clipper):
    """Sutherland-Hodgman: the part of convex polygon `subject` inside convex CCW polygon `clipper`."""
    output = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = b - a

        def side(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
        points, output = output, []
        for j in range(len(points)):
            p, q = points[j], points[(j + 1) % len(points)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                output.append(p)
            if (sp >= 0) != (sq >= 0):
                output.append(p + (q - p) * (sp / (sp - sq)))
        if not output:
            break
    return output


def _area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = [p[0] for p in poly], [p[1] for p in poly]
    return 0.5 * abs(sum(x[i] * y[i - 1] - x[i - 1] * y[i] for i in range(len(poly))))


def overlap_of_smaller(a, b):
    """Intersection volume of two upright rotated boxes over the smaller box's volume."""
    reach = (np.hypot(*a["extent"][:2]) + np.hypot(*b["extent"][:2])) / 2
    if np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1]) >= reach:
        return 0.0
    h = min(a["center"][2] + a["extent"][2] / 2, b["center"][2] + b["extent"][2] / 2) - \
        max(a["center"][2] - a["extent"][2] / 2, b["center"][2] - b["extent"][2] / 2)
    if h <= 0:
        return 0.0
    area = _area(_clip(_footprint(a["center"], a["extent"], a["yaw"]), _footprint(b["center"], b["extent"], b["yaw"])))
    smaller = min(np.prod(a["extent"]), np.prod(b["extent"]))
    return float(area * h / smaller) if smaller > 0 else 0.0


def judge(scan: dict, p: dict) -> list[str]:
    """Reasons to drop this scan (empty = keep)."""
    reasons = [f"missing box: pedestrian {m['instance_id']} ({m['n_points']} pts)" for m in scan["missing"]]
    boxes = scan["boxes"]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            f = overlap_of_smaller(a, b)
            if f > p["overlap"]:
                reasons.append(f"overlap: pedestrians {a['instance_id']} & {b['instance_id']} ({f:.0%})")
    for b in boxes:
        if b["n_points"] >= p["min_points"] and b["containment"] is not None and b["containment"] < p["min_containment"]:
            reasons.append(f"containment: pedestrian {b['instance_id']} {b['containment']:.0%} of "
                           f"{b['n_points']} pts in box")
    return reasons


def box_class(box, p):
    return "Pedestrian" if box["n_points"] >= p["min_label_points"] else "Pedestrian_ignore"
