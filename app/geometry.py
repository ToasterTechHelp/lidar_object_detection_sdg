"""Frame bookkeeping and geometry helpers.

Artifacts from VGGT/meshing stay in VGGT's frame on disk. scale.json holds one 4x4 similarity transform T
(rotation + uniform scale + translation) from that frame to the "world" frame every later stage uses:
meters, z up, floor at z = 0.
"""
import json

import numpy as np
import open3d as o3d

UP_HINT_VGGT = np.array([0.0, 1.0, 0.0])  # GLB frame is the first camera in OpenGL axes: +y is roughly up


def load_T(project) -> np.ndarray:
    f = project.path / "scale.json"
    return np.array(json.loads(f.read_text())["T"]) if f.exists() else np.eye(4)


def load_scale_info(project) -> dict:
    f = project.path / "scale.json"
    return json.loads(f.read_text()) if f.exists() else {}


def apply_T(points, T) -> np.ndarray:
    points = np.asarray(points, np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def T_scale(T) -> float:
    return float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))


def rotation_between(a, b) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        perp /= np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)  # 180 degrees about a perpendicular axis
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def fit_floor(vertices, cam_centers=None, up_hint=UP_HINT_VGGT, max_planes=6, max_tilt_deg=35):
    """Find the floor plane: the largest roughly-horizontal plane with the cameras above it.
    In hallways the biggest plane is often a wall and the ceiling is horizontal too, so size alone fails.
    Returns (normal pointing up, point on plane, inlier count, tilt from up_hint in degrees)."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(vertices, np.float64)))
    extent = float(np.linalg.norm(np.ptp(np.asarray(pcd.points), axis=0)))
    pcd = pcd.voxel_down_sample(extent / 300)
    ref = np.mean(cam_centers, axis=0) if cam_centers is not None and len(cam_centers) else None
    candidates, rest = [], pcd
    for _ in range(max_planes):
        if len(rest.points) < 100:
            break
        (a, b, c, d), idx = rest.segment_plane(distance_threshold=extent / 200, ransac_n=3, num_iterations=2000)
        n = np.array([a, b, c])
        norm = np.linalg.norm(n)
        n, d = n / norm, d / norm
        if np.dot(n, up_hint) < 0:
            n, d = -n, -d
        pts = np.asarray(rest.points)[idx]
        tilt = float(np.degrees(np.arccos(np.clip(np.dot(n, up_hint), -1, 1))))
        cams_above = ref is None or float(np.dot(n, ref) + d) > 0
        candidates.append({"normal": n, "point": pts.mean(0), "inliers": len(idx), "tilt": tilt,
                           "cams_above": cams_above})
        rest = rest.select_by_index(idx, invert=True)
    if not candidates:
        raise RuntimeError("no planes found")
    floors = [c for c in candidates if c["tilt"] <= max_tilt_deg and c["cams_above"]]
    best = max(floors, key=lambda c: c["inliers"]) if floors else min(candidates, key=lambda c: c["tilt"])
    return best["normal"], best["point"], best["inliers"], best["tilt"]


def plane_from_points(p3):
    p3 = np.asarray(p3, np.float64)
    n = np.cross(p3[1] - p3[0], p3[2] - p3[0])
    if np.linalg.norm(n) < 1e-12:
        raise ValueError("the 3 points are collinear")
    n /= np.linalg.norm(n)
    return n, p3.mean(0)


def level_transform(normal, floor_point, vertices) -> np.ndarray:
    """Rigid T (VGGT frame -> leveled): floor normal -> +z, floor at z=0, the scene's long horizontal axis
    along x, centered on the scene's footprint."""
    R = rotation_between(np.asarray(normal, np.float64), np.array([0.0, 0.0, 1.0]))
    v = np.asarray(vertices, np.float64) @ R.T
    xy = v[:, :2] - np.median(v[:, :2], axis=0)
    _, vecs = np.linalg.eigh(np.cov(xy[:: max(1, len(xy) // 50_000)].T))
    major = vecs[:, -1]
    yaw = -np.arctan2(major[1], major[0])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = Rz @ R
    v = np.asarray(vertices, np.float64) @ R.T
    fz = float((R @ np.asarray(floor_point, np.float64))[2])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [-np.median(v[:, 0]), -np.median(v[:, 1]), -fz]
    return T


class Raycaster:
    """Click ray -> first hit on the displayed mesh."""

    def __init__(self, vertices, faces):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                                 o3d.core.Tensor(np.asarray(faces, np.uint32)))

    def hit(self, origin, direction):
        d = np.asarray(direction, np.float32)
        d /= np.linalg.norm(d)
        rays = o3d.core.Tensor(np.concatenate([np.asarray(origin, np.float32), d])[None])
        t = float(self.scene.cast_rays(rays)["t_hit"].numpy()[0])
        return None if not np.isfinite(t) else np.asarray(origin, np.float64) + t * d.astype(np.float64)
