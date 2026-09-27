"""Draws a project's current artifacts (VGGT points, cameras, cleaned points, mesh) into the Viser scene,
and lets stages pick points on the mesh by clicking.

Everything is drawn in the display frame: the VGGT frame mapped through scale.json's transform T (identity
until the scale & level stage has run).
"""
import json

import numpy as np
import trimesh
import viser
import viser.transforms as vtf

from app.geometry import Raycaster, T_scale, apply_T, load_T
from app.project import Project
from glb_to_mesh import load_vggt_glb

MAX_DISPLAY_POINTS = 400_000
THUMB_W = 128

# layer -> (checkbox label, visible by default)
LAYERS = {
    "vggt": ("VGGT points", True),
    "points": ("Cleaned points", False),
    "mesh": ("Mesh", True),
    "cameras": ("Cameras", True),
}


def _subsample(pts, cols):
    if len(pts) > MAX_DISPLAY_POINTS:
        idx = np.random.default_rng(0).choice(len(pts), MAX_DISPLAY_POINTS, replace=False)
        pts, cols = pts[idx], cols[idx]
    return np.asarray(pts, np.float32), np.asarray(cols, np.uint8)


def _point_size(pts):
    return float(np.linalg.norm(np.percentile(pts, 95, 0) - np.percentile(pts, 5, 0))) / 1500 or 0.01


class SceneView:
    """Owns the /project/* scene nodes, their visibility checkboxes, and click-picking on the mesh."""

    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.handles: dict[str, list] = {}
        self.checkboxes = {}
        self.T = np.eye(4)
        self.mesh: trimesh.Trimesh | None = None  # displayed mesh, in the display frame
        self._raycaster = None
        self._pick = None  # (n, callback, points, color, tag)
        self._pick_handles = []
        self._framed = None  # (project path, T) the camera was last framed for
        self._view = None    # (position, look_at, up) for the current frame, also used for new browser tabs
        self.server.scene.on_click()(self._on_click)
        self.server.on_client_connect(lambda client: self._apply_view(client))

    def build_view_panel(self):
        with self.server.gui.add_folder("View", expand_by_default=False):
            for layer, (label, default) in LAYERS.items():
                cb = self.server.gui.add_checkbox(label, default)
                cb.on_update(lambda e, layer=layer: self.set_visible(layer, e.target.value))
                self.checkboxes[layer] = cb

    def clear(self):
        for hs in self.handles.values():
            for h in hs:
                h.remove()
        self.handles.clear()
        self.clear_picks()

    def _add(self, layer, handle):
        self.handles.setdefault(layer, []).append(handle)
        handle.visible = self.checkboxes[layer].value if layer in self.checkboxes else True

    @property
    def scale(self) -> float:
        return T_scale(self.T)

    def show_project(self, project: Project, log=print):
        self.clear()
        self.T = load_T(project)
        # Before leveling, the display frame is the VGGT GLB frame: first camera, OpenGL axes
        # (visual_util.apply_scene_alignment flips y/z), so up is +y. Leveling makes it +z.
        leveled = (project.path / "scale.json").exists()
        up = np.array([0.0, 0.0, 1.0]) if leveled else np.array([0.0, 1.0, 0.0])
        self.server.scene.set_up_direction("+z" if leveled else "+y")

        mesh_file = project.mesh_path()
        glb = project.path / "vggt" / "scene.glb"
        npz = project.path / "vggt" / "scene.npz"
        if "vggt" in self.checkboxes:  # raw points are clutter once a mesh exists; still one click away
            self.checkboxes["vggt"].value = mesh_file is None

        cam_centers = None
        if glb.exists():
            pts, cols, cam_centers = load_vggt_glb(glb)
            pts, cols = _subsample(apply_T(pts, self.T), (cols * 255).astype(np.uint8))
            self._add("vggt", self.server.scene.add_point_cloud(
                "/project/vggt", pts, cols, point_size=_point_size(pts)))
            log(f"VGGT points: {len(pts):,} shown")

        points = project.path / "mesh" / "points_clean.ply"
        if points.exists():
            pc = trimesh.load(points)
            cols = pc.colors[:, :3] if getattr(pc, "colors", None) is not None and len(pc.colors) else \
                np.full((len(pc.vertices), 3), 180)
            pts, cols = _subsample(apply_T(pc.vertices, self.T), cols)
            self._add("points", self.server.scene.add_point_cloud(
                "/project/points", pts, cols, point_size=_point_size(pts)))

        self.mesh, self._raycaster = None, None
        if mesh_file is not None:
            mesh = trimesh.load(mesh_file, process=False)
            mesh.apply_transform(self.T)
            self.mesh = mesh
            self._add("mesh", self.server.scene.add_mesh_trimesh("/project/mesh", mesh))
            log(f"{mesh_file.name}: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

        # Re-aim the camera whenever the frame changes (open, level, scale, reset). Without this the scene is
        # rescaled or rotated around a camera that stays put: scaling to metres "explodes" the mesh around the
        # viewer (clicks then hit nearby geometry), and switching up axis makes orbiting feel flipped.
        key = (str(project.path), np.round(self.T, 9).tobytes())
        if self._framed != key and self.mesh is not None:
            self._framed = key
            self.frame_view(self.mesh.vertices, up)

        if npz.exists():
            self._add_frustums(npz)
        else:
            cams_file = project.path / "mesh" / "cameras.json"
            if cam_centers is None and cams_file.exists():
                cam_centers = np.asarray(json.loads(cams_file.read_text()).get("camera_centers", []))
            if cam_centers is not None and len(cam_centers):
                self._add("cameras", self.server.scene.add_point_cloud(
                    "/project/cameras", apply_T(cam_centers, self.T).astype(np.float32), (255, 60, 60),
                    point_size=0.02 * self.scale, point_shape="circle"))

    def _add_frustums(self, npz_path):
        """One frustum per input frame, textured with the frame, in the display frame."""
        pred = np.load(npz_path)
        extr, intr, images = pred["extrinsic"], pred["intrinsic"], pred["images"]
        s = len(extr)
        w2c = np.tile(np.eye(4), (s, 1, 1))
        w2c[:, :3, :4] = extr
        opengl = np.diag([1.0, -1.0, -1.0, 1.0])
        align = self.T @ np.linalg.inv(w2c[0]) @ opengl  # visual_util.apply_scene_alignment, then T
        k = self.scale
        h, w = images.shape[-2:]
        stride = max(1, w // THUMB_W)
        spread = max(np.linalg.norm(np.linalg.inv(w2c[i])[:3, 3] - np.linalg.inv(w2c[0])[:3, 3]) for i in range(s))
        size = 0.08 * (spread or 1.0) * k
        for i in range(s):
            pose = align @ np.linalg.inv(w2c[i])  # OpenCV camera -> display; Viser frustums use OpenCV axes
            img = (np.transpose(images[i], (1, 2, 0))[::stride, ::stride] * 255).clip(0, 255).astype(np.uint8)
            fov = 2 * np.arctan(h / (2 * intr[i, 1, 1]))
            self._add("cameras", self.server.scene.add_camera_frustum(
                f"/project/cameras/{i:03d}", fov=float(fov), aspect=w / h, scale=size,
                image=img, format="jpeg", wxyz=vtf.SO3.from_matrix(pose[:3, :3] / k).wxyz, position=pose[:3, 3],
                thickness=size / 25))

    def frame_view(self, vertices, up):
        """Point every connected browser (and new ones) at the whole mesh, from above and to the side."""
        v = np.asarray(vertices, np.float64)
        lo, hi = np.percentile(v, 2, axis=0), np.percentile(v, 98, axis=0)  # ignore stray far-off faces
        center, size = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
        side = np.array([1.0, 0.0, 0.0])
        back = np.cross(up, side)
        direction = side * 0.6 - back * 0.8 + up * 0.7
        direction /= np.linalg.norm(direction)
        self._view = (center + direction * size * 1.1, center, up)
        for client in self.server.get_clients().values():
            self._apply_view(client)

    def _apply_view(self, client):
        if self._view is None:
            return
        position, look_at, up = self._view
        with client.atomic():
            client.camera.up_direction = tuple(float(x) for x in up)
            client.camera.position = tuple(float(x) for x in position)
            client.camera.look_at = tuple(float(x) for x in look_at)

    def add_measure(self, p0, p1, text, color=(255, 80, 0)):
        """A labelled segment between two display-frame points (kept until the next pick or redraw)."""
        r = self._marker_radius()
        for p in (p0, p1):
            self._pick_handles.append(self.server.scene.add_icosphere(
                f"/picks/p{len(self._pick_handles)}", radius=r, color=color, position=tuple(float(x) for x in p)))
        self._pick_handles.append(self.server.scene.add_line_segments(
            f"/picks/l{len(self._pick_handles)}", np.array([[p0, p1]], np.float32), colors=color, thickness=3.0,
            thickness_units="screen"))
        self.add_label(text, (np.asarray(p0) + np.asarray(p1)) / 2)

    def set_visible(self, layer: str, visible: bool):
        for h in self.handles.get(layer, []):
            h.visible = visible

    # ---- picking ------------------------------------------------------------------------------------------
    def start_pick(self, n: int, callback, color=(255, 80, 0), log=print):
        """Collect n clicked points on the mesh, then call callback(points Nx3 in the display frame)."""
        if self.mesh is None:
            log("no mesh to pick on")
            return
        self.clear_picks()
        self._pick = (n, callback, [], color)
        log(f"click {n} point{'s' if n > 1 else ''} on the mesh")

    def cancel_pick(self):
        self._pick = None

    def clear_picks(self):
        for h in self._pick_handles:
            h.remove()
        self._pick_handles = []

    def _marker_radius(self):
        return float(np.linalg.norm(np.ptp(self.mesh.vertices, axis=0))) / 250 if self.mesh is not None else 0.01

    def _on_click(self, event):
        if self._pick is None or self.mesh is None:
            return
        if self._raycaster is None:
            self._raycaster = Raycaster(self.mesh.vertices, self.mesh.faces)
        p = self._raycaster.hit(event.ray_origin, event.ray_direction)
        if p is None:
            return
        n, callback, pts, color = self._pick
        pts.append(p)
        r = self._marker_radius()
        self._pick_handles.append(self.server.scene.add_icosphere(
            f"/picks/p{len(self._pick_handles)}", radius=r, color=color, position=tuple(p)))
        if len(pts) > 1:
            self._pick_handles.append(self.server.scene.add_line_segments(
                f"/picks/l{len(self._pick_handles)}", np.array([[pts[-2], pts[-1]]], np.float32),
                colors=color, thickness=3.0, thickness_units="screen"))
        if len(pts) >= n:
            self._pick = None
            callback(np.array(pts))

    def add_label(self, text, position):
        self._pick_handles.append(self.server.scene.add_label(f"/picks/label{len(self._pick_handles)}", text,
                                                              position=tuple(position)))
