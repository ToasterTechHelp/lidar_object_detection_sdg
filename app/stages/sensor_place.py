"""P7 LiDAR placement: where the sensor is mounted in the site (world frame), with an instant coverage preview.

Drag the gizmo or type position / roll-pitch-yaw. The preview casts the sensor's real beam pattern (from its USD,
see app/sensors.py) against the site mesh with Open3D, so coverage is visible without launching Isaac Sim, and
reports how much of the walking area a person's torso would be seen in. Writes sensor.json, which
sim/run_capture.py reads: the sensor model fields plus "position" and "wxyz" (world frame) and "rpy_deg".
"""
import json
import threading
import time

import numpy as np
import open3d as o3d
import trimesh
import viser.transforms as vtf

from app import sensors
from app.stages import sensor_model, walkarea

KEY = "sensor_place"
TORSO_Z = 1.1          # m; a torso-height grid over the walking area is tested for visibility
GRID = 0.25            # m


def rpy_to_wxyz(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    return vtf.SO3.from_rpy_radians(r, p, y).wxyz


def wxyz_to_rpy(wxyz):
    so3 = vtf.SO3(np.asarray(wxyz))
    return np.degrees([so3.compute_roll_radians(), so3.compute_pitch_radians(), so3.compute_yaw_radians()])


class Previewer:
    def __init__(self, mesh: trimesh.Trimesh):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                                 o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))

    def cast(self, origin, dirs):
        rays = np.concatenate([np.broadcast_to(origin, dirs.shape), dirs], 1).astype(np.float32)
        return self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()

    def scan(self, pattern: sensors.BeamPattern, position, wxyz, max_rays=60_000):
        R = vtf.SO3(np.asarray(wxyz)).as_matrix()
        dirs = pattern.rays(max_rays) @ R.T
        t = self.cast(np.asarray(position, np.float32), dirs)
        ok = np.isfinite(t) & (t >= pattern.near_m) & (t <= pattern.far_m)
        return np.asarray(position) + dirs[ok] * t[ok, None], t[ok]

    def torso_coverage(self, pattern, position, wxyz, rects):
        """Share of torso-height grid points over the walking area that are inside the beam's vertical FOV,
        in range, and in line of sight."""
        pts = []
        for r in rects:
            xs = np.arange(-r["size"][0] / 2 + GRID / 2, r["size"][0] / 2, GRID)
            ys = np.arange(-r["size"][1] / 2 + GRID / 2, r["size"][1] / 2, GRID)
            g = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
            a = np.radians(r.get("yaw_deg", 0.0))
            g = g @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) + r["center"]
            pts.append(np.column_stack([g, np.full(len(g), TORSO_Z)]))
        if not pts:
            return None
        pts = np.concatenate(pts)
        pos = np.asarray(position, np.float64)
        v = pts - pos
        d = np.linalg.norm(v, axis=1)
        local = v @ vtf.SO3(np.asarray(wxyz)).as_matrix()  # world -> sensor frame (R^T v, row form)
        elev = np.degrees(np.arcsin(np.clip(local[:, 2] / np.maximum(d, 1e-9), -1, 1)))
        in_fov = (elev >= pattern.elevation_deg.min()) & (elev <= pattern.elevation_deg.max())
        in_range = (d >= pattern.near_m) & (d <= pattern.far_m)
        t = self.cast(pos.astype(np.float32), (v / d[:, None]).astype(np.float32))
        visible = ~np.isfinite(t) | (t >= d - 0.05)
        return float((in_fov & in_range & visible).mean()), pts, in_fov & in_range & visible


def load(project):
    f = project.path / "sensor.json"
    return json.loads(f.read_text()) if f.exists() else None


def default_pose(project):
    wa = walkarea.load(project)
    if wa and wa["rects"]:
        r = wa["rects"][0]
        a = np.radians(r.get("yaw_deg", 0.0))
        back = np.array([np.cos(a), np.sin(a)]) * (r["size"][0] / 2 + 0.3)
        return [*(np.array(r["center"]) - back).round(2).tolist(), 1.2], [0.0, 0.0, float(r.get("yaw_deg", 0.0))]
    return [0.0, 0.0, 1.2], [0.0, 0.0, 0.0]


def save(project, model, position, rpy):
    data = {k: model.get(k) for k in ("model", "source", "usd_path", "config", "variant")}
    data.update(position=[round(float(v), 4) for v in position], rpy_deg=[round(float(v), 3) for v in rpy],
                wxyz=[float(v) for v in rpy_to_wxyz(rpy)], tick_rate=float(model.get("info", {}).get("rate_hz", 10)))
    (project.path / "sensor.json").write_text(json.dumps(data, indent=2))
    project.set_status(KEY, "done", note=f"at ({position[0]:.2f}, {position[1]:.2f}), {position[2]:.2f} m high")
    project.invalidate_downstream(KEY)


def build(app, folder):
    gui, scene = app.server.gui, app.server.scene
    with folder:
        state_md = gui.add_markdown("")
        pos_in = gui.add_vector3("Position (m)", (0.0, 0.0, 1.2), step=0.05)
        rpy_in = gui.add_vector3("Roll / pitch / yaw (deg)", (0.0, 0.0, 0.0), step=1.0)
        live = gui.add_checkbox("Live preview scan", True)
        stats_md = gui.add_markdown("")
        save_btn = gui.add_button("Save placement", icon="check", color="green")

    st = {"model": None, "pattern": None, "prev": None, "gizmo": None, "handles": [], "dirty": False,
          "syncing": False, "rects": []}

    def clear():
        for h in st["handles"]:
            h.remove()
        st["handles"] = []

    def pose():
        return np.array(pos_in.value, np.float64), np.array(rpy_in.value, np.float64)

    def run_preview():
        if st["prev"] is None or st["pattern"] is None:
            return
        position, rpy = pose()
        wxyz = rpy_to_wxyz(rpy)
        t0 = time.time()
        pts, dist = st["prev"].scan(st["pattern"], position, wxyz)
        cov = st["prev"].torso_coverage(st["pattern"], position, wxyz, st["rects"])
        dt = time.time() - t0
        clear()
        if len(pts):
            c = np.clip(dist / max(1.0, np.percentile(dist, 95)), 0, 1)
            colors = (np.column_stack([1 - c, 0.3 + 0.5 * c, c]) * 255).astype(np.uint8)
            st["handles"].append(scene.add_point_cloud("/sensor/preview", pts.astype(np.float32), colors,
                                                       point_size=0.02))
        lines = [f"Preview: **{len(pts):,}** hits of {st['pattern'].rays().shape[0]:,} rays shown ({dt * 1000:.0f} ms)",
                 f"Mount height: **{position[2]:.2f} m** above the floor"]
        if cov is not None:
            frac, grid, seen = cov
            colors = np.where(seen[:, None], [[40, 220, 90]], [[230, 60, 60]]).astype(np.uint8)
            st["handles"].append(scene.add_point_cloud("/sensor/torso", grid.astype(np.float32), colors,
                                                       point_size=0.06, point_shape="circle"))
            lines.append(f"Torso-height coverage of the walking area: **{frac:.0%}** (green seen, red not)")
        if st["pattern"].approximate:
            lines.append(f"<small>{st['pattern'].note}</small>")
        stats_md.content = "  \n".join(lines)

    def loop():
        while True:
            time.sleep(0.25)
            if st["dirty"] and live.value:
                st["dirty"] = False
                try:
                    run_preview()
                except Exception as e:  # noqa: BLE001
                    app.log(f"preview failed: {e}")

    threading.Thread(target=loop, daemon=True).start()

    def draw_gizmo():
        if st["gizmo"] is not None:
            st["gizmo"].remove()
        position, rpy = pose()
        g = scene.add_transform_controls("/sensor/gizmo", scale=0.6, position=tuple(position),
                                         wxyz=tuple(rpy_to_wxyz(rpy)))
        st["gizmo"] = g

        @g.on_update
        def _(_):
            st["syncing"] = True
            pos_in.value = tuple(float(v) for v in g.position)
            rpy_in.value = tuple(float(v) for v in wxyz_to_rpy(g.wxyz))
            st["syncing"] = False
            st["dirty"] = True

    def inputs_changed(_):
        if st["syncing"]:
            return
        if st["gizmo"] is not None:
            position, rpy = pose()
            st["gizmo"].position = tuple(position)
            st["gizmo"].wxyz = tuple(rpy_to_wxyz(rpy))
        st["dirty"] = True

    pos_in.on_update(inputs_changed)
    rpy_in.on_update(inputs_changed)
    live.on_update(lambda _: st.__setitem__("dirty", True) if live.value else clear())

    @save_btn.on_click
    def _(_):
        position, rpy = pose()
        save(app.project, st["model"], position, rpy)
        app.log(f"sensor placed at {np.round(position, 2).tolist()}, rpy {np.round(rpy, 1).tolist()}")
        app.stage_changed(KEY)  # status badges + enables Simulate
        refresh_state()

    def refresh_state():
        p = app.project
        entry = p.state["stages"][KEY]
        missing = [what for what, ok in (("Scale & level", p.world_mesh_path() is not None),
                                         ("LiDAR model", st["model"] is not None)) if not ok]
        save_btn.disabled = bool(missing)
        if missing:
            state_md.content = f"*Needs: {', '.join(missing)}.*"
        elif entry["status"] == "done":
            state_md.content = f"✅ Saved: {entry.get('note', '')}. Move it and save again to change it."
        else:
            state_md.content = f"Place the **{st['model']['model']}**: drag the gizmo or type the pose, then save."

    def refresh():
        p = app.project
        if p is None:
            return
        clear()
        if st["gizmo"] is not None:
            st["gizmo"].remove()
            st["gizmo"] = None
        st["model"] = sensor_model.load(p)
        wa = walkarea.load(p)
        st["rects"] = wa["rects"] if wa else []
        st["prev"] = st["pattern"] = None
        mesh_file = p.world_mesh_path()
        if mesh_file is not None:
            st["prev"] = Previewer(trimesh.load(mesh_file, process=False))
        if st["model"] is not None:
            try:
                st["pattern"] = sensors.pattern_for(st["model"])
            except Exception as e:  # noqa: BLE001
                app.log(f"beam pattern unavailable: {e}")
        saved = load(p)
        if saved:
            position, rpy = saved["position"], saved.get("rpy_deg") or wxyz_to_rpy(saved["wxyz"]).tolist()
        else:
            position, rpy = default_pose(p)
        st["syncing"] = True
        pos_in.value, rpy_in.value = tuple(position), tuple(rpy)
        st["syncing"] = False
        if mesh_file is not None and st["model"] is not None:
            draw_gizmo()
            st["dirty"] = True
        refresh_state()

    app.on_project_opened(refresh)
    app.on_stage_changed(lambda key: refresh() if key in ("scale", "walkarea", "sensor_model") else None)
