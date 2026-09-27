"""P4 Scale & level: VGGT's arbitrary frame -> world frame (meters, z up, floor at z = 0).

1. Level: auto-detect the floor (largest roughly-horizontal plane with the cameras above it), or click 3 floor
   points. The floor becomes z = 0 and the scene's long axis is aligned with x.
2. Scale: click 2 points and enter their real distance in meters.
The result is one transform in scale.json, plus mesh/mesh_world.{ply,obj} (the mesh in the world frame) for the
later stages and Isaac Sim.
"""
import json

import numpy as np
import trimesh

from app.geometry import T_scale, apply_T, fit_floor, level_transform, load_scale_info, load_T, plane_from_points

KEY = "scale"


def _raw_mesh(project) -> trimesh.Trimesh:
    return trimesh.load(project.mesh_path(), process=False)


def _cam_centers(project):
    f = project.path / "mesh" / "cameras.json"
    return np.asarray(json.loads(f.read_text()).get("camera_centers", [])) if f.exists() else None


def save(project, T, **info):
    """Write scale.json and the world-frame mesh; the stage is done once the scene is both leveled and scaled."""
    data = {**load_scale_info(project), **info, "T": np.asarray(T).tolist(), "scale_factor": T_scale(T)}
    (project.path / "scale.json").write_text(json.dumps(data, indent=2))
    mesh = _raw_mesh(project)
    mesh.apply_transform(T)
    mesh.export(project.dir("mesh", "mesh_world.ply"))
    mesh.export(project.dir("mesh", "mesh_world.obj"))
    done = data.get("leveled") and data.get("scaled")
    b = mesh.bounds
    note = (f"{b[1][0] - b[0][0]:.2f} x {b[1][1] - b[0][1]:.2f} x {b[1][2] - b[0][2]:.2f} m"
            if done else "leveled; scale not set yet")
    project.set_status(KEY, "done" if done else "todo", note=note)
    if done:
        project.invalidate_downstream(KEY)
    return data


def auto_level(project):
    mesh = _raw_mesh(project)
    n, p, inliers, tilt = fit_floor(mesh.vertices, _cam_centers(project))
    T = level_transform(n, p, mesh.vertices)
    return save(project, T, leveled=True, scaled=False, level_method="auto", floor_inliers=inliers,
                floor_tilt_from_camera_up_deg=tilt)


def level_from_points(project, pts_display):
    """3 clicked floor points (display frame) -> level transform, computed in the raw VGGT frame."""
    T_old = load_T(project)
    raw = apply_T(pts_display, np.linalg.inv(T_old))
    n, p = plane_from_points(raw)
    cams = _cam_centers(project)
    ref = np.mean(cams, 0) if cams is not None and len(cams) else p + np.array([0, 1.0, 0])
    if np.dot(n, ref - p) < 0:  # normal must point to the side the cameras are on
        n = -n
    T = level_transform(n, p, _raw_mesh(project).vertices)
    return save(project, T, leveled=True, scaled=False, level_method="3 points")


def apply_scale(project, p2_display, real_m: float):
    """Scale about the world origin so the floor stays at z = 0."""
    d = float(np.linalg.norm(p2_display[1] - p2_display[0]))
    if d <= 0 or real_m <= 0:
        raise ValueError("distances must be positive")
    S = np.diag([real_m / d] * 3 + [1.0])
    return save(project, S @ load_T(project), scaled=True, scale_ref_m=real_m)


def build(app, folder):
    gui = app.server.gui
    with folder:
        state_md = gui.add_markdown("")
        gui.add_markdown("**1. Level the floor**")
        auto_btn = gui.add_button("Auto-level floor", icon="ruler-2")
        pick3_btn = gui.add_button("Pick 3 floor points instead", icon="click")
        gui.add_markdown("**2. Set the scale:** click two points you know the real distance between "
                         "(a door width, a floor tile, a table).")
        pick2_btn = gui.add_button("Pick 2 points", icon="click")
        picked_md = gui.add_markdown("")
        real = gui.add_number("Real distance (m)", initial_value=1.0, min=0.01, max=1000.0, step=0.01)
        apply_btn = gui.add_button("Apply scale", icon="check", disabled=True)
        gui.add_markdown("**3. Check:** measure another known distance.")
        measure_btn = gui.add_button("Measure distance", icon="ruler-measure")
        measure_md = gui.add_markdown("")
        reset_btn = gui.add_button("Reset scale & level", color="red", icon="restore")

    picked = {"pts": None}

    def refresh():
        p = app.project
        if p is None:
            return
        has_mesh = p.mesh_path() is not None
        info = load_scale_info(p)
        for b in (auto_btn, pick3_btn):
            b.disabled = not has_mesh
        pick2_btn.disabled = measure_btn.disabled = not has_mesh
        if not has_mesh:
            state_md.content = "*Build the mesh first.*"
        elif not info:
            state_md.content = "Not leveled or scaled yet (scene in VGGT's arbitrary frame)."
        else:
            lvl = f"✅ leveled ({info.get('level_method')})" if info.get("leveled") else "⬜ not leveled"
            scl = (f"✅ scaled (1 VGGT unit = {info['scale_factor']:.3f} m)" if info.get("scaled")
                   else "⬜ scale not set")
            state_md.content = f"{lvl}  \n{scl}  \n{p.state['stages'][KEY].get('note', '')}"
        picked["pts"] = None
        picked_md.content = ""
        apply_btn.disabled = True

    def reload():
        app.scene.show_project(app.project, log=app.log)
        app.stage_changed(KEY)
        refresh()

    def guarded(fn, what):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - show geometry failures in the GUI instead of crashing
            app.log(f"{what} failed: {e}")
            return False
        reload()
        return True

    @auto_btn.on_click
    def _(_):
        app.log("fitting floor plane…")

        def run():
            info = auto_level(app.project)
            app.log(f"floor: {info['floor_inliers']:,} inlier points, "
                    f"{info['floor_tilt_from_camera_up_deg']:.1f}° from camera up")
        guarded(run, "auto-level")

    @pick3_btn.on_click
    def _(_):
        app.scene.start_pick(3, lambda pts: guarded(lambda: level_from_points(app.project, pts), "3-point level"),
                             color=(0, 180, 255), log=app.log)

    @pick2_btn.on_click
    def _(_):
        def got(pts):
            picked["pts"] = pts
            d = float(np.linalg.norm(pts[1] - pts[0]))
            unit = "m (current scale)" if load_scale_info(app.project).get("scaled") else "VGGT units"
            picked_md.content = f"Picked distance: **{d:.4f} {unit}**. Enter the real distance and apply."
            apply_btn.disabled = False
        app.scene.start_pick(2, got, color=(255, 80, 0), log=app.log)

    @apply_btn.on_click
    def _(_):
        if picked["pts"] is None:
            return
        pts = picked["pts"]

        def run():
            info = apply_scale(app.project, pts, float(real.value))
            app.log(f"scale set: 1 VGGT unit = {info['scale_factor']:.4f} m")
        if not guarded(run, "scale"):
            return
        # The scale is about the origin, so the reference points move by the same factor: show them where they
        # now are, labelled with the distance they now span.
        k = float(real.value) / float(np.linalg.norm(pts[1] - pts[0]))
        app.scene.add_measure(pts[0] * k, pts[1] * k, f"{float(real.value):.2f} m (reference)")

    @measure_btn.on_click
    def _(_):
        def got(pts):
            d = float(np.linalg.norm(pts[1] - pts[0]))
            scaled = load_scale_info(app.project).get("scaled")
            text = f"{d:.3f} m" if scaled else f"{d:.4f} units (not scaled yet)"
            measure_md.content = f"Measured: **{text}**"
            app.scene.add_label(text, (pts[0] + pts[1]) / 2)
        app.scene.start_pick(2, got, color=(80, 220, 80), log=app.log)

    @reset_btn.on_click
    def _(_):
        p = app.project
        for f in (p.path / "scale.json", p.path / "mesh" / "mesh_world.ply", p.path / "mesh" / "mesh_world.obj"):
            f.unlink(missing_ok=True)
        p.set_status(KEY, "todo", note="")
        app.log("scale & level reset")
        reload()

    app.on_project_opened(refresh)
    app.on_stage_changed(lambda key: refresh() if key == "mesh" else None)
