"""P3 Mesh: vggt/scene.glb -> Poisson mesh (glb_to_mesh.py), optionally hole-filled (fill_holes.py).

Outputs land in mesh/. The mesh later stages use is recorded as state["stages"]["mesh"]["final"].
Before the scale stage the scene has arbitrary units, so the hole-size limit is given as a percentage of the
scene's size and converted for fill_holes.py.
"""
import sys

import numpy as np
import trimesh

from app.project import ROOT

KEY = "mesh"


def build(app, folder):
    gui = app.server.gui
    with folder:
        input_md = gui.add_markdown("")
        depth = gui.add_slider("Detail (Poisson depth)", min=7, max=11, step=1, initial_value=9,
                               hint="Higher = sharper detail but more noise and memory")
        trim = gui.add_slider("Trim low-support (%)", min=0, max=30, step=1, initial_value=5,
                              hint="Remove the least-supported surface Poisson invented far from real points")
        faces = gui.add_slider("Max faces (k)", min=50, max=1000, step=50, initial_value=300,
                               hint="Decimate to this many faces; Isaac and the browser stay fast")
        fill = gui.add_checkbox("Fill holes", True)
        max_hole = gui.add_slider("Max hole size (% of scene)", min=1, max=30, step=1, initial_value=8,
                                  hint="Only close holes smaller than this; big gaps stay open")
        run_btn = gui.add_button("Build mesh", icon="triangle")
        bar = gui.add_progress_bar(100.0, animated=True, visible=False)
        summary = gui.add_markdown("")

    fill.on_update(lambda _: setattr(max_hole, "disabled", not fill.value))

    def refresh():
        p = app.project
        if p is None:
            return
        has_glb = (p.path / "vggt" / "scene.glb").exists()
        input_md.content = "Input: VGGT point cloud." if has_glb else "*Run Reconstruct first.*"
        run_btn.disabled = not has_glb
        mesh = p.mesh_path()
        if mesh is not None and p.status(KEY) == "done":
            m = trimesh.load(mesh, process=False)
            summary.content = (f"**{mesh.name}:** {len(m.vertices):,} vertices, {len(m.faces):,} faces, "
                               f"{'watertight' if m.is_watertight else 'open'}")
        else:
            summary.content = ""

    def finish(error: str | None = None, **info):
        p = app.project
        if error:
            p.set_status(KEY, "error", note=error)
        else:
            p.set_status(KEY, "done", **info)
            p.invalidate_downstream(KEY)
        bar.visible = False
        app.scene.show_project(p, log=app.log)
        app.stage_changed(KEY)  # refreshes the status badges and the later stages' panels
        refresh()

    @run_btn.on_click
    def _(_):
        p = app.project
        out = p.dir("mesh", "x").parent
        for f in out.glob("mesh_filled.*"):  # a stale fill from an earlier run must not be picked up
            f.unlink()
        cmd = [sys.executable, ROOT / "glb_to_mesh.py", p.path / "vggt" / "scene.glb", "--out", out,
               "--depth", int(depth.value), "--trim", trim.value / 100, "--target-faces", int(faces.value * 1000)]
        params = {"depth": int(depth.value), "trim": trim.value / 100, "target_faces": int(faces.value * 1000)}

        def meshed(code):
            if code != 0 or not (out / "mesh.ply").exists():
                finish(error=f"glb_to_mesh.py exited with code {code}; see log")
                return
            if not fill.value:
                finish(final="mesh.ply", note="Poisson mesh", **params)
                return
            m = trimesh.load(out / "mesh.ply", process=False)
            extent = float(np.linalg.norm(m.bounds[1] - m.bounds[0]))
            hole = extent * max_hole.value / 100
            app.log(f"filling holes up to {hole:.3f} units ({max_hole.value:.0f}% of scene size {extent:.3f})")
            fill_cmd = [sys.executable, ROOT / "fill_holes.py", out / "mesh.ply", "--out", out / "mesh_filled",
                        "--max-hole", hole]

            def filled(code2):
                if code2 == 0 and (out / "mesh_filled.ply").exists():
                    finish(final="mesh_filled.ply", note="Poisson mesh, holes filled", max_hole=hole, **params)
                else:  # the unfilled mesh is still usable
                    app.log("hole filling failed; keeping the unfilled mesh")
                    finish(final="mesh.ply", note="Poisson mesh (hole filling failed)", **params)

            if not app.jobs.run("Fill holes", fill_cmd, cwd=ROOT, on_done=filled):
                finish(final="mesh.ply", note="Poisson mesh (hole filling skipped)", **params)

        if app.jobs.run("Build mesh", cmd, cwd=ROOT, on_done=meshed):
            p.set_status(KEY, "running")
            app.refresh_status()
            bar.visible = True

    app.on_project_opened(refresh)
    app.on_stage_changed(lambda key: refresh() if key == "reconstruct" else None)
