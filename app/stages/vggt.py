"""P2 Reconstruct: frames/ -> VGGT-Omega point cloud + cameras (vggt/scene.glb, raw predictions in vggt/scene.npz).

VGGT runs once on the GPU via reconstruct.py. The confidence filter is applied when the GLB is exported, so
changing it only re-exports from the saved predictions (CPU, seconds) instead of re-running the model.
"""
import sys
import threading

import numpy as np

from app.project import ROOT

KEY = "reconstruct"
MAX_POINTS = 4_000_000  # same default as reconstruct.py; the mesh stage voxel-downsamples anyway


def export_glb(npz_path, glb_path, conf_thres: float) -> int:
    vggt_dir = str(ROOT / "vggt-omega")
    if vggt_dir not in sys.path:
        sys.path.insert(0, vggt_dir)
    from visual_util import predictions_to_glb

    pred = dict(np.load(npz_path))
    scene = predictions_to_glb(pred, conf_thres=conf_thres, max_points=MAX_POINTS)
    scene.export(glb_path)
    return sum(len(g.vertices) for g in scene.geometry.values() if g.__class__.__name__ == "PointCloud")


def build(app, folder):
    gui = app.server.gui
    with folder:
        frames_md = gui.add_markdown("")
        mode = gui.add_dropdown("Input size", ("balanced", "max_size"), initial_value="balanced",
                                hint="max_size uses smaller inputs and less VRAM, if balanced runs out of memory")
        conf = gui.add_slider("Drop low confidence (%)", min=0, max=90, step=5, initial_value=25,
                              hint="Percentile of lowest-confidence points to remove")
        run_btn = gui.add_button("Run VGGT", icon="player-play")
        apply_btn = gui.add_button("Re-apply confidence filter", icon="filter", disabled=True,
                                   hint="Re-export from the saved predictions; no GPU needed")
        bar = gui.add_progress_bar(100.0, animated=True, visible=False)
        summary = gui.add_markdown("")

    def paths():
        p = app.project
        return p.path / "frames", p.path / "vggt" / "scene.glb", p.path / "vggt" / "scene.npz"

    def refresh():
        p = app.project
        if p is None:
            return
        frames, glb, npz = paths()
        n = len(list(frames.glob("frame_*.jpg"))) if frames.exists() else 0
        frames_md.content = f"**{n} frames** ready from Capture." if n else "*Run Capture first.*"
        run_btn.disabled = n == 0
        apply_btn.disabled = not npz.exists()
        entry = p.state["stages"][KEY]
        summary.content = f"Last result: {entry['note']}" if entry.get("note") and entry["status"] == "done" else ""

    def finish(note: str | None, error: str | None = None):
        p = app.project
        if error:
            p.set_status(KEY, "error", note=error)
        else:
            p.set_status(KEY, "done", note=note, conf_thres=conf.value)
            p.invalidate_downstream(KEY)
        bar.visible = False
        app.scene.show_project(p, log=app.log)
        app.stage_changed(KEY)  # refreshes the status badges and the later stages' panels
        refresh()

    @run_btn.on_click
    def _(_):
        p = app.project
        frames, glb, npz = paths()
        glb.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, ROOT / "reconstruct.py", frames, "--out", glb, "--save-npz",
               "--mode", mode.value, "--conf-thres", conf.value, "--max-points", MAX_POINTS]

        def on_done(code):
            if code == 0 and glb.exists():
                finish(f"{len(list(frames.glob('frame_*.jpg')))} frames, {conf.value:.0f}% low-confidence dropped")
            else:
                finish(None, error=f"reconstruct.py exited with code {code}; see log")

        if app.jobs.run("VGGT reconstruction", cmd, cwd=ROOT, on_done=on_done,
                        env={"HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}):
            p.set_status(KEY, "running")
            app.refresh_status()
            bar.visible = True

    @apply_btn.on_click
    def _(_):
        _, glb, npz = paths()
        apply_btn.disabled, bar.visible = True, True
        app.log(f"re-exporting point cloud with {conf.value:.0f}% dropped")

        def work():
            try:
                n = export_glb(npz, glb, conf.value)
                app.log(f"exported {n:,} points")
                finish(f"{conf.value:.0f}% low-confidence dropped, {n:,} points")
            except Exception as e:  # noqa: BLE001
                app.log(f"export failed: {e}")
                finish(None, error=str(e))

        threading.Thread(target=work, daemon=True).start()

    app.on_project_opened(refresh)
    app.on_stage_changed(lambda key: refresh() if key == "capture" else None)
