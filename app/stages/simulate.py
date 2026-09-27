"""P8 Simulate: characters walk through the digital twin while the simulated LiDAR scans them (Isaac Sim, headless).

One press chains jobs:
  1. sim/build_stage.py       mesh_world + walkarea.json -> sim/stage.usd (NavMeshVolumes, collision floor)
  2. sim/probe_navmesh.py     bake the navmesh once and draw it (where people can actually walk); fails early
  3. sim/run_capture.py x N   one Isaac process per run: IRA pedestrians + the placed LiDAR, --boxes
While a run captures, its newest scan streams into the 3D view (pedestrian points colored per person, their boxes
drawn). Afterwards any run/scan can be browsed or played back.
Runs vary only pedestrian properties (seed, count); train/val is later split by run.
"""
import json
import random
import re
import shutil
import sys
import threading
import time

import numpy as np

from app import simdata
from app.project import ROOT
from app.scanview import ScanDrawer
from sim.ira_config import write_config

KEY = "simulate"
ISAAC_PY = r"C:\isaacsim\python.bat"
SCAN_RE = re.compile(r"\[collect\] scan (\d+)/(\d+)")


def run_dirs(project):
    return simdata.list_runs(project)


def build(app, folder):
    gui, scene = app.server.gui, app.server.scene
    with folder:
        state_md = gui.add_markdown("")
        n_runs = gui.add_slider("Runs", min=1, max=20, step=1, initial_value=4,
                                hint="Each run is a fresh Isaac process with its own seed and crowd; "
                                     "train/val is split by run, so use 3 or more")
        peds = gui.add_multi_slider("Pedestrians per run", min=1, max=12, step=1, initial_value=(2, 5))
        scans = gui.add_slider("Scans per run", min=20, max=600, step=10, initial_value=150,
                               hint="LiDAR revolutions at 10 Hz (150 = 15 s)")
        probe = gui.add_checkbox("Check navmesh first", True)
        run_btn = gui.add_button("Simulate", icon="walk")
        bar = gui.add_progress_bar(0.0, visible=False)
        progress_md = gui.add_markdown("")
        with gui.add_folder("Browse scans", expand_by_default=True):
            run_dd = gui.add_dropdown("Run", ["(none)"])
            scan_sl = gui.add_slider("Scan", min=0, max=1, step=1, initial_value=0)
            play = gui.add_checkbox("Play", False)
            scan_md = gui.add_markdown("")

    drawer = ScanDrawer(scene, "/sim/scan")
    st = {"live_run": None, "run_idx": 0, "n_runs": 0, "nav": None, "last_drawn": None, "syncing": False}

    # ---- live progress from the capture's stdout -------------------------------------------------------------
    def on_line(line):
        m = SCAN_RE.search(line)
        if m and st["live_run"] is not None:
            i, n = int(m.group(1)), int(m.group(2))
            bar.value = 100.0 * (st["run_idx"] + i / n) / max(1, st["n_runs"])
            progress_md.content = f"Run {st['run_idx'] + 1}/{st['n_runs']}: scan {i}/{n}"

    app.jobs.on_line(on_line)

    def live_loop():
        """Show the newest finished scan of the run being captured."""
        while True:
            time.sleep(0.4)
            rd = st["live_run"]
            if rd is None or not (rd / "velodyne").exists():
                continue
            n = simdata.num_scans(rd)
            if n < 2 or st["last_drawn"] == (rd, n - 2):
                continue
            try:  # n-1 may still be being written; n-2 is complete
                drawer.draw(rd, n - 2)
                st["last_drawn"] = (rd, n - 2)
            except Exception:  # noqa: BLE001 - a half-written file; next tick
                pass

    def play_loop():
        while True:
            time.sleep(0.1)
            if play.value and st["live_run"] is None and scan_sl.max > 0:
                scan_sl.value = (scan_sl.value + 1) % (scan_sl.max + 1)

    threading.Thread(target=live_loop, daemon=True).start()
    threading.Thread(target=play_loop, daemon=True).start()

    # ---- browsing ------------------------------------------------------------------------------------------
    def selected_run():
        p = app.project
        if p is None or run_dd.value == "(none)":
            return None
        rd = p.path / "sim" / "runs" / run_dd.value
        return rd if rd.exists() else None

    def show_scan():
        rd = selected_run()
        if rd is None or st["live_run"] is not None:
            return
        try:
            info = drawer.draw(rd, int(scan_sl.value))
        except Exception as e:  # noqa: BLE001
            scan_md.content = f"*{e}*"
            return
        scan_md.content = (f"{info['points']:,} points, **{info['ped_points']:,} on pedestrians**, "
                           f"{info['boxes']} boxes")

    def refresh_runs(select=None):
        runs = [r.name for r in run_dirs(app.project)] if app.project else []
        st["syncing"] = True
        run_dd.options = runs or ["(none)"]
        run_dd.value = select if select in runs else (runs[-1] if runs else "(none)")
        st["syncing"] = False
        on_run_selected()

    def on_run_selected():
        rd = selected_run()
        n = simdata.num_scans(rd) if rd else 0
        scan_sl.max = max(1, n - 1)
        scan_sl.value = 0
        if rd is None:
            drawer.clear()
            scan_md.content = ""
        else:
            show_scan()

    run_dd.on_update(lambda _: None if st["syncing"] else on_run_selected())
    scan_sl.on_update(lambda _: show_scan())

    # ---- navmesh overlay -----------------------------------------------------------------------------------
    def draw_navmesh():
        f = app.project.path / "sim" / "navmesh.json"
        if st["nav"] is not None:
            st["nav"].remove()
            st["nav"] = None
        if not f.exists():
            return
        data = json.loads(f.read_text())
        if not data.get("tris"):
            return
        tris = np.asarray(data["tris"], np.float32)
        verts = tris.reshape(-1, 3) + [0, 0, 0.02]
        faces = np.arange(len(verts)).reshape(-1, 3)
        st["nav"] = scene.add_mesh_simple("/sim/navmesh", verts, faces, color=(60, 200, 255), opacity=0.5,
                                          side="double")

    # ---- the job chain -------------------------------------------------------------------------------------
    def fail(msg):
        st["live_run"] = None
        bar.visible = False
        app.project.set_status(KEY, "error", note=msg)
        app.log(msg)
        app.refresh_status()
        refresh_state()

    @run_btn.on_click
    def _(_):
        p = app.project
        if app.jobs.refuse_if_busy():
            return
        prev = dict(p.state["stages"][KEY])
        stage = p.path / "sim" / "stage.usd"
        sensor = p.path / "sensor.json"
        walk = json.loads((p.path / "walkarea.json").read_text())
        extent = max(max(r["size"]) for r in walk["rects"])
        lo, hi = (int(v) for v in peds.value)
        rng = random.Random(int(time.time()))
        plan = [(rng.randrange(2 ** 31), rng.randint(lo, hi)) for _ in range(int(n_runs.value))]
        n_scans = int(scans.value)
        runs_root = p.path / "sim" / "runs"
        start_idx = len(run_dirs(p))
        st.update(run_idx=0, n_runs=len(plan))
        bar.visible, bar.value = True, 0.0
        p.set_status(KEY, "running")
        app.refresh_status()

        completed = []  # this press's complete runs

        def finish():
            """All runs finished, or the user stopped the chain: the complete runs go on to the dataset stage."""
            stopped = app.jobs.stop_requested
            st["live_run"] = None
            bar.visible = False
            if stopped and not completed:  # nothing new: the stage is as it was before this press
                p.set_status(KEY, prev["status"], note=prev.get("note", ""))
                progress_md.content = "Stopped before a run completed."
                app.refresh_status()
            else:
                done = [r for r in run_dirs(p) if simdata.run_info(r).get("complete")]
                note = f"{len(done)} runs, {sum(simdata.num_scans(r) for r in done):,} scans"
                if stopped:
                    note += f" (stopped after {len(completed)} of {len(plan)} runs)"
                p.set_status(KEY, "done", note=note)
                p.invalidate_downstream(KEY)
                progress_md.content = (f"Stopped after {len(completed)} of {len(plan)} run(s)." if stopped
                                       else f"Finished {len(plan)} run(s).")
                app.stage_changed(KEY)
            refresh_runs()
            refresh_state()

        def start(name, cmd, on_done):
            if not app.jobs.run(name, cmd, cwd=ROOT, on_done=on_done):
                fail("another job is running")

        def capture(k):
            if k >= len(plan):
                finish()
                return
            seed, n_peds = plan[k]
            rd = runs_root / f"run_{start_idx + k + 1:03d}_peds{n_peds}_seed{seed}"
            rd.mkdir(parents=True, exist_ok=True)
            shutil.copy(sensor, rd / "sensor.json")
            cfg = write_config(rd / "ira.yaml", stage, seed, n_peds, n_scans, extent)
            st.update(run_idx=k, live_run=rd, last_drawn=None)
            drawer.clear()
            cmd = ["cmd", "/c", ISAAC_PY, ROOT / "sim" / "run_capture.py", "--config", cfg, "--sensor",
                   rd / "sensor.json", "--out-dir", rd, "--num-scans", n_scans, "--expected-peds", n_peds, "--boxes"]

            def captured(code):
                ok = (rd / "run_info.json").exists() and simdata.run_info(rd).get("complete")
                app.log(f"run {k + 1}/{len(plan)} {'complete' if ok else f'FAILED (exit {code})'}: {rd.name}")
                if ok:
                    completed.append(rd)
                if app.jobs.stop_requested:
                    finish()
                    return
                if not ok and k == 0 and code != 0:
                    fail(f"capture failed (exit {code}); see log")
                    return
                capture(k + 1)

            start(f"Capture run {k + 1}/{len(plan)} ({n_peds} pedestrians)", cmd, captured)

        def probed(code):
            if app.jobs.stop_requested:
                finish()
                return
            if probe.value and not (p.path / "sim" / "navmesh.json").exists():
                fail(f"navmesh did not bake (exit {code}); check the walking area")
                return
            if probe.value:
                nav = json.loads((p.path / "sim" / "navmesh.json").read_text())
                app.log(f"navmesh: {nav['triangles']} triangles, {nav['area_m2']:.1f} m² walkable")
                draw_navmesh()
            capture(0)

        def built(code):
            if app.jobs.stop_requested:
                finish()
                return
            if code != 0 or not stage.exists():
                fail(f"stage build failed (exit {code})")
                return
            (p.path / "sim" / "navmesh.json").unlink(missing_ok=True)
            if not probe.value:
                probed(0)
                return
            progress_md.content = "Baking the navmesh in Isaac Sim (about a minute)…"
            start("Check navmesh", ["cmd", "/c", ISAAC_PY, ROOT / "sim" / "probe_navmesh.py", stage, "--out",
                                    p.path / "sim" / "navmesh.json"], probed)

        progress_md.content = "Building the Isaac stage…"
        start("Build Isaac stage", [sys.executable, ROOT / "sim" / "build_stage.py", p.path], built)

    def refresh_state():
        p = app.project
        missing = [t for t, ok in (("walking area", (p.path / "walkarea.json").exists()),
                                   ("sensor placement", (p.path / "sensor.json").exists()),
                                   ("world mesh", p.world_mesh_path() is not None)) if not ok]
        run_btn.disabled = bool(missing)
        entry = p.state["stages"][KEY]
        if missing:
            state_md.content = f"*Needs: {', '.join(missing)}.*"
        else:
            runs = run_dirs(p)
            total = sum(simdata.num_scans(r) for r in runs)
            state_md.content = (f"{len(runs)} run(s), {total:,} scans captured. " if runs else "No runs yet. ") + \
                ("" if entry["status"] != "error" else f"❌ {entry.get('note', '')}")

    def refresh():
        if app.project is None:
            return
        st["live_run"] = None
        draw_navmesh()
        refresh_runs()
        refresh_state()

    app.on_project_opened(refresh)
    app.on_stage_changed(lambda key: refresh_state() if key in ("walkarea", "sensor_place", "scale") else None)
