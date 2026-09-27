"""P11 Export: package the best checkpoint as a deployable model, and preview its detections on held-out scans.

export/ gets the model (weights + the config it was trained with, inside one .pth), the inference script, the
sensor mount it was trained for, and a model card that states the input frame. The preview runs the model in WSL
on validation scans and draws its boxes (green, with scores) next to the simulated ground truth (red) over the
site, in the world frame.
"""
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from app import simdata
from app.pcd import read_pcd
from app.project import ROOT
from app.scanview import EDGES
from app.wsl import pcdet_cmd, to_wsl

KEY = "export"
PREVIEW_FRAMES = 40
REAL_MIN_SCORE = 0.1  # inference keeps everything above this; the viewer's slider filters further
MODEL_NAME = "pedestrian_pointpillars.pth"

README = """# {project}: LiDAR pedestrian detector

PointPillars (OpenPCDet), fine-tuned from KITTI on simulated scans of this site's digital twin.

- Sensor: **{sensor}** mounted at {position} m (world), roll/pitch/yaw {rpy} deg, i.e. **{height:.2f} m above the
  floor**. Deploy at the same mount; the model has only seen this height and tilt.
- Input frame: the sensor's own frame **leveled** (roll and pitch removed, z up, origin at the sensor, x along the
  sensor's heading). Rotate raw scans by the mount's roll and pitch first.
- Features: x, y, z (intensity is ignored: simulated intensity does not match real sensors).
- Range: ±{range_xy} m in x and y; floor at z = {floor_z} m in the input frame.
- Validation (held-out simulated runs): AP {ap:.3f} at 3D IoU 0.5, recall {recall:.2f} / precision {precision:.2f}
  at score 0.3 ({n_gt} pedestrians).
- Trained on {train} scans ({val} validation); best checkpoint from epoch {epochs}; dataset filters: {filters}.
- Training augmentation: {augmentation}.

Run (OpenPCDet environment, e.g. training/setup_wsl.sh):

    python infer.py --ckpt {model} --points scan.npy [--score 0.3] [--out preds.json]

Output boxes are [x, y, z, dx, dy, dz, yaw] in the input frame, z at the box center.
"""


def obb_segments(boxes, color):
    segs = []
    for b in boxes:
        c, e, yaw = np.array(b[:3]), np.array(b[3:6]) / 2, b[6]
        local = np.array([[sx * e[0], sy * e[1], sz * e[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        R = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])  # row-vector
        corners = local @ R + c
        segs += [[corners[i], corners[j]] for i, j in EDGES]
    segs = np.array(segs, np.float32).reshape(-1, 2, 3)
    return segs, np.broadcast_to(np.array(color, np.uint8), segs.shape).copy()


def package(project):
    train, ds = project.path / "train", project.path / "dataset"
    state_rows = [json.loads(x) for x in (train / "metrics.jsonl").read_text().splitlines() if x.strip()]
    evals = [r for r in state_rows if r["type"] == "eval" and r["ap"] == r["ap"]]
    start = next((r for r in state_rows if r["type"] == "start"), {})
    aug = start.get("augmentation") or {}
    best = max(evals, key=lambda r: r["ap"]) if evals else {"ap": float("nan"), "recall": 0, "precision": 0,
                                                             "n_gt": 0, "epoch": -1}
    meta = json.loads((ds / "dataset.json").read_text())
    sensor = json.loads((project.path / "sensor.json").read_text())
    out = project.path / "export"
    out.mkdir(exist_ok=True)
    shutil.copy(train / "ckpt" / "best.pth", out / MODEL_NAME)
    shutil.copy(train / "pointpillar_ped.yaml", out / "pointpillar_ped.yaml")
    shutil.copy(ROOT / "training" / "infer.py", out / "infer.py")
    shutil.copy(ROOT / "training" / "train.py", out / "train.py")  # infer.py imports its OpenPCDet setup
    shutil.copy(project.path / "sensor.json", out / "sensor_mount.json")
    card = {"model": MODEL_NAME, "architecture": "PointPillars (OpenPCDet)", "classes": ["Pedestrian"],
            "sensor": sensor.get("model"), "mount_position_world_m": sensor["position"],
            "mount_rpy_deg": sensor.get("rpy_deg"), "input_frame": "sensor frame leveled (roll/pitch removed), z up",
            "features": ["x", "y", "z"], "range_xy_m": meta["range_xy"], "floor_z_m": meta["floor_z"],
            "val": {k: best[k] for k in ("ap", "recall", "precision", "n_gt")}, "best_epoch": best["epoch"] + 1,
            "dataset": meta["counts"], "filters": meta["params"], "augmentation": aug, "exported": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    p = meta["params"]
    (out / "README.md").write_text(README.format(
        project=project.name, sensor=sensor.get("model"), position=[round(v, 2) for v in sensor["position"]],
        rpy=[round(v, 1) for v in (sensor.get("rpy_deg") or [0, 0, 0])], height=sensor["position"][2],
        range_xy=meta["range_xy"], floor_z=meta["floor_z"], ap=best["ap"], recall=best["recall"],
        precision=best["precision"], n_gt=best["n_gt"], train=meta["counts"].get("train", 0),
        val=meta["counts"].get("val", 0), epochs=best["epoch"] + 1, model=MODEL_NAME,
        augmentation=", ".join(filter(None, [
            "flip, rotation, scaling",
            f"range noise σ {aug['noise_sigma'] * 100:g} cm" if aug.get("noise_sigma") else "",
            f"point dropout {aug['drop_prob']:.0%}" if aug.get("drop_prob") else "",
            "occlusion (pyramid dropout)" if aug.get("occlusion") else ""])),
        filters=f"overlap ≤ {p['overlap']:.0%}, containment ≥ {p['min_containment']:.0%} (judged from "
                f"{p['min_points']} points), labels need {p['min_label_points']} points"), encoding="utf-8")
    return card


def build(app, folder):
    gui, scene = app.server.gui, app.server.scene
    with folder:
        state_md = gui.add_markdown("")
        export_btn = gui.add_button("Export model", icon="package-export", color="green")
        with gui.add_folder("Preview detections", expand_by_default=True):
            preview_btn = gui.add_button("Run on validation scans", icon="player-play")
            frame_sl = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0, visible=False)
            frame_md = gui.add_markdown("")
        with gui.add_folder("Test on real scans", expand_by_default=True):
            gui.add_markdown("A folder of real scans (`.pcd`, `.npy` or KITTI `.bin`, one per frame, meters, sensor "
                             "origin, z up), used as they are.")
            real_dir = gui.add_text("Folder", "")
            real_btn = gui.add_button("Run model on folder", icon="player-play")
            real_frame = gui.add_slider("Real frame", min=0, max=1, step=1, initial_value=0, visible=False)
            real_play = gui.add_checkbox("Play (10 Hz)", False, visible=False)
            real_score = gui.add_slider("Min score", min=REAL_MIN_SCORE, max=0.9, step=0.05, initial_value=0.3,
                                        visible=False)
            real_yaw = gui.add_slider("Heading in site (deg)", min=-180, max=180, step=1, initial_value=0,
                                      visible=False, hint="turn the scan to line it up with the site mesh "
                                                          "(display only; the model sees the scan as is)")
            real_md = gui.add_markdown("")

    st = {"preds": None, "ids": [], "h": {}, "real": None}

    def clear():
        for h in st["h"].values():
            h.remove()
        st["h"] = {}

    def frame_to_world(fid):
        """dataset frame id r<k>_<scan> -> (run dir, scan index, training->world 4x4)."""
        meta = json.loads((app.project.path / "dataset" / "dataset.json").read_text())
        k, i = fid[1:].split("_")
        rd = app.project.path / "sim" / "runs" / list(meta["runs"])[int(k)]
        M = np.linalg.inv(simdata.train_rotation(rd)) @ simdata.sensor_to_world(rd)
        return rd, int(i), M

    def to_world_boxes(boxes, M):
        out = []
        for b in boxes:
            c = np.append(b[:3], 1.0) @ M
            heading = np.array([np.cos(b[6]), np.sin(b[6]), 0.0]) @ M[:3, :3]
            out.append([*c[:3], *b[3:6], float(np.arctan2(heading[1], heading[0]))])
        return out

    def show_frame():
        if not st["ids"]:
            return
        fid = st["ids"][int(frame_sl.value)]
        ds = app.project.path / "dataset"
        rd, _, M = frame_to_world(fid)
        pts = np.load(ds / "points" / f"{fid}.npy")
        world = simdata.transform(pts, M).astype(np.float32)
        gt = [[float(v) for v in ln.split()[:7]] for ln in (ds / "labels" / f"{fid}.txt").read_text().splitlines()
              if ln.endswith(" Pedestrian")]
        pred = st["preds"][fid]
        st["h"]["pts"] = scene.add_point_cloud("/export/points", world, np.full((len(world), 3), 170, np.uint8),
                                               point_size=0.025)
        for name, boxes, color in (("gt", gt, (230, 60, 60)), ("pred", pred["boxes"], (40, 220, 90))):
            if boxes:
                segs, cols = obb_segments(to_world_boxes(boxes, M), color)
                st["h"][name] = scene.add_line_segments(f"/export/{name}", segs, cols, thickness=3,
                                                        thickness_units="screen")
            elif name in st["h"]:
                st["h"].pop(name).remove()
        scores = ", ".join(f"{s:.2f}" for s in pred["scores"])
        frame_md.content = (f"`{fid}` ({rd.name}): **{len(pred['boxes'])} detected** (green; scores {scores or '-'}) "
                            f"vs **{len(gt)} ground truth** (red)")

    frame_sl.on_update(lambda _: show_frame())

    # ---- real scans ------------------------------------------------------------------------------------------
    def real_root(src: Path):
        return app.project.path / "export" / "real" / src.name.replace(" ", "_")

    def show_real():
        r = st["real"]
        if not r or not r["ids"]:
            return
        fid = r["ids"][int(real_frame.value)]
        pts = np.load(r["root"] / "points" / f"{fid}.npy")
        sensor = json.loads((app.project.path / "sensor.json").read_text())
        a = np.radians(real_yaw.value)
        M = np.eye(4)  # sensor frame -> world, row-vector: turn by the heading slider, put at the mount position
        M[:2, :2] = [[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]
        M[3, :3] = sensor["position"]
        world = simdata.transform(pts[:, :3], M).astype(np.float32)
        h = np.clip((pts[:, 2] - pts[:, 2].min()) / max(1e-6, np.ptp(pts[:, 2])), 0, 1)
        colors = (np.column_stack([0.35 + 0.5 * h, 0.45 + 0.3 * h, 0.9 - 0.6 * h]) * 255).astype(np.uint8)
        for k in ("pts", "gt", "pred"):
            if k in st["h"]:
                st["h"].pop(k).remove()
        st["h"]["pts"] = scene.add_point_cloud("/export/points", world, colors, point_size=0.02)
        pred = r["preds"].get(fid, {"boxes": [], "scores": []})
        keep = [k for k, sc in enumerate(pred["scores"]) if sc >= real_score.value]
        boxes = [pred["boxes"][k] for k in keep]
        if boxes:
            segs, cols = obb_segments(to_world_boxes(boxes, M), (40, 220, 90))
            st["h"]["pred"] = scene.add_line_segments("/export/pred", segs, cols, thickness=3,
                                                      thickness_units="screen")
        scores = ", ".join(f"{pred['scores'][k]:.2f}" for k in keep)
        n_any = sum(1 for f in r["ids"] if any(sc >= real_score.value for sc in r["preds"].get(f, {}).get("scores", [])))
        real_md.content = (f"`{fid}`: **{len(boxes)} pedestrian(s)** (scores {scores or '-'}); "
                           f"{len(pts):,} points  \n{n_any} of {len(r['ids'])} frames have a detection at this score")

    def load_real(root: Path):
        f = root / "preds.json"
        if not f.exists():
            return
        preds = json.loads(f.read_text())
        ids = sorted(i for i in preds if (root / "points" / f"{i}.npy").exists())
        st["real"] = {"root": root, "preds": preds, "ids": ids}
        clear()
        frame_sl.visible = False
        real_frame.max = max(1, len(ids) - 1)
        real_frame.value = 0
        for w in (real_frame, real_play, real_score, real_yaw):
            w.visible = True
        show_real()

    for w in (real_frame, real_score, real_yaw):
        w.on_update(lambda _: show_real())

    def play_loop():
        while True:
            time.sleep(0.1)
            r = st["real"]
            if real_play.value and r and r["ids"]:
                real_frame.value = (int(real_frame.value) + 1) % len(r["ids"])  # fires show_real

    threading.Thread(target=play_loop, daemon=True).start()

    @real_btn.on_click
    def _(_):
        src = Path(real_dir.value.strip().strip('"'))
        if not src.is_dir():
            app.log(f"not a folder: {src}")
            return
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in (".pcd", ".npy", ".bin"))
        if not files:
            app.log(f"no .pcd / .npy / .bin files in {src}")
            return
        job = f"Detect on {src.name} (WSL)"
        if not app.jobs.reserve(job):  # held through the conversion, before the earlier result is deleted
            return
        root = real_root(src)
        ckpt = app.project.path / "export" / MODEL_NAME
        if not ckpt.exists():
            ckpt = app.project.path / "train" / "ckpt" / "best.pth"

        def convert_then_run():
            pts_dir, out = root / "points", root / "preds.json"

            def done(code):
                if out.exists():
                    load_real(root)
                    app.log(f"real scans: detections for {len(files)} frames; use the Real frame slider or Play")
                else:
                    app.log(f"detection on real scans failed (exit {code}); see log")

            try:  # the runner stays reserved until run() takes it over; give it back if anything before that fails
                if pts_dir.exists():
                    shutil.rmtree(pts_dir)
                pts_dir.mkdir(parents=True)
                for k, f in enumerate(files):
                    ext = f.suffix.lower()
                    pts = (read_pcd(f) if ext == ".pcd" else np.load(f) if ext == ".npy"
                           else np.fromfile(f, np.float32).reshape(-1, 4))
                    np.save(pts_dir / f"{f.stem}.npy", np.asarray(pts, np.float32))
                    if k % 100 == 0:
                        app.log(f"converted {k + 1}/{len(files)} scans")
                out.unlink(missing_ok=True)
                app.log(f"{len(files)} scans in {pts_dir}; running the model in WSL…")
                app.jobs.run(job,
                             pcdet_cmd(ROOT / "training" / "infer.py", "--ckpt", to_wsl(ckpt), "--dir",
                                       to_wsl(pts_dir), "--score", str(REAL_MIN_SCORE), "--out", to_wsl(out)),
                             cwd=ROOT, on_done=done)
            except Exception as e:  # noqa: BLE001
                app.jobs.release(job)
                app.log(f"detection on real scans failed to start: {e}")

        app.log(f"reading {len(files)} scans from {src}…")
        threading.Thread(target=convert_then_run, daemon=True).start()

    @export_btn.on_click
    def _(_):
        try:
            card = package(app.project)
        except Exception as e:  # noqa: BLE001
            app.log(f"export failed: {e}")
            return
        app.project.set_status(KEY, "done", note=f"AP {card['val']['ap']:.3f}")
        app.log(f"exported to {app.project.path / 'export'}")
        app.refresh_status()
        refresh_state()

    @preview_btn.on_click
    def _(_):
        p = app.project
        job = "Detect on validation scans (WSL)"
        if not app.jobs.reserve(job):  # before the earlier preview is deleted
            return
        ds = p.path / "dataset"
        out = p.path / "export" / "preview.json"

        def done(code):
            if out.exists():
                load_preview()
            else:
                app.log(f"preview failed (exit {code}); see log")

        try:  # the runner stays reserved until run() takes it over; give it back if anything before that fails
            val = (ds / "ImageSets" / "val.txt").read_text().split()
            ids = [val[int(k)] for k in np.linspace(0, len(val) - 1, min(PREVIEW_FRAMES, len(val)))]
            out.parent.mkdir(exist_ok=True)
            out.unlink(missing_ok=True)
            app.jobs.run(job,
                         pcdet_cmd(ROOT / "training" / "infer.py", "--ckpt",
                                   to_wsl(p.path / "train" / "ckpt" / "best.pth"), "--data", to_wsl(ds),
                                   "--ids", ",".join(ids), "--out", to_wsl(out)),
                         cwd=ROOT, on_done=done)
        except Exception as e:  # noqa: BLE001
            app.jobs.release(job)
            app.log(f"preview failed to start: {e}")

    def load_preview():
        f = app.project.path / "export" / "preview.json"
        clear()
        if not f.exists():
            st["preds"], st["ids"] = None, []
            frame_sl.visible = False
            frame_md.content = ""
            return
        st["preds"] = json.loads(f.read_text())
        st["ids"] = [i for i in st["preds"] if (app.project.path / "dataset" / "points" / f"{i}.npy").exists()]
        if not st["ids"]:
            return
        frame_sl.max = max(1, len(st["ids"]) - 1)
        frame_sl.value = 0
        frame_sl.visible = True
        show_frame()

    def refresh_state():
        p = app.project
        has_ckpt = (p.path / "train" / "ckpt" / "best.pth").exists()
        export_btn.disabled = preview_btn.disabled = not has_ckpt
        card = p.path / "export" / "model_card.json"
        if not has_ckpt:
            state_md.content = "*Needs: a trained checkpoint (Train).*"
        elif card.exists():
            c = json.loads(card.read_text())
            state_md.content = (f"✅ `export/{c['model']}`: validation AP **{c['val']['ap']:.3f}**, for a "
                                f"{c['sensor']} at {c['mount_position_world_m'][2]:.2f} m. See `export/README.md`.")
        else:
            state_md.content = "Best checkpoint ready to export."

    def refresh():
        if app.project is None:
            return
        load_preview()
        st["real"] = None
        for w in (real_frame, real_play, real_score, real_yaw):
            w.visible = False
        real_md.content = ""
        done = sorted((app.project.path / "export" / "real").glob("*/preds.json"), key=lambda f: f.stat().st_mtime)
        if done:  # reopen the latest real-scan result
            load_real(done[-1].parent)
        refresh_state()

    app.on_project_opened(refresh)
    app.on_stage_changed(lambda key: refresh_state() if key == "train" else None)
