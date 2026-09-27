"""P10 Train: PointPillars pedestrian detector, fine-tuned on the exported dataset, in WSL (training/train.py).

The trainer streams metrics.jsonl into <project>/train/; this panel tails it and plots loss (per logged
iteration) and validation AP / recall (per epoch) live. Stop asks the trainer to finish the current epoch's
evaluation and checkpoint (a STOP file); the log panel's "Stop running job" kills it outright.
"""
import json
import shutil
import threading
import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.project import ROOT
from app.wsl import pcdet_cmd, to_wsl

KEY = "train"


def read_metrics(f):
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a half-written last line
            pass
    return out


def figure(rows):
    its = [r for r in rows if r["type"] == "iter"]
    evs = [r for r in rows if r["type"] == "eval"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.18,
                        subplot_titles=("Training loss", "Validation (IoU 0.5)"))
    if its:
        x = [r["epoch"] + (r["iter"] + 1) / r["iters_per_epoch"] for r in its]
        fig.add_trace(go.Scatter(x=x, y=[r["loss"] for r in its], name="loss", line=dict(color="#e4572e")), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=[r["loss_cls"] for r in its], name="cls", line=dict(color="#f3a712", width=1)),
                      1, 1)
        fig.add_trace(go.Scatter(x=x, y=[r["loss_loc"] for r in its], name="box", line=dict(color="#29335c", width=1)),
                      1, 1)
    if evs:
        x = [r["epoch"] + 1 for r in evs]
        fig.add_trace(go.Scatter(x=x, y=[r["ap"] for r in evs], name="AP", mode="lines+markers",
                                 line=dict(color="#17a34a")), 2, 1)
        fig.add_trace(go.Scatter(x=x, y=[r["recall"] for r in evs], name="recall", mode="lines+markers",
                                 line=dict(color="#2563eb", dash="dot")), 2, 1)
    fig.update_xaxes(title_text="epoch", row=2, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_layout(height=420, margin=dict(l=40, r=10, t=30, b=30), legend=dict(orientation="h", y=-0.15))
    return fig


def summary(rows):
    start = next((r for r in rows if r["type"] == "start"), None)
    if start is None:
        return "Starting up (building the dataset index in WSL)…" if rows is not None else ""
    its = [r for r in rows if r["type"] == "iter"]
    evs = [r for r in rows if r["type"] == "eval"]
    done = next((r for r in rows if r["type"] == "done"), None)
    lines = [f"{start['train_frames']:,} train / {start['val_frames']:,} val scans, range ±{start['range_xy']} m, "
             f"{'fine-tuning KITTI PointPillars' if start['pretrained'] else 'from scratch'}"]
    if its:
        r = its[-1]
        frac = (r["epoch"] + (r["iter"] + 1) / r["iters_per_epoch"]) / start["epochs"]
        eta = r["elapsed"] / max(frac, 1e-6) * (1 - frac)
        lines.append(f"Epoch {r['epoch'] + 1}/{start['epochs']}, iteration {r['iter'] + 1}/{r['iters_per_epoch']}: "
                     f"loss **{r['loss']:.3f}**" + ("" if done else f", about {eta / 60:.1f} min left"))
    if evs:
        e = evs[-1]
        best = max((x for x in evs if x["ap"] == x["ap"]), key=lambda x: x["ap"], default=e)
        lines.append(f"Last eval: AP **{e['ap']:.3f}**, recall {e['recall']:.2f}, precision {e['precision']:.2f} "
                     f"({e['n_gt']:,} pedestrians). Best AP {best['ap']:.3f} at epoch {best['epoch'] + 1}.")
    if done:
        lines.append(f"✅ Finished in {done['elapsed'] / 60:.1f} min. Best checkpoint: `train/ckpt/best.pth`.")
    return "  \n".join(lines)


def build(app, folder):
    gui = app.server.gui
    with folder:
        state_md = gui.add_markdown("")
        epochs = gui.add_slider("Epochs", min=1, max=80, step=1, initial_value=10)
        batch = gui.add_slider("Batch size", min=1, max=16, step=1, initial_value=4,
                               hint="4 fits an 8 GB GPU")
        pretrained = gui.add_checkbox("Fine-tune from KITTI", True,
                                      hint="Start from OpenPCDet's KITTI PointPillars weights (converges much faster)")
        with gui.add_folder("Augmentation (sensor noise)", expand_by_default=False):
            gui.add_markdown("<small>Off by default: Isaac's RTX LiDAR already models the sensor. Add some if real "
                             "scans look noisier than the simulated ones.</small>")
            noise_cm = gui.add_slider("Range noise σ (cm)", min=0.0, max=5.0, step=0.5, initial_value=0.0,
                                      hint="Gaussian range error along each ray (QT64 spec is about ±3 cm)")
            drop_pct = gui.add_slider("Point dropout (%)", min=0, max=40, step=1, initial_value=0,
                                      hint="randomly remove points, like returns a real sensor never gets")
            occlusion = gui.add_checkbox("Occlusion (drop parts of people)", False,
                                         hint="OpenPCDet's pyramid dropout: removes a random part of a pedestrian")
        start_btn = gui.add_button("Start training", icon="brain", color="green")
        stop_btn = gui.add_button("Stop after this epoch", icon="player-stop", visible=False)
        prog_md = gui.add_markdown("")
        plot = gui.add_plotly(figure([]), aspect=1.0, visible=False)

    st = {"running": False, "n": -1}
    out_dir = lambda: app.project.path / "train"  # noqa: E731

    def update(force=False):
        rows = read_metrics(out_dir() / "metrics.jsonl")
        if len(rows) == st["n"] and not force:
            return
        st["n"] = len(rows)
        if rows:
            plot.figure = figure(rows)
            plot.visible = True
        prog_md.content = summary(rows) if (rows or st["running"]) else ""

    def poll():
        while True:
            time.sleep(1.0)
            if st["running"] and app.project is not None:
                try:
                    update()
                except Exception as e:  # noqa: BLE001
                    app.log(f"metrics read failed: {e}")

    threading.Thread(target=poll, daemon=True).start()

    @start_btn.on_click
    def _(_):
        p = app.project
        job = "Train PointPillars (WSL)"
        if not app.jobs.reserve(job):  # before the previous model is deleted
            return
        out = out_dir()

        def finished(code):
            st["running"] = False
            update(force=True)
            rows = read_metrics(out / "metrics.jsonl")
            done = next((r for r in rows if r["type"] == "done"), None)
            if (out / "ckpt" / "best.pth").exists():
                p.set_status(KEY, "done", note=f"best AP {done['best_ap']:.3f}" if done else "stopped early")
                p.invalidate_downstream(KEY)
            else:
                p.set_status(KEY, "error", note=f"no checkpoint (exit {code}); see log")
                if code != 0 and not rows and not app.jobs.stop_requested:
                    app.log("training did not start: is WSL set up? Run: wsl -d Ubuntu -- bash "
                            f"{to_wsl(ROOT / 'training' / 'setup_wsl.sh')}")
            start_btn.disabled, stop_btn.visible = False, False
            app.stage_changed(KEY)
            refresh_state()

        try:  # the runner stays reserved until run() takes it over; give it back if anything before that fails
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            args = ["--data", to_wsl(p.path / "dataset"), "--out", to_wsl(out), "--epochs", int(epochs.value),
                    "--batch", int(batch.value)]
            if not pretrained.value:
                args += ["--pretrained", ""]
            args += ["--noise-sigma", round(float(noise_cm.value) / 100, 4),
                     "--drop-prob", round(float(drop_pct.value) / 100, 3)]
            if occlusion.value:
                args.append("--occlusion")
            st.update(running=True, n=-1)
            plot.visible = False
            start_btn.disabled, stop_btn.visible = True, True
            p.set_status(KEY, "running")
            app.refresh_status()
            prog_md.content = "Starting up (building the dataset index in WSL)…"
            app.jobs.run(job, pcdet_cmd(ROOT / "training" / "train.py", *args), cwd=ROOT, on_done=finished)
        except Exception as e:  # noqa: BLE001
            app.jobs.release(job)
            st["running"] = False
            start_btn.disabled, stop_btn.visible = False, False
            app.log(f"could not start training: {e}")

    @stop_btn.on_click
    def _(_):
        (out_dir() / "STOP").touch()
        app.log("training will stop after this epoch's evaluation")
        stop_btn.visible = False

    def refresh_state():
        p = app.project
        ds = p.path / "dataset" / "dataset.json"
        if not ds.exists():
            state_md.content = "*Needs: a built dataset (Filter & dataset).*"
            start_btn.disabled = True
            return
        m = json.loads(ds.read_text())
        c = m["counts"]
        state_md.content = f"Dataset: {c.get('train', 0):,} train / {c.get('val', 0):,} val scans."
        start_btn.disabled = st["running"]

    def refresh():
        if app.project is None:
            return
        st["n"] = -1
        plot.visible = False
        update(force=True)
        refresh_state()

    app.on_project_opened(refresh)
    app.on_stage_changed(lambda key: refresh_state() if key == "dataset" else None)
