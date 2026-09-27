"""P9 Filter & dataset: judge every simulated scan, browse what gets dropped and why, export OpenPCDet's layout.

Filters (app/simdata.py): a scan is dropped if two pedestrians' boxes overlap
more than X% of the smaller one, if a well-seen pedestrian (>= min points) has less than Y% of its points inside
its box, or if a pedestrian has points but no box. Pedestrians with fewer than the label minimum are kept as
Pedestrian_ignore (neither rewarded nor punished). Sliders re-filter instantly from the cached per-scan analysis.

Export -> <project>/dataset/: points/<id>.npy (N x 4 float32, training frame = sensor frame leveled),
labels/<id>.txt ("x y z dx dy dz yaw class"), ImageSets/{train,val}.txt split BY RUN (neighbouring scans are
near-duplicates, so a per-frame split would leak), and dataset.json (point range and floor height for training).
"""
import json
import shutil
import threading
import time
from collections import Counter

import numpy as np

from app import simdata
from app.scanview import ScanDrawer

KEY = "dataset"
RANGES = (10.24, 20.48, 40.96)  # PointPillars grid (0.16 m pillars) divides these evenly
VAL_FRACTION = 0.2
SINGLE_RUN_GAP = 20             # scans skipped between train and val when one run has to be split in time


def split_runs(runs):
    """{run name: "train" | "val"}, or None when there is only one run (split it in time instead)."""
    if len(runs) < 2:
        return None
    n_val = max(1, round(len(runs) * VAL_FRACTION))
    return {r.name: ("val" if k >= len(runs) - n_val else "train") for k, r in enumerate(runs)}


def pick_range(max_dist):
    return next((r for r in RANGES if max_dist + 1.0 <= r), RANGES[-1])


def export(project, runs, analyses, params, progress=None):
    out = project.path / "dataset"
    split = split_runs(runs)
    ids = {"train": [], "val": []}
    stats = Counter()
    # First pass, no file IO: which scans go where, so an empty split fails before the old dataset is touched.
    kept = []  # (run index, scan index, part)
    for k, (rd, scans) in enumerate(zip(runs, analyses)):
        cut = int(len(scans) * (1 - VAL_FRACTION))
        for i, scan in enumerate(scans):
            if simdata.judge(scan, params):
                stats["dropped"] += 1
                continue
            if split is not None:
                part = split[rd.name]
            elif i < cut - SINGLE_RUN_GAP:
                part = "train"
            elif i >= cut:
                part = "val"
            else:
                stats["gap"] += 1
                continue
            kept.append((k, i, part))
    for part in ids:
        if not any(pt == part for _, _, pt in kept):
            raise ValueError(f"no {part} scans left after filtering"
                             + (" (a single run is split in time; simulate more runs or longer ones)"
                                if split is None else "; relax the filters or simulate more runs"))
    (out / "dataset.json").unlink(missing_ok=True)  # written last: it exists only for a complete dataset
    for sub in ("points", "labels", "ImageSets"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True)
    dists = []
    rotations = [simdata.train_rotation(rd) for rd in runs]
    floors = [-float(simdata.sensor_to_world(rd)[3, 2]) for rd in runs]  # world floor is z = 0; the frame is leveled
    for done, (k, i, part) in enumerate(kept, 1):
        if progress and done % 25 == 0:
            progress(done, len(kept))
        fid = f"r{k:02d}_{i:06d}"
        pts, _, _ = simdata.load_scan(runs[k], i)
        xyz = simdata.transform(pts, rotations[k])
        np.save(out / "points" / f"{fid}.npy", np.column_stack([xyz, pts[:, 3]]).astype(np.float32))
        lines = []
        for b in analyses[k][i]["boxes"]:
            cls = simdata.box_class(b, params)
            stats[cls] += 1
            if cls == "Pedestrian":
                dists.append(np.hypot(*b["center"][:2]))
            lines.append("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %s" % (*b["center"], *b["extent"], b["yaw"], cls))
        (out / "labels" / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        ids[part].append(fid)
        stats[part] += 1
    for part, lst in ids.items():
        (out / "ImageSets" / f"{part}.txt").write_text("\n".join(lst) + "\n")
    range_xy = pick_range(float(np.percentile(dists, 99)) if dists else 10.0)
    meta = {"range_xy": range_xy, "floor_z": round(float(np.mean(floors)), 4), "params": params,
            "runs": {rd.name: (split[rd.name] if split else "time-split") for rd in runs},
            "single_run_time_split": split is None, "counts": dict(stats),
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "dataset.json").write_text(json.dumps(meta, indent=2))
    return meta


def build(app, folder):
    gui, scene = app.server.gui, app.server.scene
    with folder:
        state_md = gui.add_markdown("")
        analyse_btn = gui.add_button("Analyse scans", icon="scan")
        bar = gui.add_progress_bar(0.0, visible=False)
        with gui.add_folder("Filters", expand_by_default=True):
            overlap = gui.add_slider("Max overlap %", min=0, max=100, step=1, initial_value=15,
                                     hint="Drop a scan if two pedestrians' boxes overlap more than this share of the "
                                          "smaller box (people walking into each other)")
            contain = gui.add_slider("Min containment %", min=0, max=100, step=1, initial_value=90,
                                     hint="Drop a scan if a well-seen pedestrian has fewer of its points inside its "
                                          "box (a box that doesn't match the body)")
            min_pts = gui.add_slider("Min points to judge", min=1, max=100, step=1, initial_value=10,
                                     hint="Containment is only judged for pedestrians with at least this many points")
            min_lbl = gui.add_slider("Min points to label", min=1, max=100, step=1, initial_value=10,
                                     hint="Fewer points -> Pedestrian_ignore (neither rewarded nor punished)")
            count_md = gui.add_markdown("")
        with gui.add_folder("Browse", expand_by_default=True):
            run_dd = gui.add_dropdown("Run ", ["(none)"])
            scan_sl = gui.add_slider("Scan ", min=0, max=1, step=1, initial_value=0)
            with gui.add_folder("Jump", expand_by_default=True):
                prev_btn = gui.add_button("◀ previous dropped")
                next_btn = gui.add_button("next dropped ▶")
            scan_md = gui.add_markdown("")
        export_btn = gui.add_button("Build dataset", icon="database-export", color="green")
        export_md = gui.add_markdown("")

    drawer = ScanDrawer(scene, "/dataset/scan")
    st = {"runs": [], "analyses": {}, "syncing": False, "busy": False, "gen": 0, "lock": threading.Lock(),
          "exporting": False, "stale": False}

    def params():
        return {"overlap": overlap.value / 100, "min_containment": contain.value / 100,
                "min_points": int(min_pts.value), "min_label_points": int(min_lbl.value)}

    def recount():
        if not st["analyses"]:
            count_md.content = ""
            return
        p = params()
        kept = total = 0
        reasons = Counter()
        classes = Counter()
        for rd in st["runs"]:
            for scan in st["analyses"].get(rd.name, []):
                total += 1
                r = simdata.judge(scan, p)
                if r:
                    reasons.update({x.split(":")[0] for x in r})
                else:
                    kept += 1
                    classes.update(simdata.box_class(b, p) for b in scan["boxes"])
        lines = [f"**Kept {kept:,} / {total:,} scans** ({kept / max(1, total):.0%})"]
        if reasons:
            lines.append("Dropped for: " + ", ".join(f"{k} {v:,}" for k, v in reasons.most_common()))
        lines.append(f"Labels in kept scans: {classes['Pedestrian']:,} Pedestrian, "
                     f"{classes['Pedestrian_ignore']:,} ignored")
        count_md.content = "  \n".join(lines)
        show_scan()

    for s in (overlap, contain, min_pts, min_lbl):
        s.on_update(lambda _: recount())

    def current():
        name = run_dd.value
        rd = next((r for r in st["runs"] if r.name == name), None)
        return rd, st["analyses"].get(name)

    def show_scan():
        rd, scans = current()
        if rd is None:
            return
        i = int(scan_sl.value)
        info = drawer.draw(rd, i)
        lines = [f"{info['points']:,} points, {info['boxes']} pedestrian box(es)"]
        if scans and i < len(scans):
            p = params()
            r = simdata.judge(scans[i], p)
            lines.insert(0, "❌ **Dropped**: " + "; ".join(r) if r else "✅ **Kept**")
            for b in scans[i]["boxes"]:
                c = "n/a" if b["containment"] is None else f"{b['containment']:.0%}"
                lines.append(f"<small>pedestrian {b['instance_id']}: {b['n_points']} pts, {c} in box, "
                             f"{simdata.box_class(b, p)}</small>")
        scan_md.content = "  \n".join(lines)

    def on_run():
        rd, _ = current()
        scan_sl.max = max(1, simdata.num_scans(rd) - 1) if rd else 1
        scan_sl.value = 0
        show_scan()

    run_dd.on_update(lambda _: None if st["syncing"] else on_run())
    scan_sl.on_update(lambda _: show_scan())

    def jump(step):
        rd, scans = current()
        if not scans:
            return
        p = params()
        i = int(scan_sl.value) + step
        while 0 <= i < len(scans):
            if simdata.judge(scans[i], p):
                scan_sl.value = i
                return
            i += step
        app.log("no more dropped scans in this direction")

    prev_btn.on_click(lambda _: jump(-1))
    next_btn.on_click(lambda _: jump(1))

    def analyse(force=False):
        """Analyse (or load cached analyses of) the current project's runs. Serialized; a result is dropped if
        the project changed meanwhile (st["gen"])."""
        gen = st["gen"]
        with st["lock"]:
            if gen != st["gen"]:
                return
            st["busy"] = True
            bar.visible, bar.value = True, 0.0
            runs, out = list(st["runs"]), {}
            try:
                for k, rd in enumerate(runs):
                    def prog(i, n, k=k):
                        bar.value = 100.0 * (k + i / n) / len(runs)
                    out[rd.name] = simdata.analyse_run(rd, force=force, progress=prog)
            finally:
                bar.visible = False
                st["busy"] = False
            if gen != st["gen"]:
                return
            st["analyses"] = out
            app.log(f"analysed {sum(len(a) for a in out.values()):,} scans in {len(runs)} run(s) of "
                    f"{app.project.name}")
        recount()
        refresh_state()

    analyse_btn.on_click(lambda _: threading.Thread(target=analyse, args=(True,), daemon=True).start())

    @export_btn.on_click
    def _(_):
        if st["busy"] or not st["analyses"]:
            return
        st["busy"] = st["exporting"] = True
        export_btn.disabled = True
        bar.visible, bar.value = True, 0.0
        app.project.set_status(KEY, "running")
        app.refresh_status()

        def work():
            try:
                runs = [r for r in st["runs"] if r.name in st["analyses"]]
                meta = export(app.project, runs, [st["analyses"][r.name] for r in runs], params(),
                              progress=lambda i, n: setattr(bar, "value", 100.0 * i / n))
                c = meta["counts"]
                note = f"{c.get('train', 0):,} train / {c.get('val', 0):,} val scans"
                if st["stale"]:  # built from the run list as it was when the export started
                    app.project.set_status(KEY, "todo", note=note + "; runs or placement changed since, export again")
                else:
                    app.project.set_status(KEY, "done", note=note)
                app.project.invalidate_downstream(KEY)
                app.log(f"dataset written: {meta['counts']}, range ±{meta['range_xy']} m, floor z {meta['floor_z']}")
                app.stage_changed(KEY)
            except Exception as e:  # noqa: BLE001
                app.project.set_status(KEY, "error", note=str(e))
                app.log(f"dataset export failed: {e}")
                app.stage_changed(KEY)  # Train re-checks whether there is still a dataset
            finally:
                bar.visible = False
                export_btn.disabled = False
                st["busy"] = st["exporting"] = False
                if st["stale"]:  # the runs or the placement changed during the export
                    refresh()
                else:
                    refresh_state()

        threading.Thread(target=work, daemon=True).start()

    def refresh_state():
        p = app.project
        f = p.path / "dataset" / "dataset.json"
        if not st["runs"]:
            state_md.content = ("*Needs: at least one complete Simulate run at the current sensor placement.*"
                                + (f" ({st['skipped']} older run(s) skipped.)" if st.get("skipped") else ""))
        elif not st["analyses"]:
            state_md.content = f"{len(st['runs'])} run(s) ready. Analysing…"
        else:
            split = split_runs(st["runs"])
            how = (f"train/val split by run: {sum(v == 'val' for v in split.values())} of {len(split)} run(s) "
                   "held out for validation") if split else \
                ("⚠️ only one run: validation is its last 20% in time (neighbouring scans are near-duplicates, "
                 "so val scores will be optimistic; simulate more runs)")
            state_md.content = how + (f"  \n⚠️ {st['skipped']} run(s) skipped: captured at an older sensor "
                                      "placement" if st.get("skipped") else "")
        export_btn.disabled = not st["analyses"] or st["busy"]
        if f.exists():
            m = json.loads(f.read_text())
            c = m["counts"]
            export_md.content = (f"✅ `dataset/`: **{c.get('train', 0):,} train, {c.get('val', 0):,} val** scans, "
                                 f"{c.get('Pedestrian', 0):,} pedestrian labels; range ±{m['range_xy']} m, floor at "
                                 f"z = {m['floor_z']} m (built {m['created']})")
        else:
            export_md.content = ""

    def refresh():
        p = app.project
        if p is None:
            return
        st["stale"] = False
        st["gen"] += 1
        complete = [r for r in simdata.list_runs(p) if simdata.run_info(r).get("complete")]
        # Only runs captured at the current sensor placement: the dataset has one training frame and one floor
        # height, so scans from an older mount would be mislabelled for it.
        sensor = p.path / "sensor.json"
        if sensor.exists():
            ref = simdata.pose_matrix(json.loads(sensor.read_text()))
        else:
            ref = simdata.sensor_to_world(complete[0]) if complete else None
        st["runs"] = [r for r in complete if simdata.same_pose(simdata.sensor_to_world(r), ref)]
        st["skipped"] = len(complete) - len(st["runs"])
        st["analyses"] = {}
        names = [r.name for r in st["runs"]]
        st["syncing"] = True
        run_dd.options = names or ["(none)"]
        run_dd.value = names[0] if names else "(none)"
        st["syncing"] = False
        drawer.clear()
        refresh_state()
        if st["runs"]:
            on_run()
            threading.Thread(target=analyse, daemon=True).start()

    def on_upstream(key):
        if key not in ("simulate", "sensor_place"):  # the run list is filtered by the current placement
            return
        if st["exporting"]:  # not underneath a running export; it refreshes when it ends
            st["stale"] = True
        else:
            refresh()

    app.on_project_opened(refresh)
    app.on_stage_changed(on_upstream)
