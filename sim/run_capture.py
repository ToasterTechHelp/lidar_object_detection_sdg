"""Capture one IRA + RTX lidar run: point clouds plus per-point pedestrian labels and 3D boxes.

The stage, lidar asset and lidar pose are parameters (the pipeline's Simulate stage launches it once per run):
    C:\\isaacsim\\python.bat sim\\run_capture.py --config <ira.yaml> --sensor <project>/sensor.json
        --out-dir <run folder> [--num-scans 200] [--expected-peds N] [--boxes]
The IRA config's environment.base_stage_asset_path is the stage (sim/build_stage.py output); see
sim/ira_config.py.

Per completed revolution this writes, into --out-dir:
    velodyne/XXXXXX.bin    float32 x y z intensity (KITTI velodyne layout)
    labels/XXXXXX.label    uint32 per point, same order as the .bin (SemanticKITTI layout):
                           low 16 bits = class (0 other, 30 pedestrian),
                           high 16 bits = pedestrian instance id (0 = not a pedestrian)
and once per run: times.txt, pedestrian_points.csv, instances.json, run_info.json, config.yaml.

All point outputs are in the lidar's own sensor frame, exactly as the GMO reports them.
The lidar's world transform is saved in run_info.json.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="IRA config YAML for this run")
parser.add_argument("--sensor", required=True,
                    help="sensor.json: usd_path (or config + variant), position, wxyz, tick_rate")
parser.add_argument("--out-dir", required=True, help="Run output folder")
parser.add_argument("--num-scans", type=int, default=200, help="Complete revolutions to capture (10 Hz)")
parser.add_argument("--expected-peds", type=int, default=None, help="Abort if IRA spawns a different count")
parser.add_argument("--box-sample", choices=("end", "mid", "sweep"), default="end",
                    help="When in the lidar revolution the box is sampled. 'end' (default, and "
                         "measured to be the right one) reads the annotator at scan completion. "
                         "'mid' interpolates back to the middle of the revolution -- this was "
                         "tried to cancel motion smear and makes containment worse. 'sweep' also "
                         "writes boxes at a range of lags to boxes_sweep/, which is how that was "
                         "measured.")
parser.add_argument("--boxes", action="store_true",
                    help="Also write amodal oriented 3D boxes per scan to boxes/XXXXXX.json "
                         "(Replicator bounding_box_3d_fast)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

# StableIdMap (object id -> prim path) only emits data with stableIds enabled.
# See C:\isaacsim\standalone_examples\api\isaacsim.sensors.experimental.rtx\resolve_lidar_object_ids.py
simulation_app = SimulationApp(
    {
        "headless": True,
        # Nothing reads the default viewport in a headless capture (the lidar and the GT
        # box camera have their own render products), but it was still rendered every
        # app update. Sim performance handbook: disable it.
        "disable_viewport_updates": True,
        "enable_motion_bvh": True,
        "extra_args": ["--/rtx-transient/stableIds/enabled=true"],
    }
)

import omni.usd
import omni.timeline
import omni.kit.app
import carb.eventdispatcher
import carb.settings

# isaacsim.exp.base.kit -- the experience SimulationApp loads by default -- ships
# player.useFixedTimeStepping = false, so the timeline advances by the loop
# runner's measured wall-clock dt. That couples simulated time to frame rate: a
# heavier scene means a larger dt per update, fewer renders feeding each 10 Hz
# revolution, and past dt > 0.1s, whole revolutions skipped with no error.
# Forcing fixed stepping pins dt = 1/timeCodesPerSecond so the captured data is
# identical whether the run happens at 25 fps or 3 fps.
carb.settings.get_settings().set_bool("/app/player/useFixedTimeStepping", True)
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom
from isaacsim.core.experimental.utils.app import enable_extension

sys.path.insert(0, str(Path(__file__).parent))

import yaml

CONFIG_PATH = Path(args.config).resolve()
STAGE_PATH = yaml.safe_load(CONFIG_PATH.read_text())["isaacsim.replicator.agent"]["environment"][
    "base_stage_asset_path"]
SENSOR = json.loads(Path(args.sensor).read_text())
REPO = Path(__file__).resolve().parents[1]
LIDAR_USD_PATH = str((REPO / SENSOR["usd_path"]).resolve()) if SENSOR.get("usd_path") else None
LIDAR_PATH = "/World/Lidar"
OUTPUT_DIR = Path(args.out_dir).resolve()

# Number of complete lidar revolutions to capture, NOT app updates. The app
# renders faster than the sensor's 10 Hz tick, so several app updates feed each
# revolution; MAX_UPDATES is a safety bound in case scans stop completing.
NUM_SCANS = args.num_scans
MAX_UPDATES = NUM_SCANS * 8

PEDESTRIAN_CLASS = 30  # SemanticKITTI "person"

omni.usd.get_context().open_stage(STAGE_PATH)
stage = omni.usd.get_context().get_stage()
for prim_path in [LIDAR_PATH, "/World/Characters"]:
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

enable_extension("isaacsim.replicator.agent.ui")


def _patch_navmesh_wait(max_frames=3000):
    """IRA gives the navmesh bake 100 app updates (omni.metropolis.pipeline.simulation_util), which a freshly
    loaded reconstructed mesh can exceed; the same bake standalone (sim/probe_navmesh.py) succeeds. Give it
    longer. The loaders import the function by name, so patch their module globals."""
    import omni.anim.navigation.core as nav
    from isaacsim.replicator.agent.core.scene_assembly import character_loader, environment_loader

    async def ensure_navmesh_ready() -> bool:
        inav = nav.acquire_interface()
        if inav.get_navmesh() is not None:
            print("[collect] navmesh already present", flush=True)
            return True
        if not inav.start_navmesh_baking():
            print("[collect] ERROR: navmesh bake could not start (no NavMeshVolume?)", flush=True)
            return False
        for frame in range(max_frames):
            if inav.get_navmesh() is not None:
                print(f"[collect] navmesh baked after {frame} updates", flush=True)
                return True
            await omni.kit.app.get_app().next_update_async()
        print(f"[collect] ERROR: navmesh not baked after {max_frames} updates", flush=True)
        return False

    for mod in (character_loader, environment_loader):
        mod.ensure_navmesh_ready = ensure_navmesh_ready


_patch_navmesh_wait()
from isaacsim.replicator.agent.core.events import IRAEvents
from isaacsim.replicator.agent.ui import load_config_file, setup_simulation

setup_done = {"flag": False}


def _on_setup_done(event):
    setup_done["flag"] = True


dispatcher = carb.eventdispatcher.get_eventdispatcher()
handle_setup = dispatcher.observe_event(
    event_name=IRAEvents.SET_UP_SIMULATION_DONE_EVENT,
    on_event=_on_setup_done,
    observer_name="setup_done_observer",
)

load_config_file(str(CONFIG_PATH), set_config=True)
# setup_simulation() is fire-and-forget (asyncio.ensure_future) and reopens the
# stage with force_load=True, so the lidar must not be created until this
# completes -- otherwise the stage reload destroys the prim and render product.
# The reload also means anything saved under /World/Characters in the base stage
# comes back, and IRA skips spawning over existing paths -- keep the base stage
# free of characters.
setup_simulation()

app = omni.kit.app.get_app()
while not setup_done["flag"]:
    app.update()
handle_setup.reset()

# The stage was reopened by setup_simulation(); the handle from above is stale.
stage = omni.usd.get_context().get_stage()

# Pedestrian instances IRA spawned: /World/Characters/<group>/<group>_<i>. Ids are
# assigned in (group, index) order so they are stable across reruns of a config.
_INSTANCE_RE = re.compile(r"(.+)_(\d+)")
chars_root = stage.GetPrimAtPath(Sdf.Path("/World/Characters"))
instances = []
if chars_root.IsValid():
    for grp in chars_root.GetChildren():
        for inst in grp.GetChildren():
            m = _INSTANCE_RE.fullmatch(inst.GetName())
            if m and m.group(1) == grp.GetName():
                instances.append((grp.GetName(), int(m.group(2)), inst))
instances.sort(key=lambda t: (t[0], t[1]))
INSTANCE_IDS = {inst.GetName(): k + 1 for k, (_, _, inst) in enumerate(instances)}
print(f"[collect] {len(INSTANCE_IDS)} pedestrians: {list(INSTANCE_IDS)}", flush=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if args.expected_peds is not None and len(INSTANCE_IDS) != args.expected_peds:
    msg = f"expected {args.expected_peds} pedestrians, IRA spawned {len(INSTANCE_IDS)}"
    print(f"[collect] ERROR: {msg}")
    (OUTPUT_DIR / "run_info.json").write_text(json.dumps({"complete": False, "error": msg}, indent=2))
    simulation_app.close()
    sys.exit(2)

for sub in ("velodyne", "labels") + (("boxes",) if args.boxes else ()):
    (OUTPUT_DIR / sub).mkdir(exist_ok=True)
shutil.copy(CONFIG_PATH, OUTPUT_DIR / "config.yaml")

enable_extension("isaacsim.sensors.rtx.nodes")
import omni.replicator.core as rep
from omni.replicator.core import Writer
from isaacsim.sensors.experimental.rtx import (
    Lidar,
    LidarSensor,
    parse_generic_model_output_data,
    parse_object_ids,
    parse_stable_id_map_data,
)


def _unwrap(raw):
    return raw.get("data") if isinstance(raw, dict) else raw


class BoxSource:
    """Amodal oriented 3D boxes for the pedestrians, from Replicator.

    bounding_box_3d_fast reports every labelled prim inside the camera frustum,
    ignoring occlusion (it sees through walls), so the camera here is a dummy sized
    from /World/NavMeshVolume by gt_camera.place_gt_camera -- nothing renders it.
    rep.orchestrator.set_capture_on_play(True) is what makes the annotator emit at
    all; without it every read returns zero rows and no error. Attaching to the
    lidar's own render product instead looks tempting and does work, but it only
    covers a ~+/-42 deg azimuth wedge.

    Boxes are amodal (full body, floor to head, regardless of what the lidar sees),
    skinned (extents follow the walk cycle) and oriented (the 4x4 carries a yaw), and
    they agree with the behaviour agent's skeleton centre to ~0.10 m.

    **Sampling time -- measured, not assumed.** A scan is a whole revolution, so its
    points are swept over ~0.1 s while the annotator can only be read at one instant.
    The obvious worry is that reading at scan completion puts the box half a sweep
    ahead of the points. It does not: sweeping the sample point across the revolution
    inside a single run (mode "sweep", 884 pedestrian-scans, identical points) peaks
    at lag 0.0, i.e. exactly the read at scan completion --

        lag (revolutions back)  -0.50   -0.25    0.00    0.25    0.50    0.75    1.00
        mean containment %      96.61   98.90   99.67   99.64   99.01   97.57   94.95

    -- so Replicator's annotator data is already effectively synchronised with the
    accumulated scan, and shifting it back only makes things worse. Mode "end" is
    therefore the default. Modes "mid" and "sweep" are kept because they are how that
    was established and how to re-establish it if the pipeline timing ever changes.

    The interpolation used by "mid"/"sweep" is exact for this data: one annotator read
    lands per revolution boundary, and the box transform is a pure yaw plus a
    translation, so translation, yaw and the local extents each interpolate linearly
    (yaw along the shortest arc). Anything that is not a pure yaw falls back to the
    nearest sample.
    """

    HISTORY = 12  # writer calls; one per revolution, so this is generous

    # Lags to try in "sweep" mode, in revolutions back from scan completion.
    # Negative values extrapolate past the end of the sweep.
    SWEEP_LAGS = (-0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00)

    def __init__(self, stage, lidar_to_world, mode="mid"):
        from collections import deque
        from gt_camera import place_gt_camera

        self.cam, self.render_product, self.info = place_gt_camera(stage, rep)
        # Points are written in the sensor frame, so carry the boxes there too.
        self.world_to_sensor = np.linalg.inv(np.asarray(lidar_to_world, dtype=np.float64))
        self.annot = rep.annotators.get("bounding_box_3d_fast")
        self.annot.attach(self.render_product)
        rep.orchestrator.set_capture_on_play(True)

        self.mode = mode
        self.tick = 0                    # app updates since the writer started
        self.history = deque(maxlen=self.HISTORY)   # (tick, {inst: raw box})
        self.rev_updates = 3.0           # app updates per revolution, measured as we go
        self._last_complete_tick = None
        self.lags = []                   # realised lag per scan, in revolutions
        self.fallbacks = 0               # scans that could not be interpolated
        self.last_sweep = None           # {lag: boxes} when mode == "sweep"

        print(f"[collect] GT box camera at {np.round(self.info['camera_position'], 2).tolist()}, "
              f"navmesh {np.round(np.array(self.info['navmesh_max']) - np.array(self.info['navmesh_min']), 2).tolist()} m "
              f"fills {self.info['corner_fill_fraction'] * 100:.0f}% of the frustum "
              f"(worst corner {self.info['worst_corner_deg']:.1f} of "
              f"{self.info['half_fov_deg']:.1f} deg); box sample = {mode}-revolution")

    # -- reading -------------------------------------------------------------

    def sample(self):
        """Read the annotator once. Called on every writer call, i.e. every app update."""
        self.tick += 1
        raw = self._read_raw()
        if raw:
            self.history.append((self.tick, raw))

    def _read_raw(self):
        d = self.annot.get_data()
        data = d.get("data") if isinstance(d, dict) else d
        if data is None or len(data) == 0:
            return {}
        paths = [str(x) for x in (d.get("info", {}) or {}).get("primPaths", [])]
        out = {}
        for k, row in enumerate(data):
            inst_id = _instance_id_for_path(paths[k] if k < len(paths) else "")
            if not inst_id:
                continue
            out[inst_id] = {
                "lo": np.array([row["x_min"], row["y_min"], row["z_min"]], dtype=np.float64),
                "hi": np.array([row["x_max"], row["y_max"], row["z_max"]], dtype=np.float64),
                "b2w": np.asarray(row["transform"], dtype=np.float64).reshape(4, 4),
            }
        return out

    # -- picking the moment --------------------------------------------------

    def at_scan_complete(self):
        """Boxes for the revolution that just finished, plus how they were sampled."""
        if self._last_complete_tick is not None:
            delta = self.tick - self._last_complete_tick
            if 1 <= delta <= 8:
                self.rev_updates = float(delta)
        self._last_complete_tick = self.tick

        if not self.history:
            return {}, None

        if self.mode == "sweep":
            self.last_sweep = {f"{lag:+.2f}": self._emit(self._at_lag(lag))
                               for lag in self.SWEEP_LAGS}
            self.last_sweep = {k: v for k, v in self.last_sweep.items() if v}

        if self.mode in ("end", "sweep"):
            tick, raw = self.history[-1]
            lag = (self.tick - tick) / self.rev_updates
            self.lags.append(lag)
            return self._emit(raw), {"mode": self.mode, "lag_revolutions": lag,
                                     "rev_updates": self.rev_updates}

        target = self.tick - self.rev_updates / 2.0
        older = [h for h in self.history if h[0] <= target]
        newer = [h for h in self.history if h[0] > target]
        if not older:
            # Not enough history yet (first scans). Nearest available sample.
            self.fallbacks += 1
            tick, raw = self.history[0]
            self.lags.append((self.tick - tick) / self.rev_updates)
            return self._emit(raw), {"mode": "mid-fallback", "lag_revolutions": self.lags[-1],
                                     "rev_updates": self.rev_updates}

        t0, a = older[-1]
        if not newer:
            self.fallbacks += 1
            self.lags.append((self.tick - t0) / self.rev_updates)
            return self._emit(a), {"mode": "mid-fallback", "lag_revolutions": self.lags[-1],
                                   "rev_updates": self.rev_updates}
        t1, b = newer[0]
        w = (target - t0) / (t1 - t0)
        self.lags.append((self.tick - target) / self.rev_updates)
        return self._emit(_lerp_boxes(a, b, w)), {
            "mode": "mid", "lag_revolutions": self.lags[-1],
            "rev_updates": self.rev_updates, "weight": w,
            "between_ticks": [t0, t1], "target_tick": target,
        }

    def _at_lag(self, lag):
        """Raw boxes `lag` revolutions before scan completion, by interpolating the
        last two revolution-boundary samples. lag 0 is the end of the sweep, 1.0 the
        start; outside [0, 1] this extrapolates along the same straight line."""
        if not self.history:
            return {}
        if len(self.history) < 2:
            return self.history[-1][1]
        (t0, a), (t1, b) = self.history[-2], self.history[-1]
        if t1 - t0 != 1:
            return self.history[-1][1]
        return _lerp_boxes(a, b, 1.0 - lag)

    def _emit(self, raw):
        out = {}
        for inst_id, r in raw.items():
            out[inst_id] = {
                "instance_id": inst_id,
                "local_min": r["lo"].tolist(),
                "local_max": r["hi"].tolist(),
                # USD row-vector convention throughout: p_sensor = [p_local, 1] @ box_to_sensor
                "box_to_world": r["b2w"].tolist(),
                "box_to_sensor": (r["b2w"] @ self.world_to_sensor).tolist(),
            }
        return out


def _is_pure_yaw(m, tol=1e-6):
    return (abs(m[0][2]) < tol and abs(m[1][2]) < tol
            and abs(m[2][0]) < tol and abs(m[2][1]) < tol and abs(m[2][2] - 1.0) < tol)


def _lerp_boxes(a, b, w):
    """Interpolate two per-instance box dicts. Instances in only one are taken as-is."""
    out = {}
    for inst_id in set(a) | set(b):
        ra, rb = a.get(inst_id), b.get(inst_id)
        if ra is None or rb is None:
            out[inst_id] = ra if ra is not None else rb
            continue
        ma, mb = ra["b2w"], rb["b2w"]
        if not (_is_pure_yaw(ma) and _is_pure_yaw(mb)):
            out[inst_id] = ra if w < 0.5 else rb
            continue
        ya = math.atan2(ma[0][1], ma[0][0])
        yb = math.atan2(mb[0][1], mb[0][0])
        dy = (yb - ya + math.pi) % (2 * math.pi) - math.pi   # shortest arc, in (-pi, pi]
        y = ya + dy * w
        c, sn = math.cos(y), math.sin(y)
        t = ma[3, :3] + (mb[3, :3] - ma[3, :3]) * w
        m = np.array([[c, sn, 0.0, 0.0],
                      [-sn, c, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [t[0], t[1], t[2], 1.0]], dtype=np.float64)
        out[inst_id] = {
            "lo": ra["lo"] + (rb["lo"] - ra["lo"]) * w,
            "hi": ra["hi"] + (rb["hi"] - ra["hi"]) * w,
            "b2w": m,
        }
    return out


def _instance_id_for_path(path: str) -> int:
    parts = path.split("/")
    if len(parts) > 4 and parts[1] == "World" and parts[2] == "Characters":
        return INSTANCE_IDS.get(parts[4], 0)
    return 0


class LabelledScanWriter(Writer):
    """Write one labelled point cloud per completed lidar revolution.

    Driven by Replicator's write() callback rather than polled from the main
    loop. sensor.get_data() is a poll that returns the annotator's last-written
    buffer, so on the app updates between sensor ticks it hands back the same
    finished scan -- with scanComplete still set -- and the same revolution gets
    written to disk several times over.
    """

    def __init__(self) -> None:
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput"), rep.annotators.get("StableIdMap")]
        self.scan_count = 0
        self.call_count = 0
        self.stable_id_map = {}
        self._label_cache = {}
        self.times = []
        self.ped_points = []  # (scan, instance_id, n_points)
        self.box_source = None      # set after the lidar exists; None disables boxes
        self.box_counts = []        # (scan, n_boxes) so coverage can be audited offline
        self.skipped_warmup = 0

    def write(self, data: dict) -> None:
        self.call_count += 1
        if self.box_source is not None:
            # Once per app update, so the mid-revolution interpolation has history.
            self.box_source.sample()
        if "renderProducts" not in data:
            return
        for _rp_name, rp_data in data["renderProducts"].items():
            # The map is only re-sent when it changes, so accumulate it.
            sid_raw = _unwrap(rp_data.get("StableIdMap"))
            if sid_raw is not None and len(sid_raw) > 0:
                self.stable_id_map.update(parse_stable_id_map_data(sid_raw))
            gmo = parse_generic_model_output_data(_unwrap(rp_data.get("GenericModelOutput")))
            # accumulateOutputs=1 means the model accumulates across renders
            # until one scan is complete; only that frame holds a whole cloud.
            # numElements == 0 also covers the invalid-magic-number buffers the
            # annotator emits before the first render lands.
            if gmo.numElements == 0 or not gmo.scanComplete:
                continue
            boxes, box_meta = {}, None
            if self.box_source is not None:
                boxes, box_meta = self.box_source.at_scan_complete()
                if not boxes:
                    # The annotator needs a few updates before its first output. Drop
                    # these scans rather than writing a box-less one, so scan 0 already
                    # has boxes and every index stays aligned with times.txt.
                    self.skipped_warmup += 1
                    continue
            self._save_scan(gmo, boxes, box_meta)

    def _save_scan(self, gmo, boxes=None, box_meta=None) -> None:
        i = self.scan_count
        xyz = np.column_stack([gmo.x, gmo.y, gmo.z]).astype(np.float32)
        intensity = np.asarray(gmo.scalar, dtype=np.float32)
        labels = self._labels(gmo)

        np.column_stack([xyz, intensity]).astype(np.float32).tofile(OUTPUT_DIR / "velodyne" / f"{i:06d}.bin")
        labels.tofile(OUTPUT_DIR / "labels" / f"{i:06d}.label")

        if self.box_source is not None:
            (OUTPUT_DIR / "boxes" / f"{i:06d}.json").write_text(
                json.dumps({"scan": i, "sampling": box_meta,
                            "boxes": [boxes[k] for k in sorted(boxes)]})
            )
            self.box_counts.append((i, len(boxes)))
            if self.box_source.last_sweep:
                (OUTPUT_DIR / "boxes_sweep").mkdir(exist_ok=True)
                (OUTPUT_DIR / "boxes_sweep" / f"{i:06d}.json").write_text(json.dumps(
                    {lag: [bx[k] for k in sorted(bx)]
                     for lag, bx in self.box_source.last_sweep.items()}
                ))

        self.times.append(gmo.timestampNs * 1e-9)
        inst = labels >> 16
        ids, counts = np.unique(inst[inst > 0], return_counts=True)
        self.ped_points.extend((i, int(a), int(c)) for a, c in zip(ids, counts))
        self.scan_count += 1
        # progress line the GUI parses
        print(f"[collect] scan {self.scan_count}/{NUM_SCANS} peds_hit {len(ids)} points {len(xyz)}", flush=True)

    def _labels(self, gmo) -> np.ndarray:
        labels = np.zeros(gmo.numElements, dtype=np.uint32)
        # objId is only filled at aux_output_level EXTRA or FULL.
        if gmo.objId is None or gmo.objId.size == 0:
            return labels
        # Full 128-bit ids: the upper bits carry submesh indices, so truncating
        # them misses hits on multi-subset meshes like the characters' clothing.
        for k, oid in enumerate(parse_object_ids(gmo.objId)):
            labels[k] = self._label_for(oid)
        return labels

    def _label_for(self, oid: int) -> int:
        label = self._label_cache.get(oid)
        if label is not None:
            return label
        path = self.stable_id_map.get(oid)
        if path is None:
            # Not in the map (yet): label as background but don't cache, so a
            # later map update can still resolve it.
            return 0
        if isinstance(path, (bytes, bytearray)):
            path = path.decode("utf-8", errors="replace")
        inst_id = _instance_id_for_path(str(path).rstrip("\x00"))
        label = (inst_id << 16) | PEDESTRIAN_CLASS if inst_id else 0
        self._label_cache[oid] = label
        return label


rep.WriterRegistry.register(LabelledScanWriter)

# A local OmniLidar USD (registry / sensor-agent sensors) or one of Isaac's built-in configs.
_source = {"usd_path": LIDAR_USD_PATH} if LIDAR_USD_PATH else {"config": SENSOR["config"]}
if not LIDAR_USD_PATH and SENSOR.get("variant"):
    _source["variant"] = SENSOR["variant"]
lidar = Lidar.create(
    path=LIDAR_PATH,
    **_source,
    tick_rate=float(SENSOR.get("tick_rate", 10.0)),
    translations=np.array(SENSOR["position"], dtype=np.float64),
    orientations=np.array(SENSOR["wxyz"], dtype=np.float64),
    # EXTRA adds objId (per-point object ids) on top of BASIC's scanComplete etc.
    aux_output_level="EXTRA",
)

# Points are saved in the sensor frame; record the lidar's pose so they can be placed
# in the world later (USD row-vector convention: p_world = [x, y, z, 1] @ M).
_l2w = np.array(
    UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(stage.GetPrimAtPath(Sdf.Path(LIDAR_PATH))),
    dtype=np.float64,
)
print(f"[collect] lidar local-to-world (row-vector):\n{_l2w}")

# annotators=[] -- the writer supplies its own GenericModelOutput + StableIdMap annotators.
sensor = LidarSensor(lidar, annotators=[])
scan_writer = sensor.attach_writer("LabelledScanWriter")

box_source = BoxSource(stage, _l2w, mode=args.box_sample) if args.boxes else None
scan_writer.box_source = box_source

timeline = omni.timeline.get_timeline_interface()
timeline.play()

updates = 0
while scan_writer.scan_count < NUM_SCANS and updates < MAX_UPDATES:
    app.update()
    updates += 1

# Detach before stopping. Replicator's shutdown waits (up to 1000 app updates, ~16-30 s)
# for every attached writer's schedule list to empty, and a writer that ran unscheduled
# like this one never empties it, so each run ended in "Timed out while waiting for
# pending Replicator writer schedules to drain". write() already saved every scan to
# disk synchronously, so there is nothing left to flush; the Python object keeps its
# counters for the summary below.
sensor.detach_writer("LabelledScanWriter")

timeline.stop()

print(
    f"[collect] wrote {scan_writer.scan_count} scans from {updates} app updates "
    f"({scan_writer.call_count} writer calls)"
)
if box_source is not None:
    counts = [n for _, n in scan_writer.box_counts]
    want = len(INSTANCE_IDS)
    short = [(s_, n) for s_, n in scan_writer.box_counts if n != want]
    print(f"[collect] boxes: {len(counts)} scans, {min(counts, default=0)}-{max(counts, default=0)} "
          f"per scan (expected {want}); {scan_writer.skipped_warmup} warm-up scans skipped")
    if short:
        print(f"[collect] WARNING: {len(short)} scans have a box count != {want}; first 5: {short[:5]}")
    lags = box_source.lags
    if lags:
        print(f"[collect] box sample: mode={box_source.mode}, "
              f"{box_source.rev_updates:.1f} app updates per revolution, "
              f"lag {np.mean(lags):.2f} revolutions (min {min(lags):.2f}, max {max(lags):.2f}), "
              f"{box_source.fallbacks} scans fell back to the nearest sample")
complete = scan_writer.scan_count >= NUM_SCANS
if not complete:
    print(f"[collect] WARNING: stopped at MAX_UPDATES ({MAX_UPDATES}) before reaching {NUM_SCANS} scans")

(OUTPUT_DIR / "times.txt").write_text("".join(f"{t:.6f}\n" for t in scan_writer.times))
with open(OUTPUT_DIR / "pedestrian_points.csv", "w") as f:
    f.write("scan,instance_id,n_points\n")
    f.writelines(f"{s},{a},{n}\n" for s, a, n in scan_writer.ped_points)
(OUTPUT_DIR / "instances.json").write_text(
    json.dumps(
        {str(k + 1): str(inst.GetPath()) for k, (_, _, inst) in enumerate(instances)},
        indent=2,
    )
)
(OUTPUT_DIR / "run_info.json").write_text(
    json.dumps(
        {
            "complete": complete,
            "scans_written": scan_writer.scan_count,
            "scans_requested": NUM_SCANS,
            "app_updates": updates,
            "config": str(CONFIG_PATH),
            "stage": STAGE_PATH,
            "sensor": SENSOR,
            "num_pedestrians": len(INSTANCE_IDS),
            "stable_id_map_entries": len(scan_writer.stable_id_map),
            "lidar_local_to_world_row_vector": _l2w.tolist(),
            "gt_box_camera": (box_source.info if box_source is not None else None),
            "box_counts_per_scan": ([n for _, n in scan_writer.box_counts]
                                    if box_source is not None else None),
            "point_frame": "lidar sensor frame as reported by the GMO (metres); world pose in lidar_local_to_world_row_vector",
            "files": {
                "velodyne/NNNNNN.bin": "float32 x y z intensity",
                "labels/NNNNNN.label": "uint32 per point: low 16 bits class (0 other, 30 pedestrian), high 16 bits instance id (see instances.json)",
                "times.txt": "sensor timestamp (s) per scan",
                "pedestrian_points.csv": "lidar hits per pedestrian per scan",
            },
        },
        indent=2,
    )
)

simulation_app.close()
