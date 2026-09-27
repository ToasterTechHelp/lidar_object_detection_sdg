"""Bake the navmesh of a stage headlessly and report it (a pre-flight check before a capture run).

    C:\\isaacsim\\python.bat sim\\probe_navmesh.py <stage.usd> [--out navmesh.json] [--max-frames 2000]

Writes the baked navmesh triangles (world frame) to --out so the GUI can draw where people can walk.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("stage")
ap.add_argument("--out")
ap.add_argument("--max-frames", type=int, default=2000)
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "disable_viewport_updates": True})

import json

import carb
import numpy as np
import omni.kit.app
import omni.usd
from isaacsim.core.experimental.utils.app import enable_extension

enable_extension("omni.anim.navigation.core")
import omni.anim.navigation.core as nav

app = omni.kit.app.get_app()
omni.usd.get_context().open_stage(args.stage)
for _ in range(10):
    app.update()

settings = carb.settings.get_settings()
cfg = {k: settings.get(f"/exts/omni.anim.navigation.core/navMesh/config/{k}")
       for k in ("agentMinHeight", "agentMinRadius", "agentMaxRadius", "agentMaxStepHeight", "agentMaxFloorSlope",
                 "agentMinIslandRadius", "excludeRigidBodies")}
print(f"[navmesh] config {cfg}", flush=True)

inav = nav.acquire_interface()
ok = inav.start_navmesh_baking()
print(f"[navmesh] bake started: {ok}", flush=True)
frames = 0
while inav.get_navmesh() is None and frames < args.max_frames:
    app.update()
    frames += 1
navmesh = inav.get_navmesh()
result = {"baked": navmesh is not None, "frames": frames, "config": cfg}
if navmesh is not None:
    tris = np.array([[v.x, v.y, v.z] for v in navmesh.get_draw_triangles(0)], dtype=np.float64).reshape(-1, 3, 3)
    area = float(0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum())
    result.update(triangles=len(tris), area_m2=area, bounds=[tris.reshape(-1, 3).min(0).tolist(),
                                                             tris.reshape(-1, 3).max(0).tolist()])
    if args.out:
        with open(args.out, "w") as f:
            json.dump({**result, "tris": tris.round(4).tolist()}, f)
print(f"[navmesh] {json.dumps(result)}", flush=True)
simulation_app.close()
