"""Project -> Isaac Sim base stage USD (runs in the app venv with usd-core; Isaac is not needed).

    python sim/build_stage.py <project dir> [--out <project>/sim/stage.usd]

Reads mesh/mesh_world.ply (world frame: meters, z up, floor z = 0) and walkarea.json, and writes a stage
with this layout:
    /World/Scene             the reconstructed mesh (vertex colors), visual only
    /World/Floor             a flat collision plane at z = 0 under the walking areas, so the navmesh
                             bakes on a clean floor instead of VGGT's bumpy one
    /World/NavMeshVolume     Include volume covering the union of the walking areas (gt_camera sizes the
                             ground-truth box camera from this one; translate + scale only)
    /World/NavMeshVolume_k   one Include volume per rotated walking rectangle when there are several
    /PhysicsScene, /Environment/defaultLight
The stage has no characters: IRA spawns them at setup_simulation() and skips existing paths.

walkarea.json: {"rects": [{"center": [x, y], "size": [w, h], "yaw_deg": a}, ...], "height": 2.4}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

NAV_HEIGHT = 2.4  # volume spans z in [-0.2, height - 0.2]


def rect_corners(r):
    cx, cy = r["center"]
    w, h = r["size"]
    a = np.radians(r.get("yaw_deg", 0.0))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    return local @ R.T + [cx, cy]


def _xform(prim, translate, scale=(1, 1, 1), rotate_zyx=None):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, translate)))
    if rotate_zyx is not None:
        x.AddRotateZYXOp().Set(Gf.Vec3f(*map(float, rotate_zyx)))
    x.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def _nav_volume(stage, path, center, size):
    """An Include NavMeshVolume (custom schema, so defined by type name)."""
    prim = stage.DefinePrim(path, "NavMeshVolume")
    prim.CreateAttribute("nav:volume:type", Sdf.ValueTypeNames.Token).Set("Include")
    prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set(
        Vt.Vec3fArray([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)]))
    return prim


def build(project_dir: Path, out: Path | None = None) -> Path:
    project_dir = Path(project_dir)
    mesh_file = project_dir / "mesh" / "mesh_world.ply"
    if not mesh_file.exists():
        raise FileNotFoundError("mesh/mesh_world.ply missing: run Scale & level first")
    walk = json.loads((project_dir / "walkarea.json").read_text())
    rects = walk["rects"]
    if not rects:
        raise ValueError("walkarea.json has no rectangles")
    height = float(walk.get("height", NAV_HEIGHT))
    out = Path(out or project_dir / "sim" / "stage.usd")
    out.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    mesh = trimesh.load(mesh_file, process=False)
    m = UsdGeom.Mesh.Define(stage, "/World/Scene")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, np.int32).ravel()))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, mesh.bounds[0])), Gf.Vec3f(*map(float, mesh.bounds[1]))]))
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is not None and len(vc) == len(mesh.vertices):
        pv = m.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.asarray(vc[:, :3], np.float32) / 255.0))

    corners = np.concatenate([rect_corners(r) for r in rects])
    lo, hi = corners.min(0), corners.max(0)

    # Collision floor at z = 0 (a quad), 1 m wider than the walking areas on every side.
    flo, fhi = lo - 1.0, hi + 1.0
    floor = UsdGeom.Mesh.Define(stage, "/World/Floor")
    floor.CreatePointsAttr([Gf.Vec3f(flo[0], flo[1], 0), Gf.Vec3f(fhi[0], flo[1], 0),
                            Gf.Vec3f(fhi[0], fhi[1], 0), Gf.Vec3f(flo[0], fhi[1], 0)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    floor.CreateExtentAttr([Gf.Vec3f(flo[0], flo[1], 0), Gf.Vec3f(fhi[0], fhi[1], 0)])
    floor.CreatePurposeAttr(UsdGeom.Tokens.default_)
    UsdGeom.Imageable(floor).MakeInvisible()  # the scene mesh is what the lidar should see
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(floor.GetPrim()).CreateApproximationAttr("none")

    z0, z1 = -0.2, height - 0.2
    zc, zs = (z0 + z1) / 2, z1 - z0
    if len(rects) == 1 and abs(rects[0].get("yaw_deg", 0.0)) < 1e-6:
        prim = _nav_volume(stage, "/World/NavMeshVolume", None, None)
        _xform(prim, [*((lo + hi) / 2), zc], [*(hi - lo), zs], rotate_zyx=(0, 0, 0))
    else:
        # Several or rotated areas: one Include volume each, plus a plain Xform with the union's axis-aligned
        # bounds that gt_camera sizes the box camera from (a union NavMeshVolume would make extra floor walkable).
        for k, r in enumerate(rects):
            prim = _nav_volume(stage, f"/World/NavMeshVolume_{k}", None, None)
            _xform(prim, [*r["center"], zc], [*r["size"], zs], rotate_zyx=(0, 0, r.get("yaw_deg", 0.0)))
        bounds = stage.DefinePrim("/World/NavMeshBounds", "Xform")  # for gt_camera only, not a navmesh volume
        _xform(bounds, [*((lo + hi) / 2), zc], [*(hi - lo), zs])
        bounds.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set(
            Vt.Vec3fArray([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)]))

    light = UsdLux.DistantLight.Define(stage, "/Environment/defaultLight")
    light.CreateIntensityAttr(3000.0)
    light.CreateAngleAttr(1.0)
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")

    stage.GetRootLayer().Save()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--out")
    a = ap.parse_args()
    print(f"wrote {build(Path(a.project), a.out)}")
