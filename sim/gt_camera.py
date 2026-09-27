r"""Place the dummy camera that the 3D-box annotator hangs off.

`bounding_box_3d_fast` reports every labelled prim inside the camera **frustum**,
ignoring occlusion entirely (a camera outside the building, looking through the
walls, still returns every character). So the GT
camera never renders anything anyone looks at; its only job is to contain the whole
walkable region, and a coverage miss is silent -- a character that clips the frustum
edge just stops appearing in the rows.

Rather than eyeball a position, derive it: take the sphere enclosing
`/World/NavMeshVolume` (which is exactly what bounds where IRA lets characters walk),
and stand far enough back along a fixed direction that the sphere subtends less than
the camera's half-FOV, with margin. Containing the sphere guarantees containing the
box, whatever the room shape, so this keeps working if the stage or navmesh changes.

    from gt_camera import navmesh_world_bounds, place_gt_camera
    cam, rp, info = place_gt_camera(stage, rep, resolution=(512, 512))
"""

import math

import numpy as np
from pxr import Sdf, Usd, UsdGeom

NAVMESH_PATH = "/World/NavMeshVolume"
# sim/build_stage.py writes this instead when the walking area is several (or rotated) rectangles.
NAVMESH_BOUNDS_PATH = "/World/NavMeshBounds"


# A real walkable region is several metres across. If the bounds ever come back
# much smaller than this, the transform was not applied and the camera would be placed
# for a unit cube -- which frames almost nothing and drops pedestrians silently.
MIN_PLAUSIBLE_SPAN = 1.0   # metres, longest axis


def navmesh_world_bounds(stage, path=NAVMESH_PATH):
    """World-space (min, max) of the navmesh volume, as numpy arrays.

    A NavMeshVolume is a unit cube with an `extent` of +/-0.5 and its size carried in
    `xformOp:scale`, and it is not a Boundable, so UsdGeom.BBoxCache returns an empty
    range for it -- the transform has to be applied to the extent by hand.

    UsdGeom.XformCache is not trusted here either. NavMeshVolume is a custom schema, and
    where that schema is not registered (anything reading the stage outside Isaac Sim)
    XformCache returns **identity without error**, which silently yields a 1 m cube. So
    read the xformOps directly, use XformCache only to pick up any parent transform, and
    refuse to return something implausibly small.
    """
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if not prim.IsValid() and path == NAVMESH_PATH:
        prim = stage.GetPrimAtPath(Sdf.Path(NAVMESH_BOUNDS_PATH))
    if not prim.IsValid():
        raise RuntimeError(f"{path} not found; the GT camera cannot be sized without it")

    extent = prim.GetAttribute("extent").Get()
    if extent is None:
        raise RuntimeError(f"{path} has no extent attribute")
    local = np.array([[float(c) for c in extent[0]], [float(c) for c in extent[1]]])

    scale = prim.GetAttribute("xformOp:scale").Get()
    translate = prim.GetAttribute("xformOp:translate").Get()
    rotate = prim.GetAttribute("xformOp:rotateZYX").Get()
    if rotate is not None and any(abs(float(r)) > 1e-6 for r in rotate):
        raise RuntimeError(f"{path} carries a rotation {tuple(rotate)}; "
                           "navmesh_world_bounds only handles translate+scale")
    scale = np.array([1.0, 1.0, 1.0]) if scale is None else np.array([float(c) for c in scale])
    translate = np.zeros(3) if translate is None else np.array([float(c) for c in translate])

    corners = np.array([[local[i, 0], local[j, 1], local[k, 2]]
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    world = corners * scale + translate

    # Any transform on an ancestor still has to be composed in.
    parent = np.array(UsdGeom.XformCache(Usd.TimeCode.Default())
                      .GetLocalToWorldTransform(prim.GetParent()), dtype=float)
    if not np.allclose(parent, np.eye(4)):
        world = (np.column_stack([world, np.ones(len(world))]) @ parent)[:, :3]  # row-vector

    lo, hi = world.min(0), world.max(0)
    span = float((hi - lo).max())
    if span < MIN_PLAUSIBLE_SPAN:
        raise RuntimeError(
            f"{path} resolves to a {np.round(hi - lo, 2).tolist()} m volume, which is too "
            f"small to be the walkable region (expected a longest axis of at least "
            f"{MIN_PLAUSIBLE_SPAN} m). The transform was probably not applied; sizing the "
            "GT camera from this would silently drop pedestrians out of frame.")
    return lo, hi


def place_gt_camera(stage, rep, resolution=(512, 512), direction=(1.0, 1.0, 0.6), margin=1.6):
    """Create the GT camera and render product, sized to cover the navmesh.

    `margin` is how much bigger than the enclosing sphere the frustum is; 1.6 leaves
    room for a pedestrian's own extent and for arms swinging past the navmesh edge.
    Returns (camera, render_product, info) where info records the numbers used, so a
    run can store them and the framing can be audited afterwards.
    """
    lo, hi = navmesh_world_bounds(stage)
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0)

    # rep.create.camera defaults: 24 mm focal length, 20.955 mm aperture.
    focal_length, aperture = 24.0, 20.955
    half_fov = math.atan2(aperture / 2.0, focal_length)

    distance = margin * radius / math.sin(half_fov)
    d = np.array(direction, dtype=float)
    position = centre + d / np.linalg.norm(d) * distance

    cam = rep.create.camera(position=tuple(position), look_at=tuple(centre))
    rp = rep.create.render_product(cam, resolution)

    # The placement is sized from the enclosing sphere, which is conservative, but check
    # the real corners too -- that is the claim that matters and it is nearly free.
    axis = (centre - position) / np.linalg.norm(centre - position)
    rt = np.cross(axis, np.array([0.0, 0.0, 1.0])); rt /= np.linalg.norm(rt)
    upv = np.cross(rt, axis)
    corners = np.array([[lo[0] if i else hi[0], lo[1] if j else hi[1], lo[2] if k else hi[2]]
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    body = 0.5   # half a shoulder width plus arm swing, so a person at the edge still fits
    worst = 0.0
    for c in corners:
        v = c - position
        d = np.dot(v, axis) - body
        worst = max(worst,
                    abs(math.atan2(np.dot(v, rt) + body, d)),
                    abs(math.atan2(np.dot(v, upv) + body, d)))
    if worst >= half_fov:
        raise RuntimeError(
            f"GT camera frustum does not contain the navmesh: worst corner is "
            f"{math.degrees(worst):.1f} deg off-axis against a {math.degrees(half_fov):.1f} deg "
            "half-FOV. Increase `margin` or widen the camera.")

    info = {
        "navmesh_min": lo.tolist(),
        "navmesh_max": hi.tolist(),
        "navmesh_centre": centre.tolist(),
        "navmesh_radius": radius,
        "camera_position": position.tolist(),
        "camera_distance": distance,
        "half_fov_deg": math.degrees(half_fov),
        # Half-angle the navmesh sphere actually subtends. Must stay below half_fov_deg.
        "subtended_half_angle_deg": math.degrees(math.asin(min(1.0, radius / distance))),
        "margin": margin,
        "resolution": list(resolution),
        # Worst navmesh corner, padded for a body, as a fraction of the half-FOV.
        # 1.0 would mean grazing the frustum edge.
        "worst_corner_deg": math.degrees(worst),
        "corner_fill_fraction": math.degrees(worst) / math.degrees(half_fov),
    }
    return cam, rp, info
