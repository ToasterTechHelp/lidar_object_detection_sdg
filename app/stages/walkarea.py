"""P5 Walking area: where the simulated pedestrians may walk, as rotated rectangles on the floor (world frame).

Proposal:
- height map (always available): the world mesh is sampled into a top-down grid; a cell is walkable when it has
  floor points (z ~ 0) and nothing at body height above it. The largest inscribed rectangles of that mask are
  proposed.
- Gemini (when GEMINI_API_KEY is set): a top-down render goes to Gemini, which returns rectangles; each is
  checked against the walkable mask and dropped if it is mostly not floor.
Either way the user edits them (drag the gizmo, or type center / size / yaw), then confirms. Confirm writes
walkarea.json: {"rects": [{"center": [x, y], "size": [length_x, width_y], "yaw_deg": a}], "height": 2.4}
sim/build_stage.py turns every rectangle into a NavMeshVolume.
"""
import base64
import json
import os
import threading
import urllib.request

import cv2
import numpy as np
import trimesh
import viser.transforms as vtf
from scipy import ndimage

KEY = "walkarea"
RES = 0.1            # m per height-map cell
FLOOR_Z = (-0.12, 0.15)
BODY_Z = (0.25, 1.9)  # anything here blocks walking
MIN_SIDE = 0.6        # m; room for one person to pass (the navmesh keeps a 20 cm agent radius off the edges)
GEMINI_MODEL = "gemini-2.5-flash"


# ---- height map ------------------------------------------------------------------------------------------
class HeightMap:
    def __init__(self, mesh: trimesh.Trimesh, n_samples=400_000):
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples, seed=0)
        colors = None
        vc = getattr(mesh.visual, "vertex_colors", None)
        if vc is not None and len(vc) == len(mesh.vertices):
            colors = vc[mesh.faces[face_idx]][:, :, :3].mean(1)
        self.lo = pts[:, :2].min(0) - RES
        hi = pts[:, :2].max(0) + RES
        self.shape = tuple((np.ceil((hi - self.lo) / RES).astype(int))[::-1])  # rows (y), cols (x)
        ij = self._cells(pts[:, :2])
        z = pts[:, 2]
        floor = np.zeros(self.shape, np.int32)
        body = np.zeros(self.shape, np.int32)
        fz = (z > FLOOR_Z[0]) & (z < FLOOR_Z[1])
        bz = (z > BODY_Z[0]) & (z < BODY_Z[1])
        np.add.at(floor, (ij[fz, 0], ij[fz, 1]), 1)
        np.add.at(body, (ij[bz, 0], ij[bz, 1]), 1)
        self.top = np.full(self.shape, np.nan)
        order = np.argsort(z)  # last write wins -> highest point per cell
        self.top[ij[order, 0], ij[order, 1]] = z[order]
        self.rgb = np.zeros((*self.shape, 3), np.float64)
        if colors is not None:
            cnt = np.zeros(self.shape)
            np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
            for c in range(3):
                acc = np.zeros(self.shape)
                np.add.at(acc, (ij[:, 0], ij[:, 1]), colors[:, c])
                self.rgb[..., c] = acc / np.maximum(cnt, 1)
        blocked = ndimage.binary_dilation(body > 2, iterations=1)
        floor_ok = ndimage.binary_closing(floor > 0, iterations=2)
        self.walkable = ndimage.binary_opening(floor_ok & ~blocked, iterations=1)

    def _cells(self, xy):
        c = np.floor((np.asarray(xy) - self.lo) / RES).astype(int)
        return np.column_stack([np.clip(c[:, 1], 0, self.shape[0] - 1), np.clip(c[:, 0], 0, self.shape[1] - 1)])

    def to_px(self, xy):
        """world xy -> image pixel (col, row) with north (+y) up."""
        xy = np.atleast_2d(xy)
        col = (xy[:, 0] - self.lo[0]) / RES
        row = self.shape[0] - (xy[:, 1] - self.lo[1]) / RES
        return np.column_stack([col, row])

    def from_px(self, px):
        px = np.atleast_2d(px)
        return np.column_stack([px[:, 0] * RES + self.lo[0], (self.shape[0] - px[:, 1]) * RES + self.lo[1]])

    def image(self, rects=(), selected=None, scale=None):
        """Top-down picture: mesh color, walkable cells tinted green, rectangles outlined."""
        scale = scale or max(2, 480 // self.shape[1])
        img = np.flipud(self.rgb).copy()
        if img.max() <= 0:  # no colors: shade by height
            t = np.flipud(np.nan_to_num(self.top, nan=0))
            img = np.repeat((np.clip(t / 2.5, 0, 1) * 200 + 40)[..., None], 3, 2)
        walk = np.flipud(self.walkable)
        img[walk] = img[walk] * 0.55 + np.array([60, 200, 90]) * 0.45
        img = cv2.resize(img.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for k, r in enumerate(rects):
            pts = (self.to_px(rect_corners(r)) * scale).astype(np.int32)
            color = (255, 140, 0) if k == selected else (30, 120, 255)
            cv2.polylines(img, [pts], True, color, 2)
            cv2.putText(img, str(k + 1), tuple(pts.mean(0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    def walkable_fraction(self, r):
        mask = np.zeros(self.shape, np.uint8)
        px = self.to_px(rect_corners(r))
        px[:, 1] = self.shape[0] - px[:, 1]  # back to row index from the bottom (unflipped grid)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
        n = mask.sum()
        return float((mask.astype(bool) & self.walkable).sum() / n) if n else 0.0


def rect_corners(r):
    cx, cy = r["center"]
    w, h = r["size"]
    a = np.radians(r.get("yaw_deg", 0.0))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2) @ R.T + [cx, cy]


def _largest_rectangle(mask):
    """Largest axis-aligned all-True rectangle: (row0, col0, row1, col1) inclusive, or None."""
    rows, cols = mask.shape
    heights = np.zeros(cols, int)
    best, best_area = None, 0
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (c - s)
                if area > best_area:
                    best_area, best = area, (r - sh + 1, s, r, c - 1)
                start = s
            stack.append((start, h))
    return best


def _largest_rotated(hm: HeightMap, mask, angles):
    """Largest inscribed rectangle of `mask` over several yaw angles (the room's walls need not line up with the
    world axes, e.g. after a 3-point level). Each angle resamples the mask onto a grid rotated by that yaw."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    xy = hm.lo + (np.column_stack([cols, rows]) + 0.5) * RES
    best, best_area = None, 0.0
    for a in angles:
        t = np.radians(a)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        uv = xy @ R  # world -> rotated frame
        lo = uv.min(0) - RES
        n = np.ceil((uv.max(0) + RES - lo) / RES).astype(int)
        gu, gv = np.meshgrid(lo[0] + (np.arange(n[0]) + 0.5) * RES, lo[1] + (np.arange(n[1]) + 0.5) * RES)
        world = np.column_stack([gu.ravel(), gv.ravel()]) @ R.T
        ij = np.floor((world - hm.lo) / RES).astype(int)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < mask.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < mask.shape[0])
        sub = np.zeros(len(world), bool)
        sub[inside] = mask[ij[inside, 1], ij[inside, 0]]
        box = _largest_rectangle(sub.reshape(gu.shape))
        if box is None:
            continue
        r0, c0, r1, c1 = box
        w, h = (c1 - c0 + 1) * RES, (r1 - r0 + 1) * RES
        if min(w, h) >= MIN_SIDE and w * h > best_area + 1e-9:
            c = (lo + [(c0 + c1 + 1) / 2 * RES, (r0 + r1 + 1) / 2 * RES]) @ R.T
            best_area = w * h
            best = {"center": [round(float(c[0]), 2), round(float(c[1]), 2)], "size": [round(w, 2), round(h, 2)],
                    "yaw_deg": float(a)}
    return best


def propose_heightmap(hm: HeightMap, max_rects=3):
    """Largest inscribed rectangles of the walkable mask, searched over yaw angles in 5° steps."""
    mask = hm.walkable.copy()
    rects = []
    for _ in range(max_rects):
        r = _largest_rotated(hm, mask, range(0, 90, 5))
        if r is None or (rects and np.prod(r["size"]) < 0.25 * np.prod(rects[0]["size"])):
            break
        rects.append(r)
        # remove it (plus a 0.3 m margin) before looking for the next one
        grown = {**r, "size": [r["size"][0] + 0.6, r["size"][1] + 0.6]}
        px = hm.to_px(rect_corners(grown))
        px[:, 1] = hm.shape[0] - px[:, 1]
        cut = np.zeros(hm.shape, np.uint8)
        cv2.fillPoly(cut, [np.round(px).astype(np.int32)], 1)
        mask &= ~cut.astype(bool)
    return rects


def propose_gemini(hm: HeightMap, api_key: str, log=print):
    """Ask Gemini for walking rectangles on the top-down render. Returns rects in the world frame."""
    scale = 4
    img = hm.image(scale=scale)
    ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    h, w = img.shape[:2]
    prompt = (
        f"This is a top-down view of a 3D-scanned site, {w * RES / scale:.1f} m wide and {h * RES / scale:.1f} m tall "
        f"({RES / scale * 100:.1f} cm per pixel; +x is right, +y is up). Cells tinted green have floor and no "
        "obstacle at body height. We will simulate pedestrians walking here for LiDAR training data. Propose 1-3 "
        "rectangles covering the areas where people would realistically walk (corridors, sidewalks, open floor), "
        "avoiding furniture, walls and holes in the scan. Each side must be at least 1 m. Reply with JSON only: "
        '{"rects": [{"cx": px, "cy": px, "length": px, "width": px, "angle_deg": deg, "why": "..."}]} where '
        "cx, cy is the center in pixels, length is along the rotated x axis, and angle_deg is counter-clockwise.")
    body = {"contents": [{"parts": [{"text": prompt},
                                    {"inline_data": {"mime_type": "image/png",
                                                     "data": base64.b64encode(png.tobytes()).decode()}}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        reply = json.loads(resp.read())
    text = reply["candidates"][0]["content"]["parts"][0]["text"]
    rects = []
    for g in json.loads(text).get("rects", []):
        center = hm.from_px([[g["cx"] / scale, g["cy"] / scale]])[0]
        r = {"center": [round(float(center[0]), 2), round(float(center[1]), 2)],
             "size": [round(g["length"] / scale * RES, 2), round(g["width"] / scale * RES, 2)],
             # image y points down, so a counter-clockwise angle on screen is counter-clockwise in the world
             "yaw_deg": round(float(g.get("angle_deg", 0.0)), 1)}
        frac = hm.walkable_fraction(r)
        log(f"Gemini rect {r} ({frac:.0%} on walkable floor): {g.get('why', '')}")
        if frac >= 0.5 and min(r["size"]) >= MIN_SIDE * 0.8:
            rects.append(r)
    return rects


def save(project, rects, source):
    data = {"rects": rects, "height": 2.4, "source": source}
    (project.path / "walkarea.json").write_text(json.dumps(data, indent=2))
    area = sum(r["size"][0] * r["size"][1] for r in rects)
    project.set_status(KEY, "done", note=f"{len(rects)} area{'s' if len(rects) != 1 else ''}, {area:.1f} m²")
    project.invalidate_downstream(KEY)


def load(project):
    f = project.path / "walkarea.json"
    return json.loads(f.read_text()) if f.exists() else None


# ---- GUI -------------------------------------------------------------------------------------------------
def build(app, folder):
    gui, scene = app.server.gui, app.server.scene
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    with folder:
        state_md = gui.add_markdown("")
        prop_btn = gui.add_button("Propose from height map", icon="map")
        gem_btn = gui.add_button("Propose with Gemini", icon="sparkles", disabled=not has_key,
                                 hint=None if has_key else "set GEMINI_API_KEY and restart the GUI")
        image = gui.add_image(np.zeros((8, 8, 3), np.uint8), label="Top-down (green = walkable)")
        sel = gui.add_dropdown("Rectangle", ["(none)"])
        cx = gui.add_number("Center x (m)", 0.0, step=0.05)
        cy = gui.add_number("Center y (m)", 0.0, step=0.05)
        length = gui.add_number("Length along x (m)", 2.0, min=0.5, step=0.1)
        width = gui.add_number("Width along y (m)", 2.0, min=0.5, step=0.1)
        yaw = gui.add_slider("Yaw (deg)", min=-90, max=90, step=1, initial_value=0)
        with gui.add_folder("Edit", expand_by_default=True):
            add_btn = gui.add_button("Add rectangle", icon="plus")
            del_btn = gui.add_button("Delete selected", icon="trash", color="red")
        confirm_btn = gui.add_button("Confirm walking area", icon="check", color="green")

    st = {"hm": None, "rects": [], "sel": None, "syncing": False, "handles": [], "gizmo": None}
    edit_widgets = (cx, cy, length, width, yaw)

    def clear_scene():
        for h in st["handles"]:
            h.remove()
        st["handles"] = []
        if st["gizmo"] is not None:
            st["gizmo"].remove()
            st["gizmo"] = None

    def draw():
        clear_scene()
        for k, r in enumerate(st["rects"]):
            q = vtf.SO3.from_z_radians(np.radians(r["yaw_deg"])).wxyz
            color = (255, 140, 0) if k == st["sel"] else (40, 200, 90)
            st["handles"].append(scene.add_box(f"/walkarea/rect{k}", color=color,
                                               dimensions=(r["size"][0], r["size"][1], 0.04), opacity=0.45,
                                               wxyz=q, position=(*r["center"], 0.03)))
            st["handles"].append(scene.add_label(f"/walkarea/label{k}", f"area {k + 1}",
                                                 position=(*r["center"], 0.3)))
        if st["sel"] is not None:
            r = st["rects"][st["sel"]]
            g = scene.add_transform_controls("/walkarea/gizmo", scale=0.8, active_axes=(True, True, False),
                                             disable_rotations=True, disable_sliders=False,
                                             position=(*r["center"], 0.05))
            g.on_update(lambda _: gizmo_moved())
            st["gizmo"] = g
        if st["hm"] is not None:
            image.image = st["hm"].image(st["rects"], st["sel"])

    def sync_widgets():
        st["syncing"] = True
        names = [f"Area {k + 1}" for k in range(len(st["rects"]))] or ["(none)"]
        sel.options = names
        if st["sel"] is not None:
            sel.value = names[st["sel"]]
            r = st["rects"][st["sel"]]
            cx.value, cy.value = r["center"]
            length.value, width.value = r["size"]
            yaw.value = r["yaw_deg"]
        for w in (*edit_widgets, del_btn):
            w.disabled = st["sel"] is None
        confirm_btn.disabled = not st["rects"]
        st["syncing"] = False

    def set_rects(rects, select=0):
        st["rects"] = rects
        st["sel"] = select if rects else None
        sync_widgets()
        draw()

    def gizmo_moved():
        if st["sel"] is None:
            return
        p = st["gizmo"].position
        r = st["rects"][st["sel"]]
        r["center"] = [round(float(p[0]), 3), round(float(p[1]), 3)]
        st["syncing"] = True
        cx.value, cy.value = r["center"]
        st["syncing"] = False
        # move just the box, not a full redraw (the gizmo is being dragged)
        box = st["handles"][2 * st["sel"]]
        box.position = (*r["center"], 0.03)
        st["handles"][2 * st["sel"] + 1].position = (*r["center"], 0.3)
        image.image = st["hm"].image(st["rects"], st["sel"]) if st["hm"] is not None else image.image

    def widgets_changed(_):
        if st["syncing"] or st["sel"] is None:
            return
        r = st["rects"][st["sel"]]
        r["center"] = [float(cx.value), float(cy.value)]
        r["size"] = [float(length.value), float(width.value)]
        r["yaw_deg"] = float(yaw.value)
        draw()

    for w in edit_widgets:
        w.on_update(widgets_changed)

    @sel.on_update
    def _(_):
        if st["syncing"] or not st["rects"] or sel.value == "(none)":
            return
        st["sel"] = int(sel.value.split()[-1]) - 1
        sync_widgets()
        draw()

    def ensure_heightmap():
        if st["hm"] is None:
            mesh_file = app.project.world_mesh_path()
            if mesh_file is None:
                raise RuntimeError("run Scale & level first")
            st["hm"] = HeightMap(trimesh.load(mesh_file, process=False))
        return st["hm"]

    @prop_btn.on_click
    def _(_):
        try:
            rects = propose_heightmap(ensure_heightmap())
        except Exception as e:  # noqa: BLE001
            app.log(f"height-map proposal failed: {e}")
            return
        app.log(f"height map proposed {len(rects)} walking area(s)")
        if not rects:
            app.log(f"no open floor at least {MIN_SIDE} m wide was found; use 'Add rectangle' and size it by hand")
            return
        st["source"] = "height map"
        set_rects(rects)

    @gem_btn.on_click
    def _(_):
        def run():
            try:
                rects = propose_gemini(ensure_heightmap(), os.environ["GEMINI_API_KEY"], log=app.log)
            except Exception as e:  # noqa: BLE001
                app.log(f"Gemini proposal failed: {e}")
                return
            if not rects:
                app.log("Gemini returned no usable rectangles; try the height-map proposal")
                return
            st["source"] = f"Gemini ({GEMINI_MODEL})"
            set_rects(rects)
        app.log("asking Gemini for walking areas…")
        threading.Thread(target=run, daemon=True).start()

    @add_btn.on_click
    def _(_):
        hm = st["hm"]
        c = (hm.lo + np.array(hm.shape[::-1]) * RES / 2).tolist() if hm is not None else [0.0, 0.0]
        st["rects"].append({"center": [round(c[0], 2), round(c[1], 2)], "size": [2.0, 1.5], "yaw_deg": 0.0})
        set_rects(st["rects"], len(st["rects"]) - 1)

    @del_btn.on_click
    def _(_):
        if st["sel"] is None:
            return
        st["rects"].pop(st["sel"])
        set_rects(st["rects"], max(0, st["sel"] - 1))

    @confirm_btn.on_click
    def _(_):
        bad = [k + 1 for k, r in enumerate(st["rects"]) if min(r["size"]) < MIN_SIDE]
        if bad:
            app.log(f"areas {bad} are narrower than {MIN_SIDE} m; widen or delete them")
            return
        save(app.project, st["rects"], st.get("source", "manual"))
        app.log(f"walking area saved: {len(st['rects'])} rectangle(s)")
        app.stage_changed(KEY)
        refresh_state()

    def refresh_state():
        p = app.project
        entry = p.state["stages"][KEY]
        if p.world_mesh_path() is None:
            state_md.content = "*Run Scale & level first (the walking area is in meters).*"
        elif entry["status"] == "done":
            state_md.content = f"✅ Confirmed: {entry.get('note', '')}. Edit and confirm again to change it."
        else:
            state_md.content = "Propose an area, adjust it (drag the gizmo or type values), then confirm."
        prop_btn.disabled = p.world_mesh_path() is None
        gem_btn.disabled = not has_key or p.world_mesh_path() is None

    def refresh():
        if app.project is None:
            return
        st["hm"] = None
        clear_scene()
        data = load(app.project)
        if app.project.world_mesh_path() is not None:
            try:
                ensure_heightmap()
            except Exception as e:  # noqa: BLE001
                app.log(f"height map failed: {e}")
        st["source"] = data.get("source", "manual") if data else "manual"
        set_rects(data["rects"] if data else [])
        refresh_state()

    app.on_project_opened(refresh)
    app.on_stage_changed(lambda key: refresh() if key == "scale" else None)
