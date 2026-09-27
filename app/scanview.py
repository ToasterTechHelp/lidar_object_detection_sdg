"""Drawing one labelled LiDAR scan (points + pedestrian boxes) into the Viser scene, in the world frame."""
import numpy as np

from app import simdata

BG = np.array([150, 160, 175], np.uint8)
PED_COLORS = np.array([[230, 60, 60], [255, 150, 0], [220, 60, 200], [60, 140, 255], [250, 220, 40],
                       [40, 200, 200], [160, 90, 255], [255, 100, 140], [120, 220, 60], [255, 190, 120]], np.uint8)
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners_world(box):
    lo, hi = np.array(box["local_min"]), np.array(box["local_max"])
    c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return (c @ np.array(box["box_to_world"]))[:, :3]


def box_segments(boxes):
    segs, cols = [], []
    for b in boxes:
        c = box_corners_world(b)
        color = PED_COLORS[(int(b["instance_id"]) - 1) % len(PED_COLORS)]
        for i, j in EDGES:
            segs.append([c[i], c[j]])
            cols.append([color, color])
    return np.array(segs, np.float32).reshape(-1, 2, 3), np.array(cols, np.uint8).reshape(-1, 2, 3)


class ScanDrawer:
    """Keeps one scan's scene nodes under a prefix. Nodes are re-added under the same names, which replaces
    them in place (no flicker while playing scans back). Only one drawer shows at a time: drawing hides the others,
    so the Simulate and Dataset stages don't overlay two scans."""
    _all = []

    def __init__(self, scene, prefix):
        self.scene, self.prefix, self.h = scene, prefix, {}
        ScanDrawer._all.append(self)

    def clear(self):
        for h in self.h.values():
            h.remove()
        self.h = {}

    def draw(self, run_dir, i, point_size=0.025):
        pts, labels, boxes = simdata.load_scan(run_dir, i)
        for other in ScanDrawer._all:
            if other is not self and other.h:
                other.clear()
        world = simdata.transform(pts, simdata.sensor_to_world(run_dir)).astype(np.float32)
        inst = labels >> 16
        is_ped = (labels & 0xFFFF) == simdata.PEDESTRIAN_CLASS
        colors = np.tile(BG, (len(world), 1))
        colors[is_ped] = PED_COLORS[(inst[is_ped] - 1) % len(PED_COLORS)]
        self.h["points"] = self.scene.add_point_cloud(f"{self.prefix}/points", world, colors, point_size=point_size)
        if boxes:
            segs, cols = box_segments(boxes)
            self.h["boxes"] = self.scene.add_line_segments(f"{self.prefix}/boxes", segs, cols, thickness=2.5,
                                                           thickness_units="screen")
        elif "boxes" in self.h:
            self.h.pop("boxes").remove()
        return {"points": len(world), "ped_points": int(is_ped.sum()), "boxes": len(boxes)}
