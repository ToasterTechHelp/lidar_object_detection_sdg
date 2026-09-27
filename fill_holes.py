"""Fill enclosed holes in a mesh by closing them in from their edges.

For each boundary loop smaller than --max-hole:
  1. Ear-clip the loop, always cutting the sharpest corner first, so the edges converge inward.
  2. Subdivide the patch so it has vertices of roughly the surrounding edge length.
  3. Relax the new interior vertices (Laplacian fairing) with the hole rim held fixed,
     so the patch settles into a smooth membrane matching the surrounding surface.
Colors are diffused into the patch the same way.

The largest loop (the scan's outer border) is always skipped.

Usage:
    python fill_holes.py out/hallway/mesh.ply --out out/hallway/mesh_filled --max-hole 0.5
"""
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh


def boundary_loops(faces):
    """Return boundary loops, each ordered opposite to the mesh's winding so fill faces match it."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(directed, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = directed[cnt[inv.ravel()] == 1]
    nxt = {b: a for a, b in bnd}  # walk each mesh boundary edge a->b backwards
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt.get(v)
            if v is None:
                break
        if v == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def project_2d(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    return (pts - c) @ vt[:2].T


def ear_clip(loop, pts2):
    """Triangulate a loop, cutting the smallest-angle convex ear each step. Returns None on failure."""
    idx = list(range(len(loop)))
    x, y = pts2[:, 0], pts2[:, 1]
    sign = np.sign(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) or 1.0
    tris = []
    while len(idx) > 3:
        best, best_ang = None, np.inf
        n = len(idx)
        for k in range(n):
            p, c, q = pts2[idx[k - 1]], pts2[idx[k]], pts2[idx[(k + 1) % n]]
            a, b = p - c, q - c
            if sign * (a[0] * b[1] - a[1] * b[0]) >= 0:  # reflex or degenerate corner
                continue
            ang = np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))
            if ang >= best_ang:
                continue
            others = [idx[j] for j in range(n) if j not in (k - 1 if k else n - 1, k, (k + 1) % n)]
            if others and _any_inside(pts2[others], p, c, q):
                continue
            best, best_ang = k, ang
        if best is None:
            return None
        k = best
        tris.append((loop[idx[k - 1]], loop[idx[k]], loop[idx[(k + 1) % len(idx)]]))
        idx.pop(k)
    tris.append((loop[idx[0]], loop[idx[1]], loop[idx[2]]))
    return tris


def _any_inside(pts, a, b, c):
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[:, 1] - o[1]) - (p[1] - o[1]) * (q[:, 0] - o[0])
    d1, d2, d3 = cross(a, b, pts), cross(b, c, pts), cross(c, a, pts)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return np.any(~(neg & pos))


def fair(verts, faces, free, values, iters=300):
    """Uniform Laplacian relaxation of `values` on `free` vertices, others held fixed."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    n = len(verts)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    for _ in range(iters):
        values[free] = (A @ values)[free] / deg[free, None]
    return values


def refine(pv, pf, pc, loop, max_rounds=30):
    """Liepa-style refinement: split triangles at their centroid when they're large relative to
    the local edge length, then flip interior edges to keep triangles well shaped. Rim edges never change."""
    pv, pc = list(pv), list(pc)
    faces = [tuple(f) for f in pf]
    n = len(loop)
    rim_edges = {frozenset((loop[i], loop[(i + 1) % n])) for i in range(n)}
    sigma = {}
    for i, v in enumerate(loop):  # local scale = mean length of the vertex's two rim edges
        sigma[v] = 0.5 * (np.linalg.norm(pv[v] - pv[loop[i - 1]]) + np.linalg.norm(pv[v] - pv[loop[(i + 1) % n]]))
    for v in range(len(pv)):
        sigma.setdefault(v, np.mean(list(sigma.values())))
    alpha = np.sqrt(2)

    for _ in range(max_rounds):
        split, out = False, []
        for a, b, c in faces:
            cen = (pv[a] + pv[b] + pv[c]) / 3
            s = (sigma[a] + sigma[b] + sigma[c]) / 3
            if all(alpha * np.linalg.norm(cen - pv[v]) > max(s, sigma[v]) for v in (a, b, c)):
                m = len(pv)
                pv.append(cen)
                pc.append((pc[a] + pc[b] + pc[c]) / 3)
                sigma[m] = s
                out += [(a, b, m), (b, c, m), (c, a, m)]
                split = True
            else:
                out.append((a, b, c))
        faces = out
        faces = _relax_edges(np.array(pv), faces, rim_edges)
        if not split:
            break
    return np.array(pv), np.array(faces), np.array(pc)


def _angle(p, a, b):
    u, v = a - p, b - p
    return np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1))


def _relax_edges(pv, faces, rim_edges, max_passes=10):
    """Flip interior edges that aren't locally Delaunay (opposite angles sum > pi)."""
    faces = [list(f) for f in faces]
    for _ in range(max_passes):
        edge_faces = {}
        for fi, f in enumerate(faces):
            for k in range(3):
                edge_faces.setdefault((f[k], f[(k + 1) % 3]), fi)
        existing = {frozenset(e) for e in edge_faces}
        touched, flipped = set(), 0
        for (a, b), f1 in list(edge_faces.items()):
            f2 = edge_faces.get((b, a))
            if f2 is None or f1 in touched or f2 in touched or frozenset((a, b)) in rim_edges:
                continue
            c = next(v for v in faces[f1] if v not in (a, b))
            d = next(v for v in faces[f2] if v not in (a, b))
            if frozenset((c, d)) in existing:
                continue
            if _angle(pv[c], pv[a], pv[b]) + _angle(pv[d], pv[a], pv[b]) <= np.pi + 1e-9:
                continue
            # Only flip if the new faces keep the patch's orientation (quad is convex).
            n_old = np.cross(pv[b] - pv[a], pv[c] - pv[a]) + np.cross(pv[a] - pv[b], pv[d] - pv[b])
            n1 = np.cross(pv[d] - pv[a], pv[c] - pv[a])
            n2 = np.cross(pv[b] - pv[d], pv[c] - pv[d])
            if n1 @ n_old <= 0 or n2 @ n_old <= 0:
                continue
            faces[f1], faces[f2] = [a, d, c], [d, b, c]
            existing.add(frozenset((c, d)))
            touched.update((f1, f2))
            flipped += 1
        if not flipped:
            break
    return [tuple(f) for f in faces]


def fill_hole(verts, colors, loop):
    loop_pts = verts[loop]
    tris = ear_clip(loop, project_2d(loop_pts))
    if tris is None:  # projection folds over itself; fall back to a fan from the centroid
        c = len(verts)
        verts = np.vstack([verts, loop_pts.mean(0)])
        colors = np.vstack([colors, colors[loop].mean(0)])
        tris = [(loop[i], loop[(i + 1) % len(loop)], c) for i in range(len(loop))]
    tris = np.array(tris)

    # Refine just the patch (vertices local to it), then relax its interior.
    used = np.unique(tris)
    remap = {g: i for i, g in enumerate(used)}
    local_faces = np.vectorize(remap.get)(tris)
    loop_local = [remap[v] for v in loop]
    pv, pf, pc = refine(verts[used], local_faces, colors[used], loop_local)
    # The rim (original loop vertices) stays fixed; only inserted vertices move.
    rim = np.array([remap[v] for v in loop if v in remap])
    free = np.setdiff1d(np.arange(len(pv)), rim)
    if len(free):
        pv = fair(pv, pf, free, pv.copy())
        pc = fair(pv, pf, free, pc.astype(float).copy())

    # Stitch back: original loop vertices keep their global ids; everything else is new.
    glob = np.full(len(pv), -1)
    glob[: len(used)] = used
    new = glob < 0
    glob[new] = len(verts) + np.arange(new.sum())
    verts = np.vstack([verts, pv[new]])
    colors = np.vstack([colors, pc[new]])
    return verts, colors, glob[pf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--out", default=None, help="output path without extension (default: <mesh>_filled)")
    ap.add_argument("--max-hole", type=float, default=0.5, help="only fill loops whose diameter is below this (scene units)")
    args = ap.parse_args()

    m = trimesh.load(args.mesh, process=True)
    verts = np.asarray(m.vertices, dtype=float)
    faces = np.asarray(m.faces)
    colors = np.asarray(m.visual.vertex_colors, dtype=float) if m.visual.kind == "vertex" else np.full((len(verts), 4), 200.0)

    loops = boundary_loops(faces)
    loops.sort(key=len, reverse=True)
    print(f"{len(loops)} boundary loops; skipping outer border ({len(loops[0])} verts)")

    new_faces, filled = [faces], 0
    for loop in loops[1:]:
        diam = np.linalg.norm(np.ptp(verts[loop], axis=0))
        if diam > args.max_hole:
            print(f"  skip loop of {len(loop)} verts, diameter {diam:.3f} > --max-hole")
            continue
        verts, colors, f = fill_hole(verts, colors, loop)
        new_faces.append(f)
        filled += 1

    out = trimesh.Trimesh(verts, np.vstack(new_faces), vertex_colors=np.clip(colors, 0, 255).astype(np.uint8), process=True)
    out.fix_normals()
    remaining = len(boundary_loops(np.asarray(out.faces)))
    print(f"filled {filled} holes; {remaining} boundary loop(s) left; {len(out.faces):,} faces")

    base = Path(args.out) if args.out else Path(args.mesh).with_name(Path(args.mesh).stem + "_filled")
    base.parent.mkdir(parents=True, exist_ok=True)
    out.export(base.with_suffix(".ply"))
    out.export(base.with_suffix(".obj"))
    print(f"wrote {base}.ply and {base}.obj")


if __name__ == "__main__":
    main()
