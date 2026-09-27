"""Turn a VGGT demo GLB (point cloud + camera frustums) into a clean triangle mesh.

Usage:
    python glb_to_mesh.py data/hallway_vggt.glb --out out/hallway
    python glb_to_mesh.py data/hallway_vggt.glb --out out/hallway --scale 3.7 --depth 10

Outputs (in --out dir):
    points_clean.ply   filtered, downsampled point cloud with normals
    mesh.ply           mesh with vertex colors (inspect in MeshLab / Open3D)
    mesh.obj           same mesh, for Isaac Sim import
    cameras.json       camera centers recovered from the frustum meshes
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


def load_vggt_glb(path):
    """Return (points Nx3, colors Nx3 in [0,1], camera_centers Kx3) in scene coords."""
    scene = trimesh.load(path)
    points, colors, cams = [], [], []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        verts = trimesh.transform_points(np.asarray(geom.vertices), transform)
        if isinstance(geom, trimesh.PointCloud):
            points.append(verts)
            c = np.asarray(geom.colors)[:, :3] / 255.0 if geom.colors is not None and len(geom.colors) else np.full((len(verts), 3), 0.5)
            colors.append(c)
        else:
            # Camera frustum: the apex is the vertex most shared by faces; the mean is close enough.
            counts = np.bincount(np.asarray(geom.faces).ravel(), minlength=len(verts))
            cams.append(verts[np.argmax(counts)])
    if not points:
        raise ValueError(f"No point cloud found in {path}")
    return np.concatenate(points), np.concatenate(colors), np.array(cams)


def orient_normals_to_cameras(pcd, cam_centers):
    """Flip each normal to face its nearest camera (surfaces were seen from inside the room)."""
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    _, idx = cKDTree(cam_centers).query(pts)
    to_cam = cam_centers[idx] - pts
    flip = np.einsum("ij,ij->i", nrm, to_cam) < 0
    nrm[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(nrm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("--out", default="out/mesh")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all coords (fix VGGT's arbitrary scale)")
    ap.add_argument("--voxel", type=float, default=None, help="voxel size in scene units (default: extent/500)")
    ap.add_argument("--depth", type=int, default=9, help="Poisson octree depth (higher = more detail, more noise)")
    ap.add_argument("--trim", type=float, default=0.05, help="drop this fraction of lowest-density Poisson vertices")
    ap.add_argument("--target-faces", type=int, default=300_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pts, cols, cams = load_vggt_glb(args.glb)
    pts *= args.scale
    cams *= args.scale
    print(f"loaded {len(pts):,} points, {len(cams)} cameras")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols)

    extent = np.linalg.norm(pts.max(0) - pts.min(0))
    voxel = args.voxel or extent / 500
    pcd = pcd.voxel_down_sample(voxel)
    print(f"voxel {voxel:.4f} -> {len(pcd.points):,} points")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=8, radius=voxel * 4)
    print(f"after outlier removal: {len(pcd.points):,} points")

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=30))
    if len(cams):
        orient_normals_to_cameras(pcd, cams)
    else:
        pcd.orient_normals_consistent_tangent_plane(30)
    o3d.io.write_point_cloud(str(out / "points_clean.ply"), pcd)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, args.trim))
    # Poisson closes the surface into a blob; cut anything outside the observed points.
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.02, bbox.get_center())
    mesh = mesh.crop(bbox)

    # Drop small floating islands.
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[clusters] < 0.01 * counts.max())
    mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) > args.target_faces:
        mesh = mesh.simplify_quadric_decimation(args.target_faces)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    print(f"mesh: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
    o3d.io.write_triangle_mesh(str(out / "mesh.obj"), mesh)
    (out / "cameras.json").write_text(json.dumps({"scale": args.scale, "camera_centers": cams.tolist()}, indent=2))

    b = mesh.get_axis_aligned_bounding_box()
    print(f"bounds min {np.round(b.min_bound, 3)} max {np.round(b.max_bound, 3)} (size {np.round(b.get_extent(), 3)})")
    print(f"wrote {out}/mesh.ply, mesh.obj, points_clean.ply, cameras.json")


if __name__ == "__main__":
    main()
