"""A project is one deployment site: a folder under projects/ holding every stage's artifacts + state.json."""
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = ROOT / "projects"

# (key, title) in pipeline order. Each stage reads the previous stage's artifacts from the project folder.
STAGES = [
    ("capture", "Capture"),
    ("reconstruct", "Reconstruct (VGGT)"),
    ("mesh", "Mesh"),
    ("scale", "Scale & level"),
    ("walkarea", "Walking area"),
    ("sensor_model", "LiDAR model"),
    ("sensor_place", "LiDAR placement"),
    ("simulate", "Simulate"),
    ("dataset", "Filter & dataset"),
    ("train", "Train"),
    ("export", "Export"),
]
STAGE_TITLES = dict(STAGES)
STATUSES = ("todo", "running", "done", "skipped", "error")


class Project:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.state_file = path / "state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self.state.setdefault("stages", {})
        for key, _ in STAGES:
            self.state["stages"].setdefault(key, {"status": "todo"})

    @classmethod
    def create(cls, name: str) -> "Project":
        path = PROJECTS_DIR / name
        if path.exists():
            raise FileExistsError(f"project '{name}' already exists")
        path.mkdir(parents=True)
        p = cls(path)
        p.state["created"] = datetime.now().isoformat(timespec="seconds")
        p.save()
        return p

    def save(self):
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def dir(self, *parts) -> Path:
        """Path inside the project; the parent directory is created on demand."""
        p = self.path.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def status(self, stage: str) -> str:
        return self.state["stages"][stage]["status"]

    def set_status(self, stage: str, status: str, **info):
        assert status in STATUSES, status
        entry = self.state["stages"][stage]
        entry["status"] = status
        entry["updated"] = datetime.now().isoformat(timespec="seconds")
        entry.update(info)
        self.save()

    def mesh_path(self) -> Path | None:
        """The mesh later stages build on: the hole-filled one if the mesh stage produced it."""
        final = self.state["stages"]["mesh"].get("final", "mesh.ply")
        for name in (final, "mesh.ply"):
            if (self.path / "mesh" / name).exists():
                return self.path / "mesh" / name
        return None

    def world_mesh_path(self, ext: str = "ply") -> Path | None:
        """The mesh in the world frame (meters, z up, floor z = 0); exists once the scale stage has run."""
        p = self.path / "mesh" / f"mesh_world.{ext}"
        return p if p.exists() else None

    def invalidate_downstream(self, stage: str):
        """A stage's output changed: finished later stages were built from stale inputs."""
        keys = [k for k, _ in STAGES]
        for key in keys[keys.index(stage) + 1:]:
            if self.status(key) == "done":
                self.set_status(key, "todo", note=f"outdated: {STAGE_TITLES[stage]} was re-run")

    def import_reconstruction(self, out_dir: Path, glb: Path | None = None):
        """Seed the project from an earlier glb_to_mesh.py run (out/<name>/) and optionally its VGGT GLB."""
        copied = []
        for f in ("mesh.ply", "mesh.obj", "mesh_filled.ply", "mesh_filled.obj", "points_clean.ply", "cameras.json"):
            if (out_dir / f).exists():
                shutil.copy2(out_dir / f, self.dir("mesh", f))
                copied.append(f)
        if not copied:
            raise FileNotFoundError(f"no mesh outputs in {out_dir}")
        self.set_status("capture", "skipped", note=f"imported from {out_dir.name}")
        if glb is not None and glb.exists():
            shutil.copy2(glb, self.dir("vggt", "scene.glb"))
            self.set_status("reconstruct", "done", note=f"imported {glb.name}")
        else:
            self.set_status("reconstruct", "skipped", note=f"imported from {out_dir.name}")
        self.set_status("mesh", "done", note=f"imported {', '.join(copied)}")


def list_projects() -> list[str]:
    PROJECTS_DIR.mkdir(exist_ok=True)
    return sorted(p.name for p in PROJECTS_DIR.iterdir() if (p / "state.json").exists())


def list_importable() -> list[str]:
    """Earlier reconstructions in out/ that have a mesh."""
    out = ROOT / "out"
    if not out.exists():
        return []
    return sorted(p.name for p in out.iterdir() if (p / "mesh.ply").exists())


def project_path(name: str) -> Path:
    return PROJECTS_DIR / name
