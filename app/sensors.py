"""LiDAR models: resolving a typed model name to something Isaac Sim can spawn, and its beam pattern.

Resolution order (the first hit wins):
1. sensors/registry.json: sensors authored before (the bundled PandarQT64, or ones the sensor agent wrote), each a
   local OmniLidar USD;
2. Isaac Sim's built-in configs (SUPPORTED_LIDAR_CONFIGS; NVIDIA-hosted assets, spawned by config name +
   variant);
3. neither: the sensor agent (app/sensor_agent.py) writes a new OmniLidar USD from web sources.

The beam pattern (per-channel elevation and azimuth offsets, horizontal steps per revolution) is read from a local
USD so the placement stage can preview coverage without launching Isaac. Built-in configs have no local USD, so
their preview is a generic pattern guessed from the name, flagged as approximate.
"""
import difflib
import json
import re
import runpy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.project import ROOT

REGISTRY = ROOT / "sensors" / "registry.json"
ISAAC_CONFIGS = Path(r"C:\isaacsim\exts\isaacsim.sensors.experimental.rtx\isaacsim\sensors\experimental\rtx\impl"
                     r"\rtx_lidar_configs.py")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_registry() -> list[dict]:
    return json.loads(REGISTRY.read_text())["sensors"] if REGISTRY.exists() else []


def add_to_registry(entry: dict):
    data = {"sensors": [s for s in load_registry() if norm(s["name"]) != norm(entry["name"])] + [entry]}
    REGISTRY.write_text(json.dumps(data, indent=2))


def isaac_builtin() -> dict[str, list]:
    """asset path -> list of variants (strings or {set: value} dicts), from the installed Isaac Sim."""
    if not ISAAC_CONFIGS.exists():
        return {}
    cfgs = runpy.run_path(str(ISAAC_CONFIGS))["SUPPORTED_LIDAR_CONFIGS"]
    return {k: sorted(v) if isinstance(v, set) else list(v) for k, v in cfgs.items()}


@dataclass
class Resolved:
    name: str
    source: str                   # "registry" | "isaac"
    usd_path: str | None = None   # repo-relative, for registry sensors
    config: str | None = None     # Isaac asset path, for built-ins
    variants: list = field(default_factory=list)
    variant: object = None
    info: dict = field(default_factory=dict)
    score: float = 1.0

    def to_json(self):
        return {"model": self.name, "source": self.source, "usd_path": self.usd_path, "config": self.config,
                "variant": self.variant, "info": self.info}


def _variant_label(v) -> str:
    return v if isinstance(v, str) else " / ".join(v.values())


def resolve(query: str, min_score=0.6) -> Resolved | None:
    q = norm(query)
    if not q:
        return None
    best = None
    for s in load_registry():
        for alias in [s["name"], *s.get("aliases", [])]:
            a = norm(alias)
            score = 1.0 if a == q else (0.9 if q in a or a in q else difflib.SequenceMatcher(None, q, a).ratio())
            if best is None or score > best.score:
                best = Resolved(s["name"], "registry", usd_path=s["usd_path"], info=s, score=score)
    if best is not None and best.score >= 0.85:
        return best
    for path, variants in isaac_builtin().items():
        stem = Path(path).stem
        vendor = path.split("/")[3] if path.count("/") >= 4 else ""
        names = [stem, f"{vendor} {stem}", *[_variant_label(v) for v in variants]]
        for name in names:
            a = norm(name)
            score = 1.0 if a == q else (0.9 if q in a or a in q else difflib.SequenceMatcher(None, q, a).ratio())
            if best is None or score > best.score:
                pick = None
                if variants:  # the variant that best matches the query, else the first
                    pick = max(variants, key=lambda v: difflib.SequenceMatcher(None, q, norm(_variant_label(v))).ratio())
                best = Resolved(f"{vendor} {stem}".strip(), "isaac", config=path, variants=variants, variant=pick,
                                score=score)
    return best if best is not None and best.score >= min_score else None


# ---- beam pattern --------------------------------------------------------------------------------------------
@dataclass
class BeamPattern:
    elevation_deg: np.ndarray   # per channel
    azimuth_deg: np.ndarray     # per-channel azimuth offset
    steps: int                  # horizontal firings per revolution
    near_m: float
    far_m: float
    approximate: bool = False
    note: str = ""

    def rays(self, max_rays=60_000):
        """Unit directions in the sensor frame (x forward, z up), subsampled horizontally to about max_rays."""
        steps = max(8, min(self.steps, max_rays // max(1, len(self.elevation_deg))))
        base = np.linspace(0, 360, steps, endpoint=False)
        az = np.radians(base[:, None] + self.azimuth_deg[None, :]).ravel()
        el = np.radians(np.broadcast_to(self.elevation_deg, (steps, len(self.elevation_deg)))).ravel()
        return np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def beam_pattern_from_usd(usd_path: Path) -> BeamPattern:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetDefaultPrim()
    if not prim or not prim.GetAttribute("omni:sensor:Core:emitterState:s001:elevationDeg"):
        prim = next((p for p in stage.Traverse() if p.GetTypeName() == "OmniLidar"), prim)

    def get(name, default=None):
        a = prim.GetAttribute(f"omni:sensor:Core:{name}")
        v = a.Get() if a else None
        return default if v is None else v

    el = np.array(get("emitterState:s001:elevationDeg", []), np.float64)
    az = np.array(get("emitterState:s001:azimuthDeg", np.zeros(len(el))), np.float64)
    if not len(el):
        raise ValueError(f"{usd_path} has no emitterState elevation table")
    firing = float(get("patternFiringRateHz", 0) or 0)
    scan = float(get("scanRateBaseHz", 10) or 10)
    steps = int(round(firing / scan)) if firing else 1024
    return BeamPattern(el, az, steps, float(get("nearRangeM", 0.1)), float(get("farRangeM", 100.0)))


def generic_pattern(name: str) -> BeamPattern:
    """Rough pattern for a built-in config without a local USD, guessed from its name."""
    n, raw = norm(name), name.lower()
    m = re.search(r"(?<![0-9])(\d{1,3})ch", raw)
    channels = int(m.group(1)) if m else (32 if "32" in n else 16 if "16" in n else 64)
    vfov = 45.0 if "os1" in n else 90.0 if "os0" in n else 22.5 if "os2" in n else 31.0 if "xt32" in n else 30.0
    res = re.search(r"hz(\d+)res", raw)
    steps = int(res.group(1)) if res else 1024
    return BeamPattern(np.linspace(-vfov / 2, vfov / 2, channels), np.zeros(channels), steps, 0.3, 100.0,
                       approximate=True, note=f"approximate: {channels} channels over +/-{vfov / 2:.1f} deg (guessed "
                                              "from the name; the real pattern lives in NVIDIA's hosted asset)")


def pattern_for(model: dict) -> BeamPattern:
    if model.get("usd_path"):
        return beam_pattern_from_usd(ROOT / model["usd_path"])
    return generic_pattern(f"{model.get('config', '')} {_variant_label(model['variant']) if model.get('variant') else ''}")
