# Hesai PandarQT64 — Isaac Sim Asset

RTX Lidar profile + aligned CAD for the Hesai PandarQT64 (64-channel mechanical,
10 Hz, 0.6° horizontal resolution), for use in Isaac Sim.

## Files

- `PandarQT64_Assembly.usda` — **the asset to import.** Bundles the sensor and
  the visual CAD, already aligned to each other.
  - `Lidar` — the `OmniLidar` sensor prim, referencing `Hesai_PandarQT64.usda`.
    Channel angles come from this unit's factory calibration, read off the
    sensor with PandarView 2 — **not** the generic design table in the
    manual's Appendix I, which differs by up to ~1.6° in azimuth. Firing
    offsets and range/wavelength specs come from the Hesai PandarQT user
    manual, doc version Q01-en-241210.
  - `VisualOffset/CAD` — the physical CAD mesh (`pandar_qt64.usdc`), offset so
    the visible enclosure lines up with the sensor's coordinate origin.
- `Hesai_PandarQT64.usda` — sensor profile only, no CAD. Use this instead of
  the assembly when you only want geometry-free point clouds (e.g. headless
  training-data generation where the visual mesh is never rendered).

## Importing

**Drag and drop (recommended):** open the Content window in Isaac Sim,
navigate to this folder, and drag `PandarQT64_Assembly.usda` into the
viewport or stage. This is a plain USD reference — no extension or script
required.

**From Python**, reference the profile-only sensor directly:

```python
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor

lidar = Lidar.create(
    "/World/PandarQT64",
    usd_path=r"<repo>/Hesai_PandarQT64.usda",
    tick_rate=10.0,
)
sensor = LidarSensor(lidar, annotators=[])
sensor.attach_writer("draw-point-cloud", size=0.05, color=[0.0, 1.0, 0.5, 1.0])
```

## Why this isn't a Create-menu entry

Isaac Sim's Create > Isaac > Sensors > RTX Lidar menu is driven by a hardcoded
list (`SUPPORTED_LIDAR_CONFIGS` in `isaacsim.sensors.experimental.rtx`) that
only resolves against NVIDIA's own Nucleus-hosted asset library — there's no
supported way to register a third-party sensor into that menu without patching
the installed extension. Drag-and-drop (or the Python snippet above) is the
robust, update-safe way to bring in a custom sensor asset.
