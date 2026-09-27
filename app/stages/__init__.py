"""Stage panels. Each stage module exposes build(app, folder) and is registered in PANELS by stage key.
Stages without a module yet show a placeholder naming the plan piece that will implement them."""

from app.stages import capture, dataset, mesh, scale, sensor_model, sensor_place, simulate, train, vggt, walkarea
from app.stages import export as export_stage

PANELS = {m.KEY: m.build for m in (capture, vggt, mesh, scale, walkarea, sensor_model, sensor_place, simulate, dataset,
                                          train, export_stage)}

UPCOMING = {
    "capture": "P1", "reconstruct": "P2", "mesh": "P3", "scale": "P4", "walkarea": "P5",
    "sensor_model": "P6", "sensor_place": "P7", "simulate": "P8", "dataset": "P9",
    "train": "P10", "export": "P11",
}


def build_panel(app, key, folder):
    if key in PANELS:
        PANELS[key](app, folder)
    else:
        with folder:
            app.server.gui.add_markdown(f"*Not built yet (plan piece {UPCOMING[key]}).*")
