"""Site-to-Detector GUI: every pipeline stage in one browser window.

Run:  .venv\\Scripts\\python -m app.main   then open http://localhost:8080
"""
import argparse
import threading
import time

import viser

from app import stages
from app.jobs import JobRunner
from app.project import ROOT, STAGES, Project, list_importable, list_projects, project_path
from app.scene import SceneView

BADGE = {"todo": "⬜", "running": "⏳", "done": "✅", "skipped": "➖", "error": "❌"}
LOG_LINES_SHOWN = 18
NO_PROJECT = "(none)"


class App:
    def __init__(self, port: int):
        self.server = viser.ViserServer(port=port)
        self.server.gui.configure_theme(control_layout="fixed", control_width="large", show_share_button=False,
                                        titlebar_content=None, brand_color=(0, 150, 136))
        self.jobs = JobRunner()
        self.scene = SceneView(self.server)
        self.project: Project | None = None
        self._log_dirty = False
        self._project_hooks = []
        self._change_hooks = []

        self._build_project_panel()
        self._build_status_panel()
        self.scene.build_view_panel()
        self._build_stage_panels()
        self._build_log_panel()

        self.jobs.on_line(lambda _: setattr(self, "_log_dirty", True))
        threading.Thread(target=self._refresh_loop, daemon=True).start()

        projects = list_projects()
        if projects:
            self.open_project(projects[-1])

    # ---- panels -------------------------------------------------------------------------------------------
    def _build_project_panel(self):
        gui = self.server.gui
        gui.add_markdown("## Site → Detector")
        with gui.add_folder("Project"):
            self.project_dd = gui.add_dropdown("Open", [NO_PROJECT, *list_projects()], initial_value=NO_PROJECT)
            self.project_dd.on_update(lambda _: self._on_project_selected())
            self.new_name = gui.add_text("New name", "site_01")
            gui.add_button("Create project").on_click(lambda _: self._create_project())
            importable = list_importable()
            self.import_dd = gui.add_dropdown("Import from out/", importable or ["(nothing)"],
                                              disabled=not importable)
            gui.add_button("Import as new project", disabled=not importable).on_click(lambda _: self._import())

    def _build_status_panel(self):
        with self.server.gui.add_folder("Pipeline"):
            self.status_md = self.server.gui.add_markdown("")

    def _build_stage_panels(self):
        for i, (key, title) in enumerate(STAGES, 1):
            folder = self.server.gui.add_folder(f"{i}. {title}", expand_by_default=False)
            stages.build_panel(self, key, folder)

    def _build_log_panel(self):
        with self.server.gui.add_folder("Log"):
            self.log_md = self.server.gui.add_markdown("```\n(no output yet)\n```")
            self.server.gui.add_button("Stop running job", color="red").on_click(lambda _: self.jobs.stop())

    # ---- actions ------------------------------------------------------------------------------------------
    def log(self, msg: str):
        self.jobs.emit(msg)

    def on_project_opened(self, fn):
        """Stage panels register here to redraw their results when a project is opened."""
        self._project_hooks.append(fn)

    def on_stage_changed(self, fn):
        """fn(stage_key) runs after a stage saves new output, so later stages can reload their inputs."""
        self._change_hooks.append(fn)

    def stage_changed(self, key: str):
        self.refresh_status()
        for fn in self._change_hooks:
            fn(key)

    def _on_project_selected(self):
        name = self.project_dd.value
        # open_project sets the dropdown itself, which re-fires this callback; skip that echo.
        if name != NO_PROJECT and (self.project is None or self.project.name != name):
            self.open_project(name)

    def open_project(self, name: str):
        self.project = Project(project_path(name))
        self.project_dd.value = name
        self.log(f"opened project '{name}'")
        self.scene.show_project(self.project, log=self.log)
        self.refresh_status()
        for fn in self._project_hooks:
            fn()

    def _create_project(self):
        try:
            Project.create(self.new_name.value.strip())
        except (FileExistsError, OSError) as e:
            self.log(f"create failed: {e}")
            return
        self._reload_project_list(self.new_name.value.strip())

    def _import(self):
        src = self.import_dd.value
        name = f"{src}_imported"
        try:
            p = Project.create(name)
            p.import_reconstruction(ROOT / "out" / src, glb=ROOT / "data" / f"{src}_vggt.glb")
        except (FileExistsError, FileNotFoundError) as e:
            self.log(f"import failed: {e}")
            return
        self.log(f"imported out/{src} into project '{name}'")
        self._reload_project_list(name)

    def _reload_project_list(self, select: str):
        self.project_dd.options = [NO_PROJECT, *list_projects()]
        self.open_project(select)

    def refresh_status(self):
        if self.project is None:
            self.status_md.content = "*No project open. Create or import one above.*"
            return
        rows = [f"**Project:** `{self.project.name}`", ""]
        for i, (key, title) in enumerate(STAGES, 1):
            entry = self.project.state["stages"][key]
            note = f" — {entry['note']}" if entry.get("note") else ""
            rows.append(f"{BADGE[entry['status']]} {i}. {title}{note}  ")
        self.status_md.content = "\n".join(rows)

    def _refresh_loop(self):
        # Batch log updates so chatty subprocesses don't flood the websocket.
        while True:
            time.sleep(0.5)
            if self._log_dirty:
                self._log_dirty = False
                lines = list(self.jobs.log)[-LOG_LINES_SHOWN:]
                running = f"**Running:** {self.jobs.current}\n\n" if self.jobs.busy else ""
                self.log_md.content = running + "```\n" + "\n".join(lines) + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    App(args.port)
    print(f"GUI running at http://localhost:{args.port}  (Ctrl+C to quit)")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
