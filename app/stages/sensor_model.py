"""P6 LiDAR model: the user types the sensor they will deploy; it resolves to something Isaac Sim can spawn.

Registry sensors (reused, local USD) -> Isaac Sim built-ins (config + variant) -> the sensor agent, which
searches the web for the sensor's specs and writes a new OmniLidar USD into sensors/ (see app/sensor_agent.py).
Writes sensor_model.json: {"model", "source", "usd_path" | "config" + "variant", "info"}.
"""
import json
import os
import sys

from app import sensors
from app.project import ROOT

KEY = "sensor_model"


def load(project):
    f = project.path / "sensor_model.json"
    return json.loads(f.read_text()) if f.exists() else None


def save(project, model: dict):
    (project.path / "sensor_model.json").write_text(json.dumps(model, indent=2))
    where = model["usd_path"] or f"Isaac built-in {model['config']}"
    project.set_status(KEY, "done", note=f"{model['model']} ({model['source']})")
    project.invalidate_downstream(KEY)
    return where


def build(app, folder):
    gui = app.server.gui
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    with folder:
        state_md = gui.add_markdown("")
        query = gui.add_text("LiDAR model", "Hesai PandarQT64")
        find_btn = gui.add_button("Find sensor", icon="search")
        result_md = gui.add_markdown("")
        variant_dd = gui.add_dropdown("Variant", ["(none)"], visible=False)
        use_btn = gui.add_button("Use this sensor", icon="check", color="green", disabled=True)
        agent_btn = gui.add_button("Research it (web agent)", icon="world-search", visible=False,
                                   disabled=not has_key,
                                   hint=None if has_key else "set ANTHROPIC_API_KEY and restart the GUI")
        known_md = gui.add_markdown("")

    st = {"res": None}

    def show_known():
        reg = ", ".join(s["name"] for s in sensors.load_registry()) or "none"
        n = len(sensors.isaac_builtin())
        known_md.content = f"<small>Registry: {reg}. Isaac built-ins: {n} assets (Ouster, HESAI, SICK, NVIDIA…).</small>"

    @find_btn.on_click
    def _(_):
        res = sensors.resolve(query.value)
        st["res"] = res
        agent_btn.visible = res is None
        variant_dd.visible = False
        if res is None:
            result_md.content = (f"**{query.value}** is not in the registry or Isaac's built-ins. The sensor agent "
                                 "can find its specs on the web and write a sensor model (slow, costs API credits).")
            use_btn.disabled = True
            return
        if res.source == "registry":
            info = res.info
            result_md.content = (f"✅ **{res.name}**: from the registry (reused).  \n{info.get('channels', '?')} channels, "
                                 f"vertical FOV {info.get('vfov_deg', '?')}°, {info.get('rate_hz', '?')} Hz  \n"
                                 f"<small>{info.get('source', '')}</small>")
        else:
            labels = [sensors._variant_label(v) for v in res.variants]
            if labels:
                variant_dd.options = labels
                variant_dd.value = sensors._variant_label(res.variant)
                variant_dd.visible = True
            result_md.content = (f"✅ **{res.name}**: Isaac Sim built-in `{res.config}` (downloaded from NVIDIA's "
                                 f"asset server at simulation time).  \nMatch score {res.score:.2f}; check the variant.")
        use_btn.disabled = False

    @use_btn.on_click
    def _(_):
        res = st["res"]
        if res is None:
            return
        if res.source == "isaac" and res.variants:
            res.variant = next(v for v in res.variants if sensors._variant_label(v) == variant_dd.value)
        model = res.to_json()
        model["info"] = {k: v for k, v in res.info.items() if k != "aliases"}
        save(app.project, model)
        app.log(f"sensor model set: {model['model']} ({model['source']})")
        app.stage_changed(KEY)
        refresh()

    @agent_btn.on_click
    def _(_):
        name = query.value.strip()
        cmd = [sys.executable, "-m", "app.sensor_agent", name]

        def done(code):
            if code == 0:
                app.log(f"sensor agent finished; resolving '{name}' again")
                find_btn._impl.update_cb[0](None)
            else:
                app.log(f"sensor agent failed (exit {code}); see log")
            show_known()

        app.jobs.run(f"Research LiDAR '{name}'", cmd, cwd=ROOT, on_done=done)

    def refresh():
        if app.project is None:
            return
        m = load(app.project)
        if m:
            state_md.content = f"✅ Using **{m['model']}** ({m['source']}). Search again to change it."
            query.value = m["model"]
        else:
            state_md.content = "Type the LiDAR you will deploy and press Find."
        show_known()

    app.on_project_opened(refresh)
