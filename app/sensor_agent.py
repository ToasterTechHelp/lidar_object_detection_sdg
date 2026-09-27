"""Sensor agent: turns a LiDAR model name into an Isaac Sim RTX sensor (OmniLidar USD) from web sources.

    python -m app.sensor_agent "Velodyne VLP-16"                 # research + extract + write (needs API credentials)
    python -m app.sensor_agent "Velodyne VLP-16" --spec spec.json   # skip the web: write from a spec you trust

1. Research (Claude + server-side web search / web fetch): datasheets and user manuals for the per-channel beam
   table (elevation, azimuth offset, firing time), rotation rate, horizontal resolution, ranges. The report marks
   every value as published or estimated and cites its sources.
2. Extract: a second call turns the report into a strict JSON spec (structured outputs).
3. Write sensors/<Vendor_Model>/<Vendor_Model>.usda (same attribute set as the bundled hand-authored
   PandarQT64 profile), sources.md (the report), spec.json, and a registry entry, then re-read the USD's beam
   table (what the placement preview and Isaac Sim will use) as a check.
Rotating (mechanical) sensors only; solid-state scan patterns are rejected with a clear message.
"""
import argparse
import json
import re
import sys
import time

import anthropic

from app import sensors
from app.project import ROOT

MODEL = "claude-opus-5"
MAX_CONTINUATIONS = 8

RESEARCH_PROMPT = """Research the LiDAR sensor "{name}" so it can be simulated faithfully in NVIDIA Isaac Sim
(an RTX OmniLidar profile). Find the manufacturer's datasheet and, ideally, the user manual (manuals usually
contain the per-channel angle table and firing-time table in an appendix). Prefer primary sources (manufacturer
PDFs) over blogs; ROS drivers and vendor calibration files on GitHub are good secondary sources for beam tables.

Report, with the source URL for every value:
- exact vendor and model name, common aliases, whether it is a rotating (mechanical 360 degree) sensor
- number of channels (lasers)
- per-channel ELEVATION angle in degrees, for every channel, in firing / channel-id order
- per-channel AZIMUTH offset in degrees, for every channel (0 if the design has none)
- per-channel firing-time offset within one firing sequence, in nanoseconds (if published)
- rotation (frame) rate in Hz (the default one) and horizontal angular resolution in degrees at that rate
- minimum and maximum range in metres, range accuracy, laser wavelength in nm
- range at 10% reflectivity if published

Mark every value as PUBLISHED (quoted from a source) or ESTIMATED (and explain how you estimated it). Never
invent a per-channel table: if only the vertical field of view and channel count are published, say that the
table is estimated as evenly spaced. End with the complete tables written out in full."""

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "model": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "rotating": {"type": "boolean"},
        "channels": {"type": "integer"},
        "elevation_deg": {"type": "array", "items": {"type": "number"}},
        "azimuth_offset_deg": {"type": "array", "items": {"type": "number"}},
        "fire_time_ns": {"type": "array", "items": {"type": "integer"}},
        "rate_hz": {"type": "number"},
        "horizontal_resolution_deg": {"type": "number"},
        "near_m": {"type": "number"},
        "far_m": {"type": "number"},
        "range_accuracy_m": {"type": "number"},
        "wavelength_nm": {"type": "number"},
        "range_at_10pct_m": {"type": "number"},
        "estimated": {"type": "array", "items": {"type": "string"},
                      "description": "names of the fields above whose values are estimated rather than published"},
        "notes": {"type": "string"},
    },
    "required": ["vendor", "model", "aliases", "rotating", "channels", "elevation_deg", "azimuth_offset_deg",
                 "fire_time_ns", "rate_hz", "horizontal_resolution_deg", "near_m", "far_m", "range_accuracy_m",
                 "wavelength_nm", "range_at_10pct_m", "estimated", "notes"],
    "additionalProperties": False,
}

EXTRACT_PROMPT = """Below is a research report on a LiDAR sensor. Fill in the JSON spec from it.
- Arrays are per channel, in the report's channel order, one entry per channel.
- Use an empty array for azimuth_offset_deg or fire_time_ns when the report has no such table.
- Use 0 for a scalar the report does not give at all, and list it in "estimated".
- List in "estimated" every field whose value the report marks as estimated.
- "notes": one or two sentences on the main caveats.

<report>
{report}
</report>"""


def log(msg):
    print(f"[agent] {msg}", flush=True)


def research(client, name) -> str:
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 20},
             {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 20}]
    messages = [{"role": "user", "content": RESEARCH_PROMPT.format(name=name)}]
    for k in range(MAX_CONTINUATIONS):
        with client.beta.messages.stream(
            model=MODEL, max_tokens=64000, thinking={"type": "adaptive"}, output_config={"effort": "high"},
            tools=tools, messages=messages,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        ) as stream:
            for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "server_tool_use":
                    inp = event.content_block.input or {}
                    log(f"{event.content_block.name}: {inp.get('query') or inp.get('url') or '...'}")
            response = stream.get_final_message()
        if response.stop_reason == "refusal":
            sys.exit("[agent] the model declined this request")
        if response.stop_reason != "pause_turn":
            break
        log(f"long research turn, continuing ({k + 1})")
        messages = [messages[0], {"role": "assistant", "content": response.content}]
    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        sys.exit(f"[agent] research produced no report (stop reason {response.stop_reason})")
    return text


def extract(client, report) -> dict:
    response = client.messages.create(
        model=MODEL, max_tokens=16000, output_config={"format": {"type": "json_schema", "schema": SPEC_SCHEMA}},
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(report=report)}],
    )
    if response.stop_reason == "refusal":
        sys.exit("[agent] the model declined to extract the spec")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def check(spec) -> tuple[dict, list[str]]:
    """Validate and fill gaps. Returns (spec, warnings); exits on unusable specs."""
    warn = []
    n = int(spec["channels"])
    if not spec["rotating"]:
        sys.exit("[agent] this sensor is not a rotating LiDAR; only rotating (360 degree) sensors can be authored "
                 "from this template")
    if n <= 0 or len(spec["elevation_deg"]) != n:
        sys.exit(f"[agent] beam table incomplete: {len(spec['elevation_deg'])} elevations for {n} channels")
    if not all(-90 <= e <= 90 for e in spec["elevation_deg"]):
        sys.exit("[agent] elevation angles out of range")
    if len(spec["azimuth_offset_deg"]) != n:
        warn.append("no per-channel azimuth offsets: using 0")
        spec["azimuth_offset_deg"] = [0.0] * n
    if spec["rate_hz"] <= 0 or spec["horizontal_resolution_deg"] <= 0:
        sys.exit("[agent] rotation rate / horizontal resolution missing")
    ticks = round(360.0 / spec["horizontal_resolution_deg"])
    period_ns = int(1e9 / (spec["rate_hz"] * ticks))
    if len(spec["fire_time_ns"]) != n or max(spec["fire_time_ns"], default=0) >= period_ns:
        warn.append("no usable firing-time table: channels spread evenly over one firing period")
        spec["fire_time_ns"] = [int(period_ns * i / n) for i in range(n)]
    if spec["far_m"] <= 0:
        sys.exit("[agent] maximum range missing")
    if spec["near_m"] <= 0:
        spec["near_m"] = 0.1
        warn.append("no minimum range: using 0.1 m")
    return spec, warn


def slug(spec):
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{spec['vendor']} {spec['model']}").strip("_")


def usda(spec, prim) -> str:
    n = len(spec["elevation_deg"])
    ticks = round(360.0 / spec["horizontal_resolution_deg"])
    arr = lambda xs, fmt="{:.4f}": "[" + ", ".join(fmt.format(x) for x in xs) + "]"  # noqa: E731
    est = ", ".join(spec["estimated"]) or "none"
    min_refl = 0.1 if spec["range_at_10pct_m"] > 0 else 0.0
    doc = (f"RTX LiDAR profile for the {spec['vendor']} {spec['model']}, written by the pipeline's sensor agent "
           f"from web sources (see sources.md). Estimated values: {est}.")
    return f'''#usda 1.0
(
    defaultPrim = "{prim}"
    doc = "{doc.replace('"', "'")}"
)

def OmniLidar "{prim}" (
    prepend apiSchemas = ["OmniSensorGenericLidarCoreAPI"]
)
{{
    bool omni:sensor:Core:accumulateOutputs = 1
    token omni:sensor:Core:elementsCoordsType = "CARTESIAN"
    uint omni:sensor:Core:scanRateBaseHz = {round(spec["rate_hz"])}
    float omni:sensor:tickRate = {spec["rate_hz"]:g}
    # {spec["horizontal_resolution_deg"]:g} deg horizontal resolution: {ticks} firings per revolution
    uint omni:sensor:Core:patternFiringRateHz = {round(ticks * spec["rate_hz"])}
    token omni:sensor:Core:scanType = "ROTARY"
    token omni:sensor:Core:rotationDirection = "CW"
    uint omni:sensor:Core:numberOfChannels = {n}
    uint omni:sensor:Core:numberOfEmitters = {n}
    uint omni:sensor:Core:maxReturns = 1
    float omni:sensor:Core:nearRangeM = {spec["near_m"]:g}
    float omni:sensor:Core:rangeOffsetM = {min(0.09, spec["near_m"] * 0.9):g}
    float omni:sensor:Core:farRangeM = {spec["far_m"]:g}
    float omni:sensor:Core:rangeAccuracyM = {spec["range_accuracy_m"] or 0.03:g}
    float omni:sensor:Core:rangeResolutionM = 0.004
    float omni:sensor:Core:waveLengthNm = {spec["wavelength_nm"] or 905:g}
    token omni:sensor:Core:rayType = "IDEALIZED"
    token omni:sensor:Core:intensityMappingType = "LINEAR"
    token omni:sensor:Core:intensityProcessing = "NORMALIZATION"
    float omni:sensor:Core:minReflectance = {min_refl:g}
    float omni:sensor:Core:minReflectionRangeM = {spec["range_at_10pct_m"] or spec["far_m"]:g}
    bool omni:sensor:Core:skipDroppingInvalidPoints = 0
    float omni:sensor:Core:azimuthErrorStd = 0.005
    float omni:sensor:Core:elevationErrorStd = 0
    uint[] omni:sensor:Core:emitterState:s001:channelId = {arr(range(1, n + 1), "{}")}
    float[] omni:sensor:Core:emitterState:s001:elevationDeg = {arr(spec["elevation_deg"])}
    float[] omni:sensor:Core:emitterState:s001:azimuthDeg = {arr(spec["azimuth_offset_deg"])}
    uint[] omni:sensor:Core:emitterState:s001:fireTimeNs = {arr(spec["fire_time_ns"], "{}")}

    string omni:sensor:modelName = "{spec["model"]}"
    string omni:sensor:modelVendor = "{spec["vendor"]}"
}}
'''


def write(name, spec, report, warnings):
    key = slug(spec)
    folder = ROOT / "sensors" / key
    folder.mkdir(parents=True, exist_ok=True)
    usd = folder / f"{key}.usda"
    usd.write_text(usda(spec, key), encoding="utf-8")
    (folder / "spec.json").write_text(json.dumps(spec, indent=2))
    (folder / "sources.md").write_text(
        f"# {spec['vendor']} {spec['model']}\n\nResearched by the pipeline's sensor agent ({MODEL}) on "
        f"{time.strftime('%Y-%m-%d')} for the query \"{name}\".\n\n**Estimated values:** "
        f"{', '.join(spec['estimated']) or 'none'}.  \n**Writer adjustments:** {'; '.join(warnings) or 'none'}.  \n"
        f"**Notes:** {spec['notes']}\n\n---\n\n{report}\n", encoding="utf-8")
    pattern = sensors.beam_pattern_from_usd(usd)  # the same reader the preview uses
    if len(pattern.elevation_deg) != spec["channels"]:
        sys.exit(f"[agent] written USD reads back {len(pattern.elevation_deg)} channels, expected {spec['channels']}")
    el = spec["elevation_deg"]
    sensors.add_to_registry({
        "name": f"{spec['vendor']} {spec['model']}", "aliases": sorted({name, spec["model"], *spec["aliases"]}),
        "usd_path": usd.relative_to(ROOT).as_posix(),
        "source": f"sensor agent: see {folder.relative_to(ROOT).as_posix()}/sources.md"
                  + (f" (estimated: {', '.join(spec['estimated'])})" if spec["estimated"] else ""),
        "channels": spec["channels"], "vfov_deg": [round(min(el), 2), round(max(el), 2)],
        "rate_hz": spec["rate_hz"], "range_m": [spec["near_m"], spec["far_m"]], "agent": True})
    log(f"wrote {usd.relative_to(ROOT)}: {spec['channels']} channels, {min(el):.1f} to {max(el):.1f} deg, "
        f"{pattern.steps} firings per revolution at {spec['rate_hz']:g} Hz, {spec['near_m']:g}-{spec['far_m']:g} m")
    return usd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--spec", help="write from this spec JSON instead of researching")
    ap.add_argument("--report", help="skip the web research and extract from this saved report")
    args = ap.parse_args()
    if args.spec:
        spec, report = json.loads(open(args.spec, encoding="utf-8").read()), "(spec provided by hand)"
    else:
        client = anthropic.Anthropic()
        if args.report:
            report = open(args.report, encoding="utf-8").read()
        else:
            log(f"researching '{args.name}' on the web (several minutes)")
            report = research(client, args.name)
        log("extracting the beam table")
        spec = extract(client, report)
    spec, warnings = check(spec)
    for w in warnings:
        log(f"note: {w}")
    write(args.name, spec, report, warnings)


if __name__ == "__main__":
    main()
