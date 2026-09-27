"""P1 Capture: a walking video or a set of photos -> sharp, evenly spaced frames for VGGT.

Video: for each 1/fps-second window we keep the sharpest frame (variance of the Laplacian), which copes with
motion blur from walking far better than a fixed blur threshold.
Photos (a folder or a .zip): every photo is a candidate, in filename order.
Either way, frames much blurrier than the median are dropped and the rest are thinned evenly down to
max_frames (VGGT's VRAM limit).
"""
import io
import json
import re
import shutil
import threading
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

KEY = "capture"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FRAME_LONG_SIDE = 1024  # VGGT resizes to ~512 anyway; keep some headroom for texturing later
SHEET_COLS, SHEET_THUMB_W = 8, 160


def sharpness(frame) -> float:
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(video: Path, out_dir: Path, fps: float, max_frames: int, blur_reject: float,
                   progress=lambda frac: None, log=print) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV can't open {video.name}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    window = max(1, round(src_fps / fps))
    log(f"{video.name}: {n_src} frames @ {src_fps:.1f} fps, {n_src / src_fps:.1f}s -> keeping sharpest of every {window}")

    best = []  # (sharpness, frame_index, frame) per window
    cur, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        s = sharpness(frame)
        if cur is None or s > cur[0]:
            cur = (s, i, frame)
        i += 1
        if i % window == 0:
            best.append(cur)
            cur = None
        if n_src and i % 30 == 0:
            progress(min(i / n_src, 1.0))
    if cur is not None:
        best.append(cur)
    cap.release()
    if not best:
        raise RuntimeError("no frames decoded")

    keep, n_blurry = _select(best, max_frames, blur_reject)
    _write_frames([b[2] for b in keep], out_dir)
    progress(1.0)
    return {"source": video.name, "kind": "video", "source_fps": src_fps, "source_frames": i, "sample_fps": fps,
            "candidates": len(best), "dropped_blurry": n_blurry, "frames": len(keep),
            "frame_indices": [b[1] for b in keep]}


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _load_photo(data) -> np.ndarray:
    """Photo (path or bytes) -> upright BGR, downscaled to FRAME_LONG_SIDE. Applies the EXIF rotation, which
    VGGT's own loader ignores."""
    with Image.open(io.BytesIO(data) if isinstance(data, bytes) else data) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((FRAME_LONG_SIDE, FRAME_LONG_SIDE), Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def select_photos(src: Path, out_dir: Path, max_frames: int, blur_reject: float,
                  progress=lambda frac: None, log=print) -> dict:
    """src is a folder of photos or a .zip of them."""
    if src.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(src)
        names = sorted((n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTS
                        and not Path(n).name.startswith(".") and "__MACOSX" not in n), key=_natural_key)
        read = zf.read
    else:
        names = sorted((str(f) for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS), key=_natural_key)
        read = Path
    if not names:
        raise RuntimeError(f"no photos ({', '.join(sorted(IMAGE_EXTS))}) in {src.name}")
    log(f"{src.name}: {len(names)} photos")

    cands = []  # (sharpness, index into names, frame)
    for i, n in enumerate(names):
        try:
            frame = _load_photo(read(n))
        except Exception as e:  # noqa: BLE001 - skip unreadable files, keep going
            log(f"skipping {Path(n).name}: {e}")
            continue
        cands.append((sharpness(frame), i, frame))
        progress((i + 1) / len(names) * 0.9)
    if not cands:
        raise RuntimeError("none of the photos could be read")
    keep, n_blurry = _select(cands, max_frames, blur_reject)
    shapes = {b[2].shape[:2] for b in keep}
    if len(shapes) > 1:
        log(f"warning: mixed photo sizes/orientations {sorted(shapes)}; VGGT will pad them to one size")
    _write_frames([b[2] for b in keep], out_dir)
    progress(1.0)
    return {"source": src.name, "kind": "photos", "candidates": len(cands), "dropped_blurry": n_blurry,
            "frames": len(keep), "photos_used": [Path(names[b[1]]).name for b in keep]}


def _select(cands, max_frames, blur_reject):
    """Drop candidates much blurrier than the median, then thin evenly (keeping order) to max_frames."""
    scores = np.array([c[0] for c in cands])
    keep = [c for c in cands if c[0] >= blur_reject * np.median(scores)]
    n_blurry = len(cands) - len(keep)
    if len(keep) > max_frames:
        keep = [keep[j] for j in np.linspace(0, len(keep) - 1, max_frames).round().astype(int)]
    return keep, n_blurry


def _write_frames(frames, out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for n, frame in enumerate(frames):
        h, w = frame.shape[:2]
        f = FRAME_LONG_SIDE / max(h, w)
        if f < 1:
            frame = cv2.resize(frame, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"frame_{n:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])


def contact_sheet(frames_dir: Path) -> np.ndarray | None:
    files = sorted(frames_dir.glob("frame_*.jpg"))
    if not files:
        return None
    thumbs = []
    for f in files:
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        im = cv2.resize(im, (SHEET_THUMB_W, round(h * SHEET_THUMB_W / w)), interpolation=cv2.INTER_AREA)
        cv2.putText(im, str(len(thumbs)), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(im)
    th = max(t.shape[0] for t in thumbs)
    rows = []
    for r in range(0, len(thumbs), SHEET_COLS):
        row = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in thumbs[r:r + SHEET_COLS]]
        row += [np.zeros_like(row[0])] * (SHEET_COLS - len(row))
        rows.append(np.hstack(row))
    return cv2.cvtColor(np.vstack(rows), cv2.COLOR_BGR2RGB)


# ---- GUI panel -------------------------------------------------------------------------------------------
def build(app, folder):
    gui = app.server.gui
    with folder:
        gui.add_markdown("Film the site while walking slowly, or take overlapping photos. Upload a video or a "
                         ".zip of photos, or point to a video file or photo folder on disk.")
        upload = gui.add_upload_button("Upload video or .zip", icon="upload", mime_type="video/*,.zip")
        path_in = gui.add_text("…or local path", "",
                               hint="Video file, folder of photos, or .zip (better for large inputs)")
        fps = gui.add_slider("Video sample rate (fps)", min=0.5, max=10.0, step=0.5, initial_value=3.0,
                             hint="How many frames per second of video to consider (ignored for photos)")
        max_frames = gui.add_slider("Max frames", min=10, max=200, step=5, initial_value=50,
                                    hint="VGGT VRAM: ~3.3 GB + 0.07 GB/frame (measured on the RTX 4060), "
                                         "so ~50 fits in 8 GB")
        blur = gui.add_slider("Blur rejection", min=0.0, max=1.0, step=0.05, initial_value=0.35,
                              hint="Drop frames less sharp than this fraction of the median frame")
        run_btn = gui.add_button("Extract frames", icon="movie")
        bar = gui.add_progress_bar(0.0, visible=False)
        summary = gui.add_markdown("")
        sheet = gui.add_image(np.zeros((8, 8, 3), np.uint8), label="Frames", visible=False)

    def show_result():
        p = app.project
        info_file = p.path / "capture" / "capture.json" if p else None
        if not info_file or not info_file.exists():
            summary.content, sheet.visible = "", False
            return
        info = json.loads(info_file.read_text())
        src = (f"{info['source_frames'] / info['source_fps']:.0f}s video"
               if info.get("kind", "video") == "video" else "photos")
        summary.content = (f"**{info['frames']} frames** from `{info['source']}` ({src}; "
                           f"{info.get('candidates', info.get('windows'))} candidates, "
                           f"{info['dropped_blurry']} too blurry)")
        img = contact_sheet(p.path / "frames")
        if img is not None:
            sheet.image, sheet.visible = img, True

    def start(src: Path):
        p = app.project
        if p is None:
            app.log("open or create a project first")
            return
        is_video = src.suffix.lower() in VIDEO_EXTS
        if not (is_video or src.is_dir() or src.suffix.lower() == ".zip"):
            app.log(f"expected a video, a photo folder or a .zip, got: {src.name}")
            return
        run_btn.disabled, bar.visible, bar.value = True, True, 0.0
        p.set_status(KEY, "running")
        app.refresh_status()

        def work():
            try:
                prog = lambda f: setattr(bar, "value", 100 * f)  # noqa: E731
                if is_video:
                    info = extract_frames(src, p.path / "frames", fps.value, int(max_frames.value), blur.value,
                                          progress=prog, log=app.log)
                else:
                    info = select_photos(src, p.path / "frames", int(max_frames.value), blur.value,
                                         progress=prog, log=app.log)
                p.dir("capture", "capture.json").write_text(json.dumps(info, indent=2))
                p.set_status(KEY, "done", note=f"{info['frames']} frames from {info['source']}")
                p.invalidate_downstream(KEY)
                app.log(f"capture done: {info['frames']} frames")
            except Exception as e:  # noqa: BLE001 - report any decode/IO failure in the GUI
                p.set_status(KEY, "error", note=str(e))
                app.log(f"capture failed: {e}")
            finally:
                run_btn.disabled, bar.visible = False, False
                app.stage_changed(KEY)
                show_result()

        threading.Thread(target=work, daemon=True).start()

    @upload.on_upload
    def _(_):
        if app.project is None:
            app.log("open or create a project first")
            return
        f = upload.value
        for old in app.project.path.glob("capture/source.*"):
            old.unlink()
        dest = app.project.dir("capture", "source" + Path(f.name).suffix.lower())
        dest.write_bytes(f.content)
        app.log(f"uploaded {f.name} ({len(f.content) / 1e6:.1f} MB)")
        start(dest)

    @run_btn.on_click
    def _(_):
        p = app.project
        if path_in.value.strip():
            src = Path(path_in.value.strip().strip('"'))
            if not src.exists():
                app.log(f"not found: {src}")
                return
            start(src)
        elif p is not None and (existing := next(p.path.glob("capture/source.*"), None)):
            start(existing)  # re-extract the last upload with new settings
        else:
            app.log("upload a video / .zip or enter a path first")

    app.on_project_opened(show_result)
