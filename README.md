# Site → Detector

Turn a walking video (or photos) of a real place into a LiDAR pedestrian detector trained for that place.

Video → VGGT-Omega 3D reconstruction → mesh → scaled, leveled world → walking areas → LiDAR model and
placement → Isaac Sim simulation (pedestrians + RTX LiDAR) → filtered OpenPCDet dataset → PointPillars training →
exported model. Every stage runs from one browser GUI.

## What you need

- Windows 11 with an NVIDIA GPU (8 GB VRAM is enough; developed on an RTX 4060 Laptop)
- Python 3.12, git
- For simulation: Isaac Sim 6.0.1
- For training: WSL 2 with an Ubuntu distro named `Ubuntu`

You don't need everything to start. Part 1 alone gives you the GUI, reconstruction, meshing, walking areas and
LiDAR placement. Add part 2 to simulate and part 3 to train.

## 1. GUI environment (Windows)

From the project folder:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
git clone https://github.com/facebookresearch/vggt-omega.git vggt-omega
.venv\Scripts\python -m pip install -e vggt-omega
```

`vggt-omega` must be cloned into the project folder itself (the code imports `visual_util.py` from there).

The VGGT-Omega checkpoint is gated. Request access at https://huggingface.co/facebook/VGGT-Omega, then log in once:

```
.venv\Scripts\hf auth login
```

Start the GUI and open http://localhost:8080:

```
.venv\Scripts\python -m app.main          # --port 8081 to use another port
```

### Optional API keys

Set these before starting the GUI (it checks them at startup, so restart it after setting one):

- `ANTHROPIC_API_KEY`: the LiDAR sensor agent (LiDAR model stage), for sensors not already in `sensors/registry.json`
  or Isaac Sim's built-ins.
- `GEMINI_API_KEY`: automatic walking-area proposals (Walking area stage). You can always draw them by hand.

```
setx ANTHROPIC_API_KEY "sk-ant-..."
setx GEMINI_API_KEY "..."
```

## 2. Isaac Sim (Simulate stage)

Install Isaac Sim 6.0.1 to exactly `C:\isaacsim`. The path is hard-coded in `app/stages/simulate.py` and
`app/sensors.py`, so change it there if yours differs. The GUI launches `C:\isaacsim\python.bat` itself; you don't
run Isaac by hand.

Pedestrian characters come from NVIDIA's online asset library, so the first simulation needs internet access.

## 3. Training environment (WSL)

One-time setup, entirely in user space (no sudo). It installs micromamba, a `pcdet` env with CUDA 12.1 and
torch 2.1, builds OpenPCDet, and downloads the KITTI-pretrained PointPillars checkpoint:

```
wsl --install -d Ubuntu            # if you don't have it yet
wsl -d Ubuntu -- bash /mnt/c/<path to project>/training/setup_wsl.sh
```

**Set your GPU architecture first.** `training/setup_wsl.sh` builds OpenPCDet for `TORCH_CUDA_ARCH_LIST=8.9`
(RTX 40-series). Edit that line for your card before running it:

| GPU | Value |
|---|---|
| RTX 30-series | `8.6` |
| RTX 40-series | `8.9` |
| RTX 20-series | `7.5` |

RTX 50-series cards are not supported by this setup (CUDA 12.1 / torch 2.1 predate them).

The script takes 20–40 minutes and ends with `SETUP COMPLETE`. If the pretrained checkpoint download fails,
training still works but starts from scratch.

## Using it

1. In the GUI, create a project (one per site). Everything for that site goes in `projects/<name>/`.
2. Work down the stages in order; each unlocks when the one before it is done:

   | Stage | What it does |
   |---|---|
   | Capture | Import a video or photos and pick frames |
   | Reconstruct (VGGT) | 3D point cloud and cameras from the frames |
   | Mesh | Poisson mesh with hole filling |
   | Scale & level | Set real-world scale from one known distance, level the floor |
   | Walking area | Rectangles where simulated people may walk |
   | LiDAR model | Pick a sensor (the Hesai PandarQT64 ships in `sensors/`) |
   | LiDAR placement | Position and tilt, with a coverage preview |
   | Simulate | Isaac Sim runs: pedestrians walk while the LiDAR scans |
   | Filter & dataset | Drop bad scans, build the OpenPCDet dataset |
   | Train | Fine-tune PointPillars in WSL, with live loss / AP charts |
   | Export | Package the model; run it on real scans |

3. Only one heavy job (reconstruction, simulation or training) runs at a time. The log panel shows its output and
   can stop it.

Re-running a stage marks the finished stages after it as outdated; re-run those too.

## Command-line tools

Every stage can also be run as a script; each script's docstring documents its options.

```
.venv\Scripts\python reconstruct.py <images dir> [--out x.glb --conf-thres 30]
.venv\Scripts\python glb_to_mesh.py <x.glb> --out out/<name>
.venv\Scripts\python -m app.sensor_agent "Velodyne VLP-16"
.venv\Scripts\python sim\build_stage.py <project dir>
C:\isaacsim\python.bat sim\run_capture.py --config <ira.yaml> --sensor <project>\sensor.json --out-dir <run> --boxes
.venv\Scripts\python tools\view_detections.py <folder of .pcd> <preds.json>
```

## Troubleshooting

- **"training did not start: is WSL set up?"**: run `training/setup_wsl.sh`, and check that your distro is named
  `Ubuntu` (`wsl -l -v`). A different name can be set in `app/wsl.py`.
- **CUDA errors from OpenPCDet ops**: the build architecture doesn't match your GPU; fix `TORCH_CUDA_ARCH_LIST` and
  re-run the setup script.
- **Out of GPU memory while training**: lower the batch size (4 fits 8 GB).
- **Reconstruction can't download the checkpoint**: you need access to `facebook/VGGT-Omega` and `hf auth login`.
- **An API-key feature stays disabled**: set the key, then restart the GUI.
