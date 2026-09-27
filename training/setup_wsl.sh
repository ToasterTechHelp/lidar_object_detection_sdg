#!/usr/bin/env bash
# One-time OpenPCDet setup inside WSL Ubuntu, entirely in user space (no sudo, no system CUDA/gcc needed).
#   wsl -d Ubuntu -- bash /mnt/c/.../training/setup_wsl.sh
# Installs: micromamba -> env "pcdet" (python 3.10, gcc 11, CUDA 12.1 toolkit) -> torch 2.1.2+cu121, spconv-cu120
# -> OpenPCDet (built for the RTX 4060, sm_89) -> KITTI-pretrained PointPillars checkpoint.
set -euo pipefail
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
MM="$HOME/.local/bin/micromamba"
ENV=pcdet
PCDET_DIR="$HOME/OpenPCDet"
CKPT_DIR="$HOME/pcdet_ckpt"

step() { echo; echo "=== $* ==="; }

step "micromamba"
if [ ! -x "$MM" ]; then
  mkdir -p "$HOME/.local/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/mm.tar.bz2
  # no bzip2 binary on this system; python's tarfile handles .tar.bz2
  python3 -c "import tarfile; t = tarfile.open('/tmp/mm.tar.bz2'); m = t.getmember('bin/micromamba'); m.name = 'micromamba'; t.extract(m, '$HOME/.local/bin')"
fi
"$MM" --version

step "conda env: python 3.10, gcc 11, CUDA 12.1 toolkit"
if ! "$MM" env list | grep -q "^ *$ENV "; then
  "$MM" create -y -n "$ENV" -c nvidia/label/cuda-12.1.1 -c conda-forge \
    python=3.10 gxx_linux-64=11 gcc_linux-64=11 make ninja git cuda-toolkit
fi
run() { "$MM" run -n "$ENV" "$@"; }
run nvcc --version | tail -2

step "torch 2.1.2+cu121, spconv, OpenPCDet deps"
run python -m pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
run python -m pip install --no-cache-dir "numpy<2" "setuptools<70" spconv-cu120 numba llvmlite tensorboardX easydict pyyaml \
  scikit-image tqdm SharedArray opencv-python-headless pyquaternion gdown
run python -c "import torch, spconv; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# torch 2.1 cpp_extension imports pkg_resources, which setuptools>=70 dropped (pinned above)
step "OpenPCDet build (sm_89)"
[ -d "$PCDET_DIR" ] || git clone --depth 1 https://github.com/open-mmlab/OpenPCDet.git "$PCDET_DIR"
cd "$PCDET_DIR"
run bash -c "export CUDA_HOME=\$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8; python setup.py develop" 2>&1 | tail -5
run python -c "import torch, pcdet; from pcdet.ops.iou3d_nms import iou3d_nms_cuda; from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda; print('pcdet', pcdet.__version__, 'CUDA ops OK')"

step "KITTI-pretrained PointPillars"
mkdir -p "$CKPT_DIR"
if [ ! -s "$CKPT_DIR/pointpillar_7728.pth" ]; then
  # model-zoo link from OpenPCDet README (Google Drive)
  run gdown 1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm \
    -O "$CKPT_DIR/pointpillar_7728.pth" || echo "PRETRAINED DOWNLOAD FAILED (training will start from scratch)"
fi
ls -la "$CKPT_DIR"
echo; echo "SETUP COMPLETE"
