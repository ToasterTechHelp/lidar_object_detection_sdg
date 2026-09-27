"""Running the OpenPCDet side (training/*.py) inside WSL, in the micromamba env that training/setup_wsl.sh builds."""
import shlex
from pathlib import Path

DISTRO = "Ubuntu"
ENV = "pcdet"


def to_wsl(path) -> str:
    """C:\\Users\\x -> /mnt/c/Users/x"""
    p = Path(path).resolve()
    drive, rest = p.drive.rstrip(":").lower(), p.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def pcdet_cmd(script, *args) -> list[str]:
    """wsl.exe command running `python <script> <args>` in the pcdet env. Paths among args must already be WSL
    paths (see to_wsl)."""
    inner = " ".join(shlex.quote(str(a)) for a in ("python", "-u", to_wsl(script), *args))
    return ["wsl", "-d", DISTRO, "--", "bash", "-lc",
            f'export MAMBA_ROOT_PREFIX="$HOME/micromamba"; "$HOME/.local/bin/micromamba" run -n {ENV} {inner}']
