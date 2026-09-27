"""Minimal PCD reader (ascii and binary, any fields): path -> N x 4 float32 (x, y, z, intensity), NaNs dropped."""
from pathlib import Path

import numpy as np

_TYPES = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4",
          ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}


def read_pcd(path) -> np.ndarray:
    raw = Path(path).read_bytes()
    header, pos = {}, 0
    while True:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode("ascii", "replace").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        header[key.upper()] = vals
        if key.upper() == "DATA":
            break
    fields, sizes, types = header["FIELDS"], map(int, header["SIZE"]), header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    dtype = np.dtype([(f, _TYPES[(t, s)], (c,) if c > 1 else ()) for f, s, t, c in zip(fields, sizes, types, counts)])
    n = int(header["POINTS"][0])
    mode = header["DATA"][0].lower()
    if mode == "binary":
        arr = np.frombuffer(raw, dtype, count=n, offset=pos)
    elif mode == "ascii":
        table = np.loadtxt(raw[pos:].decode().splitlines(), ndmin=2)
        arr = np.zeros(n, dtype)
        for k, f in enumerate(fields):
            arr[f] = table[:, k]
    else:
        raise ValueError(f"{path}: PCD DATA {mode} not supported (save as binary or ascii)")
    inten = arr["intensity"] if "intensity" in fields else np.zeros(n)
    pts = np.column_stack([arr["x"], arr["y"], arr["z"], inten]).astype(np.float32)
    return pts[np.isfinite(pts[:, :3]).all(1) & (np.abs(pts[:, :3]).sum(1) > 0)]
