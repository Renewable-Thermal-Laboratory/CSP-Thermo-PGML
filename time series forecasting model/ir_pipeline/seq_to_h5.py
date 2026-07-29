#!/usr/bin/env python3
"""Convert FLIR .seq recordings to compact HDF5 for the IR profile-reconstruction project.

For each input .seq this writes <basename>.h5 containing:
    temperature      (T, H, W) uint16, gzip'd. Celsius = raw * temp_scale + temp_offset
    time             (T,) float64, seconds since first frame in the file
    time_abs         (T,) float64, camera-clock epoch seconds (clock may be unset!)
    frames_averaged  (T,) uint16, source frames averaged into each output frame

Frames are averaged in raw-count space within each output time bucket (the scene is
quasi-static at conduction timescales), then converted to temperature once per bucket
with flirpy's raw2temp — identical math to FLIR's software, ~30x cheaper.

The .seq container is parsed directly (FFF frame headers at documented ExifTool
offsets) from a memory-map: no whole-file reads, so 20GB recordings stream fine,
and malformed/truncated frames are skipped with a count instead of aborting.

Requires: pip install flirpy h5py numpy   (matplotlib only for --preview)

Examples:
    python seq_to_h5.py Rec-000207.seq -o ./h5
    python seq_to_h5.py "C:/Users/me/Desktop/IR data" -o C:/Users/me/Desktop/IR_h5
"""

import argparse
import mmap
import struct
import sys
import time as _time
from pathlib import Path

import h5py
import numpy as np
from flirpy.util.raw import raw2temp

MAGIC = b"FFF\x00"

# uint16 encoding: covers -100 .. 1210.7 C at 0.02 C resolution
TEMP_SCALE = 0.02
TEMP_OFFSET = -100.0

PROGRESS_EVERY_S = 300  # print progress every N seconds of footage


# ---------------------------------------------------------------------------
# FFF container parsing (offsets per ExifTool FLIR.pm; same source flirpy uses)
# ---------------------------------------------------------------------------

def _detect_bigendian(buf, pos):
    return int.from_bytes(buf[pos + 0x14:pos + 0x18], "little") > 200


def parse_frame(buf, pos):
    """Validate + parse the FFF frame starting at pos.

    Returns dict with frame length and key record offsets, or None if pos is not
    a plausible frame start (magic bytes can also occur inside pixel data).
    """
    if pos + 0x40 > len(buf):
        return None
    try:
        big = _detect_bigendian(buf, pos)
        e = ">" if big else "<"
        magic, = struct.unpack_from("4s", buf, pos)
        version, dir_offset, rec_count = struct.unpack_from(e + "III", buf, pos + 0x14)
    except struct.error:
        return None
    if magic != MAGIC or not (100 <= version < 200) or not (0 < rec_count <= 256):
        return None
    dir_end = dir_offset + rec_count * 32
    if pos + dir_end > len(buf):
        return None

    frame_len = dir_end
    raw_rec = None      # record type 1: the image
    caminfo_rec = None  # record type 32: settings, calibration, timestamp
    try:
        for i in range(rec_count):
            r_off = pos + dir_offset + i * 32
            rtype, _sub, _ver, _idx, offset, length = struct.unpack_from(e + "HHIIII", buf, r_off)
            frame_len = max(frame_len, offset + length)
            if rtype == 1:
                raw_rec = (offset, length)
            elif rtype == 32:
                caminfo_rec = (offset, length)
    except struct.error:
        return None
    if raw_rec is None or frame_len > (1 << 31):
        return None
    return {"len": frame_len, "raw": raw_rec, "caminfo": caminfo_rec, "endian": e}


def get_raw_image(buf, pos, frame):
    """Raw uint16 counts as a zero-copy view into the mmap."""
    offset, length = frame["raw"]
    e = frame["endian"]
    w, h = struct.unpack_from(e + "HH", buf, pos + offset + 0x02)
    n = w * h
    if 0x20 + 2 * n > length or w == 0 or h == 0:
        raise ValueError(f"raw record too small for {w}x{h} image")
    dt = ">u2" if e == ">" else "<u2"
    return np.frombuffer(buf, dtype=dt, count=n, offset=pos + offset + 0x20).reshape(h, w)


def get_timestamp(buf, pos, frame):
    """Camera-clock time in seconds (millisecond resolution), or None."""
    if frame["caminfo"] is None:
        return None
    offset, length = frame["caminfo"]
    if length < 0x38C:
        return None
    sec, ms, _tz = struct.unpack_from(frame["endian"] + "IHh", buf, pos + offset + 0x384)
    return sec + ms / 1000.0


# (name, offset, type): f=float32, fK=float32 Kelvin->C, H=uint16, i=int32, s<n>=string
_CAMINFO_FIELDS = [
    ("Width", 0x02, "H"), ("Height", 0x04, "H"),
    ("Emissivity", 0x20, "f"), ("Object Distance", 0x24, "f"),
    ("Reflected Apparent Temperature", 0x28, "fK"),
    ("Atmospheric Temperature", 0x2C, "fK"),
    ("IR Window Temperature", 0x30, "fK"), ("IR Window Transmission", 0x34, "f"),
    ("Relative Humidity", 0x3C, "f"),
    ("Planck R1", 0x58, "f"), ("Planck B", 0x5C, "f"), ("Planck F", 0x60, "f"),
    ("Atmospheric Trans Alpha 1", 0x70, "f"), ("Atmospheric Trans Alpha 2", 0x74, "f"),
    ("Atmospheric Trans Beta 1", 0x78, "f"), ("Atmospheric Trans Beta 2", 0x7C, "f"),
    ("Atmospheric Trans X", 0x80, "f"),
    ("CameraModel", 0xD4, "s32"), ("LensModel", 0x170, "s32"),
    ("FieldOfView", 0x1B4, "f"),
    ("Planck O", 0x308, "i"), ("Planck R2", 0x30C, "f"),
]


def get_meta(buf, pos, frame):
    """Camera settings + calibration constants (everything raw2temp needs)."""
    if frame["caminfo"] is None:
        raise ValueError("frame has no camera-info record")
    offset, _length = frame["caminfo"]
    e = frame["endian"]
    base = pos + offset
    meta = {}
    for name, off, kind in _CAMINFO_FIELDS:
        if kind == "f":
            v = struct.unpack_from(e + "f", buf, base + off)[0]
        elif kind == "fK":
            v = struct.unpack_from(e + "f", buf, base + off)[0] - 273.14
        elif kind == "H":
            v = struct.unpack_from(e + "H", buf, base + off)[0]
        elif kind == "i":
            v = struct.unpack_from(e + "i", buf, base + off)[0]
        else:  # s<n>
            n = int(kind[1:])
            raw = struct.unpack_from(f"{n}s", buf, base + off)[0]
            v = raw.split(b"\x00")[0].decode(errors="replace")
        meta[name] = v
    if meta["Relative Humidity"] / 100 > 2:  # same normalisation flirpy applies
        meta["Relative Humidity"] /= 100.0
    return meta


def iter_frames(buf):
    """Yield (pos, frame_dict) for each complete, valid FFF frame in the buffer."""
    stats = {"truncated": 0, "bad": 0}
    pos = buf.find(MAGIC, 0)
    while pos != -1:
        frame = parse_frame(buf, pos)
        if frame is None:
            stats["bad"] += 1
            pos = buf.find(MAGIC, pos + 4)
            continue
        if pos + frame["len"] > len(buf):
            stats["truncated"] += 1  # incomplete trailing frame; recording cut mid-write
            break
        yield pos, frame
        pos = buf.find(MAGIC, pos + frame["len"])
    if stats["truncated"] or stats["bad"] > 10:
        print(f"  (skipped: {stats['truncated']} truncated frame(s), "
              f"{stats['bad']} spurious marker(s))", flush=True)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(seq_path, out_dir, rate=1.0, crop=None, preview=False):
    seq_path = Path(seq_path)
    out_path = Path(out_dir) / (seq_path.stem + ".h5")
    bucket_dt = 1.0 / rate
    t0 = _time.time()

    with open(seq_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        total = len(buf)
        meta = None
        h5 = None
        temp_ds = None
        times_rel, times_abs, counts = [], [], []
        accum = None
        n_in_bucket = 0
        bucket_idx = None
        bucket_t_sum = 0.0
        first_ts = None
        n_frames = 0
        next_progress = PROGRESS_EVERY_S
        fallback_dt = 1.0 / 30.0  # used only if frames lack a timestamp record

        def flush_bucket():
            nonlocal accum, n_in_bucket, bucket_t_sum, temp_ds
            if n_in_bucket == 0:
                return
            mean_raw = accum / n_in_bucket
            temp_c = raw2temp(mean_raw, meta)
            if crop is not None:
                x0, y0, x1, y1 = crop
                temp_c = temp_c[y0:y1, x0:x1]
            frame_u16 = np.clip(np.rint((temp_c - TEMP_OFFSET) / TEMP_SCALE), 0, 65535).astype(np.uint16)
            temp_ds.resize(temp_ds.shape[0] + 1, axis=0)
            temp_ds[-1] = frame_u16
            times_rel.append(bucket_t_sum / n_in_bucket)
            times_abs.append(first_ts + bucket_t_sum / n_in_bucket)
            counts.append(n_in_bucket)
            accum = None
            n_in_bucket = 0
            bucket_t_sum = 0.0

        for pos, frame in iter_frames(buf):
            ts = get_timestamp(buf, pos, frame)
            if ts is None:
                ts = (first_ts or 0.0) + n_frames * fallback_dt
            if first_ts is None:
                first_ts = ts
                meta = get_meta(buf, pos, frame)
                # .shape only — holding the zero-copy view would block mmap close
                h, w = get_raw_image(buf, pos, frame).shape
                if crop is not None:
                    x0, y0, x1, y1 = crop
                    h, w = y1 - y0, x1 - x0
                out_path.parent.mkdir(parents=True, exist_ok=True)
                h5 = h5py.File(out_path, "w")
                temp_ds = h5.create_dataset(
                    "temperature", shape=(0, h, w), maxshape=(None, h, w),
                    dtype=np.uint16, chunks=(1, h, w),
                    compression="gzip", compression_opts=4, shuffle=True,
                )
            t_rel = ts - first_ts
            idx = int(t_rel // bucket_dt)
            if bucket_idx is None:
                bucket_idx = idx
            if idx != bucket_idx:
                flush_bucket()
                bucket_idx = idx
            raw = get_raw_image(buf, pos, frame).astype(np.float64)
            accum = raw if accum is None else accum + raw
            n_in_bucket += 1
            bucket_t_sum += t_rel
            n_frames += 1
            if t_rel >= next_progress:
                print(f"    ... {t_rel:6.0f} s of footage ({100.0 * pos / total:3.0f}% of file)", flush=True)
                next_progress += PROGRESS_EVERY_S

        if n_frames == 0:
            print(f"  !! no valid FFF frames found in {seq_path.name}, skipping", flush=True)
            return None
        flush_bucket()

        h5.create_dataset("time", data=np.asarray(times_rel, dtype=np.float64))
        h5.create_dataset("time_abs", data=np.asarray(times_abs, dtype=np.float64))
        h5.create_dataset("frames_averaged", data=np.asarray(counts, dtype=np.uint16))
        h5.attrs["source_file"] = seq_path.name
        h5.attrs["temp_scale"] = TEMP_SCALE
        h5.attrs["temp_offset"] = TEMP_OFFSET
        h5.attrs["temp_units"] = "celsius = raw*temp_scale + temp_offset"
        h5.attrs["target_rate_hz"] = rate
        h5.attrs["source_frames"] = n_frames
        h5.attrs["camera_clock_start"] = first_ts
        if crop is not None:
            h5.attrs["crop_x0y0x1y1"] = crop
        for k, v in meta.items():
            h5.attrs[k] = v

        if preview:
            _write_preview(temp_ds, out_path)

        n_out = temp_ds.shape[0]
        h5.close()

    in_mb = seq_path.stat().st_size / 1e6
    out_mb = out_path.stat().st_size / 1e6
    print(f"  {seq_path.name}: {n_frames} frames -> {n_out} @ {rate} Hz | "
          f"{in_mb:.0f} MB -> {out_mb:.1f} MB ({in_mb/max(out_mb,0.01):.0f}x) | "
          f"{_time.time()-t0:.1f}s", flush=True)
    return out_path


def _write_preview(temp_ds, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = temp_ds.shape[0]
    picks = sorted({0, n // 2, n - 1})
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, picks):
        t_c = temp_ds[i].astype(np.float32) * TEMP_SCALE + TEMP_OFFSET
        im = ax.imshow(t_c, cmap="inferno")
        ax.set_title(f"frame {i}")
        plt.colorbar(im, ax=ax, label="degC")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".preview.png"), dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help=".seq files or directories to search recursively")
    p.add_argument("-o", "--out", required=True, help="output directory for .h5 files")
    p.add_argument("--rate", type=float, default=1.0, help="output frame rate in Hz (default 1.0)")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="crop region in pixels (applied after averaging)")
    p.add_argument("--preview", action="store_true", help="write a preview PNG per file")
    p.add_argument("--skip-existing", action="store_true", help="skip files whose .h5 already exists")
    args = p.parse_args()

    files = []
    for inp in args.inputs:
        path = Path(inp)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.seq")))
        elif path.exists():
            files.append(path)
        else:
            print(f"!! not found: {inp}", file=sys.stderr, flush=True)
    if not files:
        sys.exit("no .seq files found")

    print(f"{len(files)} file(s) -> {args.out}", flush=True)
    failed = []
    for i, f in enumerate(files, 1):
        out_file = Path(args.out) / (f.stem + ".h5")
        if args.skip_existing and out_file.exists():
            print(f"[{i}/{len(files)}] skip {f.name} (exists)", flush=True)
            continue
        print(f"[{i}/{len(files)}] {f}", flush=True)
        try:
            convert(f, args.out, rate=args.rate, crop=args.crop, preview=args.preview)
        except Exception as e:
            print(f"  !! FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            failed.append(str(f))
    if failed:
        print(f"\n{len(failed)} file(s) failed:\n  " + "\n  ".join(failed), file=sys.stderr, flush=True)
    else:
        print("\nall done", flush=True)


if __name__ == "__main__":
    main()
