"""Converting a file: the rules that do not need a card, and one that does.

The engine drives a worker with no capture and no window - the shape
compatibility_runtime proved - so most of what can go wrong is decided
before any GPU is involved: which files are offered, where the output
lands, and what resolution the network is asked for.

Two of the checks here are regressions from the first real run, and both
were the same mistake: the bytes are written to a `.partial` name first
(the recorder's rule - a file that exists is a file someone will open), and
BOTH Pillow and PyAV pick their format from the extension. `.partial` is
not one, so a conversion died at the last step, after the frame had already
been through the network. Naming the format explicitly is the fix, and
these two assertions are what would have caught it.

The GPU stage is last and reports a canonical SKIP without the worker, so
this test is still worth running on a machine that cannot execute it.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402

import media_convert  # noqa: E402
from media_convert import (IMAGE_SUFFIXES, VIDEO_SUFFIXES, _AV_FORMATS,  # noqa: E402
                           _PIL_FORMATS, _as_rgba, classify,
                           default_output_path, processing_size)
from paths import WORKER_EXE  # noqa: E402
from protocol import WORK_MAX_H, WORK_MAX_W  # noqa: E402


def main() -> int:
    failures = []

    # 1. Only what the engine can actually open is offered.
    for suffix in IMAGE_SUFFIXES:
        if classify(Path(f"a{suffix}")) != "image":
            failures.append(f"{suffix} is offered but does not classify as image")
    for suffix in VIDEO_SUFFIXES:
        if classify(Path(f"a{suffix}")) != "video":
            failures.append(f"{suffix} is offered but does not classify as video")
    for suffix in (".txt", ".exe", ".zip", ""):
        if classify(Path(f"a{suffix}")):
            failures.append(f"{suffix!r} must not be treated as media")

    # 2. THE REGRESSION: every offered suffix has a named writer. Pillow and
    #    PyAV both read the format off the extension, and the bytes go to a
    #    ".partial" file that has none.
    for suffix in IMAGE_SUFFIXES:
        if suffix not in _PIL_FORMATS:
            failures.append(f"{suffix} is offered but _PIL_FORMATS cannot name "
                            f"its writer - it would fail after processing")
    for suffix in VIDEO_SUFFIXES:
        if suffix not in _AV_FORMATS:
            failures.append(f"{suffix} is offered but _AV_FORMATS cannot name "
                            f"its container - it would fail after processing")

    # 3. The output never overwrites the input, and never an existing file.
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        source = folder / "holiday.png"
        source.write_bytes(b"x")
        first = default_output_path(source)
        if first == source:
            failures.append("the output path is the source - a second run "
                            "would destroy the original")
        if first.suffix != source.suffix:
            failures.append(f"the output changed format: {first.name}")
        first.write_bytes(b"x")
        second = default_output_path(source)
        if second == first or second.exists():
            failures.append("a second conversion would overwrite the first")
        elsewhere = folder / "out"
        elsewhere.mkdir()
        if default_output_path(source, elsewhere).parent != elsewhere:
            failures.append("a chosen folder was ignored")

    # 4. The work size obeys the same two rules as the live pipeline.
    for (w, h) in ((640, 360), (1920, 1080), (3840, 2160), (7680, 4320),
                   (1000, 1000), (65, 65)):
        for scale in (0.1, 0.5, 0.65, 1.0):
            ww, wh = processing_size(w, h, scale, True)
            if ww > w or wh > h:
                failures.append(f"work {ww}x{wh} exceeds the frame {w}x{h}")
            if ww > WORK_MAX_W or wh > WORK_MAX_H:
                failures.append(f"work {ww}x{wh} is over the NGX cap at {w}x{h}")
        # With Boost off the scale is inert: the network gets the whole frame
        # (capped), which is the measurement in TECHNICAL.md.
        off_low = processing_size(w, h, 0.1, False)
        off_high = processing_size(w, h, 1.0, False)
        if off_low != off_high:
            failures.append(f"with Boost off the scale still moved the work "
                            f"size at {w}x{h}: {off_low} vs {off_high}")

    # 5. Whatever a decoder hands back becomes a contiguous HxWx4 uint8 frame.
    for array in (np.zeros((8, 8), np.uint8),
                  np.zeros((8, 8, 3), np.uint8),
                  np.zeros((8, 8, 4), np.uint8)):
        out = _as_rgba(array)
        if out.shape != (8, 8, 4) or out.dtype != np.uint8:
            failures.append(f"_as_rgba({array.shape}) -> {out.shape}/{out.dtype}")
        elif not out.flags["C_CONTIGUOUS"]:
            failures.append(f"_as_rgba({array.shape}) is not contiguous")
    if _as_rgba(np.zeros((4, 4, 3), np.uint8))[..., 3].min() != 255:
        failures.append("a frame without alpha must come back fully opaque")

    # 6. The picker cannot offer a file the converter refuses.
    try:
        import dialogs
        ofn, _buf = dialogs._open_dialog_struct(0, None, "Convert")
        # The filter is asked of the function, not of the struct:
        # ofn.lpstrFilter reads back through ctypes truncated at the
        # first NUL, so the struct can only ever show its heading.
        spec = dialogs.media_filter()
        if spec.count(chr(0)) < 8 or not spec.endswith(chr(0)):
            failures.append("the filter is not a NUL-terminated "
                            "Win32 multi-string")
        for suffix in IMAGE_SUFFIXES + VIDEO_SUFFIXES:
            if f"*{suffix}" not in spec:
                failures.append(f"the open dialog does not offer {suffix}")
                break
        if ofn.lStructSize <= 0 or ofn.nMaxFile <= 0:
            failures.append("the open dialog struct is not initialised")
    except Exception as exc:
        failures.append(f"the open dialog struct could not be built: {exc!r}")

    # 7. The GPU stage: a real still through a real worker.
    if not WORKER_EXE.is_file():
        print("SKIP: native/nvngx.dll is not built - the GPU stage cannot run")
        if failures:
            print("=" * 60)
            print(f"FAIL: {len(failures)} - {failures}")
            return 1
        return 0

    try:
        from PIL import Image
        from settings_io import PROFILES
        params = dict(PROFILES["Natural"])
        params["style"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            src = folder / "frame.png"
            yy, xx = np.mgrid[0:256, 0:448]
            img = np.zeros((256, 448, 3), np.uint8)
            img[..., 0] = ((xx * 3) ^ (yy * 5)) & 0xFF
            img[..., 1] = (np.sin(xx / 7.0) * 110 + 128).astype(np.uint8)
            img[..., 2] = (np.cos(yy / 11.0) * 110 + 128).astype(np.uint8)
            Image.fromarray(img, "RGB").save(src)

            result = media_convert.convert(src, None, params,
                                           work_scale=0.65, nr_small=True)
            if not result.output.is_file():
                failures.append("the converted file was not written")
            elif list(folder.glob("*.partial")):
                failures.append("a .partial file was left behind")
            else:
                before = np.asarray(Image.open(src).convert("RGB")).astype(np.int16)
                after = np.asarray(
                    Image.open(result.output).convert("RGB")).astype(np.int16)
                if after.shape != before.shape:
                    failures.append(f"the output changed size: {after.shape} "
                                    f"from {before.shape}")
                elif float(np.abs(after - before).mean()) < 0.5:
                    failures.append("the output is the input - the network "
                                    "did not run")
    except Exception as exc:
        failures.append(f"the GPU stage raised {type(exc).__name__}: {exc}")

    print("=" * 60)
    if failures:
        print(f"FAIL: {len(failures)} - {failures}")
        return 1
    print("OK: the formats offered can be written, the output never "
          "overwrites, and a still really goes through the network")
    return 0


if __name__ == "__main__":
    sys.exit(main())
