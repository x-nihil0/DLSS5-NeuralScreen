r"""The NR cascade on MOVING content, against exact ground truth.

    runtime\python.exe tests\experiment_cascade_motion.py . [pan] [profile]

An experiment, not a test: it needs the card and it reports numbers rather
than passing or failing. It exists because every other measurement of the
cascade was a still, which is the one case a temporal network cannot show
what it is for - and the conclusion drawn from stills ("four passes buys
almost nothing") turned out to be an artefact of asking a moving thing to
hold still.

WHAT IT MEASURES

A detailed panorama is panned by an EXACT integer number of pixels per
frame, so the true motion is known to the pixel. Motion-compensated
temporal error shifts output frame N back by the displacement it was built
from and compares it with N-1: the same object, one frame apart, should
come back the same, and whatever does not is the network changing its mind
about a pixel it has already seen. On the source frames that error is 0 by
construction, printed first as the control - a metric that cannot show zero
on a perfect input is measuring itself.

WHAT IT FOUND (RTX 5080, 960x540, network 624x350, 20 frames)

Motion-compensated error, lower is steadier:

        pan     1 pass   2      3      4      1 -> 4
        1px/f   0.8390  0.6320 0.5316 0.4598   -45%
        4px/f   0.7437  0.5579 0.4657 0.4004   -46%
        12px/f  0.6556  0.4741 0.3859 0.3279   -50%

Monotonic at every speed across a twelvefold range, and it costs no
sharpness: detail is flat at 1 and 4 px/frame (-0.2%, -0.4%) and RISES at
12 px/frame (1165.9 -> 1205.7), where a single pass loses the most and the
cascade recovers some of it. Edge transitions stay at 1.00 px throughout -
no ghosting at any depth or speed.

So the cascade is a temporal stabiliser. That is what it is for, and stills
cannot show it.

AND WHY THE STRENGTH NORMALISATION IS LOAD-BEARING

With the flat strength the cascade shipped with, the same clip goes the
other way - it is worse than not cascading at all:

        1 pass                       mc 0.6641  sat 107.30  detail 1148.4
        4 passes, strength 0.25      mc 0.3781  sat 103.47  detail 1144.4
        4 passes, strength 1.00      mc 1.5294  sat  97.57  detail 1050.6

2.3x less stable than a single pass, and 4x less than the normalised one.
The normalisation is not a tidy-up that preserves the benefit; it is what
makes the benefit exist.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
os.environ["NS_NR_SMALL"] = "1"
os.environ.pop("NS_NR_RESIDUAL_STRENGTH", None)

import cv2
import numpy as np

from guides import TemporalGuideGenerator
from media_convert import _Engine
from settings_io import PROFILES

W, H = 960, 540
WORK_W, WORK_H = 624, 350        # one geometry for every pass count
PAN = int(sys.argv[2]) if len(sys.argv) > 2 else 4   # exact px/frame, integer
FRAMES = 20
MARGIN = 24                      # ignore the strip that scrolls in
params = dict(PROFILES[sys.argv[3] if len(sys.argv) > 3 else "Natural"])
params["style"] = 1


def wide_source() -> np.ndarray:
    """A panorama wide enough that every frame is a crop of ONE image.

    That is what makes the ground truth exact: frame N is not a re-render,
    it is the same pixels at a different offset, so two consecutive frames
    share their content exactly and any difference after alignment is the
    network's.
    """
    width = W + PAN * FRAMES + 8
    yy, xx = np.mgrid[0:H, 0:width]
    img = np.zeros((H, width, 3), np.uint8)
    img[..., 0] = (np.sin(xx / 21.0) * 55 + 100).astype(np.uint8)
    img[..., 1] = (np.cos(yy / 29.0) * 55 + 115).astype(np.uint8)
    img[..., 2] = ((xx + yy) // 7 % 190 + 35).astype(np.uint8)
    img[60:180, :] = (20, 24, 30)                      # a dark band
    for k in range(0, width, 14):                      # fine repeating structure
        img[300:430, k:k + 2] = 240
    for k in range(0, width, 160):                     # hard vertical edges
        img[200:290, k:k + 40] = 250
    for k in range(0, width, 240):
        cv2.putText(img, "EDGE", (k + 12, 480), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (248, 248, 248), 2)
    return img


PANORAMA = wide_source()


def frame_at(index: int) -> np.ndarray:
    x = index * PAN
    crop = PANORAMA[:, x:x + W]
    rgba = np.dstack([crop, np.full((H, W, 1), 255, np.uint8)])
    return np.ascontiguousarray(rgba)


def mc_error(stack: list) -> float:
    """Mean |frame N aligned onto N-1| over the interior. 0 = perfectly stable."""
    errs = []
    for i in range(1, len(stack)):
        a = stack[i - 1].astype(np.int16)
        b = stack[i].astype(np.int16)
        # frame i is frame i-1 shifted LEFT by PAN, so shifting i right by PAN
        # puts them in the same frame of reference.
        a_cut = a[:, MARGIN + PAN:W - MARGIN]
        b_cut = b[:, MARGIN:W - MARGIN - PAN]
        errs.append(float(np.abs(a_cut - b_cut).mean()))
    return float(np.mean(errs))


def flicker(stack: list) -> float:
    """Per-pixel std through the motion-aligned stack."""
    aligned = []
    for i, f in enumerate(stack):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        shift = i * PAN
        aligned.append(g[:, MARGIN + shift:W - MARGIN - (FRAMES * PAN) + shift])
    cube = np.stack(aligned, axis=0)
    return float(cube.std(axis=0).mean())


def ghosting(frame: np.ndarray) -> float:
    """Transition width of the hard moving edges, in pixels.

    A temporal network that trails smears a step over more columns. Measured
    as the mean run of columns whose gradient is above a tenth of the local
    maximum, across the band that carries the bright blocks.
    """
    g = cv2.cvtColor(frame[200:290], cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.abs(np.diff(g.mean(axis=0)))
    if grad.max() <= 0:
        return 0.0
    above = grad > grad.max() * 0.10
    runs, run = [], 0
    for flag in above:
        if flag:
            run += 1
        elif run:
            runs.append(run); run = 0
    if run:
        runs.append(run)
    return float(np.mean(runs)) if runs else 0.0


def detail(frame: np.ndarray) -> float:
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def run(passes: int) -> list:
    """The clip through the network, with the live optical-flow guides."""
    guides = TemporalGuideGenerator(WORK_W, WORK_H, preset="fast")
    eng = _Engine(params, W, H, WORK_W, WORK_H, passes)
    eng.__enter__()
    try:
        out = []
        for i in range(FRAMES):
            rgba = frame_at(i)
            guide = guides.process(rgba)
            pixels = eng.evaluate(i, rgba, guide.motion, guide.reset)
            if pixels is None:
                raise RuntimeError(f"no pixels for frame {i}")
            out.append(np.ascontiguousarray(pixels)[:, :, :3].copy())
        built = any("cascade built" in l for l in eng.logs)
        return out, built
    finally:
        eng.__exit__(None, None, None)


source = [frame_at(i)[:, :, :3] for i in range(FRAMES)]
print(f"  {W}x{H}, network {WORK_W}x{WORK_H}, panning {PAN}px/frame, "
      f"{FRAMES} frames, profile {params.get('style')}\n")
print(f"{'':>10} {'mc-error':>9} {'flicker':>8} {'ghost px':>9} {'detail':>9}")
print(f"{'SOURCE':>10} {mc_error(source):9.4f} {flicker(source):8.4f} "
      f"{ghosting(source[-1]):9.2f} {detail(source[-1]):9.1f}   <- control")

for n in (1, 2, 3, 4):
    frames, built = run(n)
    tag = "" if (n == 1 or built) else "  (cascade NOT built!)"
    print(f"{'passes=' + str(n):>10} {mc_error(frames):9.4f} {flicker(frames):8.4f} "
          f"{ghosting(frames[-1]):9.2f} {detail(frames[-1]):9.1f}{tag}")
