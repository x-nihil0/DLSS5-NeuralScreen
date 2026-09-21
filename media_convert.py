"""Run a file through the neural pass: an image or a video, off the desktop.

The overlay exists to process what is on screen, which means the one thing
the network could always do - take a picture and give it back better - was
the one thing this program could not do to a FILE. The machinery was already
here: compatibility_runtime drives a worker with synthetic frames and no
capture and no window at all, which is a converter with the file part
missing. This module is that missing part.

What it does NOT do, on purpose:

* It does not touch the live pipeline. The caller decides whether the
  overlay's worker is stood down first (see commands.convert_media); this
  module starts its own worker, converts, and reaps it. Two NGX features
  initialising on one card at the same time is the failure restart_worker's
  2 s sleep exists for, and a conversion is not worth risking the overlay.
* It does not interpret settings. It is handed the same `params` dict the
  overlay builds, so a conversion is the picture the sliders were showing,
  not a second tuning surface that could drift from them.
* It knows nothing about the menu. Progress is a callback and cancellation
  is an Event, so the engine can be driven by a test, by the UI thread, or
  from a shell, and none of those is the "real" caller.

The motion field is the part worth understanding. The network is temporal:
it is handed motion vectors and a reset flag, and it accumulates across
frames. A still image is therefore ONE frame with reset=True and zero
motion - there is no history to correlate against and pretending otherwise
smears it. A video is the opposite: the frames are a real sequence, so the
guides run exactly as they do live, and the result is temporally stable for
the same reason the desktop is.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

from guides import TemporalGuideGenerator
from pipeline import shutdown_worker, start_worker
from protocol import send_frame, send_resize

#: Stills the converter will open. Kept to what the bundled Pillow can both
#: read and write without extra plugins - a format that opens and then fails
#: to save is a worse experience than one that was never offered.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

#: Containers PyAV can demux with the codecs in the bundled runtime
#: (h264/hevc/av1/vp9 are all present - verified against av.codecs_available).
VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv")

#: Pillow's own name for each suffix we offer. Needed because the bytes are
#: written to a ".partial" file first and Pillow derives the format from the
#: extension, which that name does not have.
_PIL_FORMATS = {
    ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "BMP",
    ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP",
}

#: The container name for each suffix, for the same reason as _PIL_FORMATS:
#: the bytes go to a ".partial" file first and PyAV, like Pillow, reads the
#: format off the extension - "Could not determine output format" otherwise.
_AV_FORMATS = {
    ".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".mkv": "matroska",
    ".webm": "webm", ".avi": "avi", ".wmv": "asf",
}

#: NVENC encoders, best first - the same chain and the same reason as
#: recorder.VideoRecorder: the RTX 30 series has no AV1 encoder at all, and
#: add_stream() succeeds there anyway, so the failure only surfaces when the
#: encoder is opened.
CODEC_CHAIN = ("av1_nvenc", "hevc_nvenc", "h264_nvenc")

#: Quality-targeted VBR, as the recorder uses. A conversion is not realtime,
#: so it can afford p7 where the recorder settles for p6.
ENCODER_OPTIONS = {
    "preset": "p7",
    "tune": "hq",
    "rc": "vbr",
    "cq": "16",
    "maxrate": "250M",
    "bufsize": "500M",
}
BIT_RATE = 120_000_000

#: The worker is handed one frame at a time and answers before the next is
#: sent, so this is a per-frame ceiling and not a whole-file one. A 4K frame
#: through four NR passes is the slow case that sets it.
FRAME_TIMEOUT_S = 60.0

#: Discarded evaluations before the first real frame. The live pipeline pays
#: 120 on a cold card because the frame watchdog would otherwise kill the
#: worker on frame 0; a conversion has no watchdog and no picture to keep
#: alive, so it pays the smallest warm-up that still lets NGX settle.
WARMUP_FRAMES = 8


class ConversionCancelled(RuntimeError):
    """Raised inside the engine when the caller's cancel event is set."""


class ConversionError(RuntimeError):
    """A conversion that could not be completed, with the stage that failed."""

    def __init__(self, stage: str, cause: BaseException | str):
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage}: {cause}")


@dataclass
class Progress:
    """What the caller is told while a conversion runs.

    `total` is 0 when it is not knowable - a container that does not declare
    a frame count, which is common enough that a progress bar has to cope
    rather than lie about it.
    """
    stage: str
    done: int = 0
    total: int = 0
    detail: str = ""

    @property
    def fraction(self) -> float:
        return (self.done / self.total) if self.total > 0 else 0.0


@dataclass
class ConversionResult:
    """Where the output went and what it cost."""
    source: Path
    output: Path
    kind: str                     # "image" | "video"
    frames: int = 0
    seconds: float = 0.0
    codec: str = ""
    width: int = 0
    height: int = 0
    work_width: int = 0
    work_height: int = 0
    skipped: int = 0
    notes: list = field(default_factory=list)


def classify(path: Path) -> str:
    """"image", "video", or "" for something this module will not open."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return ""


def default_output_path(source: Path, out_dir: Path | None = None) -> Path:
    """`<name>-nr<suffix>`, beside the source unless a folder was chosen.

    Never the source itself: a converter that can overwrite its own input
    destroys the original on a second run, and the second run is exactly
    what someone does after changing a slider.
    """
    source = Path(source)
    folder = Path(out_dir) if out_dir else source.parent
    stem, suffix = source.stem, source.suffix
    candidate = folder / f"{stem}-nr{suffix}"
    n = 2
    while candidate.exists():
        candidate = folder / f"{stem}-nr-{n}{suffix}"
        n += 1
    return candidate


def processing_size(width: int, height: int, work_scale: float,
                    nr_small: bool, nr_passes: int = 1) -> tuple[int, int]:
    """The resolution the network runs at for a frame of this size.

    The same two rules the live pipeline obeys, for the same reasons: the
    NGX cap is real (feature 18 goes silent above 2560x1440), and a work
    size must never exceed the frame it came from. With Boost off the
    network is handed the whole frame and the scale is inert - which is the
    measurement in TECHNICAL.md, not a decision made here.
    """
    from settings_io import _work_size
    if not nr_small:
        return _work_size(width, height, 1.0, nr_passes)
    return _work_size(width, height, float(work_scale), nr_passes)


class _Engine:
    """One worker, held open for the frames of one file."""

    def __init__(self, params: dict, width: int, height: int,
                 work_w: int, work_h: int, nr_passes: int = 1):
        self.params = dict(params)
        self.nr_passes = max(1, min(4, int(nr_passes or 1)))
        self.width, self.height = int(width), int(height)
        self.work_w, self.work_h = int(work_w), int(work_h)
        # Upscale mode exactly as the live pipeline decides it: the worker is
        # told the full size only when it differs from the work size, because
        # at work == full the legacy 1:1 path is the one that is known to work.
        self.upscale = (self.work_w != self.width or self.work_h != self.height)
        self.worker = None
        self.reader = None
        self.logs: list = []
        self.stop = None

    def __enter__(self) -> "_Engine":
        full_w = self.width if self.upscale else 0
        full_h = self.height if self.upscale else 0
        # The residual strength is NOT derived here. It was, briefly, as
        # 1/passes - which quietly made a two-pass conversion apply half the
        # effect of a one-pass one, the same surprise the overlay had. It is
        # a setting now (config residual_strength, or the environment), and a
        # conversion inherits whatever the process was started with so that a
        # converted file matches what the panel is showing.
        self.worker, self.logs, self.reader, self.stop = start_worker(
            self.params, self.work_w, self.work_h, WARMUP_FRAMES,
            full_w, full_h, None)
        # The cascade depth reaches the worker ONLY on a resize: the stream
        # header has no field for it (see tests/test_nr_passes_wire). A
        # converter that never sent one therefore ran a single pass whatever
        # the panel said - which is what this call fixes.
        if self.nr_passes > 1:
            send_resize(self.worker, self.params, self.work_w, self.work_h,
                        WARMUP_FRAMES, full_w, full_h,
                        True, False, self.nr_passes)
            self.reader.wait_rack(timeout=60.0)
            self.reader.set_output_size(full_w or self.work_w,
                                        full_h or self.work_h)
        return self

    def __exit__(self, *exc) -> None:
        if self.worker is not None:
            try:
                shutdown_worker(self.worker, self.stop)
            except Exception:
                pass
            self.worker = None

    def evaluate(self, index: int, rgba: np.ndarray,
                 motion: np.ndarray, reset: bool) -> np.ndarray | None:
        """One frame in, the processed frame out (or None if it was skipped)."""
        if self.worker is None or self.worker.poll() is not None:
            tail = "\n".join(self.logs[-12:]) or "(no worker output)"
            raise ConversionError("worker", f"the worker exited:\n{tail}")
        send_frame(self.worker, index, rgba, motion, reset, index,
                   shm=None, want_pixels=True, skip_static=False)
        return self.reader.recv(index, timeout=FRAME_TIMEOUT_S)


def _zero_motion(work_w: int, work_h: int) -> np.ndarray:
    """The motion field the worker reads exactly work_w*work_h*4 bytes of."""
    return np.zeros((work_h, work_w, 2), dtype=np.float16)


def _as_rgba(array: np.ndarray) -> np.ndarray:
    """A contiguous HxWx4 uint8 view, whatever the decoder handed back."""
    if array.ndim == 2:
        array = np.dstack([array] * 3)
    if array.shape[2] == 3:
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
        array = np.concatenate([array, alpha], axis=2)
    return np.ascontiguousarray(array, dtype=np.uint8)


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ConversionCancelled("cancelled")


def convert_image(source: Path, output: Path, params: dict, *,
                  work_scale: float = 0.65, nr_small: bool = True,
                  nr_passes: int = 1,
                  progress: Callable[[Progress], None] | None = None,
                  cancel: threading.Event | None = None) -> ConversionResult:
    """One still through the network.

    One frame, reset=True, zero motion. The network is temporal and a single
    image has no history: handing it anything but a reset asks it to
    correlate against whatever the feature was last shown, which on a fresh
    worker is nothing at all.
    """
    from PIL import Image

    source, output = Path(source), Path(output)
    started = time.perf_counter()

    def say(stage: str, done: int = 0, total: int = 1, detail: str = "") -> None:
        if progress is not None:
            progress(Progress(stage, done, total, detail))

    say("decoding", 0, 1, source.name)
    try:
        with Image.open(source) as handle:
            handle.load()
            frame = _as_rgba(np.asarray(handle.convert("RGBA")))
    except Exception as exc:
        raise ConversionError("decode", exc) from exc

    height, width = frame.shape[0], frame.shape[1]
    if width < 64 or height < 64:
        raise ConversionError(
            "decode", f"{width}x{height} is below the 64x64 the worker accepts")
    work_w, work_h = processing_size(width, height, work_scale, nr_small,
                                     nr_passes)
    _check(cancel)

    say("processing", 0, 1, f"{width}x{height} -> network {work_w}x{work_h}")
    with _Engine(params, width, height, work_w, work_h, nr_passes) as engine:
        pixels = engine.evaluate(0, frame, _zero_motion(work_w, work_h), True)
    if pixels is None:
        raise ConversionError("process", "the worker returned no pixels")
    _check(cancel)

    say("writing", 1, 1, output.name)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        out = np.ascontiguousarray(pixels)[:, :, :3]
        image = Image.fromarray(out, mode="RGB")
        # The partial name is the recorder's rule, for the recorder's reason:
        # a file that exists is a file someone will open, and a conversion
        # that died halfway must not leave one that looks finished.
        partial = output.with_name(output.name + ".partial")
        # The format is named, never inferred: Pillow picks it from the
        # EXTENSION, and the partial name ends in ".partial", so letting it
        # guess raises "unknown file extension" after the frame has already
        # been through the network - the whole conversion lost at the last
        # step (found by the first real run).
        fmt = _PIL_FORMATS.get(output.suffix.lower(), "PNG")
        if fmt == "JPEG":
            image.save(partial, format=fmt, quality=97, subsampling=0)
        else:
            image.save(partial, format=fmt)
        os.replace(partial, output)
    except Exception as exc:
        raise ConversionError("encode", exc) from exc

    return ConversionResult(
        source=source, output=output, kind="image", frames=1,
        seconds=time.perf_counter() - started,
        width=width, height=height, work_width=work_w, work_height=work_h,
        codec=output.suffix.lstrip(".").lower())


def convert_video(source: Path, output: Path, params: dict, *,
                  work_scale: float = 0.65, nr_small: bool = True,
                  nr_passes: int = 1, flow_preset: str = "fast",
                  copy_audio: bool = True,
                  progress: Callable[[Progress], None] | None = None,
                  cancel: threading.Event | None = None) -> ConversionResult:
    """A video through the network, frame by frame, with real motion guides.

    The guides are the live ones, at the live work size, for the reason the
    pipeline has them at all: the network accumulates across frames, and a
    sequence handed zero motion flickers where a sequence handed real
    vectors is stable. A scene cut is reported as a reset by the same
    scene-score rule the desktop uses.

    Audio is remuxed, not re-encoded: the samples are the user's and nothing
    here improves them, so they are copied packet for packet when the output
    container will take them.
    """
    import av

    source, output = Path(source), Path(output)
    started = time.perf_counter()
    notes: list = []

    def say(stage: str, done: int, total: int, detail: str = "") -> None:
        if progress is not None:
            progress(Progress(stage, done, total, detail))

    try:
        container = av.open(str(source))
    except Exception as exc:
        raise ConversionError("decode", exc) from exc

    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ConversionError("decode", "the file carries no video stream")
        stream.thread_type = "AUTO"
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        if width < 64 or height < 64:
            raise ConversionError(
                "decode", f"{width}x{height} is below the 64x64 the worker accepts")
        # An odd dimension cannot be encoded as yuv420p and the worker's own
        # sizing assumes even frames; rounding DOWN keeps us inside the source.
        even_w, even_h = width - (width % 2), height - (height % 2)
        total = int(getattr(stream, "frames", 0) or 0)
        rate = stream.average_rate or stream.guessed_rate or Fraction(30, 1)
        work_w, work_h = processing_size(even_w, even_h, work_scale,
                                         nr_small, nr_passes)

        guides = TemporalGuideGenerator(work_w, work_h, preset=flow_preset)
        out_container = None
        out_stream = None
        audio_out = None
        audio_in = None
        codec_used = ""
        partial = output.with_name(output.name + ".partial")
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            out_container = av.open(
                str(partial), mode="w",
                format=_AV_FORMATS.get(output.suffix.lower(), "mp4"))
            last_error: BaseException | None = None
            for codec in CODEC_CHAIN:
                try:
                    candidate = out_container.add_stream(codec, rate=rate)
                    candidate.width, candidate.height = even_w, even_h
                    candidate.pix_fmt = "yuv420p"
                    candidate.bit_rate = BIT_RATE
                    candidate.options = dict(ENCODER_OPTIONS)
                    # Opening is what actually proves the encoder exists on
                    # this card; add_stream succeeds on hardware without it.
                    candidate.open()
                    out_stream, codec_used = candidate, codec
                    break
                except Exception as exc:
                    last_error = exc
                    continue
            if out_stream is None:
                raise ConversionError("encode", last_error or "no NVENC encoder opened")

            if copy_audio:
                audio_in = next((s for s in container.streams if s.type == "audio"), None)
                if audio_in is not None:
                    try:
                        audio_out = out_container.add_stream(template=audio_in)
                    except Exception as exc:
                        audio_out = None
                        notes.append(f"audio not copied ({exc})")

            done = 0
            skipped = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    _check(cancel)
                    rgba = _as_rgba(frame.to_ndarray(format="rgba"))
                    if rgba.shape[1] != even_w or rgba.shape[0] != even_h:
                        rgba = np.ascontiguousarray(rgba[:even_h, :even_w])
                    if done == 0:
                        engine = _Engine(params, even_w, even_h,
                                         work_w, work_h, nr_passes)
                        engine.__enter__()
                    guide = guides.process(rgba)
                    pixels = engine.evaluate(done, rgba, guide.motion, guide.reset)
                    if pixels is None:
                        skipped += 1
                        pixels = rgba
                    out = np.ascontiguousarray(pixels)[:even_h, :even_w]
                    video_frame = av.VideoFrame.from_ndarray(out, format="rgba")
                    # The colour tags belong on the FRAME as well as the
                    # stream: swscale takes its matrix from the frame while
                    # the player reads the stream, and that mismatch is what
                    # the recorder's own comment calls "the contrast".
                    video_frame.color_range = 2          # full (sRGB)
                    video_frame.colorspace = 1           # BT.709
                    video_frame.color_primaries = 1
                    video_frame.color_trc = 13           # sRGB
                    video_frame.pts = done
                    video_frame.time_base = Fraction(1, 1) / rate
                    for out_packet in out_stream.encode(video_frame):
                        out_container.mux(out_packet)
                    done += 1
                    say("processing", done, total, f"{even_w}x{even_h}")

            if done == 0:
                raise ConversionError("decode", "no frames could be decoded")

            for out_packet in out_stream.encode(None):
                out_container.mux(out_packet)

            if audio_out is not None and audio_in is not None:
                say("audio", done, total, "copying the original track")
                try:
                    container.seek(0)
                    for packet in container.demux(audio_in):
                        if packet.dts is None:
                            continue
                        packet.stream = audio_out
                        out_container.mux(packet)
                except Exception as exc:
                    notes.append(f"audio not copied ({exc})")
        finally:
            try:
                if 'engine' in dir() and done:
                    engine.__exit__(None, None, None)
            except Exception:
                pass
            if out_container is not None:
                try:
                    out_container.close()
                except Exception:
                    pass

        os.replace(partial, output)
    except ConversionCancelled:
        try:
            partial.unlink()
        except Exception:
            pass
        raise
    finally:
        try:
            container.close()
        except Exception:
            pass

    return ConversionResult(
        source=source, output=output, kind="video", frames=done,
        seconds=time.perf_counter() - started, codec=codec_used,
        width=even_w, height=even_h, work_width=work_w, work_height=work_h,
        skipped=skipped, notes=notes)


def convert(source: Path, output: Path | None, params: dict, **kwargs) -> ConversionResult:
    """Convert by kind, so a caller does not have to know which this is."""
    source = Path(source)
    kind = classify(source)
    if not kind:
        raise ConversionError(
            "decode", f"{source.suffix or 'that file'} is not a format this converts")
    if output is None:
        output = default_output_path(source, kwargs.pop("out_dir", None))
    kwargs.pop("out_dir", None)
    if kind == "image":
        kwargs.pop("flow_preset", None)
        kwargs.pop("copy_audio", None)
        return convert_image(source, Path(output), params, **kwargs)
    return convert_video(source, Path(output), params, **kwargs)
