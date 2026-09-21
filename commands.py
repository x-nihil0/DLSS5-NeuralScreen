"""What the user asked for, turned into something that happens.

Three ways in, one subject. The overlay menu reports an action and says
nothing about what it means; the tray, the taskbar button and the hotkeys all
push a word into one queue; and the screenshot flow freezes a processed frame
before it opens a system dialog, then finishes with a file on disk. The
decision in every case is taken here, where the config, the params and the
pipeline are all reachable.

The screenshot is the part worth explaining. GetSaveFileNameW is modal: run
from the main loop it freezes the overlay on the last frame, and with a
recording running that pause lands in the MP4 as a still, because the PTS
comes from the clock rather than from the frame count. So the dialog runs in
its own thread and the answer comes back through a queue the loop drains. The
frame is copied before the dialog starts: Desktop Duplication sees ordinary
Win32 dialogs, and a frame requested after Save As closes can still be the
dialog's buffered frame (#89).
"""
from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
import webbrowser
from pathlib import Path

import pygame  # the menu is baked into the screenshot

import channels
import dialogs
import pipeline
import settings_io
from hotkeys import build_bindings, parse_binding
from i18n import STRINGS as UI_STRINGS
from paths import BASE_DIR
from pipeline import restart_worker
from recorder import (RecordingError, RecordingStatus, VideoRecorder)
from settings_io import (CHANNEL_URL, PROFILES, REPO_URL,
                         WORK_SCALE_MIN, WORK_SCALE_STEP,
                         _autostart_enabled, _next_preset_name,
                         _set_autostart, _work_size, hotkey_labels)
from winapi import window_frame_rect, window_under_cursor


#: ``pending_shot`` has requested one worker frame but has no destination yet.
#: Identity comparison ensures this sentinel cannot be confused with a future
#: path-like pending-shot state.
SHOT_FRAME_PENDING = object()


def _configured_directory(value, fallback: Path) -> Path:
    """Create and return a configured media directory or its portable default."""
    directory = (Path(value) if isinstance(value, str) and value.strip()
                 else fallback)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _unique_media_path(directory: Path, prefix: str, suffix: str) -> Path:
    """A timestamped path that cannot overwrite an existing result/staging file."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    millis = (time.time_ns() // 1_000_000) % 1000
    stem = f"{prefix}-{stamp}-{millis:03d}"
    for serial in range(10_000):
        tail = "" if serial == 0 else f"-{serial}"
        candidate = directory / f"{stem}{tail}{suffix}"
        if not candidate.exists() and not Path(f"{candidate}.partial").exists():
            return candidate
    raise RuntimeError("could not allocate a unique media filename")


def save_screenshot(st, path: Path, rgba) -> None:
    """Save the frame as a maximum-quality JPEG.

    An open menu ends up in the screenshot: our layer is excluded from
    capture, so we draw it onto the frame ourselves.
    """
    try:
        surf = pygame.image.frombuffer(
            rgba, (rgba.shape[1], rgba.shape[0]), "RGBX")
        st.display.draw_capture_overlay(surf)
    except Exception as exc:
        print(f"[main] menu was not baked into the screenshot: {exc}", file=sys.stderr)
    try:
        ok = dialogs.save_image(path, rgba)
        if ok:
            print(f"[main] screenshot: {path}")
            st.display.alert(UI_STRINGS[st.lang].get(
                "shot_saved", "Screenshot saved: {path}").format(path=path))
        else:
            print(f"[main] failed to write the screenshot: {path}", file=sys.stderr)
            st.display.alert(UI_STRINGS[st.lang]["shot_fail"])
    except Exception as exc:
        print(f"[main] screenshot failed: {exc}", file=sys.stderr)
        st.display.alert(UI_STRINGS[st.lang]["shot_fail"])


def request_screenshot(st) -> None:
    """Request pixels before showing Save As, so the dialog cannot be captured.

    In presentation/DDA mode Python normally receives no pixels.  The next
    frame is explicitly requested through ``pending_shot``; main freezes that
    processed frame and only then starts the native dialog.  The copy matters:
    the shared-memory output slot is reused on the next worker response.
    """
    if st.shot_dialog_open or st.pending_shot is not None:
        return
    st.pending_shot = SHOT_FRAME_PENDING
    st.shot_requested_at = time.monotonic()
    print("[main] screenshot requested - capturing before Save As")


def freeze_screenshot_frame(st, rgba) -> bool:
    """Copy the requested result frame, then start the Save As dialog.

    Returns true only for the one frame requested by ``request_screenshot``.
    Keeping the sequencing in this module makes it testable without a worker:
    opening the dialog is structurally after the immutable frame copy (#89).
    """
    if st.pending_shot is not SHOT_FRAME_PENDING:
        return False
    st.pending_shot = None
    asked = getattr(st, "shot_requested_at", None)
    if asked is not None:
        # The click-to-frame wait. Small here + a slow dialog means the FRAME
        # was never the problem, and the next mark says where the time went.
        print(f"[main] screenshot: frame in hand "
              f"{(time.monotonic() - asked) * 1000:.0f} ms after the request")
    try:
        st.shot_rgba = rgba.copy()
    except Exception as exc:
        print(f"[main] could not freeze the screenshot frame: {exc}",
              file=sys.stderr)
        st.shot_rgba = None
        st.display.alert("No frame yet")
        return False
    if str(st.cfg.get("screenshot_mode", "ask")) == "auto":
        suffix = ".png" if str(st.cfg.get("screenshot_format", "png")) == "png" \
            else ".jpg"
        try:
            directory = _configured_directory(
                st.cfg.get("screenshot_dir"), BASE_DIR / "screenshots")
            path = _unique_media_path(directory, "neuralscreen", suffix)
            frozen = st.shot_rgba
            st.shot_rgba = None
            save_screenshot(st, path, frozen)
        except Exception as exc:
            st.shot_rgba = None
            print(f"[main] automatic screenshot failed: {exc}", file=sys.stderr)
            st.display.alert(UI_STRINGS[st.lang]["shot_fail"])
        return True
    open_save_dialog(st)
    return True


def open_save_dialog(st) -> None:
    """Show "Save as" without stalling the pipeline.

    GetSaveFileNameW is modal: in the main loop it would freeze the
    overlay on the last frame, and with a recording running the pause
    over the dialog would land in the MP4 as a still (PTS comes from
    the clock). So the dialog lives in its own thread and the path
    comes back through a queue. A second dialog is not opened - one
    window is already up.

    A configured screenshot_dir is the folder the dialog opens in,
    not a replacement for it (issue #20). ``freeze_screenshot_frame`` has
    already copied the selected image before this function is called.
    """
    if st.shot_dialog_open:
        return
    st.shot_dialog_open = True
    hwnd = st.display.get_hwnd()
    suffix = ".png" if str(st.cfg.get("screenshot_format", "png")) == "png" \
        else ".jpg"
    default_name = f"neuralscreen-{time.strftime('%Y%m%d-%H%M%S')}{suffix}"
    shot_dir = st.cfg.get("screenshot_dir")
    # The folder the dialog opens in has to exist, or the dialog silently falls
    # back to the library's own default location and the user's choice looks
    # lost (audit M2). A deleted folder, or one on a drive that is not
    # connected right now, is not an error - it just cannot be the start point.
    initial_dir = None
    if isinstance(shot_dir, str) and shot_dir.strip():
        candidate = Path(shot_dir)
        if candidate.is_dir():
            initial_dir = str(candidate)
        else:
            print(f"[main] the configured screenshot folder is not available "
                  f"({shot_dir}) - opening the dialog at the default")

    def _run() -> None:
        try:
            st.shot_paths.put(("save", dialogs.ask_save_path(
                hwnd, default_name, initial_dir,
                fallback_dir=BASE_DIR / "screenshots")))
        except Exception as exc:
            print(f"[main] the save dialog crashed: {exc}", file=sys.stderr)
            st.shot_paths.put(("save", None))

    asked = getattr(st, "shot_requested_at", None)
    if asked is not None:
        print(f"[main] screenshot: opening the dialog "
              f"{(time.monotonic() - asked) * 1000:.0f} ms after the request")
    threading.Thread(target=_run, name="save-dialog", daemon=True).start()


def drain_save_dialog(st) -> None:
    """Take the path from the dialog if the user has already answered."""
    try:
        while True:
            answer = st.shot_paths.get_nowait()
            st.shot_dialog_open = False
            if (isinstance(answer, tuple) and len(answer) == 2
                    and answer[0] in (
                        "save", "screenshot_dir", "recording_dir", "diagnostics",
                        "convert_pick", "convert_done")):
                kind, shot_path = answer
            else:
                # Backward compatibility for tests and older producers.
                kind, shot_path = "save", answer
            if kind == "convert_pick":
                # The picker answered. Nothing was started yet: the
                # conversion begins here, on the main thread, so the state
                # the worker thread copies is read in one place.
                if shot_path is not None:
                    begin_conversion(st, Path(shot_path))
                continue
            if kind == "convert_done":
                ok, detail = shot_path
                st.convert_busy = False
                st.convert_status = ""
                if ok:
                    st.display.alert(UI_STRINGS[st.lang].get(
                        "convert_done", "Converted: {path}").format(path=detail),
                        duration=8.0)
                else:
                    st.display.alert(UI_STRINGS[st.lang].get(
                        "convert_failed", "Conversion failed: {details}").format(
                            details=detail), duration=8.0)
                continue
            if kind == "diagnostics":
                ok, detail = shot_path
                if ok:
                    print(f"[main] diagnostic package: {detail}")
                    st.display.alert(UI_STRINGS[st.lang].get(
                        "diagnostics_saved",
                        "Diagnostic package: {path}").format(path=detail),
                        duration=8.0)
                else:
                    st.display.alert(UI_STRINGS[st.lang].get(
                        "diagnostics_failed",
                        "Could not create diagnostic package"), duration=6.0)
                continue
            if kind in ("screenshot_dir", "recording_dir"):
                if shot_path is None:
                    continue
                cfg_key = kind
                st.cfg[cfg_key] = str(shot_path)
                settings_io.save_menu_layout(st)
                st.display.menu.set_state({cfg_key: str(shot_path)})
                label_key = ("shot_folder_set" if kind == "screenshot_dir"
                             else "record_folder_set")
                fallback = ("Screenshot folder: {path}" if kind == "screenshot_dir"
                            else "Recording folder: {path}")
                message = UI_STRINGS[st.lang].get(label_key, fallback).format(
                    path=shot_path)
                print(f"[main] {kind} -> {shot_path}")
                st.display.alert(message)
                continue
            asked = getattr(st, "shot_requested_at", None)
            if asked is not None:
                print(f"[main] screenshot: the dialog answered "
                      f"{(time.monotonic() - asked) * 1000:.0f} ms after the "
                      f"request")
            if shot_path is None:
                st.shot_rgba = None
                print("[main] screenshot cancelled by the user")
                continue
            rgba = st.shot_rgba
            st.shot_rgba = None
            if rgba is None:
                st.display.alert("No frame yet")
                continue
            save_screenshot(st, shot_path, rgba)
    except queue.Empty:
        pass


def begin_recording_finalization(st, *, announce: bool = True) -> bool:
    """Move the active recorder to background finalization without waiting."""
    rec = getattr(st, "recorder", None)
    if rec is None:
        return False
    if getattr(st, "recording_finalizer", None) is not None:
        raise RuntimeError("a recording is already finalizing")
    st.recorder = None
    rec.finish()
    st.recording_finalizer = rec
    st.recording_finalize_deadline = (
        time.monotonic() + float(rec.FINISH_TIMEOUT_S))
    print(f"[main] recording finalizing: {rec.path}")
    if announce:
        st.display.alert(UI_STRINGS[st.lang].get(
            "record_finalizing", "Finalizing recording..."))
    return True


def _recording_metadata(rec, result=None) -> dict:
    return {
        "container": "MP4",
        "codec": str(getattr(rec, "codec", "unknown")),
        "fps": float(getattr(rec, "fps", 0.0)),
        "audio": bool(getattr(rec, "audio_enabled", False)),
        "path": str((getattr(result, "path", None)
                     if result is not None else None) or rec.path),
        "status": str((getattr(result, "status", None)
                       if result is not None else rec.status).value),
    }


def poll_recording_finalizer(st) -> None:
    """Publish one terminal recording result; never wait on the UI thread."""
    rec = getattr(st, "recording_finalizer", None)
    if rec is None:
        return
    result = rec.wait(0)
    deadline = float(getattr(st, "recording_finalize_deadline", 0.0) or 0.0)
    if result is None and deadline and time.monotonic() >= deadline:
        try:
            rec.close(timeout=0)
        except RecordingError:
            pass
        result = rec.result
    if result is None:
        return

    metadata = _recording_metadata(rec, result)
    st.last_recording = metadata
    st.recording_finalizer = None
    st.recording_finalize_deadline = 0.0
    audio = "AAC" if metadata["audio"] else "no audio"
    detail = (f"{metadata['container']} | {metadata['codec']} | "
              f"{metadata['fps']:g} fps | {audio}")
    if result.status is RecordingStatus.PUBLISHED:
        print(f"[main] recording published: {metadata['path']} | {detail} | "
              f"{rec.written} frames, {rec.duration_ms / 1000.0:.1f}s, "
              f"dropped {rec.dropped}")
        st.display.alert(UI_STRINGS[st.lang].get(
            "record_saved", "Recording saved: {path}").format(
                path=metadata["path"]))
        return

    error = result.error
    stage = getattr(error, "stage", "finalize")
    partial = metadata["path"] if result.path else "—"
    print(f"[main] recording failed at {stage}: {error}; partial={partial}",
          file=sys.stderr)
    st.display.alert(UI_STRINGS[st.lang].get(
        "record_failed_stage", "Recording failed ({stage}); partial: {path}").format(
            stage=stage, path=partial))


def open_folder_picker(st, kind: str) -> None:
    """Open one media-directory picker and return a tagged queue result."""
    if kind not in ("screenshot_dir", "recording_dir") or st.shot_dialog_open:
        return
    st.shot_dialog_open = True
    hwnd = st.display.get_hwnd()
    title_key = ("select_shot_dir" if kind == "screenshot_dir"
                 else "select_record_dir")
    fallback = ("Select the screenshot folder" if kind == "screenshot_dir"
                else "Select the recording folder")
    title = UI_STRINGS[st.lang].get(title_key, fallback)

    def _pick_dir() -> None:
        try:
            selected = dialogs.pick_directory(hwnd, title)
        except Exception as exc:
            print(f"[main] folder picker crashed: {exc}", file=sys.stderr)
            selected = None
        st.shot_paths.put((kind, selected))

    threading.Thread(target=_pick_dir, name="folder-picker", daemon=True).start()


def convert_media(st) -> None:
    """Pick a file and run it through the network, off the UI thread.

    The conversion starts its OWN worker and leaves the overlay's alone.
    That is allowed: two NGX features on one card at the same time was the
    open question, and it was measured rather than assumed - a second worker
    created and evaluated in 1.17 s while the first kept answering, and the
    first kept answering after the second exited. So this is the diagnostic
    bundle's shape (a thread and a queue) and not a pipeline teardown, and
    the picture on screen never stops.

    The settings are the live ones - st.params, st.work_scale, st.nr_small -
    so the converted file is the picture the panel is showing.
    """
    if getattr(st, "convert_busy", False):
        st.display.alert(UI_STRINGS[st.lang].get(
            "convert_busy", "A conversion is already running"))
        return
    if st.shot_dialog_open:
        return
    st.shot_dialog_open = True
    # pygame is not thread-safe: the handle is read here, on the main thread.
    hwnd = st.display.get_hwnd()
    start_dir = _configured_directory(
        st.cfg.get("recording_dir"), BASE_DIR / "recordings")

    def _pick() -> None:
        try:
            chosen = dialogs.ask_open_path(
                hwnd, str(start_dir),
                UI_STRINGS[st.lang].get("convert_pick", "Convert a file..."))
        except Exception as exc:
            print(f"[main] the convert picker crashed: {exc}", file=sys.stderr)
            chosen = None
        st.shot_paths.put(("convert_pick", chosen))

    threading.Thread(target=_pick, name="convert-picker", daemon=True).start()


def begin_conversion(st, source) -> None:
    """Start the conversion itself, once a file has been chosen."""
    import media_convert

    st.convert_busy = True
    st.convert_status = UI_STRINGS[st.lang].get(
        "convert_working", "Converting {name}...").format(name=source.name)
    st.display.alert(st.convert_status, duration=6.0)
    params = dict(st.params)
    work_scale = float(st.work_scale)
    nr_small = bool(st.nr_small)
    flow_preset = str(st.cfg.get("flow_preset", "fast"))
    nr_passes = int(getattr(st, "nr_passes", 1) or 1)
    out_dir = st.cfg.get("recording_dir") or None

    def _run() -> None:
        try:
            result = media_convert.convert(
                source, None, params, work_scale=work_scale,
                nr_small=nr_small, nr_passes=nr_passes,
                flow_preset=flow_preset, out_dir=out_dir)
            print(f"[main] converted {source.name} -> {result.output} "
                  f"({result.frames} frame(s), {result.seconds:.1f}s)")
            answer = (True, str(result.output))
        except Exception as exc:
            print(f"[main] conversion failed: {exc}", file=sys.stderr)
            answer = (False, f"{type(exc).__name__}: {exc}")
        st.shot_paths.put(("convert_done", answer))

    threading.Thread(target=_run, name="media-convert", daemon=True).start()


def create_diagnostics(st) -> None:
    """Build a sanitized support ZIP off the UI thread and report its path."""
    if st.shot_dialog_open:
        return
    st.shot_dialog_open = True
    st.display.alert(UI_STRINGS[st.lang].get(
        "diagnostics_working", "Creating diagnostic package..."))

    def _run() -> None:
        try:
            from compatibility_runtime import create_support_bundle
            result = (True, str(create_support_bundle(st, stage="manual")))
        except Exception as exc:
            print(f"[main] diagnostic package failed: {exc}", file=sys.stderr)
            result = (False, f"{type(exc).__name__}: {exc}")
        st.shot_paths.put(("diagnostics", result))

    threading.Thread(target=_run, name="diagnostic-bundle", daemon=True).start()


def _window_action_hwnd(value) -> int:
    """Read HWND from the action identity, never from a displayed title."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, dict) and "hwnd" in value:
        return int(value["hwnd"])
    if isinstance(value, (tuple, list)) and value:
        return int(value[0])

    # Pre-1.13 menus emitted one combined identity token.  Keep that narrow
    # upgrade path, but new payloads/actions always carry an integer HWND and
    # titles containing colons never enter this branch.
    prefix, marker, _title = str(value).partition(": ")
    if not marker:
        raise ValueError("window action has no HWND")
    return int(prefix, 16)


def apply_menu_action(st, action: tuple) -> None:
    """A menu action -> a real setting.

    The menu changes nothing on its own: it reports what the user
    wants and the decision is taken here, where params and cfg live.
    """
    kind = action[0]
    if kind == "nr":
        st.tray_commands.put("toggle")
    elif kind == "nr_res":
        # How much resolution the network sees. The slider is only offered
        # while Boost is on, so every position is a real work resolution;
        # whether the reduced mode runs at all is the switch's business.
        want = min(settings_io.work_scale_cap(st), float(action[1]))
        pipeline.request_apply(st, want, st.cfg["profile"], st.params, new_small=True)
    elif kind == "split":
        # No need to recreate the worker: the wipe position rides in
        # every frame's header.
        st.split_pos = min(1.0, max(0.0, float(action[1])))
    elif kind == "toggle" and action[1] == "boost":
        # Boost: run the network at the work resolution instead of the full
        # frame, and composite its edit back onto the native 1:1 picture.
        # The work scale is NOT touched here - it is the user's setting and
        # has to survive the switch going off and on again, so the slider
        # comes back where it was left.
        want = not st.nr_small
        scale = min(settings_io.work_scale_cap(st), st.work_scale) if want \
            else st.work_scale
        pipeline.request_apply(st, scale, st.cfg["profile"], st.params,
                               new_small=want)
    elif kind == "toggle" and action[1] == "open_on_start":
        st.startup_menu = not st.startup_menu
        settings_io.save_menu_layout(st)
        print(f"[main] menu at startup: {'yes' if st.startup_menu else 'no'}")
    elif kind == "toggle" and action[1] == "autostart":
        # Autostart with Windows (HKCU Run). The state lives in the
        # registry, not in the config - read it and invert.
        new_state = not _autostart_enabled()
        if _set_autostart(new_state):
            print(f"[main] autostart with Windows: {'on' if new_state else 'off'}")
            st.display.alert(UI_STRINGS[st.lang].get(
                "autostart_on" if new_state else "autostart_off",
                "Autostart ON" if new_state else "Autostart OFF"))
        else:
            st.display.alert(UI_STRINGS[st.lang].get("autostart_err", "Autostart failed"))
    elif kind == "toggle" and action[1] in ("tray_on_minimise", "tray_on_close"):
        # A switch arrives as ("toggle", name) - the name is action[1], never
        # the kind. Written as its own `kind` at first, these two clicks fell
        # into the generic toggle branch and were dropped in silence: the
        # cells simply did not react (user, 20.09).
        name = action[1]
        st.cfg[name] = not bool(st.cfg.get(name, False))
        # The window procedure runs on its own thread and must not read the
        # config, so both answers are pushed to it on every change.
        tb = getattr(st, "taskbar", None)
        if tb is not None:
            tb.to_tray_on_minimise = bool(st.cfg.get("tray_on_minimise", False))
            tb.to_tray_on_close = bool(st.cfg.get("tray_on_close", False))
        settings_io.save_menu_layout(st)
        print(f"[main] {name}: {'on' if st.cfg[name] else 'off'}")
    elif kind == "toggle" and action[1] == "rec_indicator":
        # The recording indicator outside the menu: a config flag,
        # the HUD reads it on every redraw.
        st.cfg["rec_indicator"] = not bool(st.cfg.get("rec_indicator", True))
        settings_io.save_menu_layout(st)
        print(f"[main] recording indicator: {'on' if st.cfg['rec_indicator'] else 'off'}")
    elif kind == "toggle" and action[1] == "frame_generation":
        st.cfg["frame_generation"] = not bool(st.cfg.get("frame_generation", False))
        # A fresh attempt re-arms the refusal alert: if the runtime says no
        # again, the user is told again (issue #76 - the silent ON).
        if st.cfg["frame_generation"]:
            st.fg_alerted = False
        settings_io.save_menu_layout(st)
    elif kind == "frame_multiplier":
        st.cfg["frame_multiplier"] = min(4, max(2, int(action[1])))
        # A different multiplier is a fresh attempt at the feature: if the
        # runtime refused the previous one, it gets to answer again (issue
        # #82 - 2x may work where 3x does not).
        if st.cfg.get("frame_generation"):
            st.fg_alerted = False
        settings_io.save_menu_layout(st)
    elif kind == "frame_limit_mode":
        mode = str(action[1])
        if mode in settings_io.FRAME_LIMIT_MODES:
            st.cfg["frame_limit_mode"] = mode
            settings_io.save_menu_layout(st)
            print(f"[main] frame limit: "
                  f"{settings_io.frame_limit_fps(st.cfg) or 'unlimited'}")
    elif kind == "frame_limit_custom":
        try:
            value = int(action[1])
        except (TypeError, ValueError, OverflowError):
            return
        st.cfg["frame_limit_custom"] = min(
            settings_io.FRAME_LIMIT_CUSTOM_MAX,
            max(settings_io.FRAME_LIMIT_CUSTOM_MIN, value))
        settings_io.save_menu_layout(st)
        print(f"[main] custom frame limit: {st.cfg['frame_limit_custom']} fps")
    elif kind == "toggle" and action[1] == "skip_static":
        # A per-frame flag in the header, not a worker setting: no restart,
        # the next frame already carries the new state.
        st.cfg["skip_static"] = not bool(st.cfg.get("skip_static", True))
        settings_io.save_menu_layout(st)
        print(f"[main] skip static frames: "
              f"{'on' if st.cfg['skip_static'] else 'off'}")
    elif kind == "toggle" and action[1] == "spout":
        # The Spout2 bridge: the worker reads NS_SPOUT only at startup,
        # so the toggle goes through a worker restart (pipeline.apply_spout
        # owns the whole path, including the config write).
        pipeline.apply_spout(st, not bool(st.cfg.get("spout", False)))
    elif kind == "toggle" and action[1] == "hdr":
        # HDR compatibility: NS_HDR is read once per worker process too,
        # and it decides the capture format, so this is a restart as well.
        pipeline.apply_hdr(st, not bool(st.cfg.get("hdr", False)))
    elif kind == "style":
        # Style travels with the parameters, so it applies the same way they
        # do since C1: a resize command with unchanged sizes, no feature
        # rebuild. Measured in test_param_effect - the frame changes and the
        # log says "parameters only".
        try:
            value = int(action[1])
        except (TypeError, ValueError):
            value = 1
        if 0 <= value <= 2:
            new_params = dict(st.params)
            new_params["style"] = value
            pipeline.request_apply(st, st.work_scale, st.cfg["profile"],
                                   new_params)
    elif kind == "motion_backend":
        pipeline.apply_motion_backend(st, action[1])
    elif kind == "screenshot_mode":
        mode = str(action[1])
        if mode in ("ask", "auto"):
            st.cfg["screenshot_mode"] = mode
            settings_io.save_menu_layout(st)
    elif kind == "screenshot_format":
        image_format = str(action[1]).lower()
        if image_format in ("png", "jpg"):
            st.cfg["screenshot_format"] = image_format
            settings_io.save_menu_layout(st)
    elif kind == "param":
        new_params = dict(st.params)
        new_params[action[1]] = float(action[2])
        pipeline.request_apply(st, st.work_scale, st.cfg["profile"], new_params)
    elif kind == "profile":
        if action[1] in PROFILES:
            # A built-in profile moves the four sliders only (user rule
            # 15.09): the model is its own control and survives a profile
            # change. Taking the style from the profile made "Natural"
            # silently overwrite a Cinematic the user had just picked.
            new_params = dict(PROFILES[action[1]])
            new_params["style"] = int(st.params.get("style", 1))
            pipeline.request_apply(st, st.work_scale, action[1], new_params)
        elif action[1] in st.presets:
            # A user preset DOES carry its own model - that is what saving
            # it promised.
            pipeline.request_apply(st, st.work_scale, action[1],
                                   dict(st.presets[action[1]]))
        else:
            print(f"[main] unknown profile {action[1]!r} - ignored",
                  file=sys.stderr)
    elif kind == "lang":
        if action[1] in UI_STRINGS and action[1] != st.lang:
            st.lang = action[1]
            st.display.set_lang(st.lang)
            st.display.menu.set_state({"lang": st.lang})
            print(f"[main] interface language -> {st.lang}")
    elif kind == "refresh_windows":
        # The windows page freezes its list while it is open, so the rows cannot
        # shuffle under the cursor mid-click (13.09). The cache is the freeze;
        # clearing it is a fresh reading, and the page stays where it is.
        st.window_list = None
    elif kind == "capture":
        # While the menu waits for a keypress the global hotkeys must
        # be suspended: otherwise Num2 toggles the menu instead of
        # landing in the field.
        if action[1]:
            st.hotkeys.suspend()
        else:
            st.hotkeys.resume()
    elif kind == "hotkey":
        cmd, text = action[1], action[2]
        parsed = parse_binding(text)
        if parsed is None:
            print(f"[main] could not parse the combination {text!r}", file=sys.stderr)
            st.display.alert(UI_STRINGS[st.lang]["hotkey_bad"])
        else:
            over = st.cfg.get("hotkeys")
            over = dict(over) if isinstance(over, dict) else {}
            over[cmd] = text
            st.cfg["hotkeys"] = over
            st.hotkey_bindings = build_bindings(over)
            st.hotkeys.rebind(st.hotkey_bindings)
            st.display.menu.set_hotkeys(hotkey_labels(st.hotkey_bindings))
            if not settings_io.save_hotkeys(st, over):
                # The assignment works for this session but will not
                # survive a restart - the user must know.
                st.display.alert(UI_STRINGS[st.lang]["save_fail"])
                return
            print(f"[main] {cmd} -> {text}")
            st.display.alert(UI_STRINGS[st.lang]["settings_applied"])
    elif kind == "theme":
        # The menu has already applied the theme to itself (overlay_ui).
        # It lands in st.cfg RIGHT HERE, not only in the file: the file is
        # written on menu close and on exit, but a rebuild in between -
        # a monitor switch, a GPU switch, one-window mode - recreates the
        # menu and restores the theme from st.cfg. With the value still
        # missing there, a monitor switch threw the user back to light
        # (issue #33).
        if action[1] in ("light", "dark"):
            st.cfg["theme"] = action[1]
        print(f"[main] menu theme -> {action[1]}")
    elif kind == "nr_passes":
        try:
            passes = int(action[1])
        except (TypeError, ValueError):
            print(f"[main] invalid pass count: {action[1]!r}", file=sys.stderr)
            return
        passes = min(4, max(1, passes))
        st.nr_passes = passes
        st.cfg["nr_passes"] = passes
        print(f"[main] NR cascade -> {passes} pass(es)")
        # The count travels with the parameters, so the ordinary apply
        # carries it - no teardown, no warm-up, no frozen picture. It went
        # through a full restart briefly, to re-read a residual strength
        # derived from the count; that strength is a setting now and the
        # seconds it cost were not worth it (user, 20.09).
        pipeline.request_apply(st, st.work_scale, st.cfg["profile"], st.params)
    elif kind == "fps_overlay":
        corner = str(action[1])
        if corner not in ("off", "tl", "tr", "bl", "br"):
            print(f"[main] invalid counter corner: {action[1]!r}",
                  file=sys.stderr)
            return
        st.cfg["fps_overlay"] = corner
        print(f"[main] on-screen frame counter -> {corner}")
    elif kind == "menu_scale":
        # The menu has already applied it to itself. It lands in st.cfg here
        # for the same reason the theme does: a rebuild between now and the
        # next file write recreates the menu from st.cfg, and a size the user
        # just picked must survive that.
        try:
            step = round(float(action[1]), 2)
        except (TypeError, ValueError):
            print(f"[main] invalid interface scale: {action[1]!r}",
                  file=sys.stderr)
            return
        st.cfg["menu_scale"] = step
        # The automatic fit is spent the moment the user picks a size: it is a
        # starting point for a first launch, not a preference that keeps
        # correcting them.
        st.cfg["menu_scale_auto"] = False
        print(f"[main] interface scale -> {step:g}")
    elif kind == "gpu":
        # The value arrives as "N: NVIDIA GeForce ..." - the index is the
        # identity here (it is what NS_GPU takes), the name is the label.
        try:
            index = int(str(action[1]).split(":")[0])
        except (ValueError, IndexError):
            print(f"[main] invalid GPU: {action[1]!r}", file=sys.stderr)
            return
        pipeline.apply_gpu(st, index)
    elif kind == "monitor":
        # The value arrives as "N: WxH (\\\\.\\DISPLAY1)" - the
        # devicename is the identity, the index is only a label.
        try:
            new_monitor = str(action[1]).split(" (")[1].rstrip(")")
        except (ValueError, IndexError):
            print(f"[main] invalid monitor: {action[1]!r}", file=sys.stderr)
            return
        if new_monitor != st.capture.devicename:
            pipeline.switch_monitor(st, new_monitor)
    elif kind == "window":
        # The row carries HWND separately from its clean display title.
        try:
            target = _window_action_hwnd(action[1])
        except (TypeError, ValueError, IndexError, KeyError):
            print(f"[main] invalid window: {action[1]!r}", file=sys.stderr)
            return
        if not ctypes.windll.user32.IsWindow(ctypes.c_void_p(target)):
            print(f"[main] the window 0x{target:X} is gone", file=sys.stderr)
            st.display.alert(UI_STRINGS[st.lang]["win_fail"])
            return
        # Bring the chosen window to the front: the capture follows
        # it, and a window buried under others would show through
        # the overlay as a half-covered picture (user: the chosen
        # window must come to the foreground, no overlaps).
        user32 = ctypes.windll.user32
        user32.BringWindowToTop(ctypes.c_void_p(target))
        user32.SetForegroundWindow(ctypes.c_void_p(target))
        print(f"[main] window mode on from the menu - target hwnd "
              f"0x{target:X}")
        pipeline.switch_window(st, target)
    elif kind == "button":
        name = action[1]
        if name == "close":
            # Whatever route got here, the keyboard comes back. suspend() has
            # exactly one counterpart and closing the menu is the last moment
            # it can be reached; resume() on a controller that was never
            # suspended posts a message nobody acts on.
            st.hotkeys.resume()
            st.display.menu.visible = False
            st.display.set_menu_opaque(False)
            st.display.set_menu_input(False)
            settings_io.save_menu_layout(st)
            # The close button and Esc both land here, and neither used to say
            # so: the settings paths printed opened/closed, this one printed
            # nothing. A diagnostic package then read "21 opened against 0
            # closed", which cannot be interpreted - a menu that was never
            # closed and a close that was never logged look identical. The
            # duration is what dates it against the events around it.
            opened_at = getattr(st, "menu_opened_at", 0.0)
            if opened_at:
                print(f"[main] overlay menu closed (the close button, "
                      f"open for {time.monotonic() - opened_at:.1f} s)")
            else:
                print("[main] overlay menu closed (the close button)")
        elif name == "exit":
            print(f"[main] exit: button in the overlay menu "
                  f"(frames processed {st.frame_index})")
            st.running = False
        elif name.startswith("frame_multiplier:"):
            st.cfg["frame_multiplier"] = min(4, max(2, int(name.split(":", 1)[1])))
            # Same re-arm as the menu's multiplier buttons: a new value is
            # a new attempt.
            if st.cfg.get("frame_generation"):
                st.fg_alerted = False
            else:
                # Picking a step with Frame Generation OFF turns it ON: the
                # steps and Off are one segment group, so a click on x3 has to
                # mean "run at x3" - otherwise the group would show a preference
                # nobody can see and the user would need two clicks for one
                # decision.
                #
                # NO tray command here. `toggle` is the NR switch, and sending
                # it turned the neural pass off 51 ms after FG came on (seen
                # live). The worker needs no command: it reads
                # frame_generation/frame_multiplier from every frame and
                # rebuilds its own resources when they change.
                st.cfg["frame_generation"] = True
                st.fg_alerted = False
            settings_io.save_menu_layout(st)
        elif name == "frame_generation:off":
            # The Off cell of the group. Same rule as above: write the config,
            # and the next frame carries it. `toggle` here would pause NR.
            if st.cfg.get("frame_generation"):
                st.cfg["frame_generation"] = False
                st.fg_alerted = False
            settings_io.save_menu_layout(st)
        elif name == "record":
            st.tray_commands.put("record")
        elif name == "screenshot":
            st.tray_commands.put("screenshot_menu")
        elif name == "window_mode":
            # The fullscreen button in the footer: the same action
            # as the Num5 hotkey - in window mode it returns to the
            # whole screen, in fullscreen mode it is a no-op with an
            # alert (the user asked for a visible "already active").
            if st.window_hwnd is not None:
                print("[main] window mode off - back to the whole screen")
                pipeline.switch_window(st, 0)
            else:
                st.display.alert(UI_STRINGS[st.lang]["fs_active"])
        elif name == "shot_dir":
            open_folder_picker(st, "screenshot_dir")
        elif name == "record_dir":
            open_folder_picker(st, "recording_dir")
        elif name == "convert_pick":
            convert_media(st)
        elif name == "diagnostics":
            create_diagnostics(st)
        elif name == "github":
            # The hotkeys, profiles and requirements are described
            # only in the README - there was no way to learn about
            # them from the program itself.
            try:
                webbrowser.open(REPO_URL)
                st.display.alert(UI_STRINGS[st.lang]["github_opened"])
            except Exception as exc:
                print(f"[main] could not open {REPO_URL}: {exc}",
                      file=sys.stderr)
        elif name == "save_preset":
            # The current slider values, snapshotted as a named
            # preset. The NGX plumbing of the active profile rides
            # along, so the preset reproduces the exact look it was
            # saved with.
            name = _next_preset_name(st.presets)
            st.presets[name] = dict(st.params)
            st.cfg["presets"] = st.presets
            if not settings_io.save_menu_layout(st):
                # The preset lives in memory but not on disk - the
                # user must know it will not survive a restart.
                del st.presets[name]
                st.cfg["presets"] = st.presets
                st.display.alert(UI_STRINGS[st.lang]["save_fail"])
                return
            st.display.menu.set_state(
                {"profiles": list(PROFILES) + list(st.presets)})
            print(f"[main] preset saved: {name}")
            st.display.alert(f"Preset saved: {name}")
        elif name == "delete_preset":
            # Only a user preset can be deleted - the built-in
            # profiles are not deletable.
            if st.cfg["profile"] in st.presets:
                del st.presets[st.cfg["profile"]]
                st.cfg["presets"] = st.presets
                if not settings_io.save_menu_layout(st):
                    st.display.alert(UI_STRINGS[st.lang]["save_fail"])
                    return
                st.display.menu.set_state(
                    {"profiles": list(PROFILES) + list(st.presets)})
                print(f"[main] preset deleted: {st.cfg['profile']}")
                st.display.alert(f"Preset deleted: {st.cfg['profile']}")
                new_params = dict(PROFILES["Natural"])
                new_params["style"] = int(st.params.get("style", 1))
                pipeline.request_apply(st, st.work_scale, "Natural", new_params)
        elif name == "channel":
            # The channel label in the settings page opens the
            # channel (user rule 2026-09-08).
            try:
                webbrowser.open(CHANNEL_URL)
                st.display.alert(UI_STRINGS[st.lang]["github_opened"])
            except Exception as exc:
                print(f"[main] could not open {CHANNEL_URL}: {exc}",
                      file=sys.stderr)


def drain_commands(st) -> bool:
    """Handle the tray/hotkey commands; False when the program must quit.

    Called from the main loop AND from inside the recv wait: at 4K a
    heavy scene can take ~1 s per NGX frame, and the hotkeys must
    stay responsive while main waits for the worker (user: "NR toggle
    does not always fire in Cyberpunk").
    """
    try:
        while True:
            cmd = st.tray_commands.get_nowait()
            if cmd == "quit":
                print(f"[main] exit: tray or the quit hotkey "
                      f"(frames processed {st.frame_index})")
                st.running = False
            elif cmd == "to_tray":
                # #93: the panel goes away, the taskbar button goes away, the
                # tray icon stays - and the PROCESSING does not stop. That is
                # the one thing this must not do quietly: the picture on screen
                # is the program's output, so a "minimise" that also stopped
                # the pass would change what the user sees without saying so.
                # The tray menu is the way back.
                # The tray icon is the ONLY way back once the button is
                # hidden, so it is checked first. It runs in a daemon thread
                # whose death is silent; hiding the button on the strength of
                # that would leave Task Manager as the way out.
                tray = getattr(st, "tray", None)
                if tray is None or not tray.alive():
                    print("[main] to the tray refused: there is no tray icon "
                          "to come back from", file=sys.stderr)
                    st.display.alert(UI_STRINGS[st.lang].get(
                        "tray_missing", "The tray icon is not available"))
                    continue
                if st.display.menu.visible:
                    st.hotkeys.resume()
                    st.display.menu.visible = False
                    st.display.set_menu_opaque(False)
                    st.display.set_menu_input(False)
                    # Everything the ordinary close does, because this IS a
                    # close: the HUD layer goes back onto the captured window
                    # (in one-window mode it was stretched to the whole
                    # monitor while the menu was up), and the remap field is
                    # dropped - a menu left in `capturing` swallows the first
                    # keydown of the next open as a remap, and in the tray
                    # that next open can be hours away.
                    if st.window_hwnd is not None:
                        rect = window_frame_rect(st.window_hwnd)
                        if rect is not None:
                            st.display.set_window_layer(*rect)
                    st.display.menu.capturing = None
                    settings_io.save_menu_layout(st)
                if getattr(st, "taskbar", None) is not None:
                    st.taskbar.set_visible(False)
                st.in_tray = True
                print("[main] to the tray: the button is hidden, the neural "
                      "pass keeps running")
            elif cmd in ("settings", "show_settings"):
                # Coming back from the tray restores the button first: the
                # menu is about to be shown, and a menu with no button in the
                # taskbar is the state #93 is complaining about.
                if getattr(st, "in_tray", False):
                    if getattr(st, "taskbar", None) is not None:
                        st.taskbar.set_visible(True)
                    st.in_tray = False
                    print("[main] back from the tray")
                # Num2 and the tray keep their useful toggle semantics. The
                # taskbar is different: Windows may deliver several activation
                # messages for one click, so it asks only to SHOW the menu.
                # Otherwise a duplicate activation can close a visible menu
                # and make NeuralScreen appear to hide itself (#87).
                st.display.menu.set_state(settings_io.menu_payload(st))
                was_open = st.display.menu.visible
                if cmd == "show_settings":
                    st.display.menu.visible = True
                    opened = True
                    # This is also the explicit recovery path for a layered
                    # overlay whose visual attributes were disturbed by the
                    # shell: retain its input state, but reapply the actual
                    # key/alpha attributes before claiming it is shown.
                    st.display.refresh_colorkey()
                else:
                    opened = st.display.menu.toggle()
                st.display.set_menu_opaque(opened)
                st.display.set_menu_input(opened)
                if opened:
                    st.menu_opened_at = time.monotonic()
                    if not was_open:
                        # In one-window mode the HUD layer is the size of
                        # the captured window - a menu near the edge would
                        # be clipped by it. Expand the layer to the whole
                        # monitor while the menu is open, so the menu is
                        # always fully visible (user: menu lost outside a
                        # small window). The saved offset is honoured -
                        # layout() clamps it to the screen (user rule
                        # 10.09: fixed position until the user drags it).
                        if st.window_hwnd is not None:
                            st.display.set_fullscreen_layer(st.mon_w, st.mon_h)
                        # The mouse lands on the title bar, so the user
                        # does not have to hunt for the pointer (user
                        # request). The layout must be current for the
                        # title rect to be valid.
                        try:
                            st.display.menu.layout(
                                st.display.screen.get_width(),
                                st.display.screen.get_height())
                            cx, cy = st.display.menu.title_center()
                            ctypes.windll.user32.SetCursorPos(cx, cy)
                        except Exception:
                            pass
                    # Logical visibility is not physical visibility. After a
                    # monitor/GPU rebuild or a shell/compositor disturbance the
                    # log could say "opened" while the HWND stayed hidden or
                    # below the native presenter (#87/#88). Every open/show is
                    # therefore an explicit recovery operation; duplicate
                    # taskbar activations remain idempotent.
                    if not st.display.is_visible():
                        st.display.reveal()
                    st.display.set_visible(True)
                    # The user may have moved us to another virtual desktop
                    # through Task View: the 1x1 taskbar window travels there,
                    # the borderless overlay windows are left behind and the
                    # menu would be drawn on a desktop nobody is looking at
                    # ("the program does not expand on desktop 2", #93). Put
                    # the overlay pair where the taskbar window is. Unknown
                    # (interface absent) leaves everything untouched.
                    st.display.follow_taskbar_desktop()
                    st.display.raise_topmost()
                    st.display.draw_overlay(0.0)
                else:
                    # The menu closed: put the HUD layer back on the
                    # captured window.
                    if st.window_hwnd is not None:
                        rect = window_frame_rect(st.window_hwnd)
                        if rect is not None:
                            st.display.set_window_layer(*rect)
                    settings_io.save_menu_layout(st)
                    # Whatever route closed the menu, the keyboard comes back
                    # (audit H2). Only the close button used to resume, so a
                    # hotkey field that was waiting for a key left the global
                    # hotkeys unregistered for the rest of the session when the
                    # menu was closed from the tray or the taskbar: Num0-Num7
                    # and Ctrl+Alt+Q all dead, with no way back but a restart.
                    # The menu's own capturing flag goes with it - otherwise
                    # the next open silently swallows the first keydown as a
                    # remap. resume() on a controller that was never suspended
                    # is harmless (it posts a message nobody acts on).
                    st.hotkeys.resume()
                    st.display.menu.capturing = None
                print(f"[main] overlay menu {'opened' if opened else 'closed'}")
            elif cmd == "toggle":
                st.paused = not st.paused
                if not st.paused:
                    st.work_frame = None  # a fresh grab after the pause
                    if st.worker_failed:
                        # The worker died and was shut down (issue #3):
                        # revive it - a fresh process may succeed (a
                        # transient GPU conflict, a driver hiccup).
                        st.worker_failed = False
                        # The automatic revive is disarmed by the manual one.
                        # It used to stay armed: the user brought the worker
                        # back by hand, and up to 30 s later the deadline came
                        # round and restarted the healthy worker underneath
                        # them - a black screen out of nowhere (audit F4).
                        st.next_auto_revive = 0.0
                        print("[main] reviving the worker after the failure")
                        try:
                            pipeline.require_compatibility(st)
                            st.worker, st.worker_logs, st.reader, st.worker_stop = restart_worker(
                                st.worker, st.params, st.work_w, st.work_h,
                                st.effective_warmup,
                                st.width if (st.work_w != st.width or st.work_h != st.height) else 0,
                                st.height if (st.work_w != st.width or st.work_h != st.height) else 0,
                                st.worker_stop, st.shm)
                            channels.forget_present(st)
                            channels.forget_dda(st)
                            channels.forget_out(st)
                            # The manual revive restarts the worker like any
                            # other path, so the feature-18 verdict has to die
                            # with the worker that gave it (channels.py's
                            # contract). It was the one path that skipped this,
                            # so a card that started working kept the red dot
                            # and a pipeline that came back broken kept the
                            # green one - while TECHNICAL.md tells the user
                            # that dot is the thing to trust.
                            channels.forget_verdict(st)
                            channels.sync_motion_size(st)
                            st.frame_index = 0
                            st.pts = 0
                        except Exception as exc:
                            print(f"[main] worker revive failed ({exc}) - "
                                  f"staying NR OFF", file=sys.stderr)
                            st.paused = True
                            st.worker_failed = True
                    st.display.set_visible(True)
                print(f"[main] NR {'OFF' if st.paused else 'ON'}")
                st.display.alert(UI_STRINGS[st.lang]["nr_off" if st.paused else "nr_on"])
                st.tray._set_state(nr=not st.paused)
            elif cmd == "screenshot_menu":
                request_screenshot(st)
            elif cmd == "framegen":
                # A plain on/off for Frame Generation (user request 15.09).
                # The same path the menu switch takes, so the config write,
                # the alert re-arm and the header pair all behave the same.
                apply_menu_action(st, ("toggle", "frame_generation"))
                state = bool(st.cfg.get("frame_generation", False))
                st.display.menu.set_state({"frame_generation": state})
                print(f"[main] frame generation: {'on' if state else 'off'} "
                      f"({int(st.cfg.get('frame_multiplier', 2))}x)")
                st.display.alert(UI_STRINGS[st.lang].get(
                    "fg_on" if state else "fg_off",
                    "DLSS FG ON" if state else "DLSS FG OFF"))
            elif cmd == "record":
                # Num0: record the NR frame into an MP4. The frames
                # are requested from the worker through
                # FRAME_FLAG_WANT_PIXELS (the screenshot mechanism,
                # but for every recorded frame).
                if st.recorder is None:
                    if getattr(st, "recording_finalizer", None) is not None:
                        st.display.alert(UI_STRINGS[st.lang].get(
                            "record_finalizing", "Finalizing recording..."))
                        continue
                    try:
                        rec_dir = _configured_directory(
                            st.cfg.get("recording_dir"), BASE_DIR / "recordings")
                        path = str(_unique_media_path(
                            rec_dir, "neuralscreen", ".mp4"))
                        # 30 fps, not 60: every recorded frame is a
                        # full 33 MB round-trip from the worker
                        # (FRAME_FLAG_WANT_PIXELS -> pipe), and the
                        # measurement showed 60 fps recording costs
                        # ~36% of the FPS (101 -> 65). Halving the
                        # frame rate halves that cost; the picture
                        # quality per frame is identical.
                        st.recorder = VideoRecorder(path, st.width, st.height, fps=30,
                                                 audio=st.record_audio)
                    except Exception as exc:
                        print(f"[main] recording did not start: {exc}", file=sys.stderr)
                        st.display.alert(f"REC ERROR: {exc}")
                        st.recorder = None
                    else:
                        audio = "AAC" if st.recorder.audio_enabled else "no audio"
                        detail = (f"MP4 | {st.recorder.codec} | "
                                  f"{st.recorder.fps:g} fps | {audio}")
                        print(f"[main] recording started: {path} | {detail}")
                        st.display.alert(UI_STRINGS[st.lang].get(
                            "record_started", "Recording: {details}").format(
                                details=detail))
                else:
                    try:
                        begin_recording_finalization(st)
                    except Exception as exc:
                        print(f"[main] failed to finalize the recording: {exc}", file=sys.stderr)
                        st.display.alert(UI_STRINGS[st.lang]["rec_save_fail"])
            elif cmd == "window_mode":
                # The window under the cursor wins: it works on the
                # desktop too (the focused window there is Progman,
                # which is not capturable), and it is what the user
                # is looking at. Fall back to the last focused
                # foreign window when the cursor is over nothing
                # capturable (our own overlay, the desktop).
                if st.window_hwnd is not None:
                    print("[main] window mode off - back to the whole screen")
                    pipeline.switch_window(st, 0)
                else:
                    target = window_under_cursor() or st.last_foreground
                    if target:
                        print(f"[main] window mode on - target hwnd "
                              f"0x{target:X}")
                        pipeline.switch_window(st, target)
                    else:
                        # Nothing but our own windows has had the
                        # focus, so there is nothing to capture but
                        # ourselves.
                        print("[main] window mode: no window to capture "
                              "(only our own windows have had the focus)",
                              file=sys.stderr)
                        st.display.alert(UI_STRINGS[st.lang]["win_none"])
            elif cmd in ("scale_up", "scale_down"):
                # The hotkeys walk one ladder: every step below the cap is a
                # work resolution with Boost on, and the step above it is the
                # full frame with Boost off. Without that last part the keys
                # would change the number in the alert and nothing in the
                # picture whenever Boost happened to be off - the network
                # runs at the full size then whatever the scale says
                # (measured bit for bit, 12.09).
                cap = settings_io.work_scale_cap(st)
                cur = st.work_scale if st.nr_small else cap + WORK_SCALE_STEP
                delta = WORK_SCALE_STEP if cmd == "scale_up" else -WORK_SCALE_STEP
                new_scale = min(cap + WORK_SCALE_STEP,
                                max(WORK_SCALE_MIN, cur + delta))
                if abs(new_scale - cur) > 1e-6:
                    want_small = new_scale <= cap + 1e-6
                    applied = new_scale if want_small else st.work_scale
                    new_w, new_h = _work_size(st.width, st.height, applied,
                                              getattr(st, 'nr_passes', 1))
                    print(f"[main] work_scale -> {new_scale:.2f} ({new_w}x{new_h}), "
                          f"boost {'on' if want_small else 'off'}")
                    # Off the ladder's top step the numbers the user is shown
                    # are the full frame, not the scale that stays stored for
                    # when Boost comes back: an alert reading "0.70" would
                    # name a position that does not exist.
                    st.display.alert(UI_STRINGS[st.lang]["work_scale_changed"].format(
                        new_scale if want_small else 1.0,
                        new_w if want_small else st.width,
                        new_h if want_small else st.height))
                    pipeline.request_apply(st, applied, st.cfg["profile"], st.params,
                                           new_small=want_small)
    except queue.Empty:
        pass
    return st.running
