r"""A dead parent cannot leave a worker running.

THE BUG. The worker owns a D3D12 swap-chain window and presents to the
screen itself. On a normal exit shutdown_worker closes its stdin and it
leaves; but a force-kill, an unhandled crash, or the GPU-access-loss cascade
that ends a session with exit 0x40010004 never runs that path. With no tie
between the two processes the worker kept presenting - on an HDR session, an
HDR10 PQ window left over the desktop as a washed-out layer that survived
every app restart and cleared only on reboot. Windows named it at shutdown:
"waiting for nvngx.dll to close", a process nothing owned any more.

THE FIX. Every worker is bound to a Job Object created with
KILL_ON_JOB_CLOSE. The handle lives for the life of the parent and is never
closed by the program; when the parent dies for ANY reason the OS closes it
and the kernel terminates every process in the job.

WHAT THIS TESTS. Not a mock - the real guarantee. A short-lived helper uses
pipeline's own _ensure_worker_job + bind_worker_to_job to put a long sleeper
into the job, prints the sleeper's PID, and then EXITS. If the tie holds,
the sleeper is dead within a moment of the helper leaving, though it was
told to sleep for a minute and nothing ever signalled it.
"""
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

PY = sys.executable

# The helper runs in its own process so that its death - not ours - is what
# has to kill the sleeper. It reaches into pipeline for the real job code.
HELPER = r"""
import os, sys, subprocess
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, r"{base}")
import pipeline
sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
ok = pipeline.bind_worker_to_job(sleeper)
sys.stdout.write(f"{{sleeper.pid}}|{{ok}}\n")
sys.stdout.flush()
# Leave at once. The job handle closes with this process; if the tie holds,
# KILL_ON_JOB_CLOSE takes the sleeper down with it.
os._exit(0)
"""


def _alive(pid: int) -> bool:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = wintypes.DWORD()
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    got = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(got) and code.value == 259    # STILL_ACTIVE


def main() -> int:
    failures = []

    out = subprocess.run([PY, "-c", HELPER.format(base=BASE)],
                         capture_output=True, text=True, timeout=60)
    line = (out.stdout or "").strip().splitlines()[-1] if out.stdout.strip() else ""
    if "|" not in line:
        print("=" * 60)
        print(f"FAIL: the helper printed no PID ({out.stdout!r} / {out.stderr[-300:]!r})")
        return 1
    pid_s, ok_s = line.split("|", 1)
    sleeper_pid = int(pid_s)
    bound = ok_s == "True"

    if not bound:
        # The platform refused the job (a parent job that forbids nesting).
        # That is a documented best-effort fallback, not a test failure - but
        # then this machine cannot prove the guarantee, so say so and stop.
        try:
            os.kill(sleeper_pid, 9)
        except Exception:
            pass
        print("=" * 60)
        print("SKIP: this platform would not bind the worker to a job "
              "(nested-job restriction) - the guarantee cannot be shown here")
        return 0

    # The helper has already exited. Give the kernel a beat to act on the
    # closed handle, then the sleeper - told to sleep 60 s - must be gone.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _alive(sleeper_pid):
        time.sleep(0.1)

    if _alive(sleeper_pid):
        failures.append(f"the sleeper (pid {sleeper_pid}) outlived its parent "
                        f"- the job did not kill it, a worker could orphan")
        try:
            os.kill(sleeper_pid, 9)
        except Exception:
            pass

    print("=" * 60)
    if failures:
        print(f"FAIL: {len(failures)} - {failures}")
        return 1
    print("OK: a bound worker dies with its parent - no orphan survives a "
          "crash, a force-kill or the access-loss cascade")
    return 0


if __name__ == "__main__":
    sys.exit(main())
