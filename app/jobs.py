"""Runs one pipeline subprocess at a time (the 8 GB GPU can't hold VGGT, Isaac and training together)
and streams its output lines to listeners."""
import os
import subprocess
import threading
import time
from collections import deque
from typing import Callable


class JobRunner:
    def __init__(self, max_log_lines: int = 500):
        self.log = deque(maxlen=max_log_lines)
        self.current: str | None = None
        self.stop_requested = False  # the last job was stopped by the user; on_done callbacks read it
        self._proc: subprocess.Popen | None = None
        self._reserved: str | None = None  # a job name holding the runner while its panel prepares inputs
        self._listeners: list[Callable[[str], None]] = []
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.current is not None

    def on_line(self, fn: Callable[[str], None]):
        self._listeners.append(fn)

    def emit(self, line: str):
        line = f"[{time.strftime('%H:%M:%S')}] {line.rstrip()}"
        self.log.append(line)
        for fn in self._listeners:
            fn(line)

    def refuse_if_busy(self) -> bool:
        """True (and a log line naming the running job) if a job is running. Panels call it before they delete
        the outputs a new job would replace."""
        current = self.current
        if current is not None:
            self.emit(f"'{current}' is still running; wait for it or stop it first")
        return current is not None

    def run(self, name: str, cmd: list[str], cwd=None, env: dict | None = None,
            on_done: Callable[[int], None] | None = None) -> bool:
        """Start cmd in a background thread. Returns False if another job is still running. A job reserved under
        the same name takes over its reservation."""
        with self._lock:
            if self._reserved == name:
                self._reserved = None  # keep stop_requested: a Stop during the preparation cancels the job
            elif self.refuse_if_busy():
                return False
            else:
                self.stop_requested = False
            self.current = name
        threading.Thread(target=self._run, args=(name, cmd, cwd, env, on_done), daemon=True).start()
        return True

    def reserve(self, name: str) -> bool:
        """Hold the runner for job `name` while its panel deletes old outputs or converts inputs, so no other job
        starts in between. False (and a log line) if busy. Follow with run(name, ...) or release(name)."""
        with self._lock:
            if self.refuse_if_busy():
                return False
            self.current = self._reserved = name
            self.stop_requested = False
            return True

    def release(self, name: str):
        """Give up a reservation that won't be run."""
        with self._lock:
            if self._reserved == name:
                self.current = self._reserved = None

    def _run(self, name, cmd, cwd, env, on_done):
        self.emit(f"$ {' '.join(map(str, cmd))}")
        full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
        code = -1
        try:
            with self._lock:  # stop() reads _proc under the lock, so it can't miss a job that is starting
                if self.stop_requested:
                    self.emit(f"'{name}' stopped before it started")
                    return
                proc = self._proc = subprocess.Popen(
                    [str(c) for c in cmd], cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in proc.stdout:
                self.emit(line)
            code = proc.wait()
            self.emit(f"'{name}' finished with exit code {code}")
        except Exception as e:  # noqa: BLE001 - surface any launch failure in the GUI log
            self.emit(f"'{name}' failed to start: {e}")
        finally:
            with self._lock:
                self._proc = None
                self.current = None
            if on_done:
                on_done(code)

    def stop(self):
        with self._lock:  # flag first, then kill the process read under the same lock
            name, proc = self.current, self._proc
            if name is None:
                return
            self.stop_requested = True
        self.emit(f"stopping '{name}'")
        if proc is None:  # still preparing or launching; _run sees the flag and doesn't start it
            return
        if os.name == "nt":  # kill the whole tree: Isaac runs under cmd /c python.bat
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
