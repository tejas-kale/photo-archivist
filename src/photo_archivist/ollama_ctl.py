import subprocess
import time

import httpx


def stop():
    subprocess.run(["pkill", "-f", "ollama serve"], check=False)


def wait(timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get("http://localhost:11434/api/tags", timeout=2).raise_for_status()
            return
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError("Ollama did not start")


def restart(cooldown=0):
    stop()
    proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait()
    time.sleep(cooldown)
    return proc
