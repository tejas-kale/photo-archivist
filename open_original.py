import subprocess
from pathlib import Path


def open_original(path):
    subprocess.run(["open", "-R", str(path)], check=True)
