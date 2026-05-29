import subprocess


def open_original(source, source_id, path):
    if source == "photos":
        script = f'tell application "Photos" to spotlight id "{source_id}"'
        subprocess.run(["osascript", "-e", script], check=True)
        return
    subprocess.run(["open", "-R", str(path)], check=True)
