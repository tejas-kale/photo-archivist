import argparse
import json
import shutil
import sqlite3
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import mlflow

import describe
from sources import onedrive


def sidecar_path(image):
    image = Path(image)
    return image.with_name(f"{image.stem}.description.md")


def processed_images(db_path, limit):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con.execute(
        "select id, original_path, description, indexed_at from media where original_path is not null and description is not null order by random() limit ?",
        (limit,),
    ).fetchall()


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def log_image(row, model, root):
    image = onedrive.ensure_local(Path(row["original_path"]))
    existing = sidecar_path(image)
    start = time.monotonic()
    result = describe.describe(image, backend="mlx-vlm", model=model, retries=0)
    seconds = time.monotonic() - start
    out = root / str(row["id"])
    new_sidecar = write_text(out / "mlx.description.md", "---\n" + json.dumps({"model": model, "source_id": row["id"]}, indent=2) + "\n---\n\n## Description\n" + result.description_prose + "\n")
    existing_copy = write_text(out / "existing.description.md", existing.read_text() if existing.exists() else row["description"] or "")
    metadata = write_text(out / "metadata.json", json.dumps({"id": row["id"], "image": str(image), "indexed_at": row["indexed_at"], "seconds": seconds, "mlx": asdict(result)}, indent=2, default=str))
    image_copy = out / image.name
    shutil.copy2(image, image_copy)
    mlflow.log_artifacts(out, artifact_path=f"images/{row['id']}")
    return seconds


def run(db_path="archive.db", limit=50, model=describe.MLX_MODEL, tracking_uri="sqlite:///mlflow.db", experiment="mlx-description-comparison"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    rows = processed_images(db_path, limit)
    with tempfile.TemporaryDirectory() as d, mlflow.start_run(run_name=f"{model}-{limit}"):
        mlflow.log_params({"model": model, "db_path": str(db_path), "limit": limit, "backend": "mlx-vlm"})
        ok = 0
        failed = 0
        total_seconds = 0.0
        root = Path(d)
        for row in rows:
            try:
                total_seconds += log_image(row, model, root)
                ok += 1
            except Exception as e:
                failed += 1
                write_text(root / str(row["id"]) / "error.txt", str(e))
                mlflow.log_artifacts(root / str(row["id"]), artifact_path=f"images/{row['id']}")
        mlflow.log_metrics({"images_ok": ok, "images_failed": failed, "total_seconds": total_seconds})
        return ok, failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="archive.db")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--model", default=describe.MLX_MODEL)
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="mlx-description-comparison")
    args = p.parse_args()
    ok, failed = run(args.db, args.limit, args.model, args.tracking_uri, args.experiment)
    print(f"logged {ok} images, {failed} failures")


if __name__ == "__main__":
    main()
