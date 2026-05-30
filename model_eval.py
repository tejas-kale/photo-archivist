import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import describe


RAW_FIELDS = ["image", "model", "seconds", "ok", "error", "rating", "people_count", "time_of_day", "lighting", "activity", "description_prose"]
BLIND_FIELDS = ["image", "variant", "seconds", "ok", "error", "rating", "people_count", "time_of_day", "lighting", "activity", "description_prose"]
KEY_FIELDS = ["image", "variant", "model"]
FEEDBACK_FIELDS = ["image", "variant", "correct_people_count", "correct_rating", "correct_time_of_day", "correct_lighting", "correct_activity", "people_count_score", "description_score", "activity_score", "lighting_time_score", "rating_score", "json_score", "notes"]


def images(path):
    return [Path(line.strip()).expanduser() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]


def evaluate(image, model, retries):
    start = time.monotonic()
    try:
        result = describe.describe(image, backend="ollama", model=model, retries=retries)
        row = asdict(result)
        row.update({"image": str(image), "model": model, "seconds": round(time.monotonic() - start, 3), "ok": True, "error": ""})
        return row
    except Exception as e:
        return {"image": str(image), "model": model, "seconds": round(time.monotonic() - start, 3), "ok": False, "error": str(e)}


def blind_rows(rows):
    blind = []
    key = []
    grouped = {}
    for row in rows:
        grouped.setdefault(row["image"], []).append(row)
    for image in grouped:
        variants = grouped[image][:]
        random.shuffle(variants)
        for i, row in enumerate(variants):
            variant = chr(ord("A") + i)
            visible = {k: row.get(k, "") for k in BLIND_FIELDS if k != "variant"}
            visible["variant"] = variant
            blind.append(visible)
            key.append({"image": image, "variant": variant, "model": row["model"]})
    return blind, key


def feedback_rows(blind):
    return [{field: row.get(field, "") if field in {"image", "variant"} else "" for field in FEEDBACK_FIELDS} for row in blind]


def write_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blind, key = blind_rows(rows)
    write_csv(out.with_name(out.name + "_raw.csv"), RAW_FIELDS, rows)
    write_jsonl(out.with_name(out.name + "_raw.jsonl"), rows)
    write_csv(out.with_name(out.name + "_blind.csv"), BLIND_FIELDS, blind)
    write_jsonl(out.with_name(out.name + "_blind.jsonl"), blind)
    write_csv(out.with_name(out.name + "_key.csv"), KEY_FIELDS, key)
    write_csv(out.with_name(out.name + "_feedback_template.csv"), FEEDBACK_FIELDS, feedback_rows(blind))


def run(image_list, models, out, retries):
    rows = []
    for image in images(image_list):
        for model in models:
            rows.append(evaluate(image, model, retries))
    write_outputs(rows, out)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="eval/model_quality_images.txt")
    p.add_argument("--models", nargs="+", default=["gemma4:e2b", "gemma4:e4b"])
    p.add_argument("--out", default="eval/model_quality_results")
    p.add_argument("--retries", type=int, default=0)
    args = p.parse_args()
    run(args.images, args.models, args.out, args.retries)


if __name__ == "__main__":
    main()
