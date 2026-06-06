import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="eval_upload")
parser.add_argument("--output", default="eval_out")
parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
parser.add_argument("--max-new-tokens", default=512, type=int)
parser.add_argument("--max-pixels", default=1280 * 1280, type=int)
parser.add_argument("--load-in-4bit", action="store_true")
parser.add_argument("--attempts", default=3, type=int)
args = parser.parse_args()

root = Path(args.input).expanduser()
outroot = Path(args.output).expanduser()
outroot.mkdir(parents=True, exist_ok=True)
manifest = json.loads((root / "manifest.json").read_text())

kwargs = {"device_map": "auto"}
if args.load_in_4bit:
    from transformers import BitsAndBytesConfig
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
else:
    kwargs["torch_dtype"] = "auto"
model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

prompt = """Create draft golden labels for evaluating a personal photo archive vision model.

Return strict JSON only:
{
  "rating": "keep|review|cull",
  "cull_reason": "",
  "focus": "sharp|acceptable|soft",
  "exposure": "strong|adequate|poor|clipped",
  "depth_of_field": "shallow|standard|deep",
  "noise": "clean|some|heavy",
  "lighting": "",
  "time_of_day": "",
  "dominant_color_palette": "",
  "dominant_colors": [],
  "people_count": 0,
  "keywords": [],
  "description_prose": "",
  "activity": "",
  "source": "qwen3.5-9b-gcp-draft"
}

Use generic people terms only. Do not identify names. Do not include private addresses, phone numbers, IDs, or full document text."""


def strict_json(text):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object", text, 0)
    text = text[start:end + 1]
    text = text.replace(",\n}", "\n}").replace(",\n]", "\n]")
    return json.loads(text)


def fallback(row, raw, error):
    return {
        "rating": "review",
        "cull_reason": "",
        "focus": "acceptable",
        "exposure": "adequate",
        "depth_of_field": "standard",
        "noise": "clean",
        "lighting": "unknown",
        "time_of_day": "unknown",
        "dominant_color_palette": "unknown",
        "dominant_colors": [],
        "people_count": 0,
        "keywords": [row["category"]],
        "description_prose": "",
        "activity": "unknown",
        "draft_parse_error": str(error),
        "draft_raw": raw,
    }


def generate(image, extra=""):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt + extra}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated = generated[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


for row in manifest:
    image_path = root / row["file"]
    outdir = outroot / row["category"]
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{row['id']}.json"
    if dest.exists():
        print(row["id"], "exists", flush=True)
        continue
    image = Image.open(image_path).convert("RGB")
    raw = ""
    data = None
    error = None
    for attempt in range(args.attempts):
        extra = "\nReturn only a complete JSON object. No prose, no markdown, no trailing commas." if attempt else ""
        raw = generate(image, extra)
        try:
            data = strict_json(raw)
            break
        except json.JSONDecodeError as e:
            error = e
            print(row["id"], "parse-failed", attempt + 1, e, flush=True)
    if data is None:
        data = fallback(row, raw, error)
    data["category"] = row["category"]
    data["candidate_id"] = row["id"]
    data["original_path"] = row["original_path"]
    data["source"] = f"{args.model}-gcp-draft"
    dest.write_text(json.dumps(data, indent=2))
    print(row["id"], flush=True)
