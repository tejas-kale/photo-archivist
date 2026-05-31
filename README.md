# photo-archivist

`photo-archivist` is a local CLI for archiving images from OneDrive or a local folder. It writes a SQLite catalogue, Markdown sidecars, image descriptions, optional embeddings, EXIF metadata, reverse geocoding, and face embeddings for later labelling.

It is built for personal photo-library triage, not as a lightweight viewer. Vision, CLIP, and face detection are CPU/GPU intensive.

## Requirements

- macOS
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- `exiftool` for EXIF metadata
- Ollama for the default vision backend

Install system tools:

```bash
brew install exiftool ollama uv
```

Start Ollama before archiving:

```bash
ollama serve
```

The default model is `gemma4:e2b`. Pull it if needed:

```bash
ollama pull gemma4:e2b
```

## Install the CLI

From the project checkout:

```bash
uv tool install --reinstall .
```

Check it is available:

```bash
photo-archivist --help
```

During development, this also works without installing:

```bash
uv run photo-archivist --help
```

## What gets written

For each archived image, the tool can write:

- `archive.db` in the current directory, unless `--db` is set
- `<image-stem>.description.md` beside the image
- face records in `~/.photo-archivist/faces.db`
- face crop JPEGs in `~/.photo-archivist/faces/`
- a geocoding cache in `~/.photo-archivist/`
- a trained face classifier at `~/.photo-archivist/face_classifier.pkl`

Sidecars contain YAML frontmatter plus a short description.

## Archive images

Archive from the default OneDrive folder:

```bash
photo-archivist --source onedrive
```

The default OneDrive folder is:

```text
~/Library/CloudStorage/OneDrive-Personal/tejas/Pictures
```

Archive a local folder:

```bash
photo-archivist --source ~/Pictures/export
```

Archive one image:

```bash
photo-archivist --image ~/Pictures/example.jpg
```

Limit a run. When `--limit` is used with a source, the tool chooses a random sample instead of taking the first files in folder order:

```bash
photo-archivist --source onedrive --limit 10
```

Show more progress:

```bash
photo-archivist --image ~/Pictures/example.jpg --verbose
```

Open each image in Preview while processing:

```bash
photo-archivist --source onedrive --limit 5 --preview
```

## Reduce load

The full pipeline is expensive. Embeddings are off by default because CLIP currently needs PyTorch, which is not installed with the tool. For a gentler run, disable other parts you do not need:

```bash
photo-archivist --image ~/Pictures/example.jpg --no-faces --no-geocode
```

Useful flags:

- `--embed`: enables CLIP whole-image embeddings. By default each embedding runs in a subprocess so PyTorch memory is released after each image.
- `--no-embed-subprocess`: keeps CLIP in the main process; faster, but unsafe for long runs on low-memory machines.
- `--no-faces`: skips InsightFace detection
- `--no-geocode`: skips reverse geocoding
- `--no-sidecar`: skips Markdown sidecar writing

Avoid running multiple archive commands at once. If a run makes the laptop unusable, stop only the archiver:

```bash
pkill -f 'photo-archivist --image'
pkill -f 'photo-archivist --source'
```

Stop Ollama only if needed:

```bash
pkill -f 'ollama serve'
```

For overnight runs, let the CLI own Ollama and restart it periodically:

```bash
photo-archivist \
  --source onedrive \
  --limit 500 \
  --verbose \
  --embed \
  --manage-ollama \
  --restart-ollama-every 25 \
  --cooldown 5
```

`--restart-ollama-every` requires `--manage-ollama`, because otherwise the CLI would kill a server it did not start.

## Vision backends

Default backend:

```bash
photo-archivist --image ~/Pictures/example.jpg --backend ollama
```

Use another Ollama model:

```bash
photo-archivist --image ~/Pictures/example.jpg --model llava:latest
```

Environment variables:

```bash
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=gemma4:e2b
export VISION_BACKEND=ollama
```

MLX-VLM is available as an alternative backend:

```bash
photo-archivist --image ~/Pictures/example.jpg --backend mlx-vlm
```

Optional model override:

```bash
export MLX_VLM_MODEL=mlx-community/Qwen3.5-VL-9B-Instruct-4bit
```

Compare Ollama model quality on a fixed image set without archiving anything:

```bash
uv run python model_eval.py --images eval/model_quality_images.txt --models gemma4:e2b gemma4:e4b --out eval/model_quality_results
```

This writes:

- `eval/model_quality_results_blind.csv` — review this; model names are hidden behind variants `A`, `B`, ...
- `eval/model_quality_results_blind.jsonl`
- `eval/model_quality_results_feedback_template.csv` — fill your labels and scores here
- `eval/model_quality_results_key.csv` — open only after scoring to reveal model names
- `eval/model_quality_results_raw.csv` and `.jsonl` — unblinded raw data for analysis

Use `eval/model_quality_rubric.md` for scoring guidance. Static template examples are in `eval/model_quality_feedback_template.csv` and `.json`.

## Face labelling

Face detection stores embeddings in:

```text
~/.photo-archivist/faces.db
```

Crops are stored in:

```text
~/.photo-archivist/faces/<face_id>.jpg
```

Open the labelling UI:

```bash
photo-archivist serve-faces
```

Then open:

```text
http://127.0.0.1:8714/
```

Use a different port:

```bash
photo-archivist serve-faces --port 8715
```

Label one face from the CLI:

```bash
photo-archivist label-face 24 Tejas
```

Train the local classifier after labelling at least two faces across at least two names:

```bash
photo-archivist train-faces
```

Train only on people with enough examples:

```bash
photo-archivist train-faces --min-labels 30
```

After training, sidecars can include predicted names (`name_source: predicted`) when confidence exceeds the classifier threshold. The current UI still shows blank inputs; it does not yet prefill or sort by predicted names.

Refresh existing sidecars after training or relabelling:

```bash
photo-archivist refresh-sidecars ~/Library/CloudStorage/OneDrive-Personal/tejas/Pictures
```

Backfill missing face crops:

```bash
photo-archivist backfill-crops
```

Backfill only works when `faces.db.source_id` points to a file that still exists. Older rows from removed Photos/iCloud flows may not have usable paths.

The UI shows a random set of unlabelled faces each time the page loads. After **Save All**, labelled faces are persisted and the browser redirects to a fresh random set. Faces without crop files are hidden, so broken `?` images do not keep resurfacing.

### Why are face crops low resolution?

The UI shows detected face crops, not full photos. Crops are saved from the face bounding box with 15% padding. If the face is small in the source image, the crop will be small too.

Open the original image from Finder when you need more context.

## Sidecars

A sidecar is written beside each image:

```text
IMG_1234.description.md
```

It includes:

- original path and parent folder
- resolution and file size
- EXIF camera metadata
- GPS and reverse-geocoded place when available
- rating and technical assessment
- description and keywords
- face bounding boxes and labels or predictions

Refresh sidecars after changing labels:

```bash
photo-archivist refresh-sidecars /path/to/photos
```

## SQLite catalogue

The default catalogue is:

```text
archive.db
```

Use another database:

```bash
photo-archivist --db ~/photo-archive.db --source onedrive --limit 10
```

Inspect it:

```bash
sqlite3 archive.db '.tables'
sqlite3 archive.db 'select indexed_at, original_path, description from media order by indexed_at desc limit 5;'
```

## Troubleshooting

### `RuntimeError: No description after retries`

Usually Ollama is not running or the model is unavailable.

```bash
ollama serve
ollama pull gemma4:e2b
```

During source runs, images that exhaust description retries are skipped and the next image is attempted.

### The face UI shows question marks or broken images

The face rows exist, but crop JPEGs are missing.

```bash
photo-archivist backfill-crops
```

If backfill skips rows, the original source files are unavailable.

### `templates/grid.html` missing after `uv tool install`

The current package configuration does not include template files in the installed wheel. Run the server from the checkout with `PYTHONPATH`:

```bash
cd /path/to/photo-archivist
PYTHONPATH=$PWD photo-archivist serve-faces
```

### `--source photos` fails

Apple Photos support was removed. Use OneDrive or a local path.

```bash
photo-archivist --source onedrive
photo-archivist --source ~/Pictures/export
```

### `CLIPModel requires the PyTorch library`

Embeddings were requested but PyTorch is not installed in the CLI environment. Omit `--embed`, or install PyTorch into the tool environment before using embeddings. `--embed` runs CLIP in a subprocess by default to reduce long-run swap growth, but the subprocess still needs PyTorch installed.

### HEIC files fail to open

Images are converted to bounded JPEG payloads before vision requests; HEIC/HEIF files are supported for both vision and CLIP embeddings. If embedding a specific image fails, the image is still archived with a missing embedding. Make sure dependencies are installed from the project and use the CLI environment created by `uv`:

```bash
uv tool install --reinstall .
```

## Development

Run tests:

```bash
uv run python -m unittest discover -s tests
```

Run the CLI from source:

```bash
uv run photo-archivist --help
```

Project layout:

- `archive.py`: CLI orchestration
- `sources/`: OneDrive and local path source handling
- `metadata.py`: EXIF extraction
- `geocode.py`: reverse geocoding
- `describe.py`: vision inference
- `embed.py`: CLIP embeddings
- `faces.py`: face detection, crops, labels, classifier
- `faceui.py`: FastAPI face labelling UI
- `store.py`: SQLite catalogue
- `sidecar.py`: Markdown sidecars
- `open_original.py`: Finder helpers
