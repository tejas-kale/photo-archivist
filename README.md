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

The full pipeline is expensive. CLI embeddings are off by default, while UI archive runs generate embeddings by default. For a gentler CLI run, disable other parts you do not need:

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

## MLflow description comparison

Run a local MLflow experiment that re-describes already archived images without overwriting existing sidecars:

```bash
uv run python -m photo_archivist.experiments.mlflow_experiment --limit 50
```

For each sampled image, MLflow logs:

- the original image
- the existing `.description.md` content, or DB description if the sidecar is missing
- the new `generated.description.md`
- metadata including timing and structured output

Start the MLflow UI:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000/`.

## Web UI

Open the unified local UI, shown as **Photo Archiver**:

```bash
photo-archivist serve-ui
```

Then open:

```text
http://127.0.0.1:8714/
```

The UI has three tabs:

- Archive: runs the same backend archive pipeline as the CLI, with controls for source, image count, model, random/latest selection, and file-modified date range
- Faces: shows a random set of unlabelled face crops and saves labels
- Search: plain-text searches archived descriptions/sidecars and shows matching images

Use another database or port:

```bash
photo-archivist serve-ui --db ~/photo-archive.db --port 8720
```

Only one archive job can run at a time. The UI generates CLIP embeddings by default, manages Ollama for archive runs, restarts it before the run, and restarts it every 25 attempted images with a 5s cooldown. The UI uses file modified time for latest/date-range selection. Image routes call OneDrive hydration before serving the original file, so the browser receives the available high-quality original rather than a thumbnail.

## Search descriptions

Search from the CLI:

```bash
photo-archivist query "beach sunset" --db archive.db --limit 20
```

The first version is plain text search over database descriptions, sidecar text when present, activity, place, and original path. Semantic search is planned later.

## Face labelling

Face detection stores embeddings in:

```text
~/.photo-archivist/faces.db
```

Crops are stored in:

```text
~/.photo-archivist/faces/<face_id>.jpg
```

Label faces from the Web UI, or label one face from the CLI:

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

After training, the face UI prefills classifier predictions with a confidence tooltip. Predictions require at least 95% classifier confidence by default. Predictions are not written to sidecars. Only labels you save in the UI or via `label-face` are treated as approved and written to sidecars.

Refresh existing sidecars after training or relabelling:

```bash
photo-archivist refresh-sidecars ~/Library/CloudStorage/OneDrive-Personal/tejas/Pictures
```

Backfill missing CLIP embeddings:

```bash
uv run photo-archivist backfill-embeddings --limit 100
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

Refresh sidecars after changing labels or after OneDrive timed out while writing a sidecar:

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

### `--source photos` fails

Apple Photos support was removed. Use OneDrive or a local path.

```bash
photo-archivist --source onedrive
photo-archivist --source ~/Pictures/export
```

### `CLIPModel requires the PyTorch library`

PyTorch is an explicit dependency because the UI generates embeddings by default. If an older installed tool still reports this error, reinstall it:

```bash
uv tool install --reinstall .
```

### HEIC files fail to open

Images are converted to bounded JPEG payloads before vision requests; HEIC/HEIF files are supported for both vision and CLIP embeddings. If embedding or face detection fails for a specific image, the image is still archived with the missing field. Make sure dependencies are installed from the project and use the CLI environment created by `uv`:

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

- `src/photo_archivist/cli.py`: CLI entrypoints
- `src/photo_archivist/archive_runner.py`: shared archive pipeline
- `src/photo_archivist/sources/`: OneDrive and local path source handling
- `src/photo_archivist/metadata.py`: EXIF extraction
- `src/photo_archivist/geocode.py`: reverse geocoding
- `src/photo_archivist/describe.py`: vision inference
- `src/photo_archivist/embed.py`: CLIP embeddings
- `src/photo_archivist/faces.py`: face detection, crops, labels, classifier
- `src/photo_archivist/web/app.py`: unified FastAPI UI
- `src/photo_archivist/search.py`: shared plain-text search
- `src/photo_archivist/store.py`: SQLite catalogue
- `src/photo_archivist/sidecar.py`: Markdown sidecars
- `src/photo_archivist/open_original.py`: Finder helpers
- `src/photo_archivist/experiments/`: local experiment scripts
