# photo-archivist — Development Notes

## What was done

photo-archivist is a Python 3.12+ CLI that archives images from OneDrive or local paths into SQLite + Markdown sidecars. Each image gets: vision description (structured JSON via Ollama), CLIP embeddings (optional), face detection + labelling, EXIF extraction, reverse geocoding, and a framedex-style sidecar.

### Unified CLI/UI planning (Session 8, 2 June)

- Added a two-mode shape: existing CLI behaviour remains stable, and a new FastAPI-backed UI exposes the same core backend functions through a lightweight HTML/CSS/JavaScript interface
- Added `archive_runner.py` so CLI and UI use one event-emitting archive backend instead of two archive loops
- Added `serve-ui` with Archive, Faces, and Search tabs; archive controls start with a subset only: source, image count, model, random/latest selection, and file-modified date range
- Archive selection v1 uses file modified time for "latest first" and time-period filtering; EXIF capture-time filtering is deferred because it requires slower metadata reads before the run
- Added `search.py` and `photo-archivist query`; search v1 is plain text over archived descriptions/sidecars/metadata, and semantic search is deferred until text-query embeddings and image-embedding backfill are designed
- Cancellation is deferred for v1; stopping cleanly needs cooperative job state in the archive runner, request-safe status updates, and defined behaviour for the currently processing image
- Replaced `serve-faces` and `serve-review` with `serve-ui`; the older UI modules remain in the repo while compatibility tests still cover their behaviour
- Renamed the browser app label to Photo Archiver, increased UI text size, centred the app at 80% viewport width, aligned archive controls with a responsive grid, made source/model dropdowns, and hid date fields unless Within period is selected
- Face UI actions now use a same-line toolbar with Refresh on the left and Save on the right, using matching button styling
- UI archive runs now generate CLIP embeddings by default, manage Ollama, restart it before the run, restart every 25 attempted images, use a 5s cooldown, and show Ollama lifecycle messages in the log stream
- Added `torch` as an explicit dependency so the default UI embedding path has the runtime required by Transformers CLIP

### Per-image face detection failure handling (Session 6, 30 May)

- Archive runs now catch `OSError` from face detection, print `⚠️ faces skipped <path>: <error>`, save the image with `face_count = 0`, and continue
- This covers corrupt/unsupported/transient OneDrive HEIC decode failures in Pillow/InsightFace

### Sidecar write failure handling (Session 6, 30 May)

- Archive runs now catch `OSError` from sidecar writes, print `⚠️ sidecar skipped <path>: <error>`, and continue
- This handles OneDrive timeout failures after DB save / embeddings / face storage have already succeeded
- Missing sidecars can be regenerated later with `refresh-sidecars`

### Face classifier minimum labels (Session 6, 30 May)

- `train-faces` now accepts `--min-labels N` and trains only on people with at least `N` labelled faces
- Default classifier prediction threshold is now 0.95, so UI suggestions appear only for high-confidence predictions
- The classifier pickle stores `min_labels` alongside labels, scaler, threshold, and normalisation metadata
- Classifier predictions are now UI suggestions only: the face UI prefills predicted names with confidence, but sidecars only write manually labelled names

### Random face labelling UI (Session 6, 30 May)

- Face UI now samples a random set of unlabelled faces on each page load instead of walking sequential IDs
- `POST /label` redirects back to `/` after Save All so the next view is another random set
- Faces without crop files are excluded from the grid to avoid broken `?` images resurfacing

### Overnight resource controls (Session 6, 30 May)

- `--embed` now uses `embed.embedding_blob_subprocess()` by default, invoking `python -m embed <image>` per image so PyTorch/CLIP memory exits with the worker process
- CLIP embedding now registers the HEIF opener for `.heic`/`.heif`, and archive runs skip failed per-image embeddings instead of aborting the run
- Added `--no-embed-subprocess` for faster in-process embeddings when memory pressure is acceptable
- Added `ollama_ctl.py` plus `--manage-ollama`, `--restart-ollama-every N`, and `--cooldown SECONDS` so long runs can restart Ollama every 20-25 images
- `--restart-ollama-every` requires `--manage-ollama` to avoid killing a manually managed Ollama server

### MLflow comparison retained; MLX experiment removed (Session 7, 1 June)

- Replaced `mlx_mlflow_experiment.py` with `mlflow_experiment.py`, keeping local MLflow tracking backed by SQLite (`sqlite:///mlflow.db`)
- The script still samples already processed `archive.db` rows and logs original image, existing sidecar/DB description, generated description, metadata, timings, and failures to MLflow
- Removed the `mlx-vlm` backend, `mlx-vlm` dependency, `MLX_VLM_MODEL`, `describe_mlx()`, MLX README instructions, and the old MLX-specific generated artefact name
- Rationale: the observed Unsloth UD 4-bit MLX run used roughly the same RAM/swap as Ollama Gemma 4 E2B, so it did not justify the extra backend/dependency surface

### Embedding backfill (Session 7, 1 June)

- Added `backfill-embeddings --db archive.db --limit N` to fill `media.embedding` for archived rows where embedding is null
- Embedding subprocess failures now include stderr/stdout so missing PyTorch is visible instead of only `CalledProcessError`
- Use `uv run photo-archivist backfill-embeddings` from the repo when the installed uv tool lacks PyTorch

### Photo review UI (Session 7, 31 May)

- Removed the model evaluation script, fixtures, and tests
- Added `reviewui.py` and `serve-review` for a view-only browser over `archive.db`
- Review UI shows up to three archived images per page, newest first, with original image and verbatim `.description.md` sidecar side-by-side in a scrollable panel
- Image routes call OneDrive `ensure_local()` before returning `FileResponse`, so cloud-backed originals are hydrated before display

### HEIC vision conversion (Session 5, 30 May)

- `describe.image_data()` now converts image payloads to RGB JPEG and resizes them within 1280x1280 before sending them to Ollama
- Ollama repeatedly failed on raw HEIC payloads and later consumed too much memory on full-resolution images; bounded JPEG conversion keeps source support without changing stored originals or sidecars

### Per-image description failure handling (Session 5, 30 May)

- Archive runs now catch exhausted description retries per image, print `⚠️ skipped <path>: <error>`, and continue to the next image
- The failed image is not saved to `archive.db` and no sidecar is written, because there is no `VisionResult` to persist
- Other unexpected failures still raise loudly

### Embeddings opt-in (Session 5, 30 May)

- `--embed/--no-embed` now defaults to `--no-embed`
- The installed tool does not include PyTorch, so the previous default crashed at `CLIPModel.from_pretrained()` after description/geocoding had already run
- Kept `--embed` for explicit CLIP runs when PyTorch is installed in the CLI environment
- Updated README troubleshooting for the PyTorch/CLIPModel error

### Random source limits (Session 5, 30 May)

- `--source ... --limit N` now samples `N` images randomly instead of taking the first `N` paths from filesystem traversal
- Sampling happens in `sources.onedrive.media()` after enumerating candidate image paths but before `ensure_local()`, so skipped OneDrive files are not read/downloaded
- `--image` remains deterministic and returns only the requested file
- Updated README to document random limit behaviour

### Removed Apple Photos; OneDrive-only (Session 4, 30 May)

- Dropped `sources/apple_photos.py` and `osxphotos` dependency
- Removed `--source photos` semantics, `open-photos` subcommand, and `db_path`/iCloud eviction plumbing
- `source_media()` now only resolves OneDrive default + local path / `--image`
- Collapsed `open_original.py` to just `open -R` (removed AppleScript Photos branch)
- `sidecar.py` always writes sidecar beside the image (`<image>.description.md`); no more `apple_photos/` subdirectory
- Deleted `tests/test_apple_photos.py`; stripped Photos branches from all tests

### Face crops + embedding normalisation (Session 4, 30 May)

- `detect_faces` now returns `(detections, image_array)` tuple so crops can be saved from the loaded RGB array
- Crop stored at `~/.photo-archivist/faces/<face_id>.jpg` with 15% padding clamped to image bounds
- Added `normalized()` helper that L2-normalises raw embedding bytes on read (DB stores raw float32)
- Added `backfill-crops` CLI subcommand: regenerates missing crops for existing `faces` rows, skips (warns) when source file is gone

### FastAPI face labelling UI (Session 4, 30 May)

- New module `faceui.py` — FastAPI app with Jinja2 template `templates/grid.html`
- `serve-faces` CLI subcommand (host/port options) launches uvicorn
- `GET /` — paginated grid of unlabelled faces with crop images + name inputs with autocomplete
- `POST /label` — batch form submit `{face_id: name}`; blank fields skipped
- `GET /faces/<id>.jpg` — serve crop JPEGs; `GET /names` — JSON list of existing names
- Added `fastapi`, `uvicorn`, `jinja2`, `python-multipart` to pyproject.toml

### Local face classifier (Session 4, 30 May)

- Added `train_faces()` and `predict_name()` in `faces.py`
- `train-faces` CLI subcommand trains a Logistic Regression classifier on L2-normalised labelled face embeddings
- Classifier persists to `~/.photo-archivist/face_classifier.pkl` with label list, threshold, and normalisation flag
- `predict_name()` returns `(None, confidence)` below threshold or when no model exists
- Uses `scikit-learn`; reviewed Phase 0/1 behaviour: `--source` now defaults to OneDrive, and re-running face storage recreates a missing crop for an existing face row

### Predicted names in sidecars (Session 4, 30 May)

- `name_for_face()` now returns manual labels first, then classifier predictions, then `None` if no model or below threshold
- `sidecar.py` writes `person_name`, `name_source`, and `confidence` per named face
- Added `refresh-sidecars [path]` CLI command to rewrite existing `.description.md` files with current labels/predictions
- Retained legacy `inferred_name()` as unused compatibility code for now; classifier is the active fallback

### Bootstrapping (Session 1, 28 May)

- Wired OpenRouter client in `describe.py` using httpx, then **replaced with Ollama** at user's request
- Verified osxphotos access to iCloud Photos library (reads UUIDs/paths without Photos.app open)
- Built Click CLI with emoji-logged progress (`🎲 🔎 🖼️ 🧠 📝 ✅`)
- Added `--preview` flag to open images in Preview.app during processing
- Changed from explicit image path to random Photos DB pick
- Added retry logic for empty/truncated Ollama responses (timeout=600s, num_predict=768)
- Set up GitHub repo, branch + PR workflow from the start

### Restructuring & Expansion (Session 2, 29 May)

- Created AGENTS.md, restructured monolith into modular layout:
  - `sources/` (base dataclass, apple_photos, onedrive adapters)
  - `archive.py` (Click CLI orchestration)
  - `describe.py` (vision inference), `embed.py` (CLIP embeddings)
  - `store.py` (SQLite via sqlite-utils), `sidecar.py` (Markdown+frontmatter)
  - `open_original.py` (AppleScript for Photos, `open -R` for local files)
- Made package installable via `uv` with `photo-archivist` entrypoint
- **Re-implemented entire project with TDD** after user called out lack of tests
- Added `--no-embed` flag (CLIP model download was 605MB, too slow)
- Added structured vision output: JSON with `rating` (keep/review/cull), `focus`, `exposure`, `depth_of_field`, `lighting`, `time_of_day`, `people_count`, `keywords`, `description_prose`, `activity`
- Fixed sidecar null values — model JSON wasn't being parsed into schema fields
- Added three new modules:
  - `metadata.py` — EXIF extraction via `exiftool -j -n` (subprocess, numeric GPS)
  - `geocode.py` — reverse geocoding via Nominatim (geopy) with 1.1s rate limit + SQLite cache
  - `faces.py` — face detection via insightface (buffalo_s model), plus labelling and similarity inference
- Updated sidecar to framedex-style: YAML frontmatter with resolution, camera, location, technical ratings, face bboxes, source metadata
- Added `--image` flag for single-image archiving, `open-photos` CLI command
- Added `label-face` subcommand for naming detected faces
- Inferred face labels: unlabelled faces get cosine-similarity-matched against labelled ones (threshold 0.7)

### iCloud Storage Constraints (Session 3, 29 May)

- Added `brctl download`/`brctl evict` support for pinning/releasing iCloud photos
- Stub detection: `is_stub()` checks path=None, size=0, xattr `com.apple.icloud.itemName`
- `ensure_local()` with polling loop and timeout (120s default)
- `evict_local()` and `evict_already_indexed()` for cleanup on startup
- `iter_photos()` generator with `evict_after=True` and try/finally
- **Critical fix**: `brctl download` fails for Photos library paths with `BRCloudDocsErrorDomain Code=6` ("Path is outside of any CloudDocs app library"). Fallback: `osxphotos.PhotoExporter` with `download_missing=True, use_photokit=True`
- Temporary exports tracked and cleaned up (`temporary_exports` set)
- `max_consecutive_download_failures=10` to bail out on persistent iCloud issues

---

## Decisions made and rationale

### Removed Apple Photos

| Decision | Rationale |
|---|---|
| Drop Apple Photos support entirely | OneDrive is the primary repository; Photos adds significant complexity (iCloud stubs, brctl, PhotoExporter, osxphotos) with diminishing value |
| Drop `osxphotos` dependency | Only used for Photos source; removing it simplifies the dependency tree |
| `source_media("photos")` raises ValueError | Clear feedback that Photos is unsupported rather than silently interpreting as a path |
| Drop `with_source_gps()` fallback chain | GPS passthrough from osxphotos metadata was the only consumer; EXIF is now the sole GPS source |
| `sidecar.py` always writes beside image | No more `~/.photo-archivist/sidecars/apple_photos/` subdirectory; sidecars always live at `<image>.description.md` |

### Face crop storage

| Decision | Rationale |
|---|---|
| `CROP_PADDING = 0.15` (15% of bbox width/height) | Enough to include jaw/hairline without excessive background; clamped to image bounds |
| Return `(detections, image_array)` from `detect_faces` | Avoids reading the file twice (once for insightface, once for crop); image is already in memory as RGB array |
| Crops at `~/.photo-archivist/faces/<face_id>.jpg` | No schema change needed; filename derived from primary key |
| Raw embedding stored in DB; `normalized()` helper for L2 | Keeps round-trip precision; `inferred_name()` already normalises via cosine, and the upcoming classifier will use `normalized()` consistently |
| No `crop_status` column | Unavailable-source info is logged during backfill; a column adds schema migration cost for minimal value |

### Local face classifier

| Decision | Rationale |
|---|---|
| Logistic Regression as first persisted model | Small, deterministic, probability output is enough for a local labelling loop |
| Train on L2-normalised embeddings | Keeps classifier input consistent with cosine matching and future prototype comparisons |
| Default abstain threshold `0.7` | Rejects weak binary decisions (~0.5) while accepting clear synthetic held-out matches; tune after real labels accumulate |
| Persist pickle with model, labels, threshold, and normalisation flag | Single local artefact is easy to retrain and inspect; metadata avoids guessing preprocessing later |
| Raise on fewer than two labelled faces/classes | A classifier trained on one class cannot make useful name decisions |

### Predicted names in sidecars

| Decision | Rationale |
|---|---|
| Manual labels override model predictions | `face_labels` is ground truth; classifier predictions are only a fallback |
| Sidecars include `name_source` and `confidence` | Lets stale or weak predictions be audited without opening the DB |
| Add `refresh-sidecars [path]` now | Retraining should update existing `.description.md` files, not just newly archived images |
| Leave missing model as `None` | Sidecar generation should not crash just because the classifier has not been trained yet |

### Vision backend: Ollama, not OpenRouter

| Decision | Rationale |
|---|---|
| Ollama (`gemma4:e2b`) as primary backend | User explicitly rejected OpenRouter; local, no API key, faster iteration; switched from `gemma4:e4b` for lower load |
| `DEFAULT_BACKEND = "ollama"` | User's explicit choice; only supported vision backend after the MLX experiment was removed |
| Remove `mlx-vlm` backend and dependency | Unsloth UD 4-bit MLX did not materially reduce observed RAM/swap versus Ollama Gemma 4 E2B, and keeping another backend adds test/docs/dependency load |
| Keep MLflow experiment tracking | MLflow remains useful for comparing generated descriptions, timings, failures, and artefacts even without MLX |

### Structured vision output

| Decision | Rationale |
|---|---|
| JSON schema with `rating` (keep/review/cull) | Primary use case: culling a photo library |
| 768 output tokens (`num_predict`) | Enough for full JSON; Ollama `"format": "json"` flag enforces structure |
| Fallback to plain text on parse failure | Ollama occasionally returns non-JSON despite format flag |
| Retries with progressively stricter prompts | First attempt: standard prompt. Second: "Return valid JSON only." Third: original prompt |

### CLIP embeddings: optional (`--embed`)

| Decision | Rationale |
|---|---|
| `openai/clip-vit-base-patch32` via transformers | Standard 512-dim embeddings for similarity search |
| `--embed/--no-embed` flag (CLI default: no-embed) | CLI stays gentler by default; UI enables embeddings by default for richer archive records |
| User hit missing PyTorch after the previous default tried CLIP | PyTorch dependency is heavy and unnecessary for simple archiving |

### Metadata extraction

| Decision | Rationale |
|---|---|
| `exiftool -j -n` via subprocess | De facto standard for EXIF; `-n` gives numeric GPS (not DMS) |
| `PhotoMetadata` frozen dataclass | Immutable, hashable, explicit nullable fields |
| `DateTimeOriginal` → fallback `file mtime` | Some images lack EXIF dates; mtime is the best available proxy |
| GPS passthrough from osxphotos metadata | Derivative paths lack EXIF GPS tags; osxphotos stores GPS on PhotoInfo directly |
| EXIF GPS takes priority over osxphotos GPS | EXIF is ground truth; osxphotos metadata is only used as fallback |

### Geocoding

| Decision | Rationale |
|---|---|
| Nominatim (OpenStreetMap) via geopy | Free, no API key, good enough for personal archive |
| 1.1s rate limit between calls | Nominatim's fair use policy |
| SQLite cache (`geocode_cache.db`) | GPS coords repeat across photos taken at same location |
| Rounding to 4 decimal places (~11m precision) | Good enough for place-level geocoding; increases cache hit rate |

### Face detection

| Decision | Rationale |
|---|---|
| insightface `buffalo_s` (ONNX, CPU) | Open-source, no GPU needed, good accuracy for personal photo libraries |
| `pillow-heif` for HEIC support | Apple Photos uses HEIC extensively |
| Face embeddings stored as float32 blobs in `faces.db` | Enables cosine similarity matching without re-detection |
| Cosine similarity ≥ 0.7 for inferred labels | Threshold found to balance precision/recall for family photos |
| Unique index on `(source, source_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2)` | Prevents duplicate face records when re-running on same image |

### iCloud storage strategy

| Decision | Rationale |
|---|---|
| `brctl download` first, then `PhotoExporter` fallback | brctl is faster when it works; PhotoExporter is reliable but uses temp space |
| Evict after each photo (`evict_after=True`) | MacBook Air has limited SSD; can't keep entire library pinned |
| Startup cleanup via `evict_already_indexed()` | Recovers from crashes without leaving GBs of pinned files |
| `max_consecutive_download_failures=10` | Prevents infinite loop on persistent iCloud errors |
| Temporary exports via `tempfile.mkdtemp` + manual cleanup | PhotoExporter creates copies; must track them separately from brctl files |

### Sidecar format

| Decision | Rationale |
|---|---|
| Framedex-aligned YAML frontmatter | User specified `github.com/Simbastack-hq/framedex` as reference schema |
| Markdown file alongside image or in `~/.photo-archivist/sidecars/` | Apple Photos doesn't own the derivative directory; need central store |
| Source type in sidecar path: `apple_photos/` vs local | Prevents filename collisions across sources |
| Face bboxes in sidecar with quality + inferred names | Makes sidecar self-contained for browsing without querying faces.db |

### Unified CLI/UI architecture

| Decision | Rationale |
|---|---|
| Keep the CLI as the stable default mode | Existing command behaviour and flags are already documented and tested; UI work should not regress batch archiving |
| Extract the archive loop into a shared runner that emits events | CLI can print events while FastAPI can expose progress/log state; avoids two archive implementations |
| Preselect candidate paths before processing | Gives the UI a real total for progress bars and supports random/latest/time-period selection; costs one source scan up front, but avoids OneDrive hydration until each image is actually read |
| Use file modified time for latest/time-period filters | Fast and available for unarchived images; less semantically precise than EXIF capture time |
| Start with one archive job at a time | Avoids SQLite locks, Ollama contention, InsightFace memory pressure, and an unusable laptop |
| Use simple polling or SSE for progress/logs | Keeps the frontend small and responsive; polling is easiest, SSE is cleaner for streaming logs |
| Use plain text search first | Smallest shared CLI/UI feature; semantic search needs text embeddings, ranking design, and fallback/backfill behaviour |
| Serve the UI on `127.0.0.1` by default | The app serves local original images, so LAN exposure without auth is unsafe |
| Replace `serve-faces`/`serve-review` with `serve-ui` | One local web surface is simpler than separate servers; old modules remain until the unified routes have enough real-use soak time |

### Future UI/search improvements

| Improvement | Trade-off |
|---|---|
| Cooperative cancellation | Needs shared job state and clear stop points; current-image processing may still finish before stopping |
| Semantic search | Requires CLIP/text embedding support, optional image embeddings, ranking tests, and a backfill path |
| EXIF capture-time filtering | More accurate photo chronology; slower because metadata must be read before selecting candidates |
| Thumbnail cache | Faster grids and lower memory/bandwidth; adds cache invalidation and storage management |
| Retry failed images from the UI | Useful for transient OneDrive/Ollama failures; requires persisted failure state |
| Broader archive controls | More parity with CLI; risks turning the first UI into a dense settings panel |

### Architecture patterns

| Decision | Rationale |
|---|---|
| `handle(url, config) -> tuple[dict, str]` handler convention | AGENTS.md mandate: no base classes, no decorators, just plain functions |
| `SourceMedia` frozen dataclass | Immutable passthrough from source adapters to archive pipeline |
| sqlite-utils for schema management | Auto-creates tables, handles migrations, simpler than raw sqlite3 |
| Click for CLI | User explicitly requested "based on click" |
| British English in docstrings | AGENTS.md convention |
| Red-green TDD | AGENTS.md mandate; user enforced by making re-implement entire project |

---

## Evidence and alternatives explored

### Ollama vs OpenRouter vs MLX

- **OpenRouter** was implemented first (httpx to `/api/v1/chat/completions`). User said "Remove Openrouter configuration and add support for Ollama instead."
- **Ollama** worked immediately with `gemma4:e4b`, then defaulted to `gemma4:e2b` to reduce load. On an M1 MacBook Air, `gemma4:e2b` processes each image noticeably faster and avoids making the machine unusable; output quality still needs comparison against `gemma4:e4b`. Occasional empty responses fixed with retries. JSON format flag works most of the time but not always.
- **MLX / Unsloth UD 4-bit** looked promising because 4-bit weights should shrink the model files, but the observed runtime memory was still about 4GB RAM plus 6-8GB swap, similar to Ollama Gemma 4 E2B.
- Explanation: 4-bit quantisation mainly shrinks weights, not total inference memory. KV cache, activations, prompt/image buffers, tokenizer/runtime allocations, and Metal unified-memory buffers still count. Unsloth Dynamic quantisation also keeps precision-sensitive tensors above 4-bit for quality. Ollama's model is already quantised and efficient, and macOS swap is system-wide/sticky after memory pressure.
- Decision: remove the MLX backend/experiment code and keep the generic MLflow comparison path for future Ollama model evaluations.

### brctl download failures

- `brctl download` produces `BRCloudDocsErrorDomain Code=6 "Path is outside of any CloudDocs app library"` for all Photos library paths. This is expected: the Photos library is managed by `bird`/`cloudphotosd`, not `brctl`.
- **Solution**: detect Photos library paths with `is_photos_library_path()` and use `osxphotos.PhotoExporter(download_missing=True, use_photokit=True)` instead.
- PhotoExporter creates temp copies — tracked in `temporary_exports` set for proper cleanup.

### Face embedding deduplication

- Initial implementation treated each detection as unique — same person got different face_embedding_ids.
- **Fix**: added `inferred_name()` that cosine-similarity-matches unlabelled embeddings against all labelled ones. Threshold 0.7 was chosen after testing with family photos.
- Alternative considered: DBSCAN clustering of all embeddings. Rejected as overengineered for personal library scale.

### Location metadata gap

- Initial `metadata.py` only extracted GPS from EXIF via exiftool. Many iCloud/Apple Photos images have GPS in osxphotos metadata but not in derivative file EXIF.
- **Fix**: `with_source_gps()` in archive.py fallback-chain: EXIF first, then `media.metadata.get("gps_lat")` from osxphotos.
- Evidence: User ran CLI, got no location. Checked with exiftool on derivative path — no GPS tags. osxphotos `PhotoInfo.latitude` had the value.

### CLIP model download time

- First run: 605MB `pytorch_model.bin` download + weights loading (39s for model, 5s for processor).
- User asked "Why is Pytorch used?" → added `--no-embed` flag.
- Alternative considered: mlx-embedding-models (MLX-native, no PyTorch). Rejected because CLIP specifically was required for image similarity search.
