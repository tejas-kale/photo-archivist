# AGENTS.md

- `photo-archivist` is a Python 3.10+ CLI for archiving images from Apple Photos, OneDrive, or a local path into SQLite, sidecars, descriptions, and embeddings.
- Use `uv` for development. Run tests with `uv run python -m unittest discover -s tests`. Run the CLI with `uv run photo-archivist --source photos|onedrive|<path>`. Add dependencies in `pyproject.toml`.
- CLI orchestration lives in `archive.py`. Source adapters live in `sources/`. EXIF extraction lives in `metadata.py`, reverse geocoding in `geocode.py`, face embeddings in `faces.py`, vision inference in `describe.py`, CLIP embeddings in `embed.py`, SQLite persistence in `store.py`, sidecars in `sidecar.py`, and original-opening helpers in `open_original.py`.
- Develop Red Green TDD: write a failing test first, make it pass with the smallest change, then refactor only if it improves clarity.
- Write terse code. Prefer direct functions and explicit branches. Avoid unnecessary abstractions, base classes, registries, decorators, and framework-shaped designs.
- Let code fail loudly. Do not wrap broad `try/except`; only catch errors when adding clear user-facing value.
- Keep CLI output and option names stable. Any user-visible flag, environment variable, model default, prompt, or workflow change needs documentation.
- Treat source selection, Photos library access, OneDrive path handling, image paths, EXIF extraction, geocoding, face detection, Ollama/mlx-vlm requests, embeddings, SQLite rows, sidecars, and open-original behaviour as behaviours that need tests when touched.
- Keep config strict. Missing required values should raise rather than silently invent defaults.
- Make atomic commits: one coherent change per commit, with tests/docs included when relevant.
- Do not add comments unless they explain a non-obvious constraint.
