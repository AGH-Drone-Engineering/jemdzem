# Jem Dżem

Image analysis service powered by Google Gemini. The project
exposes a small FastAPI backend with three endpoints:

* `/ocr` &ndash; extract text from an uploaded image
* `/multi-detect` &ndash; detect multiple object classes at once
* `/single-detect` &ndash; detect multiple object classes with individual Gemini calls, optionally using reference images

The examples located in `examples/` demonstrate how to call these endpoints.

## Requirements

* Python 3.12
* A Gemini API key from [Google&nbsp;AI Studio](https://aistudio.google.com/)
* [`uv`](https://github.com/astral-sh/uv) for dependency management (install
  with `pip install uv`)

## Installation

Install [`uv`](https://github.com/astral-sh/uv) if it is not already available:

```bash
pip install uv
```

Project dependencies are installed automatically the first time you run
any script with `uv run`.

### Linting

Code style is enforced with [Ruff](https://docs.astral.sh/ruff/). Run

```bash
uv run ruff check . --fix
```

before committing changes to automatically format and lint the project.

## Running the server

1. Export your Gemini API key
   ```bash
   export GOOGLE_API_KEY=<your-api-key>
   ```
2. Start the FastAPI server
   ```bash
   uv run uvicorn jemdzem.backend:app --reload
   ```

The API expects an `X-API-Key` header. For local development the default key is
`tym_razem_to_musi_poleciec`, which is used by the example scripts.

## Examples

Run the provided examples while the server is running:

```bash
./examples/ocr.sh           # OCR demo
uv run examples/single_detect.py  # single detector with optional reference image
uv run examples/multi_detect.py   # multi-class detector
```

## Repository layout

* `jemdzem/backend.py` &ndash; FastAPI application exposing the API
* `jemdzem/ai/` &ndash; wrappers around Gemini models for OCR and detection
* `jemdzem/api_utils.py` &ndash; helper utilities for image handling
* `jemdzem/auth.py` &ndash; simple API key authentication
