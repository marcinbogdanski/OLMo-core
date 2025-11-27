# Repository Guidelines

## Project Structure & Module Organization
- `src/olmo_core/` holds the library: `data/`, `nn/`, `optim/` plus `ops/`/`kernels/`, `distributed/`, `launch/`, `train/`, and `generate/` utilities.
- `src/scripts/` provides CLIs for training, evaluation, and Beaker tooling; `src/examples/` has runnable demos; `src/test/` mirrors modules with `*_test.py` files.
- `docs/` contains the Sphinx site; the root `Makefile` aggregates common workflows.

## Setup, Build, and Development Commands
- Use Python >=3.10; install extras with `pip install -e '.[all]'` (or `uv sync --all-extras`).
- `make checks` runs formatting, lint, and type suites; singles: `make style-check`, `make lint-check`, `make type-check`.
- Run tests with `pytest -v src/test`; select subsets by path or `-k <expr>`. GPU suites: `pytest -m gpu`; skip on CPU via `pytest -m "not gpu"`.
- Build artifacts with `make build`; live docs with `make docs` (Sphinx autobuild). Docker builds use `src/Dockerfile` and CUDA args in `Makefile`.

## Coding Style & Naming Conventions
- Code is formatted by `black` (100-char lines) and ordered by `isort`; run them before committing.
- `ruff` enforces lint rules; avoid unused or wildcard imports. Keep public APIs typed; `mypy` runs in CI, so annotate functions and configs.
- Modules/files use `snake_case`; classes `CamelCase`; functions and variables `snake_case`. Tests mirror source module paths.

## Testing Guidelines
- `pytest` is configured with `src/test` as `testpaths`; test classes match `Test*` or `*Test`.
- Name files `*_test.py`; prefer `pytest.mark.parametrize` for variants. Share fixtures where possible to keep runtime low.
- GPU-only paths are marked `gpu` and skip when CUDA or optional deps (`flash-attn`, `grouped_gemm`, etc.) are missing. Add CPU fallbacks where feasible and document any required hardware.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood, concise (~72 chars), prefix scope when helpful (e.g., `train:`), and reference issues (`#123`) when applicable.
- PRs should summarize intent, behavior changes, and risks; list commands executed (`pytest`, `make checks`, targeted GPU runs) and note skipped suites. Update docs or `CHANGELOG.md` when user-facing behavior changes.
- Keep PRs focused; split large refactors into follow-ups for faster reviews.
