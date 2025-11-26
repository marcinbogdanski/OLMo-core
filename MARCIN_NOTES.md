# Marcin Notes

## Setup

```bash
# Recommended in CONTRIBUTING.md
uv venv --python 3.11
source .venv/bin/activate

# Confirm nothing will be compiled
uv sync --all-extras --no-build --no-install-project --dry-run

# Actually install
uv sync --all-extras


# Option 2: Temporarily pin in pyproject.toml:
sed -i 's/"omegaconf"/"omegaconf>=2.3.0"/' pyproject.toml
rm uv.lock
uv sync --all-extras
```