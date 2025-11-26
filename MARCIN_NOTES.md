# Marcin Notes

## Setup

```bash
# Recommended in CONTRIBUTING.md
uv venv --python 3.11
source .venv/bin/activate

# Actually install - env is safe, doesn't build anywithng wierd
# NOTE: omegaconf>=2.3.0 is required
uv sync --all-extras
```

Install `flash-attn`
```bash
# Check torch, CUDA versions
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
2.9.1+cu128
True

# Check ABI availability
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
True

# Confirm torch 2.8 resolves ok
uv pip install 'torch<2.9' --index-url https://download.pytorch.org/whl/cu128 --dry-run
Resolved 25 packages in 2.23s
Would download 3 packages
Would uninstall 3 packages
Would install 3 packages
 - nvidia-nccl-cu12==2.27.5
 + nvidia-nccl-cu12==2.27.3
 - torch==2.9.1
 + torch==2.8.0+cu128
 - triton==3.5.1
 + triton==3.4.0

# Install apropriate flash-attn from the weel
```
