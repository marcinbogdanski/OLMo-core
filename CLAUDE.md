# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OLMo-core is the official training framework for Allen Institute for AI's Open Language Model (OLMo) series. It provides modular building blocks for training large language models from scratch, implementing transformer architectures, and distributed training with FSDP.

**Package name**: `ai2-olmo-core`
**Python**: >=3.10
**PyTorch**: >=2.6.0

## Common Commands

```bash
# Install for development
pip install -e '.[all]'

# Run all checks (style + lint + type)
make checks

# Auto-format code
make style

# Individual checks
make style-check    # isort + black
make lint-check     # ruff
make type-check     # mypy

# Run tests
pytest -v src/test                    # All tests
pytest src/test/nn/rope_test.py       # Single file
pytest -k rope                        # By keyword
pytest -m gpu                         # GPU-only tests
pytest -m "not gpu"                   # CPU-only tests

# Build docs locally
make docs
```

## Code Structure

```
src/
├── olmo_core/           # Main library
│   ├── nn/              # Neural network modules
│   │   ├── transformer/ # Transformer model and blocks
│   │   ├── attention/   # Attention mechanisms (flash, TE, etc.)
│   │   ├── rope.py      # Rotary position embeddings
│   │   └── moe/         # Mixture-of-Experts
│   ├── train/           # Training orchestration
│   │   ├── trainer.py   # Main Trainer class
│   │   ├── callbacks/   # Training callbacks
│   │   └── train_module/# TrainModule base classes
│   ├── data/            # Data loading and datasets
│   ├── distributed/     # FSDP, collectives, distributed checkpointing
│   ├── optim/           # Optimizers and LR schedulers
│   ├── float8/          # Float8 training support
│   └── config.py        # Base Config dataclass
├── examples/            # Reference training scripts
├── scripts/official/    # Official OLMo-2 and OLMo-3 training scripts
└── test/                # Tests (mirrors olmo_core/ structure)
```

## Architecture Highlights

- **Config system**: All configurations use dataclasses inheriting from `Config` (in `config.py`) with built-in serialization and command-line override support via OmegaConf
- **Trainer**: Central orchestrator in `train/trainer.py` managing training loop, checkpointing, callbacks, and distributed coordination
- **TrainModule**: Abstraction for model + loss computation in `train/train_module/`
- **Attention backends**: Pluggable attention implementations (Flash-Attention, TransformerEngine, standard PyTorch) in `nn/attention/`
- **Distributed training**: FSDP-based parallelism with checkpoint sharding support in `distributed/`

## Testing Conventions

- Tests live in `src/test/`, mirroring the library structure (e.g., `src/test/nn/rope_test.py` for `src/olmo_core/nn/rope.py`)
- Test files: `*_test.py`, test functions: `test_*`
- Use `pytest.mark.parametrize` for variations
- GPU tests use `@requires_gpu` decorator and skip automatically on CPU-only machines
- Multi-GPU tests use `@requires_multi_gpu` and `run_distributed_test()` helper

## Code Style

- Line length: 100
- Formatters: isort (black profile), black
- Linter: ruff (ignores F403, F405, E501)
- Type checker: mypy
- All PRs must update CHANGELOG.md

## Running Training Scripts

Official training scripts use `torchrun`:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/checkpoints

# Override config options via CLI
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/checkpoints \
  --train_module.optim.lr=6e-3
```
