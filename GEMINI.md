# Gemini Code Understanding

This document provides a high-level overview of the OLMo-core project, its structure, and how to work with it.

## Project Overview

OLMo-core is a Python library that provides the building blocks for training and using Open Language Models (OLMo). It is built on top of PyTorch and is designed to be flexible, efficient, and easy to use.

The project is structured as follows:

-   `src/olmo_core`: The main source code for the library.
-   `src/test`: The tests for the library.
-   `src/scripts`: Contains scripts for various tasks, such as training models.
-   `docs`: The documentation for the library.

### Key Technologies

-   **Python:** The primary programming language.
-   **PyTorch:** The deep learning framework used to build and train the models.
-   **Hugging Face Transformers:** Used for model sharing and inference.
-   **vLLM:** Used for high-throughput inference.
-   **OmegaConf:** Used for configuration management.

### Architecture

The core of the library is the `Transformer` class, which is a Llama-style transformer implementation. This class can be configured to create a variety of different models, including models with different numbers of layers, different model dimensionalities, and different attention mechanisms.

The library also provides a flexible configuration system that allows you to easily define and modify the architecture of your models. The configuration system is based on Python dataclasses and OmegaConf, and it allows you to define your models in a type-safe and declarative way.

## Building and Running

### Installation

To install the project for development, clone the repository and run the following command:

```bash
pip install -e .[all]
```

### Running Tests

To run the tests, use the following command:

```bash
pytest -v src/test
```

### Training a Model

To train a model, you can use the `torchrun` command to execute one of the training scripts in the `src/scripts/official` directory. For example, to train the `OLMo-2-0325-32B` model, you can use the following command:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/save/checkpoints
```

You can override most of the configuration options from the command line.

### Building Documentation

To build the documentation, run the following command:

```bash
make docs
```

## Development Conventions

### Code Style

The project uses `black` for code formatting and `isort` for import sorting. You can run the following command to check if your code is formatted correctly:

```bash
make style-check
```

### Linting

The project uses `ruff` for linting. You can run the following command to check for linting errors:

```bash
make lint-check
```

### Type Checking

The project uses `mypy` for type checking. You can run the following command to check for type errors:

```bash
make type-check
```
