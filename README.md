# Multi-Layer Perceptron Continuation (Part 3)

This repository documents and runs the Part 3 continuation of a character-level language modeling series, with the main focus on building, training, and analyzing a deeper Multi-Layer Perceptron (MLP).

The primary notebook in this repo is:

- MLP_continuation_part3.ipynb

Parts 1 and 2 are conceptually related, but they belong to their own separate repositories/codebases. This repo is intentionally centered on Part 3.

## Project Scope

Part 3 moves beyond the earlier baseline MLP setup and focuses on:

- Batch normalization in a character-level next-token model
- Improved initialization and optimization behavior
- Transition from a shallow MLP to a deeper pytorch stacked architecture
- Activation and gradient diagnostics for training stability
- Sampling generated names from the trained model

## Main Notebook: Part 3 Walkthrough

The notebook MLP_continuation_part3.ipynb is organized in roughly three progressive blocks.

### 1) Data and Character Vocabulary Pipeline

Using names.txt, the notebook:

- Reads names into memory
- Builds character-to-index and index-to-character mappings
- Reserves index 0 for the special stop token .
- Uses a context window (block size) of 3 characters
- Creates train/dev/test splits

This creates supervised training pairs where the model predicts the next character from the preceding context.

### 2) MLP Revisit with Batch Normalization

The notebook first revisits the earlier MLP setup and adds controlled training behavior:

- Character embedding table
- Hidden layer + tanh nonlinearity
- Batch normalization with running mean/std tracking
- Cross-entropy loss
- Learning-rate step decay schedule

It also computes split losses (train/val/test) and samples names to evaluate generation quality qualitatively.

### 3) Deeper MLP from Scratch

The second major section introduces lightweight custom module classes:

- Linear
- BatchNorm1d
- Tanh

These are composed into a deeper network stack to inspect how internal activations and gradients behave across depth. The notebook includes plots for:

- Activation distributions
- Activation-gradient distributions
- Parameter-gradient distributions
- Update-to-data ratio trends

This section is especially useful for understanding why initialization, normalization, and monitoring matter in deep networks.

## Files in This Repository

- MLP_continuation_part3.ipynb: Main Part 3 notebook (core of this repo)
- names.txt: Training corpus (names dataset)
- intro_to_language_modeling_part1.ipynb: Context notebook only; Part 1 has its own dedicated repo/codebase
- multi_layer_perceptron_part2.ipynb: Context notebook only; Part 2 has its own dedicated repo/codebase

## Requirements

Install dependencies in your Python environment:

```bash
pip install torch matplotlib
```

Recommended:

- Python 3.10+
- Jupyter Notebook or VS Code notebook support

## How To Run

1. Open MLP_continuation_part3.ipynb in Jupyter or VS Code.
2. Run cells from top to bottom in order.
3. Inspect:
   - Printed training progress
   - Train/validation loss outputs
   - Plots for loss and gradient/activation statistics
4. Run sampling cells to generate names from the learned model.

## What You Should Observe

During execution, you should see:

- Stable optimization compared to naive setups
- Useful regularization effects from batch normalization
- Better understanding of saturation and gradient flow in deep tanh stacks
- Generated names that reflect dataset style without memorizing exact entries

## Notes on Repository Boundaries

To avoid duplication across projects:

- Part 3 is the official focus of this repository
- Part 1 and Part 2 are included only as local context artifacts
- Their primary development and documentation live in separate repositories/codebases

## Suggested Next Improvements

- Replace manual module classes with torch.nn.Module equivalents for cleaner scaling
- Add experiment tracking for hyperparameters and runs
- Save and reload trained weights
- Add a reproducible script version of the notebook workflow
