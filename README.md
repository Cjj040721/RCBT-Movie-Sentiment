<div align="center">

# RCBT Movie Sentiment

**A sequence-preserving RoBERTa–CNN–BiLSTM–Transformer classifier**

A 2026 rewrite of the architecture introduced in Junjia Cao's IEEE ISPCEM 2024 paper

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](#quickstart)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C)](#architecture)
[![Transformers](https://img.shields.io/badge/Transformers-4.x-6E56CF)](#quickstart)
[![Licence](https://img.shields.io/badge/Licence-MIT-2AA887)](LICENSE)

[Architecture](#architecture) · [Quickstart](#quickstart) · [Training Protocol](#training-protocol) · [Verification](#verification)

</div>

---

## Overview

RCBT combines contextual language representations, local convolution, bidirectional recurrence, and Transformer self-attention for movie-review sentiment classification.

This repository is a **new implementation**, not a re-upload of the historical experiment and not a claimed reproduction of the paper's accuracy. It retains the useful model-composition idea while rebuilding the tensor flow, data splitting, checkpoint selection, and evaluation protocol.

Paper: [An improved RCBT model and its application to sentiment analysis of movie comments](https://doi.org/10.1109/ISPCEM64498.2024.00180), IEEE ISPCEM 2024, sole author Junjia Cao.

## Architecture

```mermaid
flowchart LR
    A[Token IDs and padding mask] --> B[RoBERTa contextual embeddings]
    B --> C[Parallel Conv1d kernels 3, 4 and 5]
    C --> D[Feature concatenation and LayerNorm]
    D --> E[Packed bidirectional LSTM]
    E --> F[Masked Transformer encoder]
    F --> G[Mask-aware mean pooling]
    G --> H[Binary sentiment logits]
```

### Sequence integrity

- No global pooling is applied before the recurrent or Transformer layers.
- CNN kernels operate along the complete token sequence.
- Padded embeddings are zeroed before convolution so padding cannot leak into boundary features.
- The BiLSTM uses `pack_padded_sequence`, preventing padded tokens from affecting forward or backward recurrent states.
- Transformer attention and final pooling both use the padding mask.
- The classifier returns logits directly; softmax is applied only for inference probabilities.

### Default configuration

| Component | Default |
| --- | --- |
| Pretrained encoder | `roberta-base` |
| CNN | 128 channels per kernel; kernels 3, 4, 5 |
| BiLSTM | 320 hidden units per direction |
| Transformer | 2 layers, 8 heads, feed-forward size 2048 |
| Classifier | Masked mean pooling, dropout, linear layer |

## Quickstart

Requires Python 3.11 or newer. Verified with Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first run downloads the Hugging Face `imdb` dataset and the selected pretrained encoder. Dataset files, checkpoints, and generated metrics remain under ignored cache/output directories.

### Full training command

```bash
rcbt-train \
  --model-name roberta-base \
  --output outputs/roberta-rcbt \
  --epochs 5 \
  --batch-size 8 \
  --max-length 256 \
  --learning-rate 1e-5 \
  --seed 42
```

For a small end-to-end pipeline check rather than a quality benchmark:

```bash
rcbt-train \
  --model-name sshleifer/tiny-distilroberta-base \
  --train-limit 64 \
  --validation-limit 32 \
  --test-limit 32 \
  --epochs 1 \
  --batch-size 8 \
  --max-length 64 \
  --cnn-channels 16 \
  --lstm-hidden-size 32 \
  --transformer-layers 1 \
  --device cpu \
  --output outputs/tiny-smoke
```

### Inference

After training:

```bash
rcbt-predict \
  --checkpoint outputs/roberta-rcbt/best.pt \
  --text "An ambitious story with excellent performances and a memorable ending."
```

The command loads the selected encoder, RCBT configuration, tokenizer, and best-validation checkpoint, then returns negative/positive probabilities.

## Training Protocol

1. Load the official IMDB train/test datasets.
2. Split 10% of the official training data into a stratified validation set.
3. Tokenize with truncation and batch-specific dynamic padding.
4. Train with AdamW and gradient-norm clipping.
5. Evaluate on validation after every epoch.
6. Save the checkpoint with the highest validation macro-F1.
7. Restore that checkpoint and evaluate the official test set once.
8. Save training history, test metrics, tokenizer, and model configuration.

Optional limits provide deterministic subsampling for smoke runs. They do not alter the full-data defaults.

Generated files:

```text
outputs/<run>/
├── best.pt
├── tokenizer/
├── history.json
└── metrics.json
```

## Verification

The new model was executed with a small randomly initialised RoBERTa encoder and variable-length padded batches. The checks confirmed:

- output logits have the expected `[batch, 2]` shape;
- the loss is finite and backpropagates successfully;
- non-zero, finite gradients reach the CNN, BiLSTM, Transformer, and classifier;
- changing only padded token IDs does not change the output logits.

The first padding check exposed leakage through convolution boundary features and bidirectional recurrence. The implementation was corrected by masking embeddings before convolution and packing the recurrent sequence; the corrected model passed the invariance check. A separate real-data smoke run exposed the length mismatch caused by even-sized convolution kernels; asymmetric padding was then added and verified for kernels 3, 4, and 5.

The tiny-IMDB command above completed an actual one-epoch workflow with **64 training, 32 validation, and 32 test examples**. It trained successfully, selected and restored the best-validation checkpoint, evaluated the test set, and completed a real CLI inference call. Its sample size and tiny encoder are intentionally unsuitable for measuring model quality. **A full `roberta-base` training run and benchmark comparison have not been claimed or included.**

## Project Structure

```text
src/rcbt/
├── model.py       # RCBT configuration and sequence-preserving architecture
├── data.py        # Official IMDB split, stratified validation and tokenisation
├── engine.py      # Training loop and macro classification metrics
├── train.py       # Validation-based checkpointing and final test
└── predict.py     # Checkpoint loading and sentiment inference
```

## Relation to the Historical Experiment

The 2024 experiment tree included code derived from `BeiCunNan/sentiment_analysis_Imdb`. That unlicensed historical tree is not redistributed here. This repository uses a new model and training implementation; see [`NOTICE.md`](NOTICE.md).

The paper's historical metrics remain available through the DOI and the `Research-Papers` archive. They are not presented as results of this rewrite.

## Limitations and Next Steps

- Full-scale predictive performance still requires a complete training run and comparison with RoBERTa-only and ablation baselines.
- Current masking assumes at least one valid token per review.
- Useful extensions are learning-rate scheduling, mixed precision, multi-seed evaluation, calibration analysis, and frozen-encoder ablations.

## Licence

The new source code and original documentation are available under the [MIT License](LICENSE). IMDB and pretrained model assets retain their own terms. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
