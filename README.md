# Irodori-TTS (Fork)

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

日本語版はこちら → [README_ja.md](README_ja.md)

> **This project is a fork of [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS).**  
> The base model weights are subject to the original license (non-commercial). The model is downloaded automatically on first launch.

Training and inference code for **Irodori-TTS**, a Flow Matching-based Text-to-Speech model. The architecture and training design largely follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/), using [DACVAE](https://github.com/facebookresearch/dacvae) continuous latents as the generation target.

For original model weights and audio samples, refer to the [model card](https://huggingface.co/Aratako/Irodori-TTS-500M).

---

## Features

- **Flow Matching TTS** — Rectified Flow Diffusion Transformer (RF-DiT) over continuous DACVAE latents
- **Voice Cloning** — Zero-shot voice cloning from reference audio
- **Emotion Style Presets** — One-click style presets (Normal / Strong / Calm / Bright / Whisper) and fine-grained CFG control
- **Multi-Candidate Generation** — Generate up to 8 audio candidates in a single run
- **LoRA Fine-tuning** — Lightweight adapter training via `peft`; supports resume, EMA, Early Stopping
- **Full Fine-tuning** — Multi-GPU DDP training with Muon/AdamW/Lion/AdEMAMix optimizers, WSD/Cosine schedulers, gradient checkpointing
- **Dataset Tools** — Audio slicing, Whisper captioning, emoji-style annotation via LLM API
- **Model Merging** — Weighted Average, SLERP, Task Arithmetic, partial merge, LoRA-style injection
- **Gradio Web UI** — All features accessible through a single GUI

---

## Architecture

The model consists of three main components:

1. **Text Encoder** — Token embeddings initialized from a pretrained LLM (`llm-jp/llm-jp-3-150m`), followed by self-attention + SwiGLU transformer layers with RoPE
2. **Reference Latent Encoder** — Encodes patched reference audio latents for speaker/style conditioning via self-attention + SwiGLU layers
3. **Diffusion Transformer** — Joint-attention DiT blocks with Low-Rank AdaLN (timestep-conditioned adaptive layer normalization), half-RoPE, and SwiGLU MLPs

Audio is represented as continuous latent sequences via the DACVAE codec (128-dim), enabling high-quality 48 kHz waveform reconstruction.

---

## Installation

```bash
git clone <this-repo>
cd Irodori-TTS
uv sync
```

> **Note:** For Linux/Windows with CUDA, PyTorch is automatically installed from the cu128 index. For macOS (MPS) or CPU-only usage, `uv sync` installs the default PyTorch build.

---

## Quick Start

### Gradio Web UI

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

Access the UI at `http://localhost:7860`.  
On first launch, `Aratako/Irodori-TTS-500M` is downloaded automatically if no checkpoint is found.

Optional flags:

| Flag | Description |
|---|---|
| `--server-name` | Bind address (default: `127.0.0.1`) |
| `--server-port` | Port number (default: `7860`) |
| `--share` | Create a public Gradio link |
| `--debug` | Enable Gradio debug mode |

### CLI Inference

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

Without reference audio (unconditional):

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

---

## Inference

### Inference Parameters

| Parameter | Default | Description |
|---|---|---|
| `--text` | *(required)* | Text to synthesize |
| `--ref-wav` | `None` | Reference audio file for voice cloning |
| `--ref-latent` | `None` | Pre-computed reference latent (`.pt`) |
| `--no-ref` | `False` | Unconditional generation (no reference) |
| `--num-steps` | `40` | Number of Euler integration steps |
| `--cfg-scale-text` | `3.0` | CFG scale for text conditioning |
| `--cfg-scale-speaker` | `5.0` | CFG scale for speaker conditioning |
| `--guidance-mode` | `independent` | CFG mode: `independent` / `joint` / `alternating` |
| `--cfg-min-t` | `0.5` | Diffusion timestep at which CFG starts |
| `--cfg-max-t` | `1.0` | Diffusion timestep at which CFG ends |
| `--context-kv-cache` | `True` | Cache text/speaker KV projections across steps |
| `--truncation-factor` | `None` | Latent truncation for expression range control |
| `--rescale-k` | `None` | Score rescaling k (Xu et al. 2025) |
| `--rescale-sigma` | `None` | Score rescaling sigma |
| `--speaker-kv-scale` | `None` | Speaker KV attention scale factor |
| `--speaker-kv-min-t` | `None` | Minimum timestep for speaker KV scale |
| `--speaker-kv-max-layers` | `None` | Maximum number of layers to apply speaker KV scale |
| `--model-device` | auto | Device for the TTS model (`cuda` / `mps` / `cpu`) |
| `--codec-device` | auto | Device for the DACVAE codec |
| `--model-precision` | auto | Model precision (`fp32` / `bf16`) |
| `--codec-precision` | auto | Codec precision (`fp32` / `bf16`) |
| `--seed` | random | Random seed for reproducibility |
| `--compile-model` | `False` | Enable `torch.compile` for faster inference |
| `--trim-tail` | `True` | Trim trailing silence via flattening heuristic |
| `--lora-path` | `None` | Path to a LoRA adapter directory |
| `--lora-scale` | `1.0` | LoRA injection scale (0.0 = base only, >1.0 = emphasized) |

Local checkpoints (`.pt` or `.safetensors`) are also supported:

```bash
uv run python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

---

## Gradio UI — Tab Reference

### Tab 1: Inference (🔊)

- **Model loading** — Select a checkpoint (`.pt` / `.safetensors`), device, and precision. Download models directly from HuggingFace by entering a repo ID.
- **LoRA adapter** — Optionally load a LoRA adapter and adjust its scale (0.0–2.0).
- **Audio generation** — Enter text and optionally upload a reference audio file.
- **Emotion style presets** — Normal / Strong / Calm / Bright / Whisper buttons auto-configure CFG and style parameters.
- **Style sliders** — Text expressiveness, emotion strength, speaker adherence, and expression range.
- **Sampling settings** — Number of steps (1–120) and random seed.
- **CFG settings** — Guidance mode (`independent` / `joint` / `alternating`), text CFG, speaker CFG.
- **Advanced settings** — CFG timestep range, context KV cache, score rescaling, speaker KV scale.
- **Multi-candidate generation** — Generate 1–8 candidates per run. Each candidate is saved as a separate WAV file.

### Tab 2: Prepare Manifest (📂)

Converts audio data to DACVAE latents and produces a JSONL training manifest.

Supported data sources: local CSV (`audiofolder` format), local JSONL, or HuggingFace dataset.

Column names for audio, text, and speaker ID are detected automatically from the file header.

```bash
# CLI equivalent
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

Output manifest format:

```json
{"text": "こんにちは", "latent_path": "data/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### Tab 3: Training (学習)

Full fine-tuning with optional Multi-GPU DDP.

Key settings:

| Group | Options |
|---|---|
| Base model | Resume from `.safetensors` (step 0) or `.pt` checkpoint (step/optimizer state restored) |
| Batch | Batch size, gradient accumulation, DataLoader workers |
| Optimizer | `muon` / `adamw` / `lion` / `ademamix` / `sgd` |
| Scheduler | `wsd` (warmup→stable→decay) / `cosine` / `none` |
| Precision | `bf16` / `fp32` / `fp16` |
| Attention backend | `sdpa` (recommended) / `flash2` / `sage` / `eager` |
| Regularization | Text/speaker condition dropout, timestep stratified sampling |
| Validation | Validation split ratio, validation interval, Early Stopping |
| EMA | Exponential Moving Average for inference-quality checkpoints |
| Checkpoint | Save interval, EMA-only or EMA+Full |
| Logging | W&B integration, log interval |

Single-GPU:

```bash
uv run python train.py \
  --config configs/train_v1.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

Multi-GPU DDP:

```bash
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_v1.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --device cuda
```

The UI displays a live training log (last 200 lines) with ETA, loss, and step progress. A loss curve graph updates at a configurable interval.

### Tab 4: LoRA Training (🚀)

Trains a LoRA adapter on top of a frozen base model. Requires `pip install peft`.

Key settings:

| Setting | Default | Description |
|---|---|---|
| LoRA rank | `16` | Low-rank dimension |
| lora_alpha | `32.0` | Scaling factor |
| lora_dropout | `0.05` | Dropout rate |
| Target modules | `wq,wk,wv,wo` | Comma-separated module names. Extended set: `wq,wk,wv,wo,wk_text,wv_text,wk_speaker,wv_speaker,w1,w2,w3` |
| Save mode | EMA only | `EMA only` (inference) or `EMA + Full` (resumable) |
| Attention backend | `sdpa` | Same options as full training |
| EMA | enabled | EMA decay for inference-quality adapters |
| Early Stopping | disabled | Requires `valid_ratio > 0` |
| Resume | disabled | Resume from an existing `_full` adapter folder |

Presets (YAML with a `lora:` section) can be saved and loaded from `configs/`.

### Tab 5: Dataset Creation (Dataset作成)

**Slice** — Splits long audio files into segments using silence detection.

| Parameter | Description |
|---|---|
| Min / Max duration (sec) | Acceptable segment length range |
| Top dB | Silence threshold in dBFS |
| Frame / Hop length | STFT window parameters |
| Target sample rate | Optional resampling |
| Recursive | Search subdirectories |

**Caption** — Transcribes audio segments using Whisper and outputs a manifest.

| Parameter | Description |
|---|---|
| Whisper model | `tiny` / `base` / `small` / `medium` / `large-v3` |
| Language | Language code (e.g., `ja`) or auto-detect |
| Output format | `CSV` or `JSONL` |
| Speaker ID | Optional fixed speaker label |

**Pipeline** — Runs slice → caption in sequence.

**Emoji Caption** — Extracts acoustic features (pitch, energy, speech rate, MFCC delta, ZCR) from each audio segment and calls an LLM API to annotate text with Irodori-TTS compatible emotion emojis.

Supported APIs: `lm_studio` / `groq` / `openai` / `together`.

Emoji annotations supported (38 types), including: 👂 whisper, 😤 strong, 😌 calm, 🤭 laugh, 😭 crying, 😱 scream, ⏩ fast-speaking, 🐢 slow, 🎵 humming, and others.

### Tab 6: Checkpoint Conversion (🔄)

**Normal checkpoint conversion** — Converts a training `.pt` checkpoint to inference-only `.safetensors` format. Model config is embedded in the file metadata.

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
# Output: outputs/checkpoint_final.safetensors

# Force overwrite:
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt --force
```

**LoRA checkpoint conversion** — Converts a `_full` LoRA checkpoint directory (containing optimizer state) to a `_ema` adapter-only format for inference.

### Tab 7: Model Merge (🔀)

Merges two model checkpoints (`.pt` or `.safetensors`) with architecture compatibility checking.

**Merge methods:**

| Method | Description |
|---|---|
| `weighted_average` | `result = α × A + (1 − α) × B` |
| `slerp` | Spherical linear interpolation. Preserves norms; falls back to weighted average for near-zero vectors |
| `task_arithmetic` | `result = base + λA × (A − base) + λB × (B − base)`. Requires a separate base model |

**Partial merge** — Apply a different merge method to each layer group independently:

| Group | Layers |
|---|---|
| `text` | Text encoder, text norm, JointAttention text KV |
| `speaker` | Speaker encoder, speaker norm, JointAttention speaker KV |
| `diffusion_core` | Diffusion blocks, cond_module |
| `io` | in_proj, out_norm, out_proj |

**LoRA-style injection** — Injects the difference between a donor and base model into a target model: `result = base + scale × (donor − base)`. Target groups are selectable independently.

Output format: `.safetensors` (recommended for inference) or `.pt`.

---

## Training

### 1. Prepare Manifest

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

With speaker ID:

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

### 2. Training

Single-GPU:

```bash
uv run python train.py \
  --config configs/train_v1.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

Multi-GPU DDP:

```bash
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_v1.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --device cuda
```

### 3. LoRA Training

```bash
uv run python lora_train.py \
  --base-model checkpoints/Aratako_Irodori-TTS-500M/model.safetensors \
  --manifest data/train_manifest.jsonl \
  --output-dir lora/my_run \
  --lora-rank 16 \
  --lora-alpha 32.0 \
  --max-steps 1000
```

### 4. Checkpoint Conversion

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
```

---

## Project Structure

```text
Irodori-TTS/
├── train.py                              # Full fine-tuning entry point (DDP support)
├── lora_train.py                         # LoRA fine-tuning entry point
├── infer.py                              # CLI inference
├── gradio_app.py                         # Gradio web UI (all features)
├── prepare_manifest.py                   # Audio → DACVAE latent preprocessing
├── dataset_tools.py                      # Audio slice / Whisper caption / emoji annotation
├── merge.py                              # Model merge utilities
├── convert_checkpoint_to_safetensors.py  # .pt → .safetensors conversion
├── convert_lora_checkpoint.py            # LoRA _full → _ema conversion
│
├── irodori_tts/                          # Core library
│   ├── model.py                          # TextToLatentRFDiT architecture
│   ├── rf.py                             # Rectified Flow utilities & Euler CFG sampling
│   ├── codec.py                          # DACVAE codec wrapper
│   ├── dataset.py                        # Dataset and collator
│   ├── tokenizer.py                      # Pretrained LLM tokenizer wrapper
│   ├── config.py                         # Model / Train / Sampling config dataclasses
│   ├── inference_runtime.py              # Cached, thread-safe inference runtime
│   ├── text_normalization.py             # Japanese text normalization
│   ├── optim.py                          # Muon + AdamW + Lion + AdEMAMix optimizers
│   └── progress.py                       # Training progress tracker
│
├── configs/
│   ├── train_v1.yaml                     # Training config (500M, ~50 samples, RTX 5060 Ti)
│   └── *.yaml                            # Additional user configs
│
├── checkpoints/                          # Downloaded and trained model checkpoints
├── lora/                                 # LoRA adapter outputs
├── logs/                                 # Training log files
├── data/                                 # Manifest files and DACVAE latents
└── gradio_outputs/                       # Generated audio files from the GUI
```

---

## Configuration Reference (train_v1.yaml)

The default config is tuned for approximately 50 samples on an RTX 5060 Ti (16 GB VRAM), targeting ~30–60 minutes of training time.

| Key | Default | Description |
|---|---|---|
| `batch_size` | `4` | Per-GPU batch size |
| `gradient_accumulation_steps` | `2` | Effective batch = batch × accum |
| `optimizer` | `muon` | `muon` / `adamw` / `lion` / `ademamix` / `sgd` |
| `learning_rate` | `3e-4` | Peak learning rate |
| `lr_scheduler` | `wsd` | `wsd` / `cosine` / `none` |
| `warmup_steps` | `300` | Linear warmup steps |
| `stable_steps` | `2100` | Stable phase steps (wsd only) |
| `max_steps` | `3000` | Total training steps |
| `max_text_len` | `256` | Maximum text token length |
| `max_latent_steps` | `750` | Maximum latent frame length |
| `text_condition_dropout` | `0.15` | CFG text dropout rate |
| `speaker_condition_dropout` | `0.15` | CFG speaker dropout rate |
| `valid_ratio` | `0.1` | Validation split ratio (0 = disabled) |
| `valid_every` | `100` | Validation interval (steps) |
| `save_every` | `100` | Checkpoint save interval (steps) |
| `precision` | `bf16` | Training precision |
| `compile_model` | `false` | Enable torch.compile |

---

## License

- **Code**: [MIT License](LICENSE)
- **Model Weights**: Non-commercial. See the [original model card](https://huggingface.co/Aratako/Irodori-TTS-500M) for details.

---

## Acknowledgments

- [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) — Original repository this fork is based on
- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — Architecture and training design reference
- [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE codec

---

## Citation

```bibtex
@misc{irodori-tts,
  author = {Chihiro Arata},
  title = {Irodori-TTS: A Flow Matching-based Text-to-Speech Model with Emoji-driven Style Control},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Aratako/Irodori-TTS}}
}
```
