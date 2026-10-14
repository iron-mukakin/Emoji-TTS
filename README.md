# Irodori-TTS (Fork)

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

日本語版はこちら → [README_ja.md](README_ja.md)

> **This project is a fork of [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS).**
> The fork ports the v4-Small / v3-VoiceDesign architecture family (pretrained text encoder, Duration Predictor, caption conditioning, quantized checkpoints, Speaker Inversion) from upstream into both the inference and training code paths, while keeping fork-exclusive features (LoRA training, model/LoRA merging, EMA, dataset tools, extended Gradio GUI).
> Model weights are subject to the license of each Hugging Face model card. Models are downloaded automatically on first launch.

Training and inference code for **Irodori-TTS**, a Flow Matching-based Text-to-Speech model. The architecture and training design largely follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/), using [DACVAE](https://github.com/facebookresearch/dacvae) continuous latents as the generation target.

---

## Features

- **Flow Matching TTS** — Rectified Flow Diffusion Transformer (RF-DiT) over continuous DACVAE latents
- **Multi-version Model Support** — One codebase for v1 / v2 / v3 / v2 & v3-VoiceDesign / v4-Small / v4.1-Small checkpoints; codec and conditioning layout are auto-detected on load (see the model support table below)
- **Quantized Checkpoint Loading** — torchao-quantized v4-Small-Quantized checkpoints (int8 weight-only / int8 dynamic / int4 / float8) for inference (`--extra quantization` required)
- **Voice Cloning** — Zero-shot voice cloning from reference audio
- **Voice Design (Caption Conditioning)** — Describe voice style/emotion in natural language. v3-VoiceDesign / v4-Small support 3-factor control: text + reference speech + caption
- **Duration Predictor** — Automatic output length on supported models; manual control via `--seconds` / `--duration-scale`
- **Speaker Inversion (v4 family)** — Trainable speaker tokens that replace the reference-latent speaker encoder (`.speaker.safetensors`); full training (`train.py` + `configs/train_v4_small_speaker_inversion.yaml`), inference (`infer.py --ref-embed`), and a Gradio reference tab are supported
- **Pretrained Text Encoder (v4 family)** — Loads pretrained backbone + projector (v4-Small/v4.1-Small use `sbintuitions/modernbert-ja-310m`); separate lower LR for the backbone, and the encoder config is embedded into saved checkpoints for offline re-loading
- **MeanFlow Distillation Training (v4.1)** — `train_mode: meanflow_distill` distills an RF-trained checkpoint into a few-step student (default 4 NFE, `configs/train_v4_small_meanflow.yaml`); only the DiT is trained while the teacher produces targets via CFG-fused online Euler rollout. MeanFlow checkpoints are sampled automatically at 4 NFE during inference (see `docs/meanflow.md`)
- **Emoji Style Control** — Emotion and sound-effect control via emojis in the input text (38 types, `EMOJI_ANNOTATIONS` compatible)
- **Emotion Style Presets** — One-click presets (Normal / Strong / Calm / Bright / Whisper) with fine-grained CFG control
- **Speaker Library** — Register reference audio as named speaker profiles (`speakers/`) and reuse them
- **Multi-Candidate Generation** — Up to 8 candidates per run (batched sampling)
- **Multiline Split Generation** — Per-line individual output, or concatenation with silence gaps
- **Watermarking** — SilentCipher (dim32) / DACVAE (dim128), toggleable in UI and CLI
- **LoRA Fine-tuning** — `peft`-based adapters with resume, EMA, Early Stopping, per-module disable, adapter merge/bake-in
- **Full Fine-tuning** — Multi-GPU DDP, Muon/AdamW/Lion/AdEMAMix/SGD, WSD/Cosine schedulers, gradient checkpointing (with `store_every` partial activation retention), TensorBoard & metrics logging
- **Dataset Tools** — Silero VAD slicing, Whisper captioning, LLM-based emoji annotation, and VoiceDesign "voice analysis caption" generation
- **Model Merging** — Weighted Average, SLERP, Task Arithmetic, per-group partial merge, LoRA-style difference injection (v3-family compatible)
- **Gradio Web UI / Dark Mode** — All features accessible from a single GUI

---

## Model Support

The loaded checkpoint's metadata (`model_config` / `text_encoder_config_json` / quantization metadata) determines the model variant automatically.

| Model | Hugging Face repo | Codec (latent dim) | Conditioning | Duration<br>Predictor | Inference | Full training | LoRA training |
|---|---|---|---|---|---|---|---|
| v1 | `Aratako/Irodori-TTS-500M` | `facebook/dacvae-watermarked` (dim128) | Text + Reference | – | ✅ | ✅ | ✅ |
| v2 | `Aratako/Irodori-TTS-500M-v2` | `Aratako/Semantic-DACVAE-Japanese-32dim` (dim32) | Text + Reference | – | ✅ | ✅ | ✅ |
| v2-VoiceDesign | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | same as above (dim32) | Text + Caption | – | ✅ | ✅ | ✅ |
| v3 | `Aratako/Irodori-TTS-500M-v3` | same as above (dim32) | Text + Reference | ✅ | ✅ | ✅ | ✅ |
| v3-VoiceDesign | `Aratako/Irodori-TTS-600M-v3-VoiceDesign` | same as above (dim32) | Text + Reference + Caption | ✅ | ✅ | ✅ | ✅ |
| v4-Small | `Aratako/Irodori-TTS-v4-Small` | same as above (dim32) | Text + Reference + Caption + Speaker tokens | ✅ | ✅ | ✅ | ✅ |
| v4.1-Small | `Aratako/Irodori-TTS-v4.1-Small` (default) | same as above (dim32) | Text + Reference + Caption + Speaker tokens | ✅ | ✅ | ✅ | ✅ |

**Notes**

- **Quantized checkpoints** (torchao format, e.g. `Aratako/Irodori-TTS-v4-Small-Quantized`) are **inference-only**. Loading them requires `uv sync --extra quantization` (torchao). They cannot be used with `--init-checkpoint` for training — train on the full-precision base, merge, then quantize.
- Manifest preparation modes: `model_v1` (dim128) / `model_v2` / `model_v3` / `model_v4` (default; speaker + caption columns) / `voice_design` (caption column required).
- Training configs can set `use_caption_condition`, `use_duration_predictor`, `text_encoder_type: pretrained`, etc. in the `model:` section of the YAML.
- **v4-Large training configs** — `configs/train_v4_large.yaml` / `configs/train_v4_large_duration.yaml` are bundled (pretrained text encoder `google/t5gemma-2-1b-1b`, Muon for both the main model and the pretrained text encoder). The upstream HF release is still forthcoming, so only the configs are provided (no model code changes required).
- **MeanFlow checkpoints** — Checkpoints produced by `train_mode: meanflow_distill` carry `flow_parameterization: meanflow` in their metadata; inference (infer.py / Gradio) automatically uses the 4 NFE sampler. Runtime CFG is ignored (CFG is fused into the teacher target during training).

---

## Architecture

The model consists of up to four main components (availability varies by model generation):

1. **Text Encoder** — Scratch-trained (v1–v3: token embeddings initialized from `llm-jp/llm-jp-3-150m`, self-attention + SwiGLU layers with RoPE), or a pretrained backbone + projector (v4 family: `sbintuitions/modernbert-ja-310m`)
2. **Reference Latent Encoder** — Encodes patched reference audio latents for speaker/style conditioning. In v4-Small it can be replaced by learned speaker tokens (Speaker Inversion)
3. **Diffusion Transformer** — Joint-attention DiT blocks with Low-Rank AdaLN, half-RoPE, and SwiGLU MLPs; 3-branch conditioning from text / caption / speaker
4. **Duration Predictor** — Predicts audio duration from text plus speaker/caption conditioning (variable-length training, v3 and later)

Audio is represented as continuous latent sequences via the DACVAE codec. v2 and later use 32-dim latents (48 kHz); v1 uses 128-dim latents.

---

## Installation

```bash
git clone <this-repo>
cd Irodori-TTS
uv sync
```

- Python 3.10+. On Linux/Windows with CUDA, PyTorch is installed from the cu128 index automatically (macOS/CPU use the default PyTorch build).
- Only when loading quantized checkpoints:

```bash
uv sync --extra quantization   # torchao (for int8/int4/float8 checkpoints)
```

- ffmpeg is required for audio slicing / Whisper captioning (on Windows, add `ffmpeg-7.1-full_build\bin` to PATH).

---

## Quick Start

### Gradio Web UI

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
# On Windows, double-clicking start.bat also works
```

Access the UI at `http://localhost:7860`. If no checkpoint is found, `Aratako/Irodori-TTS-v4.1-Small` is downloaded by default (falls back to v4-Small / v3 / v2 / v1 if already present locally); any HF repo ID can also be entered in the UI's download box.

| Flag | Description |
|---|---|
| `--server-name` | Bind address (default: `127.0.0.1`) |
| `--server-port` | Port number (default: `7860`) |
| `--share` | Create a public Gradio link |
| `--debug` | Enable Gradio debug mode |

### CLI Inference

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs_gradio/sample.wav
```

Voice Design model with a caption:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "こんにちは、今日はいい天気ですね。" \
  --caption "明るく弾んだ声。少し早めの話速" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs_gradio/sample.wav
```

Without reference audio (unconditional):

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav outputs_gradio/sample.wav
```

v4 with a Speaker Inversion embedding (no reference audio needed; produced by `--speaker-inversion` training):

```bash
uv run python infer.py \
  --checkpoint checkpoints/Aratako_Irodori-TTS-v4-Small/model.safetensors \
  --ref-embed outputs_train/speaker_inversion/checkpoint_final.speaker.safetensors \
  --text "今日はいい天気ですね。" \
  --caption "clear bright female voice" \
  --output-wav outputs_gradio/sample.wav
```

Local checkpoints (`.pt` / `.safetensors` / quantized safetensors) are also supported:

```bash
uv run python infer.py \
  --checkpoint checkpoints/Aratako_Irodori-TTS-500M-v3/model.safetensors \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs_gradio/sample.wav
```

---

## Inference Parameters (infer.py)

| Parameter | Default | Description |
|---|---|---|
| `--checkpoint` / `--hf-checkpoint` | *(required, mutually exclusive)* | Local checkpoint / HF repo ID |
| `--text` | *(required)* | Text to synthesize (emoji style control supported) |
| `--caption` | `None` | Caption (style description) for VoiceDesign models |
| `--ref-wav` / `--ref-wavs` / `--ref-latent` / `--ref-latents` / `--ref-embed` / `--no-ref` | *(optional, mutually exclusive)* | Reference audio (single / multiple concatenated) / pre-computed latent (.pt) (single / multiple concatenated) / Speaker Inversion embedding (`.speaker.safetensors`, v4) / unconditional. Speaker-conditioned checkpoints require one of these or `--no-ref` |
| `--max-ref-seconds` | `None` (auto) | Maximum reference duration. Defaults to the checkpoint recommendation (120s for v4, 30s for legacy). <=0 disables the cap |
| `--ref-normalize-db` | `-16.0` | Reference loudness normalization target (`none` to disable) |
| `--ref-ensure-max` | `True` | Scale reference down only when peak exceeds 1.0 |
| `--num-steps` | `None` (auto) | Number of Euler integration steps. Auto-selected when omitted: 4 for MeanFlow checkpoints / 40 for RF checkpoints |
| `--t-schedule-mode` / `--sway-coeff` | `linear` / `-1.0` | RF sampler timestep schedule (`sway` = denser at low-noise side) and its coefficient |
| `--num-candidates` | `1` | Candidates generated per batched pass |
| `--decode-mode` | `sequential` | Candidate decoding (`sequential` = lower VRAM / `batch` = faster) |
| `--seconds` | `None` | Manual output length (Duration Predictor used when omitted; 30 s fallback without one) |
| `--duration-scale` | `1.0` | Multiplier on predicted duration (>1 longer, <1 shorter) |
| `--cfg-scale-text` | `3.0` | CFG scale for text conditioning |
| `--cfg-scale-caption` | `3.0` | CFG scale for caption conditioning |
| `--cfg-scale-speaker` | `5.0` | CFG scale for speaker conditioning |
| `--cfg-guidance-mode` | `independent` | CFG mode: `independent` / `joint` / `alternating` |
| `--cfg-min-t` / `--cfg-max-t` | `0.5` / `1.0` | Timestep range where CFG is applied |
| `--context-kv-cache` | `True` | Cache text/speaker KV projections across steps |
| `--truncation-factor` | `None` | Initial noise scaling (expression range control) |
| `--rescale-k` / `--rescale-sigma` | `None` | Temporal score rescaling (Xu et al. 2025) |
| `--speaker-kv-scale` | `None` | Speaker KV attention scale (>1 strengthens speaker identity) |
| `--speaker-kv-min-t` | `0.9` | Timestep at which speaker KV scaling stops |
| `--speaker-kv-max-layers` | `None` | Number of leading layers to apply speaker KV scale |
| `--speaker-uncond-mode` | `mask` | Unconditional speaker embedding formulation (`mask` = zero / `noise` = Gaussian noise) |
| `--max-text-len` / `--max-caption-len` | checkpoint value → 256 | Maximum conditioning token lengths |
| `--model-device` / `--codec-device` | auto | Device (`cuda` / `mps` / `xpu` / `cpu`) |
| `--model-precision` / `--codec-precision` | `fp32` | Precision (`fp32` / `bf16`) |
| `--enable-watermark` | `False` | Watermark embedding (dim32 = SilentCipher / dim128 = DACVAE) |
| `--codec-repo` | `Aratako/Semantic-DACVAE-Japanese-32dim` | Decoding codec (auto-selected to match the model) |
| `--lora-path` / `--lora-scale` / `--lora-disabled-modules` | `None` / `1.0` / `""` | LoRA adapter loading, scaling, per-module disable |
| `--lora-adapter` | `None` | Dynamically apply a PEFT adapter per request (not merged into the base checkpoint) |
| `--compile-model` / `--compile-dynamic` | `False` | Enable torch.compile |
| `--trim-tail` | `True` | Trim trailing silence via flattening heuristic |
| `--tail-window-size` / `--tail-std-threshold` / `--tail-mean-threshold` | `20` / `0.05` / `0.1` | Tail trimming thresholds |
| `--seed` | random | Random seed for reproducibility |
| `--show-timings` | `True` | Print per-stage timings |

---

## Gradio UI — Tab Reference

### Tab 1: Inference (🔊)

- **Model loading** — Select a checkpoint under `checkpoints/`, or download directly by HF repo ID. Device, precision, and watermark toggle. Codec auto-selection by latent dim; quantized checkpoints are detected automatically.
- **LoRA adapter** — Select an adapter under `lora/`, adjust scale (0.0–2.0), disable specific modules.
- **Reference audio** — ① Direct upload ② Speaker library ③ Speaker registration (saves `ref.wav` / `ref.pt` / `profile.json` under `speakers/{name}/`) ④ Speaker Inversion embedding (select a `*.speaker.safetensors` produced by v4 speaker-inversion training; no reference audio needed).
- **Voice Design** — Caption field and caption CFG appear only when a caption-compatible model is loaded.
- **Emotion style presets** — Normal / Strong / Calm / Bright / Whisper.
- **Style sliders** — Text expressiveness, emotion strength, speaker adherence, expression range.
- **Sampling** — Steps (0=auto: MeanFlow=4 / RF=40, or 1–120), seed, candidate count (1–8), multiline split generation (individual / concatenated with silence gaps).
- **CFG settings** — Guidance mode, text/caption/speaker CFG, timestep range, KV cache, score rescaling, speaker KV scale.
- Generated audio is saved under `outputs_gradio/`.

### Tab 2: Prepare Manifest (📂)

Converts audio data to DACVAE latents and produces a JSONL training manifest.

Data sources: local CSV (`audiofolder` format), local JSONL, or Hugging Face dataset. Column names for audio, text, and speaker ID are auto-detected from the header.

| Mode | Codec | Description |
|---|---|---|
| `model_v4` | dim32 (default) | Speaker + caption columns, for v4-Small / v4.1-Small |
| `model_v3` | dim32 | Speaker-conditioned, for v3 models |
| `model_v2` | dim32 | Speaker-conditioned, for v2 models |
| `model_v1` | dim128 | Speaker-conditioned, for v1 models |
| `voice_design` | dim32 | Caption-conditioned (caption column required) |

```bash
# CLI equivalent
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest my_manifest/train_manifest.jsonl \
  --latent-dir my_manifest/latents \
  --device cuda
```

Output manifest format:

```json
{"text": "こんにちは", "latent_path": "my_manifest/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### Tab 3: Training (🏋️)

Full fine-tuning with optional Multi-GPU DDP.

| Group | Options |
|---|---|
| Base model | Resume from `.safetensors` (step 0) or `.pt` checkpoint (step/optimizer state restored). Quantized checkpoints not allowed |
| Batch | Batch size, gradient accumulation, DataLoader workers |
| Optimizer | `muon` / `adamw` / `lion` / `ademamix` / `sgd` (via YAML config; the CLI flag accepts `muon`/`adamw` only) |
| Scheduler | `wsd` (warmup→stable→decay) / `cosine` / `none` |
| Precision | `bf16` / `fp32` |
| Attention backend | `sdpa` (recommended) / `flash2` / `eager` |
| Regularization | Text/caption/speaker condition dropout, timestep stratified sampling |
| Duration training | `duration_loss_weight` etc. (Duration Predictor models) |
| Validation | Validation split ratio, validation interval, Early Stopping |
| EMA | EMA for inference-quality checkpoints (EMA-only by default; `--save-full` also saves a resumable full checkpoint) |
| Checkpoint | Save interval, EMA-only or EMA+Full |
| Logging | Metrics JSONL (`logs/`), log interval, live loss/LR graph, TensorBoard events on completion |

Presets (YAML under `configs/`) can be selected, saved, and updated from the dropdown. A live command preview shows the exact CLI being run.

Single-GPU:

```bash
uv run python train.py \
  --config configs/train_v3.yaml \
  --manifest my_manifest/train_manifest.jsonl \
  --output-dir outputs_train/irodori_tts
```

Multi-GPU DDP:

```bash
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_v3.yaml \
  --manifest my_manifest/train_manifest.jsonl \
  --output-dir outputs_train/irodori_tts \
  --device cuda
```

### Tab 4: LoRA Training (🚀)

Trains a LoRA adapter on top of a frozen base model (`peft` bundled).

| Setting | Default | Description |
|---|---|---|
| Model version | `auto` | `auto` detects the version from the base checkpoint; the detected version is shown in the UI. Selecting a version explicitly switches the LoRA defaults |
| LoRA rank | `16` | Low-rank dimension |
| lora_alpha | `32.0` | Scaling factor |
| lora_dropout | `0.05` | Dropout rate |
| Target modules | `auto` | `auto` = per-version default (v1–v3: `wq,wk,wv,wo` / v4: diffusion-attention-only regex incl. `wk_text/wv_text/wk_speaker/wv_speaker/wk_caption/wv_caption`, pretrained backbone excluded). Comma-separated lists and a single `^`-anchored regex are also accepted |
| modules_to_save | `auto` | Modules fully trained without LoRA. `auto` = v4: `duration_predictor` / others: none |
| Save mode | EMA only | `EMA only` (inference) or `EMA + full` (resumable) |
| Attention backend | `sdpa` | `sdpa` / `flash2` / `sage` / `eager` |
| EMA / Early Stopping / Resume | — | Continue training from an existing `_full` adapter folder; early stop on validation loss |

A ready-made v4 preset is provided at `configs/train_v4_small_lora.yaml`.

Presets (YAML with a `lora:` section) can be saved to and loaded from `configs/`.
In the Inference tab, the **LoRA Disabled Modules** field zeroes out selected LoRA weights per module without reloading the model.

### Tab 5: Dataset Creation (🎙️)

**Slice** — Splits long audio into segments using Silero VAD.

| Parameter | Description |
|---|---|
| Min / Max duration (sec) | Acceptable segment length range |
| VAD threshold | Speech detection sensitivity (0.5 recommended) |
| Min silence duration (ms) | Minimum silence length to trigger a split |
| Speech pad (ms) | Padding added before/after each segment |
| Target sample rate | Optional resampling |
| Recursive | Search subdirectories |

**Caption** — Transcribes segments with Whisper and outputs a manifest.

| Parameter | Description |
|---|---|
| Whisper model | `tiny` / `base` / `small` / `medium` / `large-v3` |
| Language | Language code (e.g. `ja`) or auto-detect |
| Speaker field mode | `speaker` (speaker_id column) or `caption` (VoiceDesign mode) |
| Output format | `CSV` or `JSONL` |

**Pipeline** — Runs slice → caption in sequence.

**Emoji Caption** — Extracts acoustic features (pitch, energy, speech rate, MFCC delta, ZCR) per segment and calls an LLM API to annotate text with Irodori-TTS compatible emotion emojis. Supported APIs: `lm_studio` / `groq` / `openai` / `together`.

**Voice Analysis Caption (VoiceDesign)** — In VoiceDesign mode, a second LLM pass generates a voice-quality caption (`caption` column) from acoustic features only (no spoken content).

> Emoji Caption requires CSV output and runs automatically after the main captioning job.

### Tab 6: Checkpoint Conversion (🔄)

- **Normal conversion** — Converts a training `.pt` checkpoint to inference-only `.safetensors` (model config embedded in metadata).

```bash
uv run python convert_checkpoint_to_safetensors.py outputs_train/irodori_tts/checkpoint_final.pt
# Output: checkpoint_final.safetensors (use --force to overwrite)
```

- **LoRA conversion** — Converts a `_full` LoRA checkpoint directory (with optimizer state) to an inference `_ema` adapter format.

### Tab 7: Model Merge (🔀)

Merges two checkpoints (`.pt` / `.safetensors`) after an architecture compatibility check.

| Method | Description |
|---|---|
| `weighted_average` | `result = α × A + (1 − α) × B` |
| `slerp` | Spherical linear interpolation; falls back to weighted average near zero vectors |
| `task_arithmetic` | `result = base + λA × (A − base) + λB × (B − base)`. Requires a separate base model |

**Partial merge** — Apply a different method per layer group (`text` / `speaker` / `diffusion_core` / `io`).
**LoRA-style injection** — `result = base + scale × (donor − base)` with selectable target groups.
Output format: `.safetensors` (recommended) or `.pt`.

### Tab 8: LoRA Merge (🧬)

- **Standard LoRA Merge** — Merges two LoRA adapters into a new adapter (not baked into the base). Same merge methods as model merge, with per-group partial merge. Output goes to `lora/lora_merged_*/`.
- **Bake into Base Model** — Fuses one or two LoRA adapters (each with an independent scale) into a base model checkpoint, with optional post-bake merge between two baked results and partial baking. Output goes to `checkpoints/lora_merged/`.

---

## Training Workflow

### 1. Prepare Manifest

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker \
  --output-manifest my_manifest/train_manifest.jsonl \
  --latent-dir my_manifest/latents \
  --device cuda
```

### 2. Training (Full / LoRA / Speaker Inversion / MeanFlow Distillation)

```bash
# Full (v3)
uv run python train.py --config configs/train_v3.yaml \
  --manifest my_manifest/train_manifest.jsonl --output-dir outputs_train/irodori_tts

# Full (v4-Small / v4.1-Small; pretrained text encoder gets a separate lower LR via
# pretrained_text_encoder_learning_rate, and its config is embedded into saved checkpoints)
uv run python train.py --config configs/train_v4_small.yaml \
  --manifest my_manifest/train_manifest.jsonl --output-dir outputs_train/irodori_tts_v4

# LoRA (base checkpoint version is auto-detected; v4 defaults target the
# diffusion attention only and fully train the duration predictor)
uv run python lora_train.py \
  --base-model checkpoints/Aratako_Irodori-TTS-v4.1-Small/model.safetensors \
  --manifest my_manifest/train_manifest.jsonl \
  --output-dir lora/my_run \
  --target-modules auto --modules-to-save auto \
  --lora-rank 16 --lora-alpha 32.0 --max-steps 1000

# Speaker Inversion (v4; freezes everything except learned speaker tokens and
# writes <step>.speaker.safetensors next to each checkpoint; use with infer.py --ref-embed)
uv run python train.py --config configs/train_v4_small_speaker_inversion.yaml \
  --manifest my_manifest/train_manifest.jsonl \
  --init-checkpoint checkpoints/Aratako_Irodori-TTS-v4-Small/model.safetensors \
  --output-dir outputs_train/speaker_inversion

# MeanFlow distillation (v4.1; initializes both student and teacher from an RF-trained
# checkpoint and trains only the DiT. The teacher builds targets with CFG-fused Euler
# rollout; resulting checkpoints are sampled at 4 NFE automatically. train.teacher_checkpoint
# is required in the config; resume distillation with --resume)
uv run python train.py --config configs/train_v4_small_meanflow.yaml \
  --manifest my_manifest/train_manifest.jsonl \
  --output-dir outputs_train/meanflow

# v4-Large (t5gemma-2-1b-1b based; the pretrained text encoder also runs on Muon)
uv run python train.py --config configs/train_v4_large.yaml \
  --manifest my_manifest/train_manifest.jsonl --output-dir outputs_train/irodori_tts_v4_large
```

### 3. Checkpoint Conversion (if needed)

```bash
uv run python convert_checkpoint_to_safetensors.py outputs_train/irodori_tts/checkpoint_final.pt
```

---

## Project Structure

```text
Irodori-TTS/
├── train.py                              # Full fine-tuning entry point (DDP support)
├── lora_train.py                         # LoRA fine-tuning entry point
├── infer.py                              # CLI inference
├── gradio_app.py                         # Gradio web UI (all features)
├── mani_gui.py                           # Simple manifest GUI (experimental tool)
├── prepare_manifest.py                   # Audio → DACVAE latent preprocessing
├── dataset_tools.py                      # Slice / Whisper caption / emoji & voice-analysis captions
├── merge.py                              # Model merge utilities
├── lora_merge.py                         # LoRA adapter merge / bake-in utilities
├── convert_checkpoint_to_safetensors.py  # .pt → .safetensors conversion
├── convert_lora_checkpoint.py            # LoRA _full → _ema conversion
├── start.bat                             # Gradio launcher (Windows)
│
├── irodori_tts/                          # Core library
│   ├── model.py                          # TextToLatentRFDiT (pretrained backbone / Duration Predictor)
│   ├── config.py                         # Model / Train config dataclasses
│   ├── rf.py                             # Rectified Flow utilities & Euler CFG sampling
│   ├── codec.py                          # DACVAE codec wrapper
│   ├── dataset.py                        # Dataset and collator
│   ├── tokenizer.py                      # Tokenizer wrappers
│   ├── inference_runtime.py              # Cached, thread-safe inference runtime
│   ├── duration.py                       # Duration Predictor support
│   ├── quantization.py                   # torchao quantized checkpoint parsing / quantization
│   ├── speaker_inversion.py              # Learned speaker tokens (v4-Small)
│   ├── attention.py                      # Shared self/context attention kernels (ContextAttentionPlan)
│   ├── meanflow.py                       # MeanFlow distillation loss, teacher rollout & few-step sampler (v4.1)
│   ├── watermark.py                      # Watermark utilities
│   ├── text_normalization.py             # Japanese text normalization
│   ├── optim.py                          # Muon / AdamW / Lion / AdEMAMix / SGD
│   └── progress.py                       # Training progress tracker
│
├── configs/                              # Training / LoRA preset YAMLs (model: / train: / lora: sections)
├── docs/                                 # Feature docs (meanflow.md: MeanFlow distillation training & inference)
├── checkpoints/                          # Downloaded and trained models (codecs/ tokenizers/ whisper/ caches)
├── lora/                                 # LoRA adapter outputs
├── speakers/                             # Named speaker profiles
├── logs/                                 # Training logs (JSONL, TensorBoard)
├── my_dataset/                           # Dataset processing output (metadata.csv, etc.)
├── my_manifest/                          # Manifest files and DACVAE latents
├── outputs_gradio/                       # Generated audio from the GUI
└── outputs_train/                        # Training outputs (checkpoints, config.json)
```

---

## Configuration Reference (training YAML)

YAML files are organized into `model:` / `train:` sections (LoRA presets use a `lora:` section). `train.py --config` loads them; CLI flags override.

Main `train:` keys (defaults from `TrainConfig`):

| Key | Default | Description |
|---|---|---|
| `batch_size` | `8` | Per-GPU batch size |
| `gradient_accumulation_steps` | `1` | Effective batch = batch × accum |
| `optimizer` | `muon` | `muon` / `adamw` / `lion` / `ademamix` / `sgd` |
| `learning_rate` | `1e-4` | Peak learning rate |
| `pretrained_text_encoder_learning_rate` | `1e-5` | LR for the pretrained text encoder part (v4 family) |
| `pretrained_text_encoder_optimizer` | `adamw` | Optimizer for the pretrained text encoder split (`adamw` or `muon`; `muon` requires `optimizer: muon`) |
| `pretrained_projector_warmup_steps` | `0` | Freeze pretrained projectors (`text_encoder.*` / `caption_encoder.*` grads cleared) for the first N steps |
| `train_mode` | `rf` | `rf` / `duration_only` / `meanflow_distill` (MeanFlow distillation; see `docs/meanflow.md`) |
| `teacher_checkpoint` / `teacher_steps` | `None` / `40` | MeanFlow teacher checkpoint (must be an RF model; required) / teacher Euler steps |
| `meanflow_*` | see `docs/meanflow.md` | MeanFlow distillation parameters (time sampling / teacher rollout chunking & device offset / CFG fusion / adaptive loss weights, etc.) |
| `gradient_checkpointing_store_every` | `0` | Keep activations for 1 of every N blocks when gradient checkpointing is on (fewer recomputes; 0 = recompute all). CLI: `--grad-checkpoint-store-every` |
| `lr_scheduler` | `none` | `wsd` / `cosine` / `none` |
| `warmup_steps` / `stable_steps` / `min_lr_scale` | `0` / `0` / `0.1` | WSD schedule |
| `max_steps` | `200000` | Total training steps |
| `max_text_len` / `max_caption_len` | `256` / `None` | Maximum conditioning token lengths |
| `max_latent_steps` | `750` | Maximum latent frame length |
| `text_condition_dropout` / `caption_condition_dropout` / `speaker_condition_dropout` | `0.1` | CFG-training condition dropouts |
| `caption_warmup` / `caption_warmup_steps` | `false` / `0` | Delayed caption-condition training |
| `speaker_inversion_enabled` / `speaker_inversion_tokens` | `false` / `16` | Learned speaker tokens (v4). Freezes everything else; saves `*.speaker.safetensors` per checkpoint (use with `infer.py --ref-embed`). Requires `--init-checkpoint`, adamw, rf mode; `--resume` not supported |
| `speaker_inversion_init_std` / `speaker_inversion_init_embedding` | `0.02` / `None` | Embedding init std / initialize from an existing `.speaker.safetensors` |
| `duration_loss_weight` / `duration_huber_delta` | `0.1` / `0.1` | Duration Predictor training |
| `valid_ratio` / `valid_every` | `0.0` / `0` | Validation split and interval |
| `save_every` | `1000` | Checkpoint save interval |
| `precision` | `bf16` | Training precision |
| `gradient_checkpointing` | `false` | VRAM saving (~40% less, ~20% slower) |
| `compile_model` | `false` | torch.compile |
| `wandb_enabled` | `false` | Weights & Biases logging |

The `model:` section accepts `latent_dim`, `use_caption_condition`, `use_duration_predictor`, `text_encoder_type: pretrained`, `text_tokenizer_repo`, `flow_parameterization` (default `rf_velocity`; MeanFlow outputs are marked `meanflow` automatically), etc. When starting from an existing model's `.safetensors`, the embedded config is inherited automatically.

---

## License

- **Code**: [MIT License](LICENSE)
- **Model weights**: Subject to the license of each Hugging Face model card (some are non-commercial). See the [Aratako/Irodori-TTS collection](https://huggingface.co/collections/Aratako/irodori-tts) for details.

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
