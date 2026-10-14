# MeanFlow Distillation

Rectified Flow (RF) models generate audio latents from noise. During inference, an RF model repeatedly
predicts the instantaneous velocity at the current timestep and follows that velocity from noise
toward an audio latent. This requires multiple model evaluations along the sampling trajectory.

Using fewer RF steps makes each update larger, but the model still predicts the velocity at only
the start of that update. MeanFlow instead learns the average velocity over the interval. This
lets the model learn how to make larger updates for generation with fewer sampling steps.

In the distillation approach used here, a frozen RF teacher generates the trajectory within each sampled
interval. The student's target is the average velocity calculated from that trajectory. The
student receives the current noisy latent, timestep, interval length, and the same text, speaker,
and caption conditions as the teacher.

The Irodori-TTS implementation is based on
[dots.tts](https://github.com/studio-dots-ai/dots.tts). The provided recipe distills
[Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)
into [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF).

## Inference

MeanFlow checkpoints use the same CLI, Gradio applications, and `InferenceRuntime` interface as
RF checkpoints. The `flow_parameterization` stored in checkpoint metadata selects the sampler
automatically.

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small-MF \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_meanflow.wav
```

MeanFlow inference defaults to four sampling steps. Use `--num-steps` to change the step count.
In either Gradio app, enter the model repository in the checkpoint field and leave **Num Steps**
blank to use the model default. Enter a number to override it.

Classifier-free guidance (CFG) combines model predictions with and without a condition to control
how strongly generation follows that condition. Here, text, caption, and speaker guidance is
applied to the teacher trajectory during distillation. The student learns that guided trajectory
and needs only one conditional DiT evaluation per sampling step.

The following RF-only controls are ignored for MeanFlow checkpoints:

- CFG scales, guidance mode, and CFG time bounds
- Sway Sampling
- initial-noise truncation and temporal score rescaling
- speaker-KV scaling

Text, caption, reference audio, duration control, candidate batching, seeds, and decode mode keep
their normal meanings.

## Distillation

Set `train.teacher_checkpoint` in `configs/train_v4_small_meanflow.yaml` to a local v4.1-Small
RF checkpoint (`.pt` or `.safetensors`), then run:

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small_meanflow.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_v4_1_meanflow
```

A new run initializes both teacher and student from `teacher_checkpoint`; `--init-checkpoint`
is unnecessary. To continue a MeanFlow training run, pass its training checkpoint with `--resume`.

The training path:

1. Strictly loads the same RF weights into teacher and student.
2. Adds an interval-length embedding to the student, with its final projection initialized to zero.
3. Freezes the teacher, condition encoders, and duration predictor; trains the entire student DiT.
4. Samples interval endpoints. Zero-length intervals use the RF target (noise minus data latent).
5. Integrates the teacher over each remaining interval using `teacher_steps` Euler substeps.
6. Fuses independent text, caption, and speaker CFG into the teacher trajectory.
7. Updates the student DiT with an adaptive, per-utterance average-velocity loss.

The frozen condition encoders are evaluated once per batch. Teacher context KV tensors are cached
per sample chunk and reused across rollout steps and CFG branches.

### Main configuration fields

| Field | Recipe value | Meaning |
|-------|-------------:|---------|
| `teacher_checkpoint` | local path | RF checkpoint used to initialize teacher and student |
| `teacher_steps` | `40` | Euler substeps used for each teacher interval rollout |
| `meanflow_anchor_prob` | `0.5` | Probability of using a zero-length analytic RF anchor |
| `meanflow_time_logit_mean` | `0.4` | Mean of the Irodori-time logit-normal endpoint distribution |
| `meanflow_time_logit_std` | `1.0` | Standard deviation of the endpoint distribution |
| `meanflow_adaptive_weight_power` | `0.5` | Exponent of the detached adaptive loss weight |
| `meanflow_cfg_text_scale` | `3.0` | Text guidance fused into the teacher target |
| `meanflow_cfg_caption_scale` | `4.0` | Caption guidance fused into the teacher target |
| `meanflow_cfg_speaker_scale` | `5.0` | Speaker guidance fused into the teacher target |
| `meanflow_cfg_min_t` / `meanflow_cfg_max_t` | `0.5` / `1.0` | Teacher timestep range in which CFG is active |
| `meanflow_teacher_chunk_size` | `24` | Maximum samples processed together in a teacher rollout; smaller chunks use less memory |
| `meanflow_teacher_branch_batch_size` | `3` | Number of dropped-condition branches evaluated together |
| `meanflow_teacher_fuse_base_branch` | `true` | Evaluates the base and dropped branches in one teacher call |
| `meanflow_compile_teacher` | `false` | Optionally compiles the frozen teacher DiT callable |
| `meanflow_teacher_device_offset` | `0` | Maps each student rank to a separate teacher GPU when positive |

On resume, chunk size, branch batch size, compilation, and device placement can change. The
training code requires the saved teacher checkpoint path, teacher steps, time distribution,
anchor probability, CFG, loss, and condition-dropout settings to match the current config.

### Separate teacher GPUs

By default, each teacher shares a GPU with its student. For four visible GPUs, set the following
field under `train` in `configs/train_v4_small_meanflow.yaml`:

```yaml
train:
  meanflow_teacher_device_offset: 2
```

Launch two student processes. Students use `cuda:0,1`; their teachers use `cuda:2,3`.

```bash
uv run --no-sync torchrun --nproc_per_node=2 train.py \
  --config configs/train_v4_small_meanflow.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_v4_1_meanflow
```

With eight visible GPUs, four student ranks and an offset of four map teachers to `cuda:4..7`.
Separating devices reduces per-GPU memory pressure but does not overlap teacher and student work
automatically.

## Export

Convert a `.pt` training checkpoint to the inference-only `.safetensors` format:

```bash
uv run --no-sync python convert_checkpoint_to_safetensors.py \
  outputs/irodori_v4_1_meanflow/checkpoint_final.pt \
  --output path/to/model.safetensors
```

The converter stores `flow_parameterization: meanflow` in safetensors metadata and exports the
tokenizer beside the model. Keep `model.safetensors` and the generated `tokenizer/` directory
together.

## Reference

- [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) — Original MeanFlow paper
- [dots.tts](https://github.com/studio-dots-ai/dots.tts) — MeanFlow implementation reference
