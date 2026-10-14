# Irodori-TTS (フォーク版)

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

English → [README.md](README.md)

> **本プロジェクトは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) のフォーク版です。**
> 本家から v4-Small / v3-VoiceDesign 系のアーキテクチャ（事前学習テキストエンコーダ、Duration Predictor、Caption条件、量子化チェックポイント、Speaker Inversion）を推論・学習系へ移植し、フォーク固有機能（LoRA学習・各種マージ・EMA・データセットツール・Gradio GUI拡張）を維持しています。
> モデルの重みは各Hugging Faceモデルカードのライセンスに従います。モデルは初回起動時に自動ダウンロードされます。

**Irodori-TTS** は Flow Matching ベースの音声合成モデルです。アーキテクチャと学習設計は [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) に準拠し、[DACVAE](https://github.com/facebookresearch/dacvae) の連続潜在表現を生成ターゲットとして使用します。

---

## 機能一覧

- **Flow Matching TTS** — 連続DACVAE潜在空間上のRectified Flow拡散トランスフォーマー（RF-DiT）
- **マルチバージョンモデル対応** — v1 / v2 / v3 / v2・v3-VoiceDesign / v4-Small / v4.1-Small まで同一コードで読込。モデルロード時にコーデック・条件構成を自動判定（下表参照）
- **量子化チェックポイント読込** — torchao で量子化された v4-Small-Quantized 系（int8 weight-only / int8 dynamic / int4 / float8）を推論に使用可能（`--extra quantization` が必要）
- **ゼロショット声質クローニング** — 参照音声から話者の声質を再現
- **Voice Design（キャプション条件）** — 声質・感情・スタイルを自然文キャプションで指定。v3-VoiceDesign / v4-Small では「テキスト + 参照音声 + キャプション」の3要素同時制御に対応
- **Duration Predictor** — 対応モデルでは出力長を自動予測。`--seconds` / `--duration-scale` で手動制御も可能
- **Speaker Inversion（v4系）** — 参照音声エンコーダの代わりに使える学習可能な話者トークン（`.speaker.safetensors`）。学習（`train.py` + `configs/train_v4_small_speaker_inversion.yaml`）・推論（`infer.py --ref-embed`）・Gradio UIまで対応
- **事前学習テキストエンコーダ（v4系）** — 事前学習バックボーン + プロジェクタ構造を読み込み（v4-Small/v4.1-Small は `sbintuitions/modernbert-ja-310m` 使用）。バックボーンは別LRに分離、エンコーダ設定は保存チェックポイントに埋め込まれ再学習時にHF再ダウンロード不要
- **MeanFlow蒸留学習（v4.1）** — `train_mode: meanflow_distill` でRF学習済みチェックポイントを数ステップ（既定4ステップ）の学生モデルへ蒸留（`configs/train_v4_small_meanflow.yaml`）。DiTのみ学習し、teacherはCFG融合付きオンラインEuler rolloutで標的を生成。MeanFlowチェックポイントは推論時に自動で4 NFEサンプリング（詳細は `docs/meanflow.md`）
- **絵文字スタイル制御** — テキスト中の絵文字で感情・効果音を制御（`EMOJI_ANNOTATIONS` 準拠）
- **感情スタイルプリセット** — ワンクリックでスタイル切替（ノーマル / 力強く / おとなしく / 明るく / ひそやかに）とCFGの手動微調整
- **スピーカーライブラリ** — 参照音声を名前付きプロファイル（`speakers/`）として登録し再利用
- **複数候補生成** — 1回の実行で最大8候補をバッチ生成
- **改行分割生成** — 行単位で個別生成、または無音区間を挟んで連結
- **ウォーターマーク** — dim32は SilentCipher、dim128は DACVAE 埋め込み。UI/CLIでON・OFF
- **LoRA差分学習** — `peft` 使用。Resume・EMA・Early Stopping・モジュール単位無効化・LoRA同士のマージ/焼き込み
- **フルファインチューニング** — Multi-GPU DDP、Muon/AdamW/Lion/AdEMAMix/SGD、WSD/Cosine、勾配チェックポイント（`store_every` による活性化部分保持対応）、TensorBoard・メトリクスログ
- **データセットツール** — Silero VADスライス、Whisperキャプション、LLM APIによる絵文字注釈、VoiceDesign用「音声分析キャプション」生成
- **モデルマージ** — Weighted Average・SLERP・Task Arithmetic・層グループ別部分マージ・LoRA的差分注入（v3系対応）
- **Gradio Web UI / ダークモード** — 全機能をブラウザから操作可能

---

## モデル対応表

ロードしたチェックポイントのメタデータ（`model_config` / `text_encoder_config_json` / 量子化メタデータ）から自動判定されます。

| モデル | HuggingFace リポジトリ | コーデック（潜在次元） | 条件付け | Duration<br>Predictor | 推論 | フル学習 | LoRA学習 |
|---|---|---|---|---|---|---|---|
| v1 | `Aratako/Irodori-TTS-500M` | `facebook/dacvae-watermarked`（dim128） | テキスト + 参照音声 | – | ✅ | ✅ | ✅ |
| v2 | `Aratako/Irodori-TTS-500M-v2` | `Aratako/Semantic-DACVAE-Japanese-32dim`（dim32） | テキスト + 参照音声 | – | ✅ | ✅ | ✅ |
| v2-VoiceDesign | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | 同上（dim32） | テキスト + キャプション | – | ✅ | ✅ | ✅ |
| v3 | `Aratako/Irodori-TTS-500M-v3` | 同上（dim32） | テキスト + 参照音声 | ✅ | ✅ | ✅ | ✅ |
| v3-VoiceDesign | `Aratako/Irodori-TTS-600M-v3-VoiceDesign` | 同上（dim32） | テキスト + 参照音声 + キャプション | ✅ | ✅ | ✅ | ✅ |
| v4-Small | `Aratako/Irodori-TTS-v4-Small` | 同上（dim32） | テキスト + 参照音声 + キャプション + 話者トークン | ✅ | ✅ | ✅ | ✅ |
| v4.1-Small | `Aratako/Irodori-TTS-v4.1-Small`（既定） | 同上（dim32） | テキスト + 参照音声 + キャプション + 話者トークン | ✅ | ✅ | ✅ | ✅ |

**補足**

- **量子化チェックポイント**（`Aratako/Irodori-TTS-v4-Small-Quantized` 等の torchao 形式）は **推論専用** です。読込には `uv sync --extra quantization`（torchao）が必要です。`--init-checkpoint` での学習再開には使用できません（フル精度ベースで学習 → マージ → 量子化の順で運用）。
- マニフェスト作成のモード対応: `model_v1`(dim128) / `model_v2` / `model_v3` / `model_v4`(既定、話者+caption列) / `voice_design`(caption列必須)。
- 学習時は YAML の `model:` セクションで `use_caption_condition`、`use_duration_predictor`、`text_encoder_type: pretrained` 等を指定できます。
- **v4-Large 学習コンフィグ** — `configs/train_v4_large.yaml` / `configs/train_v4_large_duration.yaml` を同梱（事前学習テキストエンコーダは `google/t5gemma-2-1b-1b`、本体/事前TEとも muon 分離）。上流でのHF公開は未定のため設定同梱のみ（モデルコードの変更不要なし）。
- **MeanFlowチェックポイント** — `train_mode: meanflow_distill` で生成したチェックポイントはメタデータに `flow_parameterization: meanflow` を持ち、推論（infer.py / Gradio）では自動で4 NFEサンプリングになります（実行時CFGは無効。CFGは学習時にteacherへ融合済み）。

---

## アーキテクチャ

モデルは主に4つのコンポーネントで構成されます（モデル世代により有無あり）。

1. **テキストエンコーダ** — スクラッチ学習（v1〜v3: `llm-jp/llm-jp-3-150m` 初期化埋め込み + RoPE付きSelf-Attention/SwiGLU）、または事前学習バックボーン + プロジェクタ（v4系: `sbintuitions/modernbert-ja-310m`）
2. **参照潜在エンコーダ** — パッチ化した参照音声潜在をエンコードし話者・スタイル条件付けに使用。v4-Smallでは学習済み話者トークン（Speaker Inversion）で置換可能
3. **拡散トランスフォーマー** — Low-Rank AdaLN、half-RoPE、SwiGLU MLPを持つJoint-Attention DiTブロック。テキスト/キャプション/話者の3ブランチ条件付け
4. **Duration Predictor** — テキスト + 話者/キャプション条件から音声長を予測（v3系以降、可変長学習）

音声はDACVAEコーデックで連続潜在列として表現されます。v2以降は32次元潜在（48 kHz）、v1は128次元潜在を使用します。

---

## インストール

```bash
git clone <このリポジトリ>
cd Irodori-TTS
uv sync
```

- Python 3.10 以上。Linux/Windows + CUDA では PyTorch が cu128 インデックスから自動インストールされます（macOS/CPU は既定のPyTorch）。
- 量子化チェックポイントを読む場合のみ追加:

```bash
uv sync --extra quantization   # torchao (int8/int4/float8 チェックポイント読込用)
```

- 音声スライス・Whisperキャプションには ffmpeg が必要です（Windows は `ffmpeg-7.1-full_build` を PATH へ登録）。

---

## クイックスタート

### Gradio Web UI

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
# Windows では start.bat をダブルクリックして起動することもできます
```

`http://localhost:7860` にアクセス。チェックポイントが無い場合は既定で `Aratako/Irodori-TTS-v4.1-Small` を自動ダウンロードします（v4.1-Small → v4-Small → v3 → v2 → v1 の順で既存ファイルがあればそれを採用）。UI内のダウンロード欄から任意のHFリポジトリIDを指定することも可能です。

| フラグ | 説明 |
|---|---|
| `--server-name` | バインドアドレス（既定 `127.0.0.1`） |
| `--server-port` | ポート（既定 `7860`） |
| `--share` | 公開Gradioリンクの作成 |
| `--debug` | Gradioデバッグモード |

### CLI推論

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs_gradio/sample.wav
```

VoiceDesignモデルでキャプション付き生成:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-600M-v3-VoiceDesign \
  --text "こんにちは、今日はいい天気ですね。" \
  --caption "明るく弾んだ声。少し早めの話速" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs_gradio/sample.wav
```

参照音声なし（無条件生成）:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav outputs_gradio/sample.wav
```

ローカルチェックポイント（`.pt` / `.safetensors` / 量子化safetensors）も使用できます:

```bash
uv run python infer.py \
  --checkpoint checkpoints/Aratako_Irodori-TTS-500M-v3/model.safetensors \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs_gradio/sample.wav
```

v4系でSpeaker Inversion embeddingを使用（参照音声不要。`--speaker-inversion` 学習で生成されます）:

```bash
uv run python infer.py \
  --checkpoint checkpoints/Aratako_Irodori-TTS-v4-Small/model.safetensors \
  --ref-embed outputs_train/speaker_inversion/checkpoint_final.speaker.safetensors \
  --text "今日はいい天気ですね。" \
  --caption "clear bright female voice" \
  --output-wav outputs_gradio/sample.wav
```

---

## 推論パラメータ（infer.py）

| パラメータ | 既定値 | 説明 |
|---|---|---|
| `--checkpoint` / `--hf-checkpoint` | *(必須・排他)* | ローカルckpt / HFリポジトリID |
| `--text` | *(必須)* | 合成テキスト（絵文字スタイル制御可） |
| `--caption` | `None` | VoiceDesignモデル用キャプション（スタイル記述） |
| `--ref-wav` / `--ref-latent` / `--ref-embed` / `--no-ref` | *(必須・排他)* | 参照音声 / 事前計算潜在(.pt) / Speaker Inversion embedding（`.speaker.safetensors`、v4系） / 無条件 |
| `--max-ref-seconds` | `30.0` | 参照音声の最大秒数（<=0で無制限） |
| `--ref-normalize-db` | `-16.0` | 参照音声のloudness正規化目標（`none`で無効） |
| `--ref-ensure-max` | `True` | 参照音声のピークが1.0を超える時のみ縮小 |
| `--num-steps` | `None`（自動） | Euler積分ステップ数。未指定時はMeanFlow ckpt=4 / RF ckpt=40に自動設定 |
| `--num-candidates` | `1` | 1回の生成で得る候補数（バッチ生成） |
| `--decode-mode` | `sequential` | 候補の復号方式（`sequential`=低VRAM / `batch`=高速） |
| `--seconds` | `None` | 出力秒数の手動指定（省略時はDuration Predictor、非対応ckptは30秒） |
| `--duration-scale` | `1.0` | Duration Predictor予測値の倍率（>1長め、<1短め） |
| `--cfg-scale-text` | `3.0` | テキスト条件のCFGスケール |
| `--cfg-scale-caption` | `3.0` | キャプション条件のCFGスケール |
| `--cfg-scale-speaker` | `5.0` | 話者条件のCFGスケール |
| `--cfg-guidance-mode` | `independent` | CFG方式（`independent` / `joint` / `alternating`） |
| `--cfg-min-t` / `--cfg-max-t` | `0.5` / `1.0` | CFGを適用するタイムステップ範囲 |
| `--context-kv-cache` | `True` | テキスト/話者KVのキャッシュ |
| `--truncation-factor` | `None` | 初期ノイズスケーリング（表現レンジの制御） |
| `--rescale-k` / `--rescale-sigma` | `None` | 時間的スコア再スケーリング（Xu et al. 2025） |
| `--speaker-kv-scale` | `None` | 話者KV_attentionスケール（>1で話者同一性強調） |
| `--speaker-kv-min-t` | `0.9` | 話者KVスケーリングを停止するタイムステップ |
| `--speaker-kv-max-layers` | `None` | スケーリングを適用する先頭層数 |
| `--max-text-len` / `--max-caption-len` | ckpt値→256 | 条件の最大トークン長 |
| `--model-device` / `--codec-device` | auto | 実行デバイス（`cuda` / `mps` / `cpu`） |
| `--model-precision` / `--codec-precision` | `fp32` | 精度（`fp32` / `bf16`） |
| `--enable-watermark` | `False` | 透かし埋め込み（dim32=SilentCipher / dim128=DACVAE） |
| `--codec-repo` | `Aratako/Semantic-DACVAE-Japanese-32dim` | 復号コーデック（モデルに合わせ自動選択可） |
| `--lora-path` / `--lora-scale` / `--lora-disabled-modules` | `None` / `1.0` / `""` | LoRAアダプタ読込・スケール・モジュール単位無効化 |
| `--compile-model` / `--compile-dynamic` | `False` | torch.compile 有効化 |
| `--trim-tail` | `True` | 末尾無音の平坦化トリム |
| `--tail-window-size` / `--tail-std-threshold` / `--tail-mean-threshold` | `20` / `0.05` / `0.1` | 末尾トリムのしきい値調整 |
| `--seed` | ランダム | 再現性のためのシード |
| `--show-timings` | `True` | ステージ別の処理時間表示 |

---

## Gradio UI — タブ一覧

### タブ1: 推論（🔊）

- **モデル読込** — `checkpoints/` 内のチェックポイント選択、またはHFリポジトリIDで直接ダウンロード。デバイス・精度・ウォーターマークのON/OFF。潜在次元に合わせたコーデック自動選択、量子化ckptの自動判定。
- **LoRAアダプタ** — `lora/` 内のアダプタ選択、スケール（0.0–2.0）、無効化モジュールの指定。
- **参照音声** — ①直接アップロード ②スピーカーライブラリ ③スピーカー登録（`speakers/{name}/` に `ref.wav` / `ref.pt` / `profile.json` を保存） ④Speaker Inversion（v4系の話者トークン学習が生成した `*.speaker.safetensors` を選択。参照音声不要）。
- **Voice Design** — キャプション対応モデル読込時のみ、キャプション欄とキャプションCFGが表示されます。
- **感情スタイルプリセット** — Normal / Strong / Calm / Bright / Whisper。
- **スタイルスライダー** — テキスト表現力、感情強度、話者追従、表現レンジ。
- **サンプリング** — ステップ数（1–120）、シード、候補数（1–8）、改行分割生成（個別 / 連結+無音間隔）。
- **CFG設定** — ガイダンスモード、テキスト/キャプション/話者CFG、タイムステップ範囲、KVキャッシュ、スコア再スケーリング、話者KVスケール。
- 生成音声は `outputs_gradio/` に保存されます。

### タブ2: Prepare Manifest（📂）

音声データをDACVAE潜在へ変換し、学習用JSONLマニフェストを生成します。

データソース: ローカルCSV（`audiofolder`形式）/ ローカルJSONL / Hugging Faceデータセット。音声・テキスト・話者IDの列名はヘッダから自動検出。

| モード | コーデック | 説明 |
|---|---|---|
| `model_v4` | dim32（既定） | 話者 + キャプション列、v4-Small / v4.1-Small用 |
| `model_v3` | dim32 | 話者条件、v3モデル用 |
| `model_v2` | dim32 | 話者条件、v2モデル用 |
| `model_v1` | dim128 | 話者条件、v1モデル用 |
| `voice_design` | dim32 | キャプション条件（caption列必須） |

```bash
# CLI同等
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest my_manifest/train_manifest.jsonl \
  --latent-dir my_manifest/latents \
  --device cuda
```

出力マニフェスト形式:

```json
{"text": "こんにちは", "latent_path": "my_manifest/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### タブ3: 学習（🏋️）

フルファインチューニング（Multi-GPU DDP対応）。

| グループ | 内容 |
|---|---|
| ベースモデル | `.safetensors`（step 0から）/ `.pt`チェックポイント（step・optimizer状態含む）から再開。量子化ckptは使用不可 |
| バッチ | バッチサイズ、勾配蓄積、DataLoaderワーカー |
| オプティマイザ | `muon` / `adamw` / `lion` / `ademamix` / `sgd`（YAML設定経由。CLIフラグは `muon`/`adamw` のみ） |
| スケジューラ | `wsd`（warmup→stable→decay）/ `cosine` / `none` |
| 精度 | `bf16` / `fp32` |
| Attentionバックエンド | `sdpa`（推奨）/ `flash2` / `eager` |
| 正則化 | テキスト/キャプション/話者条件dropout、タイムステップ層化サンプリング |
| Duration学習 | `duration_loss_weight` 等（Duration Predictor対応モデル） |
| 検証 | 検証分割率、検証間隔、Early Stopping |
| EMA | 推論用チェックポイントのためのEMA（既定はEMA版保存、`--save-full` で学習再開可能なフル版も） |
| チェックポイント | 保存間隔、EMAのみ / EMA+フル |
| ログ | メトリクスJSONL（`logs/`）、ログ間隔、リアルタイムLoss/LRグラフ、終了時にTensorBoardイベント生成 |

プリセットは `configs/` のYAMLをプルダウン選択・保存・更新できます。コマンドプレビューで実行CLIをリアルタイム表示します。

単一GPU:

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

### タブ4: LoRA学習（🚀）

凍結したベースモデル上にLoRAアダプタを学習します（`peft` 同梱）。

| 設定 | 既定値 | 説明 |
|---|---|---|
| モデルバージョン | `auto` | `auto` はベースチェックポイントから自動判別（検出結果はUIに表示）。明示選択するとそのバージョンのLoRA既定値に切替 |
| LoRA rank | `16` | 低ランク次元 |
| lora_alpha | `32.0` | スケーリング係数 |
| lora_dropout | `0.05` | Dropout率 |
| 対象モジュール | `auto` | `auto` = バージョン既定値（v1〜v3: `wq,wk,wv,wo` / v4系: 拡散attention限定regex。事前学習バックボーンは対象外）。カンマ区切り list や `^` 始まり正規表現1本の直接指定も可 |
| modules_to_save | `auto` | LoRAなしで全パラメータ学習するモジュール。`auto` = v4系: `duration_predictor` / その他: なし |
| 保存モード | EMAのみ | `EMAのみ`（推論用）/ `EMA + full`（学習再開可） |
| Attentionバックエンド | `sdpa` | `sdpa` / `flash2` / `sage` / `eager` |
| EMA / Early Stopping / Resume | — | 学習継続・早期停止・`_full`アダプタからの再開 |

v4系向けプリセットとして `configs/train_v4_small_lora.yaml` を同梱しています。

プリセット（`lora:` セクション付きYAML）は `configs/` に保存・読込可能です。
推論タブでは「LoRA無効化モジュール」欄でモジュール名（カンマ区切り）を指定すると、モデル再読込なしで選択したLoRA重みをゼロ化できます。

### タブ5: Dataset作成（🎙️）

**スライス** — Silero VADで長い音声をセグメント分割。

| パラメータ | 説明 |
|---|---|
| 最小/最大長（秒） | 許容セグメント長 |
| VAD threshold | 音声検出感度（0.5推奨） |
| 最小無音長（ms） | 分割のトリガーとなる無音長 |
| Speech pad（ms） | 前後の余白 |
| 対象サンプルレート | 任意のリサンプリング |
| 再帰的 | サブディレクトリ探索 |

**キャプション** — Whisperで文字起こししマニフェストを出力。

| パラメータ | 説明 |
|---|---|
| Whisperモデル | `tiny` / `base` / `small` / `medium` / `large-v3` |
| 言語 | 言語コード（例 `ja`）または自動判定 |
| 話者フィールドモード | `speaker`（speaker_id列）/ `caption`（VoiceDesignモード） |
| 出力形式 | `CSV` / `JSONL` |

**パイプライン** — スライス → キャプションを連続実行。

**絵文字キャプション** — 各音声から音響特徴量（ピッチ、エネルギー、話速、MFCC差分、ZCR）を抽出し、LLM APIでIrodori-TTS互換の感情絵文字注釈を付与します。対応API: `lm_studio` / `groq` / `openai` / `together`。

**音声分析キャプション（VoiceDesign）** — VoiceDesignモード選択時、LLMへ2回目の呼び出しを行い、音響特徴のみからなる声質記述キャプション（`caption`列）を自動生成します。読み上げ内容は含みません。

> 絵文字キャプションはCSV出力形式が必須で、本体キャプション処理の後に自動実行されます。

### タブ6: チェックポイント変換（🔄）

- **通常変換** — 学習用 `.pt` を推論専用 `.safetensors` へ変換（モデル設定をメタデータへ埋め込み）。

```bash
uv run python convert_checkpoint_to_safetensors.py outputs_train/irodori_tts/checkpoint_final.pt
# 出力: checkpoint_final.safetensors（上書きは --force）
```

- **LoRA変換** — `_full` LoRAチェックポイント（optimizer状態込み）を推論用 `_ema` アダプタ形式へ変換。

### タブ7: モデルマージ（🔀）

2つのチェックポイント（`.pt` / `.safetensors`）をアーキテクチャ互換性チェック後にマージします。

| 方法 | 説明 |
|---|---|
| `weighted_average` | `result = α × A + (1 − α) × B` |
| `slerp` | 球面線形補間。ゼロ近傍は加重平均へフォールバック |
| `task_arithmetic` | `result = base + λA × (A − base) + λB × (B − base)`。ベースモデル別途必要 |

**部分マージ** — 層グループ（`text` / `speaker` / `diffusion_core` / `io`）ごとに別方法を適用。
**LoRA的差分注入** — `result = base + scale × (donor − base)`。対象グループを選択可。
出力は `.safetensors`（推奨）/ `.pt`。

### タブ8: LoRAマージ（🧬）

- **通常LoRAマージ** — 2つのLoRAアダプタを新アダプタへ統合（ベースモデルには焼き込まない）。マージ方法はモデルマージと同様、グループ別部分マージ対応。出力は `lora/lora_merged_*/`。
- **本体モデルマージ（焼き込み）** — 1〜2つのLoRAを独立スケールでベースモデルへ融合。融合結果同士のpost-bakeマージ、部分焼き込みにも対応。出力は `checkpoints/lora_merged/`。

---

## 学習フロー

### 1. マニフェスト準備

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

### 2. 学習（フル / LoRA / Speaker Inversion / MeanFlow蒸留）

```bash
# フル（v3）
uv run python train.py --config configs/train_v3.yaml \
  --manifest my_manifest/train_manifest.jsonl --output-dir outputs_train/irodori_tts

# フル（v4-Small / v4.1-Small。事前学習テキストエンコーダは pretrained_text_encoder_learning_rate
# で別LRに分離。テキストエンコーダ設定は保存チェックポイントに自動埋め込み）
uv run python train.py --config configs/train_v4_small.yaml \
  --manifest my_manifest/train_manifest.jsonl --output-dir outputs_train/irodori_tts_v4

# LoRA（ベースのバージョンを自動判別。v4系既定は拡散attention限定LoRA + duration_predictor全学習）
uv run python lora_train.py \
  --base-model checkpoints/Aratako_Irodori-TTS-v4.1-Small/model.safetensors \
  --manifest my_manifest/train_manifest.jsonl \
  --output-dir lora/my_run \
  --target-modules auto --modules-to-save auto \
  --lora-rank 16 --lora-alpha 32.0 --max-steps 1000

# Speaker Inversion（v4系。話者トークン以外を凍結し、各チェックポイント隣に
# <step>.speaker.safetensors を保存。infer.py --ref-embed で使用可能）
uv run python train.py --config configs/train_v4_small_speaker_inversion.yaml \
  --manifest my_manifest/train_manifest.jsonl \
  --init-checkpoint checkpoints/Aratako_Irodori-TTS-v4-Small/model.safetensors \
  --output-dir outputs_train/speaker_inversion

# MeanFlow蒸留（v4.1。RF学習済みckptをstudent/teacher両方の初期化に使い、
# DiTのみ学習。teacherはCFG融合付きEuler rolloutで標的生成。生成ckptの推論は
# 自動で4 NFE。teacher_checkpointはconfig側で必須、--resume で蒸留再開可）
uv run python train.py --config configs/train_v4_small_meanflow.yaml \
  --manifest my_manifest/train_manifest.jsonl \
  --output-dir outputs_train/meanflow

# v4-Large（t5gemma-2-1b-1bベース。事前学習TEもmuonに分離して学習）
uv run python train.py --config configs/train_v4_large.yaml \
  --manifest my_manifest/train_manifest.jsonl --output-dir outputs_train/irodori_tts_v4_large
```

### 3. チェックポイント変換（必要に応じて）

```bash
uv run python convert_checkpoint_to_safetensors.py outputs_train/irodori_tts/checkpoint_final.pt
```

---

## プロジェクト構成

```text
Irodori-TTS/
├── train.py                              # フル学習エントリポイント（DDP対応）
├── lora_train.py                         # LoRA学習エントリポイント
├── infer.py                              # CLI推論
├── gradio_app.py                         # Gradio Web UI（全機能）
├── mani_gui.py                           # マニフェスト生成の簡易GUI（旧版・動作しない → Prepare Manifestタブを使用）
├── prepare_manifest.py                   # 音声 → DACVAE潜在 前処理
├── dataset_tools.py                      # スライス / Whisperキャプション / 絵文字・音声分析キャプション
├── merge.py                              # モデルマージ
├── lora_merge.py                         # LoRAマージ / 焼き込み
├── convert_checkpoint_to_safetensors.py  # .pt → .safetensors 変換
├── convert_lora_checkpoint.py            # LoRA _full → _ema 変換
├── start.bat                             # Gradio起動用（Windows）
│
├── irodori_tts/                          # コアライブラリ
│   ├── model.py                          # TextToLatentRFDiT（事前学習バックボーン/Duration Predictor含む）
│   ├── config.py                         # Model / Train 設定データクラス
│   ├── rf.py                             # Rectified Flow と Euler CFG サンプリング
│   ├── codec.py                          # DACVAEコーデックラッパー
│   ├── dataset.py                        # Dataset と Collator
│   ├── tokenizer.py                      # トークナイザーラッパー
│   ├── inference_runtime.py              # キャッシュ対応・スレッドセーフな推論ランタイム
│   ├── duration.py                       # Duration Predictor関連
│   ├── quantization.py                   # torchao量子化チェックポイントの解析・量子化
│   ├── speaker_inversion.py              # 学習済み話者トークン（v4-Small）
│   ├── attention.py                      # セルフ/コンテキストattention共通実装（ContextAttentionPlan）
│   ├── meanflow.py                       # MeanFlow蒸留loss・teacher rollout・数ステップサンプラー（v4.1）
│   ├── watermark.py                      # 透かし関連ユーティリティ
│   ├── text_normalization.py             # 日本語テキスト正規化
│   ├── optim.py                          # Muon / AdamW / Lion / AdEMAMix / SGD
│   └── progress.py                       # 学習進捗トラッカー
│
├── configs/                              # 学習・LoRAプリセットYAML（model: / train: / lora: セクション）
├── docs/                                 # 機能ドキュメント（meanflow.md = MeanFlow蒸留の学習・推論手順）
├── checkpoints/                          # 学習済み・DL済みモデル（codecs/ tokenizers/ whisper/ キャッシュ含む）
├── lora/                                 # LoRAアダプタ出力
├── speakers/                             # 名前付きスピーカープロファイル
├── logs/                                 # 学習ログ（JSONL・TensorBoard）
├── my_dataset/                           # データセット処理の出力先（metadata.csv 等）
├── my_manifest/                          # マニフェストと潜在ファイル
├── outputs_gradio/                       # GUI生成音声
└── outputs_train/                        # 学習出力（チェックポイント・config.json）
```

---

## 設定リファレンス（train系YAML）

YAMLは `model:` / `train:`（LoRAプリセットは `lora:`）のセクションで構成します。`train.py --config` で読み込み、CLIフラグが上書きします。

主な `train:` キー（既定値は `TrainConfig`）:

| キー | 既定値 | 説明 |
|---|---|---|
| `batch_size` | `8` | GPUあたりバッチサイズ |
| `gradient_accumulation_steps` | `1` | 実効バッチ = batch × accum |
| `optimizer` | `muon` | `muon` / `adamw` / `lion` / `ademamix` / `sgd` |
| `learning_rate` | `1e-4` | ピーク学習率 |
| `pretrained_text_encoder_learning_rate` | `1e-5` | 事前学習テキストエンコーダ部の学習率（v4系。オプティマイザは本体と分離され `pretrained_text_encoder_optimizer` で選択） |
| `pretrained_text_encoder_optimizer` | `adamw` | 事前学習テキストエンコーダ分離用のオプティマイザ（`adamw` / `muon`。`muon` は `optimizer: muon` 必須） |
| `pretrained_projector_warmup_steps` | `0` | 事前学習プロジェクタ（`text_encoder.*` / `caption_encoder.*`）の勾配を最初のNステップ間クリア（凍結相当） |
| `train_mode` | `rf` | `rf` / `duration_only` / `meanflow_distill`（MeanFlow蒸留。詳細は `docs/meanflow.md`） |
| `teacher_checkpoint` / `teacher_steps` | `None` / `40` | MeanFlow蒸留のteacherチェックポイント（RF版・必須）/ teacherのEulerステップ数 |
| `meanflow_*` | `docs/meanflow.md` 参照 | MeanFlow蒸留パラメータ（時間サンプリング / teacher rollout分割・deviceオフセット / CFG融合 / 適応loss重み等） |
| `gradient_checkpointing_store_every` | `0` | 勾配チェックポイント時に1/Nブロックごとに活性化を保持し再計算を削減（0=全ブロック再計算。CLI `--grad-checkpoint-store-every`） |
| `lr_scheduler` | `none` | `wsd` / `cosine` / `none` |
| `warmup_steps` / `stable_steps` / `min_lr_scale` | `0` / `0` / `0.1` | WSDスケジュール |
| `max_steps` | `200000` | 総学習ステップ |
| `max_text_len` / `max_caption_len` | `256` / `None` | 条件の最大トークン長 |
| `max_latent_steps` | `750` | 潜在フレーム長上限 |
| `text_condition_dropout` / `caption_condition_dropout` / `speaker_condition_dropout` | `0.1` | CFG学習用dropout |
| `caption_warmup` / `caption_warmup_steps` | `false` / `0` | キャプション条件の学習遅延 |
| `speaker_inversion_enabled` / `speaker_inversion_tokens` | `false` / `16` | 学習済み話者トークン（v4系）。トークン以外を凍結し、各チェックポイント隣に `*.speaker.safetensors` を保存（`infer.py --ref-embed` で使用）。`--init-checkpoint` 必須・adamw・rfモード限定・`--resume` 不可 |
| `speaker_inversion_init_std` / `speaker_inversion_init_embedding` | `0.02` / `None` | 初期化std / 既存 `.speaker.safetensors` からの初期化 |
| `duration_loss_weight` / `duration_huber_delta` | `0.1` / `0.1` | Duration Predictor学習 |
| `valid_ratio` / `valid_every` | `0.0` / `0` | 検証分割・間隔 |
| `save_every` | `1000` | チェックポイント保存間隔 |
| `checkpoint_best_n` | `0` | 検証loss上位N件のチェックポイント保持（0=無効） |
| `precision` | `bf16` | 学習精度 |
| `gradient_checkpointing` | `false` | VRAM削減（約40%減・速度約20%低下） |
| `compile_model` | `false` | torch.compile |
| `wandb_enabled` | `false` | Weights & Biasesログ |

`model:` セクションでは `latent_dim`、`use_caption_condition`、`use_duration_predictor`、`text_encoder_type: pretrained`、`text_tokenizer_repo`、`flow_parameterization`（既定 `rf_velocity`。MeanFlow蒸留の成果物は自動で `meanflow`）等を設定できます。既存モデルの `.safetensors` から開始する場合はメタデータの設定が自動引き継ぎされます。

---

## ライセンス

- **コード**: [MIT License](LICENSE)
- **モデルの重み**: 各Hugging Faceモデルカードのライセンスに従います（非商用のモデルを含む）。詳細は [Aratako/Irodori-TTS コレクション](https://huggingface.co/collections/Aratako/irodori-tts) を参照してください。

---

## 謝辞

- [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) — 本フォークのオリジナル
- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — アーキテクチャ・学習設計の参考
- [DACVAE](https://github.com/facebookresearch/dacvae) — 音声VAEコーデック

---

## 引用

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
