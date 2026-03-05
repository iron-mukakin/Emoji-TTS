# Irodori-TTS (フォーク版)

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

> **本プロジェクトは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) のフォーク版です。**  
> ベースモデルの重みは元のライセンスに従い商用利用禁止です。モデルは初回起動時に自動ダウンロードされます。

**Irodori-TTS** は Flow Matching ベースの音声合成モデルです。アーキテクチャと学習設計は [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) に準拠し、[DACVAE](https://github.com/facebookresearch/dacvae) の連続潜在表現を生成ターゲットとして使用します。

オリジナルのモデル重みと音声サンプルは[モデルカード](https://huggingface.co/Aratako/Irodori-TTS-500M)を参照してください。

---

## 機能一覧

- **Flow Matching TTS** — 連続DACVAE潜在空間上のRectified Flow拡散トランスフォーマー（RF-DiT）
- **ゼロショット声質クローニング** — 参照音声から話者の声質を再現
- **感情スタイルプリセット** — ワンクリックでスタイルを切り替え（ノーマル / 力強く / おとなしく / 明るく / ひそやかに）、CFGパラメータの細かい手動調整も可能
- **複数候補生成** — 1回の実行で最大8候補の音声を同時生成
- **LoRA差分学習** — `peft` を使った軽量アダプタ学習。Resume・EMA・Early Stopping対応
- **フルファインチューニング** — Multi-GPU DDP学習。Muon/AdamW/Lion/AdEMAMixオプティマイザ、WSD/Cosineスケジューラ、勾配チェックポイント対応
- **データセットツール** — 音声スライス、Whisperキャプション、LLM APIを使った絵文字スタイルアノテーション
- **モデルマージ** — Weighted Average・SLERP・Task Arithmetic・部分マージ・LoRA的差分注入
- **Gradio Web UI** — 全機能をブラウザから操作可能

---

## アーキテクチャ

モデルは3つの主要コンポーネントで構成されます。

1. **テキストエンコーダ** — 事前学習済みLLM（`llm-jp/llm-jp-3-150m`）から初期化したトークン埋め込みと、RoPE付きSelf-Attention + SwiGLUトランスフォーマー層
2. **参照潜在エンコーダ** — パッチ化した参照音声の潜在表現をエンコードし、話者・スタイルの条件付けに使用するSelf-Attention + SwiGLU層
3. **拡散トランスフォーマー** — Low-Rank AdaLN（タイムステップ条件付き適応層正規化）、half-RoPE、SwiGLU MLPを持つJoint-Attention DiTブロック

音声はDACVAEコーデック（128次元）で連続潜在列として表現され、48 kHz高品質波形に変換されます。

---

## インストール

```bash
git clone <このリポジトリ>
cd Irodori-TTS
uv sync
```

> **注意:** Linux/Windows + CUDA環境ではPyTorchがcu128インデックスから自動インストールされます。macOS（MPS）またはCPUのみの環境では`uv sync`がデフォルトのPyTorchをインストールします。

---

## クイックスタート

### Gradio Web UI

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

ブラウザで `http://localhost:7860` にアクセスしてください。  
チェックポイントが見つからない場合、初回起動時に `Aratako/Irodori-TTS-500M` が自動ダウンロードされます。

起動オプション:

| オプション | 説明 |
|---|---|
| `--server-name` | バインドアドレス（デフォルト: `127.0.0.1`） |
| `--server-port` | ポート番号（デフォルト: `7860`） |
| `--share` | Gradioの公開URLを生成 |
| `--debug` | Gradioデバッグモードを有効化 |

### CLIで推論

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

参照音声なし（無条件生成）:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

---

## 推論

### 推論パラメータ一覧

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--text` | *(必須)* | 合成するテキスト |
| `--ref-wav` | `None` | 声質クローニング用の参照音声ファイル |
| `--ref-latent` | `None` | 事前計算済みの参照潜在表現（`.pt`） |
| `--no-ref` | `False` | 参照なしの無条件生成 |
| `--num-steps` | `40` | Euler積分のステップ数 |
| `--cfg-scale-text` | `3.0` | テキスト条件付けのCFGスケール |
| `--cfg-scale-speaker` | `5.0` | 話者条件付けのCFGスケール |
| `--guidance-mode` | `independent` | CFGモード: `independent` / `joint` / `alternating` |
| `--cfg-min-t` | `0.5` | CFGを適用し始める拡散タイムステップ |
| `--cfg-max-t` | `1.0` | CFGを適用し終える拡散タイムステップ |
| `--context-kv-cache` | `True` | テキスト・話者のKV射影をステップ間でキャッシュ（高速化） |
| `--truncation-factor` | `None` | 表現の振れ幅制御のための潜在トランケーション |
| `--rescale-k` | `None` | スコア再スケールk（Xu et al. 2025） |
| `--rescale-sigma` | `None` | スコア再スケールsigma |
| `--speaker-kv-scale` | `None` | 話者KVアテンションのスケール係数 |
| `--speaker-kv-min-t` | `None` | 話者KVスケールを適用し始めるタイムステップ |
| `--speaker-kv-max-layers` | `None` | 話者KVスケールを適用するレイヤー数上限 |
| `--model-device` | 自動 | TTSモデルのデバイス（`cuda` / `mps` / `cpu`） |
| `--codec-device` | 自動 | DACVAEコーデックのデバイス |
| `--model-precision` | 自動 | モデルの精度（`fp32` / `bf16`） |
| `--codec-precision` | 自動 | コーデックの精度（`fp32` / `bf16`） |
| `--seed` | ランダム | 再現性のための乱数シード |
| `--compile-model` | `False` | `torch.compile` を有効化（初回遅延あり） |
| `--trim-tail` | `True` | 末尾無音をフラット化ヒューリスティックで除去 |
| `--lora-path` | `None` | LoRAアダプタのフォルダパス |
| `--lora-scale` | `1.0` | LoRA注入スケール（0.0=ベースのみ、>1.0=強調） |

ローカルチェックポイント（`.pt` または `.safetensors`）も指定可能です:

```bash
uv run python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

---

## Gradio UI — タブ機能説明

### タブ1: 推論（🔊）

- **モデル読み込み** — チェックポイント（`.pt` / `.safetensors`）・デバイス・精度を選択。repo IDを入力してHuggingFaceから直接ダウンロードも可能。
- **LoRAアダプタ** — LoRAアダプタを読み込み、スケール（0.0〜2.0）を調整。
- **音声生成** — テキストを入力し、任意で参照音声をアップロード。
- **感情スタイルプリセット** — ノーマル / 力強く / おとなしく / 明るく / ひそやかに の各ボタンでCFGとスタイルパラメータを自動設定。
- **スタイルスライダー** — テキスト表現力・感情の強さ・話者密着度・表現の振れ幅を個別調整。
- **サンプリング設定** — ステップ数（1〜120）・乱数シード。
- **CFG設定** — ガイダンスモード（`independent` / `joint` / `alternating`）・テキストCFG・話者CFG。
- **詳細設定** — CFGタイムステップ範囲・コンテキストKVキャッシュ・スコア再スケール・話者KVスケール（上級者向け）。
- **複数候補生成** — 1回の実行で1〜8候補を生成。各候補は個別のWAVファイルとして保存。

### タブ2: Prepare Manifest（📂）

音声データをDACVAE潜在表現に変換し、学習用JSONL形式のマニフェストを生成します。

対応データソース: ローカルCSV（audiofolder形式）・ローカルJSONL・HuggingFaceデータセット。

ファイルを指定すると音声・テキスト・話者ID列名をヘッダーから自動検出します。

```bash
# CLIでの実行例
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

出力マニフェストの形式:

```json
{"text": "こんにちは", "latent_path": "data/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### タブ3: 学習

フルファインチューニング。Multi-GPU DDP対応。

主な設定項目:

| グループ | 設定内容 |
|---|---|
| ベースモデル | `.safetensors`（step 0から）または`.pt`チェックポイント（step/optimizer状態を引き継ぎ）からResumeしてファインチューニング |
| バッチ | バッチサイズ・勾配蓄積・DataLoaderワーカー数 |
| オプティマイザ | `muon` / `adamw` / `lion` / `ademamix` / `sgd` |
| スケジューラ | `wsd`（warmup→stable→decay）/ `cosine` / `none` |
| 精度 | `bf16` / `fp32` / `fp16` |
| Attentionバックエンド | `sdpa`（推奨）/ `flash2` / `sage` / `eager` |
| 正則化 | テキスト/話者条件ドロップアウト・タイムステップ層化サンプリング |
| バリデーション | バリデーション分割比率・実行間隔・Early Stopping |
| EMA | 推論品質向上のための指数移動平均 |
| チェックポイント | 保存間隔・EMAのみ または EMA+Fullの両方 |
| ログ | W&B連携・ログ出力間隔 |

単一GPU:

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

UIでは末尾200行のリアルタイムログ・ETA・loss値・ステップ進捗を表示します。loss曲線グラフは設定した間隔で自動更新されます。

### タブ4: LoRA学習（🚀）

凍結したベースモデルの上にLoRAアダプタを学習します。事前に `pip install peft` が必要です。

主な設定項目:

| 設定 | デフォルト | 説明 |
|---|---|---|
| LoRAランク | `16` | 低ランク次元数 |
| lora_alpha | `32.0` | スケーリング係数 |
| lora_dropout | `0.05` | ドロップアウト率 |
| ターゲットモジュール | `wq,wk,wv,wo` | カンマ区切りで指定。拡張セット: `wq,wk,wv,wo,wk_text,wv_text,wk_speaker,wv_speaker,w1,w2,w3` |
| 保存モード | EMAのみ | `EMAのみ`（推論用）/ `EMA + Full`（Resume前提） |
| Attentionバックエンド | `sdpa` | フル学習と同じ選択肢 |
| EMA | 有効 | 推論品質向上のためのEMA減衰 |
| Early Stopping | 無効 | `valid_ratio > 0` のときのみ有効 |
| Resume | 無効 | 既存の `_full` アダプタフォルダから再開 |

`lora:` セクションを持つYAMLファイルとして `configs/` フォルダへのプリセット保存・読み込みが可能です。

### タブ5: Dataset作成

**スライス** — 無音区間検出で長尺音声を分割します。

| パラメータ | 説明 |
|---|---|
| 最小/最大時間（秒） | 許容するセグメント長の範囲 |
| Top dB | 無音判定のdBFSしきい値 |
| Frame / Hop length | STFT窓パラメータ |
| ターゲットサンプリングレート | 任意のリサンプリング |
| 再帰的 | サブディレクトリも対象にする |

**キャプション** — Whisperで音声を文字起こしし、マニフェストを出力します。

| パラメータ | 説明 |
|---|---|
| Whisperモデル | `tiny` / `base` / `small` / `medium` / `large-v3` |
| 言語 | 言語コード（例: `ja`）または自動検出 |
| 出力形式 | `CSV` または `JSONL` |
| 話者ID | 固定の話者ラベル（任意） |

**パイプライン** — スライス → キャプションを一括実行します。

**絵文字キャプション** — 各音声セグメントの音響特徴量（ピッチ・エネルギー・発話速度・MFCCデルタ・零交差率）を抽出し、LLM APIを呼び出してIrodori-TTS互換の絵文字スタイルアノテーションをテキストに付与します。

対応API: `lm_studio` / `groq` / `openai` / `together`

対応絵文字アノテーション（38種）:

| 絵文字 | 意味 |
|---|---|
| 👂 | 囁き・耳元の音 |
| 😮‍💨 | 吐息・溜息・寝息 |
| ⏸️ | 間・沈黙 |
| 🤭 | 笑い・くすくす・含み笑い |
| 🥵 | 喘ぎ・うめき声・唸り声 |
| 📢 | エコー・リバーブ |
| 😏 | からかうように・甘えるように |
| 🥺 | 声を震わせながら・自信のなさげに |
| 🌬️ | 息切れ・荒い息遣い |
| 😮 | 息をのむ |
| 👅 | 舐める音・咀嚼音・水音 |
| 💋 | リップノイズ |
| 🫶 | 優しく・柔らかく |
| 😭 | 嗚咽・泣き声・悲しみ |
| 😱 | 悲鳴・叫び・絶叫 |
| 😪 | 眠そうに・気だるげに |
| ⏩ | 早口・まくしたてる |
| 📞 | 電話越し・スピーカー越し |
| 🐢 | ゆっくりと |
| 🥤 | 唾を飲み込む音 |
| 🤧 | 咳き込み・鼻をすする・くしゃみ |
| 😒 | 舌打ち |
| 😰 | 慌てて・動揺・緊張・どもり |
| 😆 | 喜びながら |
| 😠 | 怒り・不満げに・拗ねながら |
| 😲 | 驚き・感嘆 |
| 🥱 | あくび |
| 😖 | 苦しげに |
| 😟 | 心配そうに |
| 🫣 | 恥ずかしそうに・照れながら |
| 🙄 | 呆れたように |
| 😊 | 楽しげに・嬉しそうに |
| 👌 | 相槌・頷く音 |
| 🙏 | 懇願するように |
| 🥴 | 酔っ払って |
| 🎵 | 鼻歌 |
| 🤐 | 口を塞がれて |
| 😌 | 安堵・満足げに |
| 🤔 | 疑問の声 |

### タブ6: チェックポイント変換（🔄）

**通常チェックポイント変換** — 学習チェックポイント（`.pt`）を推論用の `.safetensors` 形式に変換します。モデルのconfigはファイルメタデータに埋め込まれます。

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
# 出力: outputs/checkpoint_final.safetensors

# 上書きを強制する場合:
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt --force
```

**LoRAチェックポイント変換** — optimizer状態を含む `_full` LoRAフォルダを、推論専用の `_ema` アダプタ形式に変換します。

### タブ7: モデルマージ（🔀）

2つのモデルチェックポイント（`.pt` または `.safetensors`）をマージします。実行前にアーキテクチャの互換性チェックが行われます。

**マージ手法:**

| 手法 | 説明 |
|---|---|
| `weighted_average` | `result = α × A + (1 − α) × B` |
| `slerp` | 球面線形補間。ノルムを保持。ゼロベクトル付近はWeighted Averageにフォールバック |
| `task_arithmetic` | `result = base + λA × (A − base) + λB × (B − base)` ベースモデルが別途必要 |

**部分マージ** — レイヤーグループごとに異なるマージ手法を独立して適用できます:

| グループ | 対象レイヤー |
|---|---|
| `text` | テキストエンコーダ・テキストnorm・JointAttentionのテキストKV |
| `speaker` | 話者エンコーダ・話者norm・JointAttentionの話者KV |
| `diffusion_core` | 拡散ブロック・cond_module |
| `io` | in_proj・out_norm・out_proj |

**LoRA的差分注入** — ドナーモデルとベースモデルの差分を指定スケールでターゲットに注入します: `result = base + scale × (donor − base)`。注入対象グループは個別に選択可能です。

出力形式: `.safetensors`（推論用・推奨）または `.pt`。

---

## 学習手順

### 1. マニフェスト作成

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

話者IDを含む場合:

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

### 2. フルファインチューニング

単一GPU:

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

### 3. LoRA学習

```bash
uv run python lora_train.py \
  --base-model checkpoints/Aratako_Irodori-TTS-500M/model.safetensors \
  --manifest data/train_manifest.jsonl \
  --output-dir lora/my_run \
  --lora-rank 16 \
  --lora-alpha 32.0 \
  --max-steps 1000
```

### 4. チェックポイント変換

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
```

---

## プロジェクト構成

```text
Irodori-TTS/
├── train.py                              # フルファインチューニングエントリポイント（DDP対応）
├── lora_train.py                         # LoRA学習エントリポイント
├── infer.py                              # CLI推論
├── gradio_app.py                         # Gradio Web UI（全機能）
├── prepare_manifest.py                   # 音声 → DACVAE潜在変換・マニフェスト生成
├── dataset_tools.py                      # 音声スライス / Whisperキャプション / 絵文字アノテーション
├── merge.py                              # モデルマージユーティリティ
├── convert_checkpoint_to_safetensors.py  # .pt → .safetensors 変換
├── convert_lora_checkpoint.py            # LoRA _full → _ema 変換
│
├── irodori_tts/                          # コアライブラリ
│   ├── model.py                          # TextToLatentRFDiT アーキテクチャ
│   ├── rf.py                             # Rectified Flow ユーティリティ・Euler CFGサンプリング
│   ├── codec.py                          # DACVAE コーデックラッパー
│   ├── dataset.py                        # データセット・コレーター
│   ├── tokenizer.py                      # 事前学習済みLLMトークナイザーラッパー
│   ├── config.py                         # Model / Train / Sampling 設定データクラス
│   ├── inference_runtime.py              # キャッシュ付きスレッドセーフ推論ランタイム
│   ├── text_normalization.py             # 日本語テキスト正規化
│   ├── optim.py                          # Muon + AdamW + Lion + AdEMAMix オプティマイザ
│   └── progress.py                       # 学習進捗トラッカー
│
├── configs/
│   ├── train_v1.yaml                     # 学習設定（500M・約50件・RTX 5060 Ti向け）
│   └── *.yaml                            # ユーザー追加設定
│
├── checkpoints/                          # ダウンロード済み・学習済みチェックポイント
├── lora/                                 # LoRAアダプタ出力
├── logs/                                 # 学習ログファイル
├── data/                                 # マニフェストファイルとDACVAE潜在表現
└── gradio_outputs/                       # GUIで生成した音声ファイル
```

---

## 設定ファイル リファレンス（train_v1.yaml）

デフォルト設定は約50件のサンプルデータ・RTX 5060 Ti（VRAM 16 GB）での学習を想定しており、学習時間の目安は30〜60分です。

| キー | デフォルト | 説明 |
|---|---|---|
| `batch_size` | `4` | GPU単体のバッチサイズ |
| `gradient_accumulation_steps` | `2` | 実効バッチ = batch × accum |
| `optimizer` | `muon` | `muon` / `adamw` / `lion` / `ademamix` / `sgd` |
| `learning_rate` | `3e-4` | ピーク学習率 |
| `lr_scheduler` | `wsd` | `wsd` / `cosine` / `none` |
| `warmup_steps` | `300` | 線形ウォームアップのステップ数 |
| `stable_steps` | `2100` | 安定期のステップ数（wsdのみ） |
| `max_steps` | `3000` | 総学習ステップ数 |
| `max_text_len` | `256` | テキストトークンの最大長 |
| `max_latent_steps` | `750` | 潜在フレームの最大長 |
| `text_condition_dropout` | `0.15` | CFGテキストドロップアウト率 |
| `speaker_condition_dropout` | `0.15` | CFG話者ドロップアウト率 |
| `valid_ratio` | `0.1` | バリデーション分割比率（0で無効） |
| `valid_every` | `100` | バリデーション実行間隔（ステップ） |
| `save_every` | `100` | チェックポイント保存間隔（ステップ） |
| `precision` | `bf16` | 学習精度 |
| `compile_model` | `false` | torch.compile を使用する |

---

## ライセンス

- **コード**: [MIT License](LICENSE)
- **モデル重み**: 商用利用禁止。詳細は[オリジナルモデルカード](https://huggingface.co/Aratako/Irodori-TTS-500M)を参照してください。

---

## 謝辞

- [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) — 本フォークのベースとなったオリジナルリポジトリ
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
