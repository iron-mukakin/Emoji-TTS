#!/usr/bin/env python3
"""
dataset_tools.py
================
3つの独立した機能を提供するデータセット前処理ツール。

機能1: slice_audio    -- 無音区間検出による長尺音声の自動スライス
機能2: caption_audio  -- Whisperによる音声キャプション（文字起こし）+
                         JSONL/CSV manifest 出力
機能3: emoji_caption  -- 既存CSVのtextと音声から音響特徴量を抽出し、
                         LLM（LM Studio / Groq / OpenAI / Together AI）で
                         Irodori-TTS互換の絵文字キャプションを生成する

使い方:
  # スライスのみ
  python dataset_tools.py slice \
      --input  /path/to/long_audio.wav \
      --output /path/to/sliced/

  # キャプションのみ（既存の音声フォルダに対して）
  python dataset_tools.py caption \
      --input  /path/to/sliced/ \
      --output-manifest /path/to/manifest.jsonl \
      [--format jsonl|csv] [--model large-v3] [--language ja]

  # スライス → キャプション を一括実行
  python dataset_tools.py pipeline \
      --input  /path/to/long_audio.wav \
      --slice-output /path/to/sliced/ \
      --output-manifest /path/to/manifest.jsonl

  # 絵文字キャプション生成（GUIで操作推奨）
  python dataset_tools.py emoji_caption \
      --csv /path/to/metadata.csv \
      --wav-dir /path/to/wavs/ \
      --api lm_studio \
      [--lm-studio-url http://localhost:1234] \
      [--lm-studio-model モデル名] \
      [--api-key YOUR_KEY]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac"}
DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_SR = 16000  # Whisper の内部サンプリングレート

# Irodori-TTS 絵文字アノテーション仕様（EMOJI_ANNOTATIONS.md 準拠）
EMOJI_ANNOTATIONS = """
あなたはIrodori-TTS用の絵文字キャプション生成AIです。
以下の絵文字仕様に従い、与えられた音声の音響特徴量とテキストから、
テキストに絵文字を付与したキャプションを生成してください。

【絵文字仕様一覧】
👂 囁き・耳元の音（Whisper, sounds close to the ear）
😮‍💨 吐息・溜息・寝息（Breath, sigh, sleeping breath）
⏸️ 間・沈黙（Pause, silence）
🤭 笑い・くすくす・含み笑い（Chuckle, giggle, suppressed laugh）
🥵 喘ぎ・うめき声・唸り声（Panting, moan, groan）
📢 エコー・リバーブ（Echo, reverb）
😏 からかうように・甘えるように（Teasing, playfully sweet）
🥺 声を震わせながら・自信のなさげに（Trembling voice, timidly）
🌬️ 息切れ・荒い息遣い（Shortness of breath, heavy breathing）
😮 息をのむ（Gasp）
👅 舐める音・咀嚼音・水音（Licking, chewing, wet sound）
💋 リップノイズ（Lip smack）
🫶 優しく・柔らかく（Gently, tenderly）
😭 嗚咽・泣き声・悲しみ（Sobbing, crying, sorrowfully）
😱 悲鳴・叫び・絶叫（Scream, shout, shriek）
😪 眠そうに・気だるげに（Sleepily, languidly）
⏩ 早口・一気にまくしたてる（Fast-speaking, rapidly）
📞 電話越し・スピーカー越し（Over the phone）
🐢 ゆっくりと（Slowly）
🥤 唾を飲み込む音（Gulp, swallowing sound）
🤧 咳き込み・鼻をすする・くしゃみ（Coughing, sniffling）
😒 舌打ち（Tutting, clicking tongue）
😰 慌てて・動揺・緊張・どもり（Panicked, nervous, stuttering）
😆 喜びながら（Joyfully, happily）
😠 怒り・不満げに・拗ねながら（Angry, displeased, sulking）
😲 驚き・感嘆（Surprise, awe）
🥱 あくび（Yawn）
😖 苦しげに（Painfully, agonizingly）
😟 心配そうに（Anxiously, worriedly）
🫣 恥ずかしそうに・照れながら（Shyly, bashfully）
🙄 呆れたように（Exasperatedly）
😊 楽しげに・嬉しそうに（Cheerfully, gladly）
👌 相槌・頷く音（Backchanneling, agreement）
🙏 懇願するように（Pleadingly, begging）
🥴 酔っ払って（Drunkenly）
🎵 鼻歌（Humming）
🤐 口を塞がれて（Muffled）
😌 安堵・満足げに（Relieved, contentedly）
🤔 疑問の声（Questioning voice）

【音響特徴量の読み方】
- pitch_mean_hz: 平均ピッチ。高い(>250Hz)=高音・明るい・興奮、低い(<150Hz)=低音・落ち着き・怒り
- pitch_std_hz: ピッチの揺らぎ。大きい=感情的・不安定、小さい=落ち着き・棒読み
- energy_mean: 平均音量。大きい=元気・怒り・叫び、小さい=囁き・疲れ
- energy_std: 音量の変動。大きい=抑揚豊か、小さい=単調
- speech_rate_syllables_per_sec: 発話速度。速い(>8)=早口、遅い(<4)=ゆっくり
- zcr_mean: 零交差率。高い=ノイジー・息交じり・摩擦音、低い=クリアな声
- mfcc_delta_energy: 声の変化速度。大きい=感情的変化あり
- duration_sec: 発話時間（秒）

【出力ルール】
1. テキストに絵文字を自然な位置に挿入または末尾に付与する
2. 音響特徴量を根拠に、最も適切な絵文字を1〜3個選ぶ
3. 特徴が弱い場合は絵文字なし（元のテキストそのまま）でもよい
4. 必ずテキストのみを返す。説明・理由・JSON不要
5. 元のテキストの意味・文字は変えない

【例】
テキスト:「どうして…」  音響:低ピッチ・音量大・高変動 → 「どうして…😠」
テキスト:「こんにちは」  音響:高ピッチ・明るい・標準速度 → 「こんにちは😊」
テキスト:「ねえ、聞いて」 音響:ピッチ高・音量小・ゆっくり → 「ねえ、聞いて👂」
"""

# API設定
API_PROVIDERS = {
    "なし（従来通り）": "none",
    "LM Studio（ローカル）": "lm_studio",
    "Groq": "groq",
    "OpenAI（ChatGPT）": "openai",
    "Together AI": "together",
}

API_BASE_URLS = {
    "lm_studio": "http://localhost:1234/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "together": "https://api.together.xyz/v1",
}

API_DEFAULT_MODELS = {
    "lm_studio": "",  # LM Studioは起動中のモデルを自動使用
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

VOICE_ANALYSIS_CAPTION_RULES = """
あなたは音声分析キャプション作成AIです。以下の仕様を厳守してください。

【目的】
- 音響特徴量パラメーターのみを根拠にして、caption列に入れる短い日本語の音声描写文を1つ作る。
- 読み上げ内容（文字起こしテキスト）は一切参照しない。音の特性だけを記述する。

【入力】
- 音響特徴量のみ: pitch_mean_hz, pitch_std_hz, energy_mean, energy_std,
  speech_rate_syllables_per_sec, zcr_mean, mfcc_delta_energy, duration_sec

【出力ルール】
1) 出力は1文のみ（最大60文字目安）。
2) 日本語の自然文で、次の順で書く:
   「声質/声の高さ」→「話速」→「抑揚/感情傾向」
3) 絵文字・顔文字・記号装飾（!?,♪など）・引用符は使わない。
4) 箇条書き、JSON、説明文、前置き、理由文は禁止。文だけ返す。
5) 性別・年齢・人格は、パラメーターから強く示唆される場合のみ記述し、根拠が弱い場合は断定しない。
   - pitch_mean_hz > 200Hz かつ energy_mean が低め → 「高めの声」または「女性的な高さの声」
   - pitch_mean_hz < 130Hz → 「低めの声」または「男性的な低さの声」
6) 推定不能な要素は省略し、「やや」「比較的」などで弱く表現する。
7) 読み上げ内容・単語・話題は絶対に含めない。

【判定ガイド（目安）】
- 声の高さ:
  - pitch_mean_hz > 220: 高めの声
  - pitch_mean_hz 130〜220: 中程度の声
  - pitch_mean_hz < 130: 低めの声
- 話速:
  - speech_rate_syllables_per_sec < 4.0: ゆっくり
  - 4.0〜8.0: 標準的な速さ
  - > 8.0: 早口
- 抑揚:
  - pitch_std_hz > 40 または energy_std > 0.05: 抑揚が大きい
  - pitch_std_hz < 15 かつ energy_std < 0.02: 抑揚が穏やか（単調）
- 息感/ノイズ感:
  - zcr_mean > 0.15: 息混じり
- 音量:
  - energy_mean < 0.02: 小さい声・囁き気味
  - energy_mean > 0.10: 大きい声・張りのある声

【出力例】
- 高めの声で、標準的な速さで、抑揚が穏やかな話し方
- 低めの声で、やや早口で、抑揚が大きく感情の動きがある話し方
- 息混じりの声で、ゆっくりと、落ち着いた話し方
- 中程度の声で、早口で、エネルギッシュな話し方
"""


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _require(pkg: str, pip_name: str | None = None) -> None:
    """ライブラリが存在しない場合にわかりやすいエラーを出す。"""
    import importlib
    if importlib.util.find_spec(pkg) is None:
        pip = pip_name or pkg
        print(f"[ERROR] '{pkg}' が見つかりません。\n  pip install {pip}", file=sys.stderr)
        sys.exit(1)


def _collect_audio_files(path: Path, recursive: bool) -> list[Path]:
    """フォルダまたは単一ファイルから音声ファイルを列挙。"""
    if path.is_file():
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            return [path]
        print(f"[WARN] {path} は対応音声形式ではありません。", file=sys.stderr)
        return []
    glob = path.rglob("*") if recursive else path.glob("*")
    files = sorted(p for p in glob if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    return files


# ─────────────────────────────────────────────────────────────────────────────
# 機能1: slice_audio  （Silero VAD ベース）
# ─────────────────────────────────────────────────────────────────────────────

def _load_silero_vad(device, log_fn=print):
    """
    Silero VAD モデルをロードして返す。
    キャッシュ先: dataset_tools.py と同じフォルダ内の torch_hub/
    フォルダが存在しない場合は自動ダウンロードされる。
    """
    import torch

    # dataset_tools.py があるディレクトリ / torch_hub をキャッシュ先に設定
    hub_dir = Path(__file__).resolve().parent / "torch_hub"
    hub_dir.mkdir(parents=True, exist_ok=True)

    original_hub_dir = torch.hub.get_dir()
    torch.hub.set_dir(str(hub_dir))

    try:
        cached = (hub_dir / "snakers4_silero-vad_master").exists()
        if cached:
            log_fn(f"Silero VAD キャッシュを使用: {hub_dir}")
        else:
            log_fn(f"Silero VAD を自動ダウンロード中: {hub_dir}")

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
    finally:
        # 他の torch.hub 利用に影響しないよう元のhub_dirに戻す
        torch.hub.set_dir(original_hub_dir)

    model = model.to(device)
    model.eval()
    get_speech_timestamps = utils[0]
    log_fn("Silero VAD ロード完了")
    return model, get_speech_timestamps


def _split_at_silence(wav_orig, wav_vad, start, end,
                      max_samples, min_samples, scale):
    """
    max_sec を超えるセグメントを最近傍の低エネルギー点で再帰的に分割する。
    単純均等分割ではなく、VADスコアが最も低いフレームを優先して切る。
    """
    import torch
    segments: list[tuple[int, int]] = []
    cursor = start

    while (end - cursor) > max_samples:
        search_center = cursor + max_samples
        search_half   = max(min_samples // 2, int(0.5 * (max_samples / max(scale, 1))))
        search_start  = max(cursor + min_samples, search_center - search_half)
        search_end    = min(end - min_samples,    search_center + search_half)

        if search_start >= search_end:
            split_pt = cursor + max_samples
        else:
            ss_vad = max(0, int(search_start / scale))
            se_vad = min(wav_vad.shape[-1], int(search_end / scale))
            if ss_vad >= se_vad:
                split_pt = cursor + max_samples
            else:
                chunk = wav_vad[ss_vad:se_vad].abs()
                window = max(1, int(0.02 * (1.0 / scale) if scale > 0 else 1))
                if chunk.shape[-1] > window:
                    rms = chunk.unfold(0, window, 1).pow(2).mean(-1).sqrt()
                    min_idx = int(rms.argmin().item())
                else:
                    min_idx = int(chunk.argmin().item())
                split_pt = int((ss_vad + min_idx) * scale)
                split_pt = max(cursor + min_samples, min(split_pt, end - min_samples))

        segments.append((cursor, split_pt))
        cursor = split_pt

    if (end - cursor) >= min_samples:
        segments.append((cursor, end))
    return segments


def slice_audio(
    input_path: Path,
    output_dir: Path,
    *,
    min_sec: float = 2.0,
    max_sec: float = 30.0,
    threshold: float = 0.5,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 30,
    target_sr: int | None = None,
    recursive: bool = False,
    device: str | None = None,
    log_fn=print,
) -> list[Path]:
    """
    Silero VAD（ニューラルネット音声活動検出）で音声をスライスして保存する。

    librosa.effects.split と異なり、連続発話・息継ぎ・小さい子音も
    正確に「発話」として扱えるため、キャラクター音声に適している。

    Parameters
    ----------
    input_path     : 単一ファイルまたはフォルダ
    output_dir     : スライス済みWAVの保存先
    min_sec        : この秒数より短いセグメントは破棄
    max_sec        : この秒数より長いセグメントは最近傍の無音点で分割
    threshold      : VAD発話判定閾値（0.0〜1.0）。大きいほど厳しく発話を検出
    min_silence_ms : この時間（ms）以上の無音が続かないと区切らない
    speech_pad_ms  : 発話区間の前後に追加するパディング（ms）
    target_sr      : 出力WAVのリサンプル先SR（None = 元のまま）
    recursive      : フォルダを再帰的に検索するか
    device         : "cuda" / "cpu" / None=自動
    log_fn         : ログ出力関数

    Returns
    -------
    保存されたスライスファイルのパスリスト
    """
    import torch
    import torchaudio

    SILERO_SR = 16000  # Silero VAD の動作サンプリングレート

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    audio_files = _collect_audio_files(input_path, recursive)
    if not audio_files:
        log_fn(f"[WARN] 音声ファイルが見つかりません: {input_path}")
        return []

    vad_model, get_speech_timestamps = _load_silero_vad(device_obj, log_fn)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    total_files = len(audio_files)

    for file_idx, audio_path in enumerate(audio_files, 1):
        log_fn(f"[{file_idx}/{total_files}] スライス処理: {audio_path.name}")
        try:
            wav, orig_sr = torchaudio.load(str(audio_path))
        except Exception as e:
            log_fn(f"  [ERROR] 読み込み失敗: {e}")
            continue

        # モノラル化
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # VAD用に16kHzへリサンプル
        if orig_sr != SILERO_SR:
            wav_vad = torchaudio.transforms.Resample(orig_sr, SILERO_SR)(wav)
        else:
            wav_vad = wav.clone()

        wav_vad_1d = wav_vad.squeeze(0).to(device_obj)

        # ── Silero VAD で発話区間を検出 ──
        with torch.no_grad():
            speech_timestamps = get_speech_timestamps(
                wav_vad_1d,
                vad_model,
                sampling_rate=SILERO_SR,
                threshold=threshold,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=False,
            )

        if not speech_timestamps:
            log_fn(f"  [WARN] 発話区間が検出されませんでした: {audio_path.name}")
            continue

        # VAD結果(16kHz単位) → 元SR単位へのスケール係数
        scale = orig_sr / SILERO_SR
        min_samples = int(min_sec * orig_sr)
        max_samples = int(max_sec * orig_sr)

        segments: list[tuple[int, int]] = []
        for ts in speech_timestamps:
            start = max(0, int(ts["start"] * scale))
            end   = min(wav.shape[-1], int(ts["end"] * scale))
            seg_len = end - start
            if seg_len < min_samples:
                continue
            if seg_len <= max_samples:
                segments.append((start, end))
            else:
                segments.extend(
                    _split_at_silence(
                        wav.squeeze(0), wav_vad_1d,
                        start, end, max_samples, min_samples, scale,
                    )
                )

        log_fn(f"  セグメント数: {len(segments)}")

        out_sr = target_sr if target_sr is not None else orig_sr
        resampler_out = (
            torchaudio.transforms.Resample(orig_sr, out_sr) if out_sr != orig_sr else None
        )

        stem = audio_path.stem
        for seg_idx, (start, end) in enumerate(segments):
            segment = wav[:, start:end]
            if resampler_out is not None:
                segment = resampler_out(segment)
            out_path = output_dir / f"{stem}_{seg_idx:05d}.wav"
            torchaudio.save(str(out_path), segment.cpu(), out_sr)
            saved.append(out_path)

    log_fn(f"\nスライス完了: {len(saved)} ファイル → {output_dir}")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 機能2: caption_audio
# ─────────────────────────────────────────────────────────────────────────────

def caption_audio(
    input_path: Path,
    output_manifest: Path,
    *,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str | None = "ja",
    speaker_id: str | None = None,
    voice_design: bool = False,
    voice_design_caption: str | None = None,
    output_format: str = "jsonl",   # "jsonl" | "csv"
    recursive: bool = False,
    batch_size: int = 1,
    device: str | None = None,
    model_cache_dir: Path | str | None = None,
    log_fn=print,
) -> list[dict]:
    """
    Whisper で音声ファイルを文字起こしし、manifest（JSONL または CSV）を出力する。

    出力JSONL フォーマット:
        通常モード: {"text": "...", "audio_path": "...", "speaker": "..."}
        VoiceDesign: {"text": "...", "audio_path": "...", "caption": "..."}

    出力CSV フォーマット:
        通常モード: file_name,text,speaker
        VoiceDesign: file_name,text,caption

    Parameters
    ----------
    input_path      : 単一ファイルまたはフォルダ
    output_manifest : 出力先ファイルパス（.jsonl / .csv）
    model_name      : Whisper モデル名（tiny / base / small / medium / large-v3 等）
    language        : 言語コード（ja / en / None=自動検出）
    speaker_id      : 通常モード時に全ファイルへ付与する話者ID（speaker列）
    voice_design    : True時はspeaker列を無効化しcaption列をVoiceDesign向けで出力
    voice_design_caption : VoiceDesign時に全件へ固定で使うcaption
    output_format   : "jsonl" または "csv"
    recursive       : フォルダを再帰的に検索するか
    batch_size      : バッチ処理数（faster-whisper 使用時に有効）
    device          : "cuda" / "cpu" / None=自動
    model_cache_dir : モデルのダウンロード・キャッシュ先ディレクトリ
                      None の場合は ~/.cache/huggingface/hub（HFデフォルト）
                      指定した場合はそのフォルダにモデルが保存される
    log_fn          : ログ出力関数

    Returns
    -------
    書き出したレコードのリスト
    """
    _require("faster_whisper", "faster-whisper")
    from faster_whisper import WhisperModel

    audio_files = _collect_audio_files(input_path, recursive)
    if not audio_files:
        log_fn(f"[WARN] 音声ファイルが見つかりません: {input_path}")
        return []

    # デバイス自動選択
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    # モデルキャッシュディレクトリの準備
    download_root: str | None = None
    if model_cache_dir is not None:
        cache_path = Path(model_cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        download_root = str(cache_path)
        log_fn(f"Whisperモデルキャッシュ: {download_root}")

    log_fn(f"Whisperモデル読み込み: {model_name}  device={device}  compute={compute_type}")
    if download_root:
        model = WhisperModel(model_name, device=device, compute_type=compute_type,
                             download_root=download_root)
    else:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

    if voice_design and speaker_id:
        log_fn("[INFO] VoiceDesignモードのため speaker_id は無視します。")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    total = len(audio_files)

    for idx, audio_path in enumerate(audio_files, 1):
        log_fn(f"[{idx}/{total}] キャプション: {audio_path.name}")
        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,               # 精度重視
                best_of=5,
                temperature=0.0,           # greedy decoding（精度最大）
                vad_filter=True,           # VADフィルタで無音を除去
                vad_parameters={"min_silence_duration_ms": 300},
            )
            text = "".join(seg.text for seg in segments).strip()
        except Exception as e:
            log_fn(f"  [ERROR] 文字起こし失敗: {e}")
            continue

        if not text:
            log_fn(f"  [SKIP] テキストが空: {audio_path.name}")
            continue

        record: dict = {"text": text, "audio_path": str(audio_path)}
        if voice_design:
            record["caption"] = voice_design_caption if voice_design_caption else text
        elif speaker_id:
            record["speaker"] = speaker_id

        records.append(record)
        log_fn(f"  → {text[:80]}{'...' if len(text) > 80 else ''}")

    # ── 出力 ──
    fmt = output_format.lower()
    if fmt == "csv":
        # voice_design=False（speakerモード）時は話者ID未入力でもspeaker列を必ず出力
        _write_csv(records, output_manifest, speaker_mode=not voice_design)
    else:
        _write_jsonl(records, output_manifest)

    log_fn(f"\nキャプション完了: {len(records)}/{total} 件 → {output_manifest}")
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_csv(records: list[dict], path: Path, *, speaker_mode: bool = False) -> None:
    """
    出力形式:
        通常モード(speaker_mode=True): file_name,text,speaker  ← 話者IDが空でも列は必ず出力
        VoiceDesign: file_name,text,caption
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        has_caption = any("caption" in rec for rec in records)
        columns = ["file_name", "text"]
        if has_caption:
            columns.append("caption")
        else:
            # speaker_mode=True または speaker値が1件でもある場合は speaker 列を出力
            has_speaker_value = any(
                str(rec.get("speaker", rec.get("speaker_id", ""))).strip() != ""
                for rec in records
            )
            if speaker_mode or has_speaker_value:
                columns.append("speaker")
        writer = csv.writer(f)
        writer.writerow(columns)
        for rec in records:
            file_name = Path(rec["audio_path"]).name  # フルパスからファイル名のみ抽出
            row = [file_name, rec.get("text", "")]
            if has_caption:
                row.append(rec.get("caption", ""))
            elif "speaker" in columns:
                row.append(rec.get("speaker", rec.get("speaker_id", "")))
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# 機能3: emoji_caption  音響特徴量抽出 + LLM 絵文字キャプション生成
# ─────────────────────────────────────────────────────────────────────────────

def _extract_acoustic_features(wav_path: Path) -> dict:
    """
    librosa で音響特徴量を抽出して辞書で返す。
    抽出項目:
      - pitch_mean_hz / pitch_std_hz : ピッチ（基本周波数）の平均・標準偏差
      - energy_mean / energy_std     : RMSエネルギーの平均・標準偏差
      - speech_rate_syllables_per_sec: 推定発話速度（音節数/秒）
      - zcr_mean                     : 零交差率（息交じり度の代理指標）
      - mfcc_delta_energy            : MFCCデルタのノルム（声の変化速度）
      - duration_sec                 : 発話時間
    """
    _require("librosa", "librosa")
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # ── ピッチ ──
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, frame_length=2048,
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)] if voiced_f0 is not None else np.array([])
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_std  = float(np.std(voiced_f0))  if len(voiced_f0) > 0 else 0.0

    # ── RMSエネルギー ──
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    # ── 零交差率 ──
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
    zcr_mean = float(np.mean(zcr))

    # ── MFCC デルタ（声の変化速度） ──
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_energy = float(np.mean(np.linalg.norm(mfcc_delta, axis=0)))

    # ── 発話速度推定（onset をシラブルの代理として使用） ──
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    speech_rate = len(onset_frames) / duration if duration > 0 else 0.0

    return {
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 1),
        "energy_mean": round(energy_mean, 4),
        "energy_std": round(energy_std, 4),
        "speech_rate_syllables_per_sec": round(speech_rate, 2),
        "zcr_mean": round(zcr_mean, 4),
        "mfcc_delta_energy": round(mfcc_delta_energy, 4),
        "duration_sec": round(duration, 2),
    }


def _call_llm_emoji(
    text: str,
    features: dict,
    api_provider: str,
    api_key: str = "",
    api_base_url: str = "",
    model_name: str = "",
    log_fn=print,
) -> str:
    """
    LLMにテキスト＋音響特徴量を渡して絵文字付きキャプションを返す。
    OpenAI互換APIを使用（LM Studio / Groq / OpenAI / Together AI）。
    """
    try:
        from openai import OpenAI
    except ImportError:
        log_fn("[ERROR] openai パッケージが見つかりません。pip install openai")
        return text

    base_url = api_base_url or API_BASE_URLS.get(api_provider, "")
    model    = model_name   or API_DEFAULT_MODELS.get(api_provider, "")
    key      = api_key      or ("lm-studio" if api_provider == "lm_studio" else "")

    client = OpenAI(api_key=key, base_url=base_url)

    features_str = "\n".join(f"  {k}: {v}" for k, v in features.items())
    user_message = f"【テキスト】\n{text}\n\n【音響特徴量】\n{features_str}"

    # LM Studio はモデル名を起動中のものに合わせる必要があるが、
    # 多くの場合 model 指定不要（サーバー側で無視される）。
    # 空文字だと openai クライアントがエラーを出すので最低限の値を入れる。
    if not model:
        model = "local-model"

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": EMOJI_ANNOTATIONS},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 256,
    }

    try:
        resp = client.chat.completions.create(**kwargs)
        caption = resp.choices[0].message.content.strip()
        log_fn(f"  [LLM RAW] {repr(caption[:80])}")   # デバッグ: LLMの生返答を確認
        # 空返答の場合は元テキストをフォールバック
        return caption if caption else text
    except Exception as e:
        log_fn(f"  [LLM ERROR] {e}")
        return text  # 失敗時は元テキストをそのまま返す


def _strip_emoji_characters(text: str) -> str:
    """caption列向けに絵文字を除去する。"""
    if not text:
        return ""
    emoji_re = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]",
        flags=re.UNICODE,
    )
    cleaned = emoji_re.sub("", text)
    cleaned = cleaned.replace("\u200d", "").replace("\ufe0f", "")
    return " ".join(cleaned.split()).strip()


def _call_llm_voice_caption(
    features: dict,
    api_provider: str,
    api_key: str = "",
    api_base_url: str = "",
    model_name: str = "",
    log_fn=print,
) -> str:
    """音響特徴量のみからcaption列向け音声描写文を生成する。
    文字起こしテキストは一切参照しない。
    """
    try:
        from openai import OpenAI
    except ImportError:
        log_fn("[ERROR] openai パッケージが見つかりません。pip install openai")
        return ""

    base_url = api_base_url or API_BASE_URLS.get(api_provider, "")
    model = model_name or API_DEFAULT_MODELS.get(api_provider, "")
    key = api_key or ("lm-studio" if api_provider == "lm_studio" else "")
    client = OpenAI(api_key=key, base_url=base_url)

    if not features:
        log_fn("  [VOICE CAP SKIP] 音響特徴量が空のためcaptionを生成できません")
        return ""

    # 音響特徴量のみをuserメッセージに渡す（文字起こし内容は含めない）
    features_str = "\n".join(f"  {k}: {v}" for k, v in features.items())
    user_message = f"【音響特徴量】\n{features_str}"

    if not model:
        model = "local-model"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VOICE_ANALYSIS_CAPTION_RULES},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        raw_content = resp.choices[0].message.content
        if raw_content is None:
            log_fn("  [VOICE CAP WARN] LLMがNoneを返しました（tool_call応答の可能性）")
            return ""
        raw_content = raw_content.strip()
        caption = _strip_emoji_characters(raw_content)
        # 絵文字除去後に空になった場合は除去前を返す
        if not caption and raw_content:
            caption = raw_content
        return caption
    except Exception as e:
        log_fn(f"  [VOICE CAP ERROR] {e}")
        return ""


def emoji_caption(
    csv_path: Path,
    wav_dir: Path,
    *,
    api_provider: str = "lm_studio",
    api_key: str = "",
    api_base_url: str = "",
    model_name: str = "",
    voice_design: bool = False,
    retry_count: int = 2,
    log_fn=print,
) -> list[dict]:
    """
    CSVのtextと対応するwavから音響特徴量を抽出し、LLMで処理する。

    処理内容:
      - text列（全モード共通）: 絵文字キャプション（EMOJI_ANNOTATIONSルール）
      - caption列（voice_design=True のみ）: 音声分析キャプション（VOICE_ANALYSIS_CAPTION_RULESルール）
        → 音響特徴量パラメーターのみを入力。文字起こし内容は一切参照しない。

    入力CSV: file_name, text（必須カラム）
    出力CSV:
      通常モード  : file_name, text（絵文字付き）, speaker（変更なし）
      VoiceDesign: file_name, text（絵文字付き）, caption（音声分析）

    Parameters
    ----------
    csv_path     : 入力CSVファイルパス
    wav_dir      : wavファイルが格納されているフォルダ
    api_provider : "lm_studio" / "groq" / "openai" / "together"
    api_key      : 外部APIキー（LM Studioは不要）
    api_base_url : APIベースURL（省略時はデフォルトを使用）
    model_name   : 使用モデル名（省略時はデフォルトを使用）
    voice_design : True=VoiceDesignモード（caption列を生成）/ False=通常モード（caption列なし）
    retry_count  : LLM呼び出し失敗時のリトライ回数
    log_fn       : ログ出力関数

    Returns
    -------
    処理済みレコードのリスト
    """
    if not csv_path.exists():
        log_fn(f"[ERROR] CSVファイルが見つかりません: {csv_path}")
        return []

    # CSV読み込み
    records: list[dict] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "file_name" not in (reader.fieldnames or []) or "text" not in (reader.fieldnames or []):
            log_fn("[ERROR] CSVに file_name, text カラムが必要です。")
            return []
        records = list(reader)

    total = len(records)
    log_fn(f"対象件数: {total} 件  API: {api_provider}")

    results: list[dict] = []
    for idx, rec in enumerate(records, 1):
        file_name = rec.get("file_name", "")
        text      = rec.get("text", "").strip()
        wav_path  = wav_dir / file_name

        log_fn(f"[{idx}/{total}] {file_name}")

        # 音響特徴量抽出
        if wav_path.exists():
            try:
                features = _extract_acoustic_features(wav_path)
                log_fn(f"  音響: pitch={features['pitch_mean_hz']}Hz "
                       f"energy={features['energy_mean']:.4f} "
                       f"rate={features['speech_rate_syllables_per_sec']}/s")
            except Exception as e:
                log_fn(f"  [WARN] 音響解析失敗: {e} → テキストのみでLLMに渡します")
                features = {}
        else:
            log_fn(f"  [WARN] wavファイルが見つかりません: {wav_path} → テキストのみ")
            features = {}

        # ── LLM呼び出し1: 絵文字キャプション（text列） ──
        # EMOJI_ANNOTATIONSシステムプロンプトのみ使用。音声分析ルールとは完全分離。
        emoji_text = text
        for attempt in range(retry_count + 1):
            emoji_text = _call_llm_emoji(
                text, features,
                api_provider=api_provider,
                api_key=api_key,
                api_base_url=api_base_url,
                model_name=model_name,
                log_fn=log_fn,
            )
            if emoji_text is not None:
                break
            if attempt < retry_count:
                log_fn(f"  リトライ {attempt + 1}/{retry_count}")
                time.sleep(1)

        # ── LLM呼び出し2: 音声分析キャプション（caption列・VoiceDesignモードのみ） ──
        # VOICE_ANALYSIS_CAPTION_RULESシステムプロンプトのみ使用。
        # 文字起こしテキストは渡さず、音響特徴量パラメーターのみを入力とする。
        # 通常モード（voice_design=False）では実行しない。
        if voice_design:
            voice_caption = ""
            if features:
                voice_caption = _call_llm_voice_caption(
                    features,
                    api_provider=api_provider,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    model_name=model_name,
                    log_fn=log_fn,
                )
            if not voice_caption:
                log_fn("  [WARN] 音声分析キャプション生成失敗。空文字列を設定します。")
                voice_caption = ""
            log_fn(f"  → text: {emoji_text}")
            log_fn(f"  → caption: {voice_caption}")
            updated = dict(rec)
            updated["text"] = emoji_text
            updated["caption"] = voice_caption
        else:
            # 通常モード: text列のみ絵文字付きに更新。speaker列はそのまま維持。
            log_fn(f"  → text: {emoji_text}")
            updated = dict(rec)
            updated["text"] = emoji_text

        results.append(updated)

    # CSV上書き保存
    fieldnames = list(records[0].keys()) if records else ["file_name", "text"]
    if voice_design:
        # VoiceDesignモード: caption列を追加、speaker列を除去
        if "caption" not in fieldnames:
            fieldnames.append("caption")
        fieldnames = [f for f in fieldnames if f != "speaker"]
    else:
        # 通常モード: speaker列がなければ追加（話者ID未入力でもヘッダーは維持）
        if "speaker" not in fieldnames:
            fieldnames.append("speaker")
        # 通常モードではcaption列は出力しない
        fieldnames = [f for f in fieldnames if f != "caption"]
        # speaker値がない行に空文字を保証
        for r in results:
            r.setdefault("speaker", "")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    mode_label = "絵文字＋音声分析キャプション" if voice_design else "絵文字キャプション（text列のみ）"
    log_fn(f"\n{mode_label}完了: {len(results)}/{total} 件 → {csv_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# pipeline: slice → caption を一括実行
# ─────────────────────────────────────────────────────────────────────────────

def pipeline(
    input_path: Path,
    slice_output: Path,
    output_manifest: Path,
    *,
    min_sec: float = 2.0,
    max_sec: float = 30.0,
    threshold: float = 0.5,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 30,
    target_sr: int | None = None,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str | None = "ja",
    speaker_id: str | None = None,
    voice_design: bool = False,
    voice_design_caption: str | None = None,
    output_format: str = "jsonl",
    device: str | None = None,
    model_cache_dir: Path | str | None = None,
    log_fn=print,
) -> list[dict]:
    """スライス → キャプション を連続実行するパイプライン。"""
    log_fn("=" * 60)
    log_fn("STEP 1/2: 音声スライス")
    log_fn("=" * 60)
    sliced = slice_audio(
        input_path, slice_output,
        min_sec=min_sec, max_sec=max_sec,
        threshold=threshold, min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        target_sr=target_sr, recursive=False,
        device=device, log_fn=log_fn,
    )
    if not sliced:
        log_fn("[WARN] スライス結果が0件のためキャプションをスキップします。")
        return []

    log_fn("\n" + "=" * 60)
    log_fn("STEP 2/2: キャプション生成")
    log_fn("=" * 60)
    records = caption_audio(
        slice_output, output_manifest,
        model_name=model_name, language=language, speaker_id=speaker_id,
        voice_design=voice_design, voice_design_caption=voice_design_caption,
        output_format=output_format, recursive=False, device=device,
        model_cache_dir=model_cache_dir, log_fn=log_fn,
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _add_slice_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input",  required=True, help="入力ファイルまたはフォルダ")
    p.add_argument("--output", required=True, help="スライス済みWAVの保存先フォルダ")
    p.add_argument("--min-sec",        type=float, default=2.0,  help="最小セグメント長（秒）[default: 2.0]")
    p.add_argument("--max-sec",        type=float, default=30.0, help="最大セグメント長（秒）[default: 30.0]")
    p.add_argument("--threshold",      type=float, default=0.5,  help="VAD発話判定閾値 0.0〜1.0 [default: 0.5]")
    p.add_argument("--min-silence-ms", type=int,   default=300,  help="無音と判定する最短時間ms [default: 300]")
    p.add_argument("--speech-pad-ms",  type=int,   default=30,   help="発話前後のパディングms [default: 30]")
    p.add_argument("--target-sr",      type=int,   default=None, help="出力リサンプルSR（省略=元のまま）")
    p.add_argument("--device",                     default=None, help="cuda / cpu（省略=自動）")
    p.add_argument("--recursive", action="store_true", help="フォルダを再帰検索")


def _add_caption_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True, help="入力ファイルまたはフォルダ")
    p.add_argument("--output-manifest", required=True, help="出力manidestファイルパス")
    p.add_argument("--format", dest="output_format", choices=["jsonl", "csv"], default="jsonl",
                   help="出力フォーマット [default: jsonl]")
    p.add_argument("--model", default=DEFAULT_WHISPER_MODEL,
                   help=f"Whisperモデル名 [default: {DEFAULT_WHISPER_MODEL}]")
    p.add_argument("--language", default="ja", help="言語コード（ja/en/None=自動）[default: ja]")
    p.add_argument("--speaker-id", default=None, help="全ファイルに付与する話者ID（省略可）")
    p.add_argument("--voice-design", action="store_true",
                   help="VoiceDesignモード（speaker_idは使わずcaption列を出力）")
    p.add_argument("--voice-design-caption", default=None,
                   help="VoiceDesignモード時に全ファイルへ付与するcaption（省略時はWhisper text）")
    p.add_argument("--device", default=None, help="cuda / cpu（省略=自動）")
    p.add_argument("--model-cache-dir", default=None,
                   help="Whisperモデルのキャッシュ先ディレクトリ（省略=~/.cache/huggingface/hub）")
    p.add_argument("--recursive", action="store_true", help="フォルダを再帰検索")


def _add_emoji_caption_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv",     required=True, help="入力CSVファイルパス（file_name,text）")
    p.add_argument("--wav-dir", required=True, help="wavファイルが格納されているフォルダ")
    p.add_argument("--api",     default="lm_studio",
                   choices=["none", "lm_studio", "groq", "openai", "together"],
                   help="使用するAPIプロバイダー [default: lm_studio]")
    p.add_argument("--api-key",      default="", help="外部APIキー（LM Studioは不要）")
    p.add_argument("--api-base-url", default="", help="APIベースURL（省略=自動）")
    p.add_argument("--model",        default="", help="モデル名（省略=デフォルト）")
    p.add_argument("--voice-design", action="store_true",
                   help="VoiceDesignモード: caption列（音声分析）を生成する。省略時はtext列（絵文字）のみ")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="音声スライス & Whisperキャプション & 絵文字キャプション ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gui", action="store_true", help="GUIモードで起動")
    sub = parser.add_subparsers(dest="command")

    # ── slice サブコマンド ──
    p_slice = sub.add_parser("slice", help="無音区間ベースで音声をスライス")
    _add_slice_args(p_slice)

    # ── caption サブコマンド ──
    p_cap = sub.add_parser("caption", help="Whisperで音声をキャプションしmanifest出力")
    _add_caption_args(p_cap)

    # ── pipeline サブコマンド ──
    p_pipe = sub.add_parser("pipeline", help="スライス → キャプションを一括実行")
    p_pipe.add_argument("--input",           required=True)
    p_pipe.add_argument("--slice-output",    required=True)
    p_pipe.add_argument("--output-manifest", required=True)
    p_pipe.add_argument("--format", dest="output_format", choices=["jsonl", "csv"], default="jsonl")
    p_pipe.add_argument("--min-sec",        type=float, default=2.0)
    p_pipe.add_argument("--max-sec",        type=float, default=30.0)
    p_pipe.add_argument("--threshold",      type=float, default=0.5)
    p_pipe.add_argument("--min-silence-ms", type=int,   default=300)
    p_pipe.add_argument("--speech-pad-ms",  type=int,   default=30)
    p_pipe.add_argument("--target-sr",      type=int,   default=None)
    p_pipe.add_argument("--model",          default=DEFAULT_WHISPER_MODEL)
    p_pipe.add_argument("--language",       default="ja")
    p_pipe.add_argument("--speaker-id",     default=None)
    p_pipe.add_argument("--voice-design", action="store_true")
    p_pipe.add_argument("--voice-design-caption", default=None)
    p_pipe.add_argument("--device",         default=None)
    p_pipe.add_argument("--model-cache-dir", default=None)

    # ── emoji_caption サブコマンド ──
    p_emoji = sub.add_parser("emoji_caption", help="CSVから絵文字キャプションを生成")
    _add_emoji_caption_args(p_emoji)

    args = parser.parse_args()

    # GUIモード
    if args.gui or args.command is None:
        _launch_gui()
        return

    if args.command == "slice":
        slice_audio(
            Path(args.input), Path(args.output),
            min_sec=args.min_sec, max_sec=args.max_sec,
            threshold=args.threshold, min_silence_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
            target_sr=args.target_sr, recursive=args.recursive,
            device=args.device,
        )

    elif args.command == "caption":
        lang = None if args.language.lower() in ("none", "auto", "") else args.language
        caption_audio(
            Path(args.input), Path(args.output_manifest),
            model_name=args.model, language=lang,
            speaker_id=args.speaker_id, output_format=args.output_format,
            voice_design=args.voice_design,
            voice_design_caption=args.voice_design_caption,
            recursive=args.recursive, device=args.device,
            model_cache_dir=args.model_cache_dir,
        )

    elif args.command == "pipeline":
        lang = None if args.language.lower() in ("none", "auto", "") else args.language
        pipeline(
            Path(args.input), Path(args.slice_output), Path(args.output_manifest),
            min_sec=args.min_sec, max_sec=args.max_sec,
            threshold=args.threshold, min_silence_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
            target_sr=args.target_sr,
            model_name=args.model, language=lang,
            speaker_id=args.speaker_id, output_format=args.output_format,
            voice_design=args.voice_design,
            voice_design_caption=args.voice_design_caption,
            device=args.device, model_cache_dir=args.model_cache_dir,
        )

    elif args.command == "emoji_caption":
        emoji_caption(
            Path(args.csv), Path(args.wav_dir),
            api_provider=args.api,
            api_key=args.api_key,
            api_base_url=args.api_base_url,
            model_name=args.model,
            voice_design=args.voice_design,
        )


# ─────────────────────────────────────────────────────────────────────────────
# GUI  (tkinter)
# ─────────────────────────────────────────────────────────────────────────────

def _launch_gui() -> None:
    """tkinter GUIを起動する。"""
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext

    root = tk.Tk()
    root.title("Dataset Tools")
    root.resizable(True, True)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # ── 共通ログウィジェット生成ヘルパー ──
    def _make_log(frame):
        log = scrolledtext.ScrolledText(frame, height=10, state="disabled",
                                        font=("Consolas", 9))
        log.pack(fill="both", expand=True, padx=5, pady=5)
        def write(msg):
            log.config(state="normal")
            log.insert("end", msg + "\n")
            log.see("end")
            log.config(state="disabled")
            root.update_idletasks()
        return write

    def _browse_file(var, filetypes=None):
        path = filedialog.askopenfilename(filetypes=filetypes or [("All", "*.*")])
        if path:
            var.set(path)

    def _browse_dir(var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _browse_save(var, filetypes=None):
        path = filedialog.asksaveasfilename(filetypes=filetypes or [("All", "*.*")])
        if path:
            var.set(path)

    def _row(frame, label, widget, row):
        tk.Label(frame, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=5, pady=3)
        widget.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        frame.columnconfigure(1, weight=1)

    # ══════════════════════════════════════════════════════
    # タブ1: スライス
    # ══════════════════════════════════════════════════════
    tab_slice = ttk.Frame(notebook)
    notebook.add(tab_slice, text="スライス")

    sf = ttk.LabelFrame(tab_slice, text="設定", padding=8)
    sf.pack(fill="x", padx=10, pady=5)

    s_input   = tk.StringVar()
    s_output  = tk.StringVar()
    s_min_sec = tk.StringVar(value="2.0")
    s_max_sec = tk.StringVar(value="30.0")
    s_thresh  = tk.StringVar(value="0.5")
    s_silence = tk.StringVar(value="300")
    s_pad     = tk.StringVar(value="30")
    s_device  = tk.StringVar(value="")

    for i, (lbl, var, btn) in enumerate([
        ("入力ファイル/フォルダ", s_input,  lambda: _browse_file(s_input) or _browse_dir(s_input)),
        ("出力フォルダ",         s_output, lambda: _browse_dir(s_output)),
    ]):
        e = tk.Entry(sf, textvariable=var, width=50)
        b = tk.Button(sf, text="参照", command=btn)
        _row(sf, lbl, e, i)
        b.grid(row=i, column=2, padx=5)

    for i, (lbl, var) in enumerate([
        ("最小秒数",     s_min_sec),
        ("最大秒数",     s_max_sec),
        ("VAD閾値",      s_thresh),
        ("最小無音ms",   s_silence),
        ("パディングms", s_pad),
        ("デバイス(cuda/cpu/空=自動)", s_device),
    ], start=2):
        _row(sf, lbl, tk.Entry(sf, textvariable=var, width=20), i)

    s_log = _make_log(tab_slice)

    def _run_slice():
        import threading
        def task():
            slice_audio(
                Path(s_input.get()), Path(s_output.get()),
                min_sec=float(s_min_sec.get()), max_sec=float(s_max_sec.get()),
                threshold=float(s_thresh.get()),
                min_silence_ms=int(s_silence.get()),
                speech_pad_ms=int(s_pad.get()),
                device=s_device.get() or None,
                log_fn=s_log,
            )
        threading.Thread(target=task, daemon=True).start()

    tk.Button(tab_slice, text="▶ スライス実行", command=_run_slice,
              bg="#4CAF50", fg="white", font=("", 11, "bold")).pack(pady=5)

    # ══════════════════════════════════════════════════════
    # タブ2: キャプション (Whisper)
    # ══════════════════════════════════════════════════════
    tab_cap = ttk.Frame(notebook)
    notebook.add(tab_cap, text="キャプション(Whisper)")

    cf = ttk.LabelFrame(tab_cap, text="設定", padding=8)
    cf.pack(fill="x", padx=10, pady=5)

    c_input   = tk.StringVar()
    c_output  = tk.StringVar()
    c_model   = tk.StringVar(value=DEFAULT_WHISPER_MODEL)
    c_lang    = tk.StringVar(value="ja")
    c_speaker = tk.StringVar(value="")
    c_format  = tk.StringVar(value="csv")
    c_device  = tk.StringVar(value="")

    for i, (lbl, var, is_save) in enumerate([
        ("入力フォルダ/ファイル", c_input,  False),
        ("出力manifestパス",     c_output, True),
    ]):
        e = tk.Entry(cf, textvariable=var, width=50)
        cmd = (lambda v=var: _browse_save(v, [("CSV","*.csv"),("JSONL","*.jsonl"),("All","*.*")])) \
              if is_save else (lambda v=var: _browse_dir(v))
        b = tk.Button(cf, text="参照", command=cmd)
        _row(cf, lbl, e, i)
        b.grid(row=i, column=2, padx=5)

    _row(cf, "Whisperモデル", tk.Entry(cf, textvariable=c_model, width=20), 2)
    _row(cf, "言語コード",    tk.Entry(cf, textvariable=c_lang,  width=10), 3)
    _row(cf, "話者ID(省略可)", tk.Entry(cf, textvariable=c_speaker, width=20), 4)
    _row(cf, "出力形式", ttk.Combobox(cf, textvariable=c_format,
         values=["csv","jsonl"], width=10, state="readonly"), 5)
    _row(cf, "デバイス(cuda/cpu/空=自動)", tk.Entry(cf, textvariable=c_device, width=15), 6)

    c_log = _make_log(tab_cap)

    def _run_caption():
        import threading
        def task():
            lang = None if c_lang.get().lower() in ("none","auto","") else c_lang.get()
            caption_audio(
                Path(c_input.get()), Path(c_output.get()),
                model_name=c_model.get(), language=lang,
                speaker_id=c_speaker.get() or None,
                output_format=c_format.get(),
                device=c_device.get() or None,
                log_fn=c_log,
            )
        threading.Thread(target=task, daemon=True).start()

    tk.Button(tab_cap, text="▶ キャプション実行", command=_run_caption,
              bg="#2196F3", fg="white", font=("", 11, "bold")).pack(pady=5)

    # ══════════════════════════════════════════════════════
    # タブ3: パイプライン
    # ══════════════════════════════════════════════════════
    tab_pipe = ttk.Frame(notebook)
    notebook.add(tab_pipe, text="パイプライン")

    pf = ttk.LabelFrame(tab_pipe, text="設定", padding=8)
    pf.pack(fill="x", padx=10, pady=5)

    p_input    = tk.StringVar()
    p_sliceout = tk.StringVar()
    p_manifest = tk.StringVar()
    p_model    = tk.StringVar(value=DEFAULT_WHISPER_MODEL)
    p_lang     = tk.StringVar(value="ja")
    p_format   = tk.StringVar(value="csv")

    for i, (lbl, var, is_save) in enumerate([
        ("入力音声ファイル",    p_input,    False),
        ("スライス出力フォルダ", p_sliceout, False),
        ("manifest出力パス",   p_manifest, True),
    ]):
        e = tk.Entry(pf, textvariable=var, width=50)
        cmd = (lambda v=var: _browse_save(v, [("CSV","*.csv"),("JSONL","*.jsonl")])) \
              if is_save else (lambda v=var: _browse_file(v) if i == 0 else _browse_dir(v))
        b = tk.Button(pf, text="参照", command=cmd)
        _row(pf, lbl, e, i)
        b.grid(row=i, column=2, padx=5)

    _row(pf, "Whisperモデル", tk.Entry(pf, textvariable=p_model, width=20), 3)
    _row(pf, "言語コード",    tk.Entry(pf, textvariable=p_lang,  width=10), 4)
    _row(pf, "出力形式", ttk.Combobox(pf, textvariable=p_format,
         values=["csv","jsonl"], width=10, state="readonly"), 5)

    p_log = _make_log(tab_pipe)

    def _run_pipeline():
        import threading
        def task():
            lang = None if p_lang.get().lower() in ("none","auto","") else p_lang.get()
            pipeline(
                Path(p_input.get()), Path(p_sliceout.get()), Path(p_manifest.get()),
                model_name=p_model.get(), language=lang,
                output_format=p_format.get(), log_fn=p_log,
            )
        threading.Thread(target=task, daemon=True).start()

    tk.Button(tab_pipe, text="▶ パイプライン実行", command=_run_pipeline,
              bg="#9C27B0", fg="white", font=("", 11, "bold")).pack(pady=5)

    # ══════════════════════════════════════════════════════
    # タブ4: 絵文字キャプション (新機能)
    # ══════════════════════════════════════════════════════
    tab_emoji = ttk.Frame(notebook)
    notebook.add(tab_emoji, text="🎭 絵文字キャプション")

    ef = ttk.LabelFrame(tab_emoji, text="ファイル設定", padding=8)
    ef.pack(fill="x", padx=10, pady=5)

    e_csv     = tk.StringVar()
    e_wavdir  = tk.StringVar()

    for i, (lbl, var, is_file) in enumerate([
        ("入力CSV（file_name,text）", e_csv,    True),
        ("wavファイルフォルダ",        e_wavdir, False),
    ]):
        ent = tk.Entry(ef, textvariable=var, width=50)
        cmd = (lambda v=var: _browse_file(v, [("CSV","*.csv"),("All","*.*")])) \
              if is_file else (lambda v=var: _browse_dir(v))
        b = tk.Button(ef, text="参照", command=cmd)
        _row(ef, lbl, ent, i)
        b.grid(row=i, column=2, padx=5)

    # API設定フレーム
    af = ttk.LabelFrame(tab_emoji, text="API設定", padding=8)
    af.pack(fill="x", padx=10, pady=5)

    e_api      = tk.StringVar(value="なし（従来通り）")
    e_apikey   = tk.StringVar(value="")
    e_baseurl  = tk.StringVar(value="")
    e_model    = tk.StringVar(value="")
    e_lm_model = tk.StringVar(value="")

    # API選択プルダウン
    api_label_to_key = API_PROVIDERS  # {"なし（従来通り）": "none", ...}
    api_combo = ttk.Combobox(
        af, textvariable=e_api,
        values=list(api_label_to_key.keys()),
        state="readonly", width=30,
    )
    _row(af, "APIプロバイダー", api_combo, 0)

    # 動的に表示切り替えするフレーム
    api_detail_frame = ttk.Frame(af)
    api_detail_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=3)
    af.columnconfigure(1, weight=1)

    lbl_apikey  = tk.Label(api_detail_frame, text="APIキー", anchor="w")
    ent_apikey  = tk.Entry(api_detail_frame, textvariable=e_apikey, width=40, show="*")
    lbl_baseurl = tk.Label(api_detail_frame, text="ベースURL（省略=自動）", anchor="w")
    ent_baseurl = tk.Entry(api_detail_frame, textvariable=e_baseurl, width=40)
    lbl_model   = tk.Label(api_detail_frame, text="モデル名（省略=デフォルト）", anchor="w")
    ent_model   = tk.Entry(api_detail_frame, textvariable=e_model, width=40)

    def _refresh_api_fields(*_):
        key = api_label_to_key.get(e_api.get(), "none")
        # 全ウィジェットを一度非表示
        for w in api_detail_frame.winfo_children():
            w.grid_forget()

        if key == "none":
            tk.Label(api_detail_frame,
                     text="LLM処理なし：CSVのtextをcaptionにそのままコピーします",
                     fg="gray").grid(row=0, column=0, columnspan=2, sticky="w", padx=5)
            return

        row = 0
        if key != "lm_studio":
            lbl_apikey.grid( row=row, column=0, sticky="w", padx=5, pady=2)
            ent_apikey.grid( row=row, column=1, sticky="ew", padx=5, pady=2)
            row += 1

        lbl_baseurl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ent_baseurl.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        row += 1

        lbl_model.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ent_model.grid(row=row, column=1, sticky="ew", padx=5, pady=2)

        # デフォルトモデルをヒント表示
        default_m = API_DEFAULT_MODELS.get(key, "")
        ent_model.config(fg="black")
        if not e_model.get():
            ent_model.delete(0, "end")
            ent_model.insert(0, default_m)

        api_detail_frame.columnconfigure(1, weight=1)

    api_combo.bind("<<ComboboxSelected>>", _refresh_api_fields)
    _refresh_api_fields()  # 初期表示

    e_log = _make_log(tab_emoji)

    def _run_emoji():
        import threading
        api_key_str = api_label_to_key.get(e_api.get(), "none")

        if api_key_str == "none":
            # なし：textをそのままcaptionにコピー
            def task_none():
                csv_path = Path(e_csv.get())
                if not csv_path.exists():
                    e_log(f"[ERROR] CSVが見つかりません: {csv_path}")
                    return
                rows = []
                with open(csv_path, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    fieldnames = list(reader.fieldnames or [])
                    rows = list(reader)
                if "caption" not in fieldnames:
                    fieldnames.append("caption")
                for r in rows:
                    r["caption"] = r.get("text", "")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                e_log(f"完了: textをcaptionにコピーしました ({len(rows)}件) → {csv_path}")
            threading.Thread(target=task_none, daemon=True).start()
            return

        def task():
            emoji_caption(
                Path(e_csv.get()), Path(e_wavdir.get()),
                api_provider=api_key_str,
                api_key=e_apikey.get(),
                api_base_url=e_baseurl.get(),
                model_name=e_model.get(),
                log_fn=e_log,
            )
        threading.Thread(target=task, daemon=True).start()

    tk.Button(tab_emoji, text="▶ 絵文字キャプション実行", command=_run_emoji,
              bg="#FF5722", fg="white", font=("", 11, "bold")).pack(pady=8)

    root.mainloop()


if __name__ == "__main__":
    main()
