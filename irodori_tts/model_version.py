"""モデルバージョンの自動判別ユーティリティ。

チェックポイントに埋め込まれたメタデータ（config_json / text_encoder_config_json）
や state_dict のキー構成からモデル世代を分類する。
判定結果は UI への表示、LoRA 既定値の選択、学習設定の自動補完に使用する。

モデルを読まない処理（マニフェスト作成など）では自動判別できないため、
呼び出し側でプルダウン等の手動選択を用意する（MANIFEST_MODE_* 対応表を参照）。

分類方針（config_json の内容から判定）:
  - latent_dim=128                        -> v1
  - text_encoder_type="pretrained"        -> v4（v4-Small / v4.1-Small は同一アーキテクチャ）
  - duration + caption 条件               -> v3-VoiceDesign
  - duration 条件のみ                     -> v3
  - dim32 のみ                            -> v2
メタデータが無いチェックポイントの場合は state_dict のキー構成で推定する。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# UI のプルダウンで使う選択値。"auto" はチェックポイントからの自動判別を意味する。
VERSION_AUTO = "auto"
VERSION_KEYS = ("v1", "v2", "v3", "v3_voicedesign", "v4")
VERSION_CHOICES = (VERSION_AUTO,) + VERSION_KEYS

CODEC_REPO_V1 = "facebook/dacvae-watermarked"
CODEC_REPO_DIM32 = "Aratako/Semantic-DACVAE-Japanese-32dim"

# Prepare Manifest タブ / prepare_manifest.py のモード名との対応
MANIFEST_MODE_FOR_VERSION = {
    "v1": "model_v1",
    "v2": "model_v2",
    "v3": "model_v3",
    "v3_voicedesign": "voice_design",
    "v4": "model_v4",
}
VERSION_FOR_MANIFEST_MODE = {mode: key for key, mode in MANIFEST_MODE_FOR_VERSION.items()}


@dataclass(frozen=True)
class ModelVersionInfo:
    """モデル世代ごとの識別情報と、フォーク版で推奨する既定値。"""

    key: str
    label: str
    latent_dim: int
    codec_repo: str
    has_duration: bool
    has_caption: bool
    has_pretrained_text: bool
    supports_speaker_inversion: bool
    manifest_mode: str
    # LoRA 学習の推奨既定値（フォーク版の仕様: カンマ区切り list または正規表現1本）
    lora_target_modules: str
    lora_modules_to_save: str


_VERSION_TABLE: dict[str, ModelVersionInfo] = {
    "v1": ModelVersionInfo(
        key="v1",
        label="v1",
        latent_dim=128,
        codec_repo=CODEC_REPO_V1,
        has_duration=False,
        has_caption=False,
        has_pretrained_text=False,
        supports_speaker_inversion=False,
        manifest_mode="model_v1",
        lora_target_modules="wq,wk,wv,wo",
        lora_modules_to_save="none",
    ),
    "v2": ModelVersionInfo(
        key="v2",
        label="v2",
        latent_dim=32,
        codec_repo=CODEC_REPO_DIM32,
        has_duration=False,
        has_caption=False,
        has_pretrained_text=False,
        supports_speaker_inversion=False,
        manifest_mode="model_v2",
        lora_target_modules="wq,wk,wv,wo",
        lora_modules_to_save="none",
    ),
    "v3": ModelVersionInfo(
        key="v3",
        label="v3",
        latent_dim=32,
        codec_repo=CODEC_REPO_DIM32,
        has_duration=True,
        has_caption=False,
        has_pretrained_text=False,
        supports_speaker_inversion=False,
        manifest_mode="model_v3",
        lora_target_modules="wq,wk,wv,wo",
        lora_modules_to_save="none",
    ),
    "v3_voicedesign": ModelVersionInfo(
        key="v3_voicedesign",
        label="v3-VoiceDesign",
        latent_dim=32,
        codec_repo=CODEC_REPO_DIM32,
        has_duration=True,
        has_caption=True,
        has_pretrained_text=False,
        supports_speaker_inversion=False,
        manifest_mode="voice_design",
        lora_target_modules="wq,wk,wv,wo",
        lora_modules_to_save="none",
    ),
    "v4": ModelVersionInfo(
        key="v4",
        label="v4-Small/v4.1-Small",
        latent_dim=32,
        codec_repo=CODEC_REPO_DIM32,
        has_duration=True,
        has_caption=True,
        has_pretrained_text=True,
        supports_speaker_inversion=True,
        manifest_mode="model_v4",
        # v4 系は事前学習テキストエンコーダを持つため、LoRA は拡散ブロックの
        # attention へ限定するのが上流レシピ相当（フォーク版の正規表現指定で表現）。
        lora_target_modules=(
            r"^blocks\.\d+\.attention\."
            r"(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)$"
        ),
        lora_modules_to_save="duration_predictor",
    ),
}


def version_info(key: str) -> ModelVersionInfo:
    """バージョンキー（v1 等）から情報を取得する。未知のキーは ValueError。"""
    normalized = normalize_version_choice(key)
    if normalized is None or normalized not in _VERSION_TABLE:
        raise ValueError(f"Unknown model version: {key!r}")
    return _VERSION_TABLE[normalized]


def normalize_version_choice(choice: Any) -> str | None:
    """プルダウン等の選択値を正規化する。"auto" は None（自動判別）として扱う。"""
    if choice is None:
        return None
    value = str(choice).strip().lower()
    if not value or value == VERSION_AUTO:
        return None
    if value in {"v3-voicedesign", "v3_voicedesign", "voicedesign", "voice_design"}:
        return "v3_voicedesign"
    if value in _VERSION_TABLE:
        return value
    raise ValueError(f"Unknown model version choice: {choice!r}")


def classify_model_version(
    config_dict: Mapping[str, Any] | None,
    state_dict_keys: Iterable[str] | None = None,
) -> ModelVersionInfo | None:
    """チェックポイントのメタデータ／キー構成からモデル世代を推定する。

    判定できない場合は None を返す（呼び出し側で手動選択にフォールバックする）。
    """
    if isinstance(config_dict, Mapping) and config_dict:
        return _classify_from_config(config_dict)

    if state_dict_keys is not None:
        return _classify_from_state_keys(state_dict_keys)

    return None


def resolve_model_version(
    choice: Any,
    config_dict: Mapping[str, Any] | None,
    state_dict_keys: Iterable[str] | None = None,
) -> ModelVersionInfo | None:
    """プルダウン選択（auto/明示指定）とチェックポイント情報から最終的な世代を決める。

    明示指定が最優先、"auto" または未指定のときはメタデータから自動判別する。
    """
    explicit = normalize_version_choice(choice)
    if explicit is not None:
        return _VERSION_TABLE[explicit]
    return classify_model_version(config_dict, state_dict_keys)


def describe_model_version(info: ModelVersionInfo | None) -> str:
    """UI 表示用の短い説明文字列を生成する。"""
    if info is None:
        return "model version: 不明（自動判別できず）"
    features = []
    if info.has_duration:
        features.append("duration")
    if info.has_caption:
        features.append("caption")
    if info.has_pretrained_text:
        features.append("pretrained_text")
    if info.supports_speaker_inversion:
        features.append("speaker_tokens")
    feature_text = ", ".join(features) if features else "base"
    return f"model version: {info.label} (dim{info.latent_dim}, {feature_text})"


def _classify_from_config(config: Mapping[str, Any]) -> ModelVersionInfo | None:
    latent_dim = config.get("latent_dim")
    if isinstance(latent_dim, int) and latent_dim != 32:
        # v1 のみ dim128（dim32 以外の将来世代はここでは扱わない）
        return _VERSION_TABLE["v1"]

    text_encoder_type = str(config.get("text_encoder_type") or "").strip().lower()
    if text_encoder_type == "pretrained":
        return _VERSION_TABLE["v4"]

    has_duration = bool(config.get("use_duration_predictor"))
    has_caption = bool(config.get("use_caption_condition"))
    if has_duration and has_caption:
        return _VERSION_TABLE["v3_voicedesign"]
    if has_duration:
        return _VERSION_TABLE["v3"]
    if isinstance(latent_dim, int) and latent_dim == 32:
        return _VERSION_TABLE["v2"]
    return None


def _classify_from_state_keys(keys: Iterable[str]) -> ModelVersionInfo | None:
    has_pretrained = False
    has_duration = False
    has_caption = False
    for key in keys:
        name = str(key)
        if name.startswith("pretrained_text_backbone."):
            has_pretrained = True
        elif name.startswith("duration_predictor."):
            has_duration = True
        elif name.startswith("caption_encoder."):
            has_caption = True
    if has_pretrained:
        return _VERSION_TABLE["v4"]
    if has_duration and has_caption:
        return _VERSION_TABLE["v3_voicedesign"]
    if has_duration:
        return _VERSION_TABLE["v3"]
    return None
