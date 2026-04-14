from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class ShareGPTTagConfig:
    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    observation_tag: str = "observation"
    function_tag: str = "function_call"
    system_tag: str = "system"


@dataclass
class ShareGPTDatasetConfig:
    """Configuration for normalizing ShareGPT-like samples into the intermediate dataset format."""

    messages_key: str = "conversations"
    images_key: str = "images"
    videos_key: str = "videos"
    audios_key: str = "audios"
    id_key: str = "id"
    data_source_key: str = "data_source"
    default_data_source: str = "sharegpt"
    source_file_key: str = "__sharegpt_source_file"

    ground_truth_key: str = "ground_truth"
    fallback_ground_truth_keys: list[str] = field(default_factory=list)

    drop_all_assistant_messages: bool = False

    strict_role_check: bool = True
    strict_messages_check: bool = True
    strict_image_type_check: bool = False
    strict_video_type_check: bool = False
    strict_audio_type_check: bool = False
    strict_media_token_match: bool = False

    # Indexed media reference behavior (<@image:N> / <@video:N> / <@audio:N>):
    # references in prompt text are always resolved (cheap, no collision with
    # legacy <image> tokens). Traversal of extra_info is opt-in because the
    # hot path in prompt-only datasets shouldn't pay for a deep walk it never
    # needs. AgentShareGPTDataset flips the walk flag on.
    resolve_extra_info_references: bool = False
    strict_references: bool = True

    pass_through_keys: list[str] = field(default_factory=list)
    tags: ShareGPTTagConfig = field(default_factory=ShareGPTTagConfig)


def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested mapping overrides into a base config."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(overrides: dict[str, Any] | None = None) -> ShareGPTDatasetConfig:
    """Build config from dataclass defaults + optional dict overrides."""
    if not overrides:
        return ShareGPTDatasetConfig()

    base = asdict(ShareGPTDatasetConfig())
    merged = _merge_dicts(base, overrides)

    tags_cfg = merged.pop("tags", {})
    if isinstance(tags_cfg, dict):
        tags = ShareGPTTagConfig(**tags_cfg)
    else:
        tags = tags_cfg
    return ShareGPTDatasetConfig(**merged, tags=tags)


__all__ = [
    "ShareGPTTagConfig",
    "ShareGPTDatasetConfig",
    "build_config",
]
