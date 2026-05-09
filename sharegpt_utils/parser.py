from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ShareGPTDatasetConfig
from .reference import resolve_references_in_obj

logger = logging.getLogger(__name__)

_MEDIA_TOKEN_PATTERN = re.compile(r"(<@(?:image|video|audio):\d+>|<image>|<video>|<audio>)")
_INDEXED_TOKEN_PATTERN = re.compile(r"<@(image|video|audio):(\d+)>")
_TOKEN_TO_MEDIA_TYPE = {
    "<image>": "image",
    "<video>": "video",
    "<audio>": "audio",
}
_MISSING = object()
MEDIA_TYPES = ("image", "video", "audio")
_WARNED_DROPPED_TOP_LEVEL_KEY_SETS: set[tuple[str, ...]] = set()


@dataclass
class ParsedSample:
    """Canonical typed output of ShareGPTParser."""

    id: Any
    messages: list[dict[str, Any]]
    images: list[Any]
    videos: list[Any]
    audios: list[Any]
    ground_truth: Any
    data_source: str
    extra_info: dict[str, Any]
    pass_through: dict[str, Any]


def _deep_get(mapping: dict[str, Any], dotted_key: str) -> Any:
    """Read a dotted key from a nested mapping, raising ``KeyError`` if missing."""
    current: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Key '{key}' not found in {current} using '{dotted_key}'.")
        current = current[key]
    return current


def _to_text(value: Any) -> str:
    """Convert arbitrary content into the canonical text representation."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_media_refs(raw_media: Any, strict: bool, media_name: str) -> list[Any]:
    """Normalize a media field to a list and optionally validate item types."""
    if raw_media is None:
        return []
    media_list = raw_media if isinstance(raw_media, list) else [raw_media]
    if strict:
        for i, media_item in enumerate(media_list):
            if not isinstance(media_item, str):
                raise TypeError(f"{media_name}[{i}] must be str path/url, got {type(media_item).__name__}")
    return media_list


@dataclass
class _MediaPool:
    _items: dict[str, list[Any]]
    _offsets: dict[str, int]
    _referenced: dict[str, set[int]]

    @classmethod
    def create(cls, images: list[Any], videos: list[Any], audios: list[Any]) -> _MediaPool:
        return cls(
            _items={"image": images, "video": videos, "audio": audios},
            _offsets={"image": 0, "video": 0, "audio": 0},
            _referenced={"image": set(), "video": set(), "audio": set()},
        )

    def consume(self, media_type: str) -> Any | None:
        """Consume the next not-yet-used media item of the requested type.

        Skips indices already claimed by ``<@kind:N>`` references so legacy
        ``<image>`` tokens and indexed references can coexist in one sample
        without double-counting.
        """
        items = self._items[media_type]
        referenced = self._referenced[media_type]
        offset = self._offsets[media_type]
        while offset < len(items) and offset in referenced:
            offset += 1
        if offset >= len(items):
            self._offsets[media_type] = offset
            return None
        self._offsets[media_type] = offset + 1
        return items[offset]

    def mark_referenced(self, media_type: str, index: int) -> None:
        """Flag that a ``<@kind:N>`` reference claimed this index (no offset move)."""
        self._referenced[media_type].add(index)

    def remaining(self, media_type: str) -> list[Any]:
        """Return items neither consumed in order nor claimed by reference."""
        items = self._items[media_type]
        offset = self._offsets[media_type]
        referenced = self._referenced[media_type]
        return [items[i] for i in range(offset, len(items)) if i not in referenced]

    def as_lookup(self) -> dict[str, list[Any]]:
        """Return the full pool for random-access reference resolution."""
        return self._items

    def mark_all_consumed(self) -> None:
        """Mark all media items as consumed after fallback injection."""
        for media_type in MEDIA_TYPES:
            self._offsets[media_type] = len(self._items[media_type])

    def assert_all_consumed(self) -> None:
        """Raise when strict media matching leaves unused top-level media refs."""
        for media_type in MEDIA_TYPES:
            if self.remaining(media_type):
                raise ValueError(f"Unused {media_type}s remain: {len(self.remaining(media_type))}")


class ShareGPTParser:
    """Parser that converts ShareGPT samples into framework-agnostic message samples."""

    def __init__(self, config: ShareGPTDatasetConfig | None = None):
        """Create a parser with ShareGPT field names and normalization rules."""
        self.config = config or ShareGPTDatasetConfig()

        tags = self.config.tags
        self.tag_mapping = {
            tags.system_tag: "system",
            tags.user_tag: "user",
            tags.assistant_tag: "assistant",
            tags.observation_tag: "tool",
            tags.function_tag: "assistant",
        }

    @staticmethod
    def _resolve_local_media_ref(media_ref: Any, source_file: str | None) -> Any:
        """Resolve relative local media paths against the source dataset file."""
        if not isinstance(media_ref, str):
            return media_ref

        text = media_ref.strip()
        if not text:
            return media_ref

        lowered = text.lower()
        if lowered.startswith("http://") or lowered.startswith("https://") or lowered.startswith("file://") or lowered.startswith("data:"):
            return media_ref

        path = Path(text)
        if path.is_absolute():
            return str(path)
        if not source_file:
            return media_ref
        return str((Path(source_file).resolve().parent / path).resolve())

    def _build_media_pool(self, raw_sample: dict[str, Any]) -> _MediaPool:
        """Load top-level image, video, and audio refs into an ordered pool."""
        source_file = raw_sample.get(self.config.source_file_key)
        images = _normalize_media_refs(raw_sample.get(self.config.images_key, []), self.config.strict_image_type_check, self.config.images_key)
        videos = _normalize_media_refs(raw_sample.get(self.config.videos_key, []), self.config.strict_video_type_check, self.config.videos_key)
        audios = _normalize_media_refs(raw_sample.get(self.config.audios_key, []), self.config.strict_audio_type_check, self.config.audios_key)

        return _MediaPool.create(
            images=[self._resolve_local_media_ref(ref, source_file=source_file) for ref in images],
            videos=[self._resolve_local_media_ref(ref, source_file=source_file) for ref in videos],
            audios=[self._resolve_local_media_ref(ref, source_file=source_file) for ref in audios],
        )

    def _build_text_item(self, value: Any) -> dict[str, Any]:
        """Wrap plain content into the canonical text content block."""
        return {"type": "text", "text": _to_text(value)}

    def _build_media_item(self, media_type: str, value: Any) -> dict[str, Any]:
        """Wrap a resolved media reference into the canonical content block."""
        return {"type": media_type, media_type: value}

    def _parse_text_content(self, content: str, media_pool: _MediaPool) -> list[dict[str, Any]]:
        """Parse string content into a uniform list of text/media content blocks.

        Handles two token forms in prompt text:
          * Legacy ``<image>`` / ``<video>`` / ``<audio>`` — consume the pool in order.
          * Indexed ``<@image:N>`` / ``<@video:N>`` / ``<@audio:N>`` — random-access
            the pool by index without consuming. Does not advance offsets, so
            legacy tokens intermixed in the same message still see the full pool.
        Both emit a structured media content block, not inline text.
        """
        content_list: list[dict[str, Any]] = []

        for segment in [seg for seg in _MEDIA_TOKEN_PATTERN.split(content) if seg != ""]:
            indexed = _INDEXED_TOKEN_PATTERN.fullmatch(segment)
            if indexed is not None:
                media_type = indexed.group(1)
                index = int(indexed.group(2))
                pool_items = media_pool.as_lookup().get(media_type, [])
                if 0 <= index < len(pool_items):
                    media_pool.mark_referenced(media_type, index)
                    content_list.append(self._build_media_item(media_type, pool_items[index]))
                    continue
                if self.config.strict_references:
                    raise ValueError(
                        f"Unresolvable reference {segment}: {media_type} pool has {len(pool_items)} item(s)."
                    )
                content_list.append(self._build_text_item(segment))
                continue

            media_type = _TOKEN_TO_MEDIA_TYPE.get(segment)
            if media_type is None:
                content_list.append(self._build_text_item(segment))
                continue

            media_value = media_pool.consume(media_type)
            if media_value is not None:
                content_list.append(self._build_media_item(media_type, media_value))
                continue

            if self.config.strict_media_token_match:
                raise ValueError(f"Found {segment} token but no {media_type} is available.")
            content_list.append(self._build_text_item(segment))

        if not content_list:
            content_list.append(self._build_text_item(content))
        return content_list

    def _extract_media_value(self, item: dict[str, Any], media_type: str) -> Any | None:
        """Extract a media payload from a structured content item using known aliases."""
        if media_type == "image":
            aliases = ("image", "image_url")
        elif media_type == "video":
            aliases = ("video",)
        elif media_type == "audio":
            aliases = ("audio", "audio_url")
        else:
            raise ValueError(f"Unsupported media type: {media_type}")

        item_type = item.get("type")
        explicit_match = item_type in {media_type, *aliases}
        if not explicit_match and not any(alias in item for alias in aliases):
            return _MISSING

        for alias in aliases:
            if alias in item:
                value = item[alias]
                break
        else:
            value = item.get(media_type)

        if isinstance(value, dict) and "url" in value:
            value = value["url"]
        return value

    def _parse_content_item(self, item: Any, media_pool: _MediaPool, source_file: str | None) -> list[dict[str, Any]]:
        """Normalize one structured content item into one or more canonical blocks."""
        if not isinstance(item, dict):
            return [self._build_text_item(item)]

        if item.get("type") == "text":
            return self._parse_text_content(_to_text(item.get("text", "")), media_pool)

        for media_type in ("image", "video", "audio"):
            media_value = self._extract_media_value(item, media_type)
            if media_value is _MISSING:
                continue
            media_value = self._resolve_local_media_ref(media_value, source_file=source_file)
            return [self._build_media_item(media_type, media_value)]

        if "text" in item:
            return self._parse_text_content(_to_text(item.get("text", "")), media_pool)

        return [self._build_text_item(item)]

    def _parse_message_content(self, content: Any, media_pool: _MediaPool, source_file: str | None) -> list[dict[str, Any]]:
        """Normalize one message content payload into canonical content blocks."""
        if isinstance(content, str):
            return self._parse_text_content(content, media_pool)
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for item in content:
                blocks.extend(self._parse_content_item(item, media_pool, source_file=source_file))
            return blocks
        return [self._build_text_item(content)]

    def _parse_messages(
        self,
        raw_sample: dict[str, Any],
        media_pool: _MediaPool,
    ) -> list[dict[str, Any]]:
        """Normalize raw ShareGPT conversations into canonical role/content messages."""
        raw_messages = raw_sample.get(self.config.messages_key, [])
        if not isinstance(raw_messages, list):
            if self.config.strict_messages_check:
                raise TypeError(f"{self.config.messages_key} must be a list, got {type(raw_messages).__name__}")
            return []

        source_file = raw_sample.get(self.config.source_file_key)
        messages: list[dict[str, Any]] = []

        for i, message in enumerate(raw_messages):
            if not isinstance(message, dict):
                if self.config.strict_messages_check:
                    raise TypeError(f"message at index {i} must be dict, got {type(message).__name__}")
                continue

            raw_role = message.get(self.config.tags.role_tag)
            if raw_role not in self.tag_mapping:
                if self.config.strict_role_check:
                    raise ValueError(f"Unsupported role tag: {raw_role}")
                logger.warning("Skipping unsupported role tag: %s", raw_role)
                continue

            normalized_content = self._parse_message_content(
                message.get(self.config.tags.content_tag),
                media_pool,
                source_file=source_file,
            )
            messages.append({"role": self.tag_mapping[raw_role], "content": normalized_content})

        self._inject_remaining_media(messages, media_pool)
        if self.config.strict_media_token_match:
            media_pool.assert_all_consumed()
        return messages

    def _inject_remaining_media(self, messages: list[dict[str, Any]], media_pool: _MediaPool) -> None:
        """Prepend any unused top-level media refs to the first user message."""
        remaining = {mt: media_pool.remaining(mt) for mt in MEDIA_TYPES}
        if not any(remaining.values()):
            return

        first_user_index = next(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            -1,
        )
        if first_user_index < 0:
            if self.config.strict_media_token_match:
                raise ValueError("Media inputs exist but no user message was found.")
            return

        prefix: list[dict[str, Any]] = []
        for media_type in MEDIA_TYPES:
            prefix.extend(self._build_media_item(media_type, ref) for ref in remaining[media_type])

        messages[first_user_index]["content"] = prefix + messages[first_user_index]["content"]
        media_pool.mark_all_consumed()

    def _derive_ground_truth(self, raw_sample: dict[str, Any], messages: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
        """Resolve supervision targets from explicit keys and strip any trailing assistant turn.

        Ground truth is always read from ``ground_truth_key`` (with fallbacks).
        A trailing assistant message is treated as a data-prep debugging artifact
        and silently removed so that data pipelines can include a reference answer
        without conflicting with the explicit ground truth.
        """
        candidate_keys: list[str] = []
        for key in [self.config.ground_truth_key, *self.config.fallback_ground_truth_keys]:
            if key not in candidate_keys:
                candidate_keys.append(key)

        found_keys: list[str] = []
        ground_truth: Any = None
        for key in candidate_keys:
            try:
                value = _deep_get(raw_sample, key)
            except KeyError:
                continue
            found_keys.append(key)
            ground_truth = value

        if len(found_keys) > 1:
            raise ValueError(f"Multiple ground-truth keys are present: {found_keys}")

        # Strip trailing assistant turn — it exists only for offline debugging.
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]

        if self.config.drop_all_assistant_messages:
            messages = [message for message in messages if message["role"] != "assistant"]
        return ground_truth, messages

    def _extract_media_outputs(self, messages: list[dict[str, Any]]) -> tuple[list[Any], list[Any], list[Any]]:
        """Collect normalized media refs back out of the final message sequence."""
        images: list[Any] = []
        videos: list[Any] = []
        audios: list[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                raise TypeError(f"Expected content to be a list of blocks, got {type(content).__name__}: {content}")
            for item in content:
                if not isinstance(item, dict):
                    raise TypeError(f"Content block items must be dict, got {type(item).__name__}: {item}")
                if item.get("type") == "image" and "image" in item:
                    images.append(item["image"])
                elif item.get("type") == "video" and "video" in item:
                    videos.append(item["video"])
                elif item.get("type") == "audio" and "audio" in item:
                    audios.append(item["audio"])
        return images, videos, audios

    def _resolve_data_source(self, raw_sample: dict[str, Any], sample_id: Any) -> str:
        """Validate and return the task/data source attached to a sample."""
        if self.config.data_source_key not in raw_sample:
            fallback_data_source = self.config.default_data_source
            logger.warning(
                "Missing '%s' in sample id=%s; falling back to '%s'.",
                self.config.data_source_key,
                sample_id,
                fallback_data_source,
            )
            return fallback_data_source

        data_source = raw_sample.get(self.config.data_source_key)
        if not isinstance(data_source, str) or not data_source.strip():
            raise ValueError(f"Invalid '{self.config.data_source_key}' in sample id={sample_id}: {data_source}")
        return data_source

    def _normalize_extra_info(
        self,
        raw_sample: dict[str, Any],
        index: int,
        media_pool: _MediaPool,
    ) -> dict[str, Any]:
        """Normalize ``extra_info``, stamp a dataset index, and optionally resolve refs.

        When ``resolve_extra_info_references`` is set (agent-style datasets),
        deep-walk the dict and substitute ``<@kind:N>`` leaves against the
        already-built media pool. Off by default for the prompt-only hot path.
        """
        extra_info = raw_sample.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        else:
            extra_info = dict(extra_info)
        extra_info.setdefault("index", index)
        if self.config.resolve_extra_info_references:
            extra_info = resolve_references_in_obj(
                extra_info,
                media_pool.as_lookup(),
                strict=self.config.strict_references,
                path="$.extra_info",
                on_reference=media_pool.mark_referenced,
            )
        return extra_info

    def _allowed_top_level_keys(self) -> set[str]:
        allowed = {
            self.config.id_key,
            self.config.messages_key,
            self.config.images_key,
            self.config.videos_key,
            self.config.audios_key,
            self.config.data_source_key,
            self.config.source_file_key,
            self.config.ground_truth_key,
            "extra_info",
        }
        allowed.update(key.split(".", 1)[0] for key in self.config.fallback_ground_truth_keys)
        allowed.update(self.config.pass_through_keys)
        return allowed

    def _warn_on_dropped_top_level_keys(self, raw_sample: dict[str, Any], sample_id: Any) -> None:
        """Warn when non-canonical top-level keys will be discarded during normalization."""
        dropped_keys = sorted(set(raw_sample) - self._allowed_top_level_keys())
        if not dropped_keys:
            return
        dropped_key_set = tuple(dropped_keys)
        if dropped_key_set in _WARNED_DROPPED_TOP_LEVEL_KEY_SETS:
            return
        _WARNED_DROPPED_TOP_LEVEL_KEY_SETS.add(dropped_key_set)
        logger.warning(
            "Dropping unexpected top-level keys for sample id=%s: %s. This warning is emitted once per dropped-key set. Move task metadata under 'extra_info' or add keys to pass_through_keys.",
            sample_id,
            dropped_keys,
        )

    def _build_parsed_sample(
        self,
        raw_sample: dict[str, Any],
        sample_id: Any,
        messages: list[dict[str, Any]],
        ground_truth: Any,
        data_source: str,
        extra_info: dict[str, Any],
    ) -> ParsedSample:
        """Assemble the final framework-agnostic normalized sample."""
        images, videos, audios = self._extract_media_outputs(messages)
        pass_through: dict[str, Any] = {}
        for key in self.config.pass_through_keys:
            if key in raw_sample:
                pass_through[key] = raw_sample[key]
        return ParsedSample(
            id=sample_id,
            messages=messages,
            images=images,
            videos=videos,
            audios=audios,
            ground_truth=ground_truth,
            data_source=data_source,
            extra_info=extra_info,
            pass_through=pass_through,
        )

    def parse_sample(self, raw_sample: dict[str, Any], index: int) -> ParsedSample:
        """Convert one raw ShareGPT row into the canonical intermediate sample format.

        Only canonical top-level fields plus ``extra_info`` are kept by default.
        Additional top-level fields require explicit ``pass_through_keys`` whitelisting.
        """
        sample_id = raw_sample.get(self.config.id_key, index)
        self._warn_on_dropped_top_level_keys(raw_sample, sample_id)
        media_pool = self._build_media_pool(raw_sample)
        # Walk extra_info first so any <@kind:N> references there mark pool
        # indices before _parse_messages runs _inject_remaining_media; without
        # this, a referenced image would also get auto-prepended to the prompt.
        extra_info = self._normalize_extra_info(raw_sample, index, media_pool)
        messages = self._parse_messages(raw_sample, media_pool)
        ground_truth, messages = self._derive_ground_truth(raw_sample, messages)
        data_source = self._resolve_data_source(raw_sample, sample_id)
        return self._build_parsed_sample(raw_sample, sample_id, messages, ground_truth, data_source, extra_info)
