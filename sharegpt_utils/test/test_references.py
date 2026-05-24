"""Tests for indexed media references (<@image:N>, <@video:N>, <@audio:N>).

Covers the three contracts:
  1. Prompt-side resolution (both forms — legacy <image> token and indexed
     <@image:N> — emit structured media blocks, coexist without double-count).
  2. extra_info walk (opt-in via resolve_extra_info_references) rewrites
     every leaf string that is or contains a reference.
  3. Strict mode raises with a JSON path on out-of-range references.
"""

from __future__ import annotations

import pytest

from sharegpt_utils import ShareGPTParser, build_config


def _row(conversations, images=None, extra_info=None, **kwargs):
    row = {
        "id": "test",
        "data_source": "test",
        "conversations": conversations,
        "images": images or [],
    }
    if extra_info is not None:
        row["extra_info"] = extra_info
    row.update(kwargs)
    return row


# ---------------------------------------------------------------------------
# Prompt-side: indexed references

def test_indexed_image_reference_in_prompt_emits_media_block():
    parser = ShareGPTParser(build_config())
    row = _row(
        [{"from": "human", "value": "look at <@image:0>"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/cat.jpg"],
    )
    content = parser.parse_sample(row, 0).messages[0]["content"]
    assert content == [
        {"type": "text", "text": "look at "},
        {"type": "image", "image": "/abs/cat.jpg"},
    ]


def test_indexed_reference_does_not_auto_prepend_pool_item():
    """A referenced image must not also appear via _inject_remaining_media."""
    parser = ShareGPTParser(build_config())
    row = _row(
        [{"from": "human", "value": "<@image:0>"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/a.jpg"],
    )
    msgs = parser.parse_sample(row, 0).messages
    image_blocks = [b for m in msgs for b in m["content"] if b.get("type") == "image"]
    assert image_blocks == [{"type": "image", "image": "/abs/a.jpg"}]  # exactly one


def test_legacy_token_and_indexed_reference_coexist():
    """<image> consumes index 1 when <@image:0> already claimed index 0."""
    parser = ShareGPTParser(build_config())
    row = _row(
        [
            {"from": "human", "value": "ref <@image:0> then token <image>"},
            {"from": "gpt", "value": "ok"},
        ],
        images=["/abs/a.jpg", "/abs/b.jpg"],
    )
    content = parser.parse_sample(row, 0).messages[0]["content"]
    image_values = [b["image"] for b in content if b.get("type") == "image"]
    assert image_values == ["/abs/a.jpg", "/abs/b.jpg"]


def test_video_and_audio_references():
    parser = ShareGPTParser(build_config())
    row = _row(
        [{"from": "human", "value": "<@video:0> plus <@audio:0>"}, {"from": "gpt", "value": "ok"}],
        videos=["/abs/clip.mp4"],
        audios=["/abs/track.wav"],
    )
    content = parser.parse_sample(row, 0).messages[0]["content"]
    types = [b.get("type") for b in content]
    assert "video" in types
    assert "audio" in types


# ---------------------------------------------------------------------------
# extra_info walk (off by default, on for agent mode)

def test_extra_info_walk_off_by_default_leaves_references_unresolved():
    parser = ShareGPTParser(build_config())
    extra = {"tools_kwargs": {"sam2": {"create_kwargs": {"image": "<@image:0>"}}}}
    row = _row(
        [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/x.jpg"],
        extra_info=extra,
    )
    result = parser.parse_sample(row, 0).extra_info
    assert result["tools_kwargs"]["sam2"]["create_kwargs"]["image"] == "<@image:0>"


def test_extra_info_walk_resolves_nested_reference_string():
    cfg = build_config({"resolve_extra_info_references": True})
    parser = ShareGPTParser(cfg)
    extra = {
        "tools_kwargs": {
            "sam2": {"create_kwargs": {"image": "<@image:0>", "gt_mask": [{"counts": "x"}]}},
            "other": {"create_kwargs": {"ground_truth": "42"}},
        },
    }
    row = _row(
        [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/c.jpg"],
        extra_info=extra,
    )
    result = parser.parse_sample(row, 0).extra_info
    sam = result["tools_kwargs"]["sam2"]["create_kwargs"]
    other = result["tools_kwargs"]["other"]["create_kwargs"]
    assert sam["image"] == "/abs/c.jpg"
    assert sam["gt_mask"] == [{"counts": "x"}]  # non-reference leaves untouched
    assert other == {"ground_truth": "42"}  # no ref -> no change, no pollution


def test_extra_info_substring_reference_rewritten_in_place():
    cfg = build_config({"resolve_extra_info_references": True})
    parser = ShareGPTParser(cfg)
    extra = {"note": "see <@image:0> for context"}
    row = _row(
        [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/c.jpg"],
        extra_info=extra,
    )
    result = parser.parse_sample(row, 0).extra_info
    assert result["note"] == "see /abs/c.jpg for context"


def test_extra_info_reference_marks_pool_so_no_auto_prepend():
    """If prompt has no <image> but extra_info references image 0, the pool
    entry is claimed — _inject_remaining_media must NOT auto-prepend it."""
    cfg = build_config({"resolve_extra_info_references": True})
    parser = ShareGPTParser(cfg)
    row = _row(
        [{"from": "human", "value": "no media here"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/x.jpg"],
        extra_info={"tools_kwargs": {"sam2": {"create_kwargs": {"image": "<@image:0>"}}}},
    )
    msgs = parser.parse_sample(row, 0).messages
    image_blocks = [b for m in msgs for b in m["content"] if b.get("type") == "image"]
    assert image_blocks == []  # referenced by extra_info only, not injected


# ---------------------------------------------------------------------------
# Strict mode

def test_out_of_range_reference_raises_in_strict_mode():
    parser = ShareGPTParser(build_config())  # strict_references=True default
    row = _row(
        [{"from": "human", "value": "<@image:7>"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/a.jpg"],
    )
    with pytest.raises(ValueError, match=r"<@image:7>"):
        parser.parse_sample(row, 0)


def test_out_of_range_reference_in_extra_info_raises_with_json_path():
    cfg = build_config({"resolve_extra_info_references": True})
    parser = ShareGPTParser(cfg)
    row = _row(
        [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/a.jpg"],
        extra_info={"tools_kwargs": {"sam2": {"create_kwargs": {"image": "<@image:5>"}}}},
    )
    with pytest.raises(ValueError, match=r"tools_kwargs.sam2"):
        parser.parse_sample(row, 0)


def test_non_strict_mode_leaves_unresolvable_reference_as_text():
    cfg = build_config({"strict_references": False})
    parser = ShareGPTParser(cfg)
    row = _row(
        [{"from": "human", "value": "<@image:9>"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/a.jpg"],
    )
    content = parser.parse_sample(row, 0).messages[0]["content"]
    # Unresolvable reference falls back to being kept verbatim as text.
    assert any(b == {"type": "text", "text": "<@image:9>"} for b in content)


# ---------------------------------------------------------------------------
# Backward compat with legacy format

def test_legacy_sharegpt_sample_parses_identically():
    """Legacy <image> token sample with no references parses as before."""
    parser = ShareGPTParser(build_config())
    row = _row(
        [
            {"from": "human", "value": "describe <image>"},
            {"from": "gpt", "value": "a cat"},
        ],
        images=["/abs/cat.jpg"],
    )
    sample = parser.parse_sample(row, 0)
    assert sample.messages[0]["content"] == [
        {"type": "text", "text": "describe "},
        {"type": "image", "image": "/abs/cat.jpg"},
    ]
    assert sample.images == ["/abs/cat.jpg"]


def test_malformed_extra_info_entries_do_not_break_walk():
    cfg = build_config({"resolve_extra_info_references": True})
    parser = ShareGPTParser(cfg)
    extra = {
        "good": "<@image:0>",
        "not_a_dict": "hello",
        "nested_list": ["keep", "<@image:0>", 42, None],
        "nested_none": None,
    }
    row = _row(
        [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "ok"}],
        images=["/abs/z.jpg"],
        extra_info=extra,
    )
    result = parser.parse_sample(row, 0).extra_info
    assert result["good"] == "/abs/z.jpg"
    assert result["not_a_dict"] == "hello"
    assert result["nested_list"] == ["keep", "/abs/z.jpg", 42, None]
    assert result["nested_none"] is None
