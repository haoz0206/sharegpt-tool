import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from xpoints.datasets.sharegpt_utils import (
    ParsedSample,
    ShareGPTDatasetConfig,
    ShareGPTMessageDataset,
    ShareGPTParser,
    build_config,
)
from xpoints.datasets.sharegpt_utils import parser as parser_module
from xpoints.datasets.sharegpt_utils.io import load_sharegpt_json


class ShareGPTUtilsTest(unittest.TestCase):
    def test_build_config_returns_defaults(self):
        cfg = build_config()
        self.assertEqual(cfg.messages_key, "conversations")
        self.assertEqual(cfg.audios_key, "audios")
        self.assertEqual(cfg.fallback_ground_truth_keys, [])
        self.assertEqual(cfg.pass_through_keys, [])

    def test_build_config_applies_overrides(self):
        cfg = build_config({
            "strict_media_token_match": True,
            "images_key": "image_paths",
            "tags": {"assistant_tag": "assistant"},
        })
        self.assertTrue(cfg.strict_media_token_match)
        self.assertEqual(cfg.images_key, "image_paths")
        self.assertEqual(cfg.tags.assistant_tag, "assistant")

    def test_sharegpt_parser_supports_image_video_audio_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sample_path = tmp_path / "sample.jsonl"
            parser = ShareGPTParser(ShareGPTDatasetConfig(strict_media_token_match=True))

            raw_sample = {
                "id": "sample-1",
                "data_source": "demo",
                "ground_truth": {"label": "done"},
                "images": ["images/a.jpg"],
                "videos": ["videos/a.mp4"],
                "audios": ["audios/a.wav"],
                "conversations": [
                    {"from": "system", "value": "sys"},
                    {"from": "human", "value": "<image><video><audio>hello"},
                    {"from": "gpt", "value": "done"},
                ],
                "__sharegpt_source_file": str(sample_path),
            }

            parsed = parser.parse_sample(raw_sample, 0)

            self.assertEqual(parsed.images, [str((tmp_path / "images/a.jpg").resolve())])
            self.assertEqual(parsed.videos, [str((tmp_path / "videos/a.mp4").resolve())])
            self.assertEqual(parsed.audios, [str((tmp_path / "audios/a.wav").resolve())])
            self.assertEqual(len(parsed.messages), 2)  # system + user, assistant stripped
            self.assertEqual(parsed.messages[1]["content"][0]["type"], "image")
            self.assertEqual(parsed.messages[1]["content"][1]["type"], "video")
            self.assertEqual(parsed.messages[1]["content"][2]["type"], "audio")
            self.assertEqual(parsed.ground_truth, {"label": "done"})

    def test_sharegpt_message_dataset_loads_jsonl_and_injects_remaining_media(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "dataset.jsonl"
            rows = [
                {
                    "id": "sample-2",
                    "data_source": "demo",
                    "ground_truth": {"label": "answer"},
                    "images": ["images/a.jpg", "images/b.jpg"],
                    "audios": ["audios/a.wav"],
                    "conversations": [
                        {"from": "human", "value": "<image>hello"},
                        {"from": "gpt", "value": "answer"},
                    ],
                }
            ]
            data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

            dataset = ShareGPTMessageDataset(str(data_path), config=ShareGPTDatasetConfig(strict_media_token_match=True))
            sample = dataset[0]

            self.assertEqual(sample.id, "sample-2")
            self.assertEqual(sample.ground_truth, {"label": "answer"})
            self.assertEqual(
                sample.images,
                [
                    str((tmp_path / "images/b.jpg").resolve()),
                    str((tmp_path / "images/a.jpg").resolve()),
                ],
            )
            self.assertEqual(sample.audios, [str((tmp_path / "audios/a.wav").resolve())])

            content = sample.messages[0]["content"]
            self.assertEqual([item["type"] for item in content], ["image", "audio", "image", "text"])
            self.assertEqual(content[-1]["text"], "hello")

    def test_sharegpt_json_loader_accepts_list_of_dicts_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            good_path = tmp_path / "dataset.json"
            bad_path = tmp_path / "bad.json"

            good_path.write_text(json.dumps([{"id": 1}, {"id": 2}], ensure_ascii=False), encoding="utf-8")
            bad_path.write_text(json.dumps([{"id": 1}, 2], ensure_ascii=False), encoding="utf-8")

            rows = load_sharegpt_json(good_path)

            self.assertEqual(rows, [{"id": 1}, {"id": 2}])
            with self.assertRaises(TypeError):
                load_sharegpt_json(bad_path)

    def test_sharegpt_parser_preserves_extra_info_and_pass_through_fields(self):
        parser = ShareGPTParser(
            ShareGPTDatasetConfig(
                pass_through_keys=["reward_model", "meta", "stage_meta", "answer_type"],
            )
        )
        raw_sample = {
            "id": "sample-keep",
            "data_source": "demo",
            "reward_model": {"ground_truth": "kept", "score": 1.0},
            "meta": {"source": "unit-test"},
            "stage_meta": {"stage_id": "s1"},
            "answer_type": "json",
            "extra_info": {"foo": "bar"},
            "conversations": [{"from": "human", "value": "hello"}],
        }

        parsed = parser.parse_sample(raw_sample, 7)

        self.assertEqual(parsed.extra_info, {"foo": "bar", "index": 7})
        self.assertEqual(parsed.pass_through["reward_model"], {"ground_truth": "kept", "score": 1.0})
        self.assertEqual(parsed.pass_through["meta"], {"source": "unit-test"})
        self.assertEqual(parsed.pass_through["stage_meta"], {"stage_id": "s1"})
        self.assertEqual(parsed.pass_through["answer_type"], "json")

    def test_sharegpt_parser_defaults_to_minimal_output_fields(self):
        parser = ShareGPTParser(ShareGPTDatasetConfig())
        raw_sample = {
            "id": "sample-minimal",
            "data_source": "demo",
            "reward_model": {"score": 1.0},
            "meta": {"source": "unit-test"},
            "stage_meta": {"stage_id": "s1"},
            "answer_type": "json",
            "ground_truth": {"label": 1},
            "conversations": [{"from": "human", "value": "hello"}],
        }

        parsed = parser.parse_sample(raw_sample, 0)

        self.assertEqual(parsed.ground_truth, {"label": 1})
        self.assertNotIn("reward_model", parsed.pass_through)
        self.assertNotIn("meta", parsed.pass_through)
        self.assertNotIn("stage_meta", parsed.pass_through)
        self.assertNotIn("answer_type", parsed.pass_through)

    def test_sharegpt_parser_warns_when_dropping_unexpected_top_level_keys(self):
        parser_module._WARNED_DROPPED_TOP_LEVEL_KEY_SETS.clear()
        parser = ShareGPTParser(ShareGPTDatasetConfig())
        raw_sample = {
            "id": "sample-drop",
            "data_source": "demo",
            "ground_truth": {"label": 1},
            "overlap": 0.5,
            "source": "matchbench",
            "conversations": [{"from": "human", "value": "hello"}],
        }

        with self.assertLogs("xpoints.datasets.sharegpt_utils.parser", level="WARNING") as logs:
            parsed = parser.parse_sample(raw_sample, 0)

        self.assertEqual(parsed.extra_info, {"index": 0})
        self.assertTrue(any("Dropping unexpected top-level keys" in log for log in logs.output))
        self.assertTrue(any("overlap" in log and "source" in log for log in logs.output))

    def test_sharegpt_parser_warns_once_per_dropped_top_level_key_set(self):
        parser_module._WARNED_DROPPED_TOP_LEVEL_KEY_SETS.clear()
        parser = ShareGPTParser(ShareGPTDatasetConfig())
        raw_sample = {
            "id": "sample-drop",
            "data_source": "demo",
            "ground_truth": {"label": 1},
            "agent_name": "description_agent",
            "conversations": [{"from": "human", "value": "hello"}],
        }

        with self.assertLogs("xpoints.datasets.sharegpt_utils.parser", level="WARNING") as logs:
            parser.parse_sample(raw_sample, 0)
        self.assertEqual(sum("Dropping unexpected top-level keys" in log for log in logs.output), 1)

        with self.assertNoLogs("xpoints.datasets.sharegpt_utils.parser", level="WARNING"):
            parser.parse_sample({**raw_sample, "id": "sample-drop-2"}, 1)

        with self.assertLogs("xpoints.datasets.sharegpt_utils.parser", level="WARNING") as logs:
            parser.parse_sample({**raw_sample, "id": "sample-drop-3", "source": "sa1b"}, 2)
        self.assertTrue(any("agent_name" in log and "source" in log for log in logs.output))

    def test_sharegpt_parser_keeps_explicit_pass_through_keys(self):
        parser = ShareGPTParser(
            ShareGPTDatasetConfig(
                pass_through_keys=["stage_meta"],
            )
        )
        raw_sample = {
            "id": "sample-pass-through",
            "data_source": "demo",
            "ground_truth": {"label": 1},
            "stage_meta": {"stage_id": "s1"},
            "conversations": [{"from": "human", "value": "hello"}],
        }

        with self.assertNoLogs("xpoints.datasets.sharegpt_utils.parser", level="WARNING"):
            parsed = parser.parse_sample(raw_sample, 0)

        self.assertEqual(parsed.pass_through["stage_meta"], {"stage_id": "s1"})

    def test_sharegpt_message_dataset_keeps_explicit_ground_truth_without_assistant(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "dataset.jsonl"
            rows = [
                {
                    "id": "sample-ground-truth",
                    "data_source": "demo",
                    "ground_truth": {"pair": [1, 2]},
                    "conversations": [{"from": "human", "value": "hello"}],
                }
            ]
            data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

            dataset = ShareGPTMessageDataset(
                str(data_path),
                config=ShareGPTDatasetConfig(),
            )
            sample = dataset[0]

            self.assertEqual(sample.ground_truth, {"pair": [1, 2]})
            self.assertEqual(sample.messages, [{"role": "user", "content": [{"type": "text", "text": "hello"}]}])

    def test_sharegpt_parser_reads_ground_truth_from_reward_model_fallback(self):
        parser = ShareGPTParser(ShareGPTDatasetConfig(fallback_ground_truth_keys=["reward_model.ground_truth"]))
        raw_sample = {
            "id": "sample-reward-model-gt",
            "data_source": "demo",
            "reward_model": {"ground_truth": {"pair": [1, 2]}},
            "conversations": [{"from": "human", "value": "hello"}],
        }

        parsed = parser.parse_sample(raw_sample, 0)

        self.assertEqual(parsed.ground_truth, {"pair": [1, 2]})

    def test_sharegpt_parser_strips_trailing_assistant_and_uses_explicit_ground_truth(self):
        parser = ShareGPTParser(ShareGPTDatasetConfig())
        raw_sample = {
            "id": "sample-both",
            "data_source": "demo",
            "ground_truth": {"label": 1},
            "conversations": [
                {"from": "human", "value": "hello"},
                {"from": "gpt", "value": "debug answer"},
            ],
        }

        parsed = parser.parse_sample(raw_sample, 0)

        self.assertEqual(parsed.ground_truth, {"label": 1})
        self.assertEqual(len(parsed.messages), 1)
        self.assertEqual(parsed.messages[0]["role"], "user")

    def test_sharegpt_parser_strips_trailing_assistant_before_dropping_all(self):
        parser = ShareGPTParser(
            ShareGPTDatasetConfig(
                drop_all_assistant_messages=True,
            )
        )
        raw_sample = {
            "id": "sample-gt",
            "data_source": "demo",
            "ground_truth": {"label": 1},
            "conversations": [
                {"from": "human", "value": "question-1"},
                {"from": "gpt", "value": "reasoning"},
                {"from": "human", "value": "question-2"},
                {"from": "gpt", "value": "final-answer"},
            ],
        }

        parsed = parser.parse_sample(raw_sample, 0)

        self.assertEqual(parsed.ground_truth, {"label": 1})
        self.assertEqual(
            parsed.messages,
            [
                {"role": "user", "content": [{"type": "text", "text": "question-1"}]},
                {"role": "user", "content": [{"type": "text", "text": "question-2"}]},
            ],
        )

    def test_sharegpt_parser_supports_structured_content_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            parser = ShareGPTParser(ShareGPTDatasetConfig())
            raw_sample = {
                "id": "sample-structured",
                "data_source": "demo",
                "__sharegpt_source_file": str(tmp_path / "sample.jsonl"),
                "conversations": [
                    {
                        "from": "human",
                        "value": [
                            {"type": "text", "text": "hello"},
                            {"type": "image_url", "image_url": {"url": "images/a.jpg"}},
                            {"type": "video", "video": "videos/a.mp4"},
                            {"type": "audio_url", "audio_url": {"url": "audios/a.wav"}},
                        ],
                    }
                ],
            }

            parsed = parser.parse_sample(raw_sample, 0)

            self.assertEqual(parsed.images, [str((tmp_path / "images/a.jpg").resolve())])
            self.assertEqual(parsed.videos, [str((tmp_path / "videos/a.mp4").resolve())])
            self.assertEqual(parsed.audios, [str((tmp_path / "audios/a.wav").resolve())])
            self.assertEqual([item["type"] for item in parsed.messages[0]["content"]], ["text", "image", "video", "audio"])

    def test_sharegpt_parser_warns_and_falls_back_when_data_source_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            parser = ShareGPTParser(ShareGPTDatasetConfig())
            raw_sample = {
                "id": "sample-3",
                "conversations": [{"from": "human", "value": "hello"}],
                "__sharegpt_source_file": str(tmp_path / "sample.jsonl"),
            }

            with self.assertLogs("xpoints.datasets.sharegpt_utils.parser", level="WARNING") as logs:
                parsed = parser.parse_sample(raw_sample, 0)

            self.assertEqual(parsed.data_source, "sharegpt")
            self.assertTrue(any("falling back to 'sharegpt'" in log for log in logs.output))

    def test_sharegpt_parser_uses_configured_default_data_source(self):
        parser = ShareGPTParser(ShareGPTDatasetConfig(default_data_source="metric"))
        raw_sample = {
            "id": "sample-default-source",
            "conversations": [{"from": "human", "value": "hello"}],
        }

        with self.assertLogs("xpoints.datasets.sharegpt_utils.parser", level="WARNING") as logs:
            parsed = parser.parse_sample(raw_sample, 0)

        self.assertEqual(parsed.data_source, "metric")
        self.assertTrue(any("falling back to 'metric'" in log for log in logs.output))


    def test_parsed_sample_is_dataclass(self):
        sample = ParsedSample(
            id="test-1",
            messages=[],
            images=[],
            videos=[],
            audios=[],
            ground_truth=None,
            data_source="demo",
            extra_info={},
            pass_through={},
        )
        self.assertTrue(dataclasses.is_dataclass(sample))
        self.assertEqual(sample.id, "test-1")
        self.assertEqual(sample.data_source, "demo")
        self.assertEqual(sample.pass_through, {})

    def test_parsed_sample_asdict(self):
        sample = ParsedSample(
            id="test-1",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            images=[],
            videos=[],
            audios=[],
            ground_truth={"label": 1},
            data_source="demo",
            extra_info={"index": 0},
            pass_through={},
        )
        d = dataclasses.asdict(sample)
        self.assertEqual(d["id"], "test-1")
        self.assertEqual(d["ground_truth"], {"label": 1})
        self.assertIsInstance(d, dict)


if __name__ == "__main__":
    unittest.main()
