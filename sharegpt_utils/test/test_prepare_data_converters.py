import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareDataConvertersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.match_converter = _load_module("match_sharegpt_converter", "prepare_data/matchbench/sharegpt_converter.py")
        cls.metric_converter = _load_module("metric_sharegpt_converter", "prepare_data/metricbench/sharegpt_converter.py")

    def test_matchbench_converter_moves_task_metadata_into_extra_info(self):
        row = {
            "id": "match-1",
            "source": "lmdb_scannet",
            "db_idx": 12,
            "image1_path": "scannet/00000012_A.jpg",
            "image2_path": "scannet/00000012_B.jpg",
            "answer": {"matches": [[1, 7], [2, 3]]},
            "overlap": 0.61,
            "stage_meta": {"stage_id": "basic", "num_core_pairs": 5},
        }

        record = self.match_converter.to_sharegpt_record(
            row=row,
            row_idx=0,
            include_answer=False,
            system_prompt="sys",
            data_source="match",
        )

        self.assertEqual(record["ground_truth"], {"matches": [[1, 7], [2, 3]]})
        self.assertEqual(
            record["extra_info"],
            {
                "source": "lmdb_scannet",
                "db_idx": 12,
                "overlap": 0.61,
                "stage_meta": {"stage_id": "basic", "num_core_pairs": 5},
            },
        )
        self.assertNotIn("source", record)
        self.assertNotIn("db_idx", record)
        self.assertNotIn("overlap", record)
        self.assertNotIn("stage_meta", record)

    def test_metricbench_converter_keeps_question_type_and_extra_columns_in_extra_info(self):
        row = {
            "id": "metric-1",
            "image_path": "images/00000001.jpg",
            "Generated_Sentence": "Estimate the chair height.",
            "Type_in_Answer": "height",
            "Value_in_Answer": "1.2",
            "scene_id": "scene-7",
            "camera_id": "cam-2",
        }

        record = self.metric_converter.to_sharegpt_record(
            row=row,
            row_idx=0,
            include_answer=False,
            data_source="metric",
        )

        self.assertEqual(record["ground_truth"], {"height": 1.2})
        self.assertEqual(
            record["extra_info"],
            {
                "question_type": "height",
                "scene_id": "scene-7",
                "camera_id": "cam-2",
            },
        )
        self.assertNotIn("scene_id", record)
        self.assertNotIn("camera_id", record)

    def test_matchbench_converter_routes_question_type_into_extra_info(self):
        row = {
            "id": "match-ground-1",
            "source": "lmdb_scannet",
            "db_idx": 5,
            "image1_path": "scannet/00000005_A.jpg",
            "image2_path": "scannet/00000005_B.jpg",
            "answer": [{"label": "1", "point_2d": [345, 612]}],
            "question_type": "grounding",
            "overlap": 0.45,
            "stage_meta": {"stage_id": "ground_basic", "num_core_pairs": 3},
        }

        record = self.match_converter.to_sharegpt_record(
            row=row,
            row_idx=0,
            include_answer=False,
            system_prompt="sys",
            data_source="match",
        )

        self.assertEqual(record["ground_truth"], [{"label": "1", "point_2d": [345, 612]}])
        self.assertEqual(record["extra_info"]["question_type"], "grounding")
        self.assertEqual(record["extra_info"]["source"], "lmdb_scannet")

    def test_matchbench_converter_selects_grounding_template_by_question_type(self):
        row = {
            "id": "match-ground-2",
            "source": "lmdb_scannet",
            "db_idx": 6,
            "image1_path": "scannet/00000006_A.jpg",
            "image2_path": "scannet/00000006_B.jpg",
            "answer": [{"label": "1", "point_2d": [100, 200]}],
            "question_type": "grounding",
        }

        record = self.match_converter.to_sharegpt_record(
            row=row,
            row_idx=0,
            include_answer=False,
            system_prompt="sys",
            data_source="match",
        )

        user_msg = record["conversations"][1]["value"]
        self.assertIn("unannotated", user_msg)
        self.assertIn("0-1000", user_msg)

    def test_metricbench_converter_uses_answer_tag(self):
        row = {
            "id": "metric-2",
            "image_path": "images/00000002.jpg",
            "Generated_Sentence": "Estimate the chair height.",
            "Type_in_Answer": "height",
            "Value_in_Answer": "1.2",
        }

        record = self.metric_converter.to_sharegpt_record(
            row=row,
            row_idx=0,
            include_answer=True,
            data_source="metric",
        )

        assistant_turn = record["conversations"][-1]
        self.assertEqual(assistant_turn["from"], "gpt")
        self.assertTrue(assistant_turn["value"].startswith("<answer>"))
        self.assertIn("</answer>", assistant_turn["value"])


if __name__ == "__main__":
    unittest.main()
