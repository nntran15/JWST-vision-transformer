import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"


def load_evaluate_module():
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ResolveEmbeddingsPathTests(unittest.TestCase):
    def test_uses_existing_requested_file(self):
        evaluate = load_evaluate_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            embeddings_path = tmp_path / "custom" / "embeddings.h5"
            embeddings_path.parent.mkdir(parents=True)
            embeddings_path.write_bytes(b"test")

            resolved = evaluate.resolve_embeddings_path(str(embeddings_path), root_dir=tmp_path)

            self.assertEqual(resolved, embeddings_path)

    def test_falls_back_to_newest_experiment_artifact(self):
        evaluate = load_evaluate_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            older = tmp_path / "output" / "experiments" / "exp_a" / "embeddings.h5"
            newer = tmp_path / "output" / "experiments" / "exp_b" / "embeddings.h5"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.write_bytes(b"older")
            newer.write_bytes(b"newer")
            older.touch()
            newer.touch()

            resolved = evaluate.resolve_embeddings_path("output/embeddings.h5", root_dir=tmp_path)

            self.assertEqual(resolved, newer)

    def test_reports_candidates_for_missing_explicit_path(self):
        evaluate = load_evaluate_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidate = tmp_path / "output" / "experiments" / "exp_a" / "embeddings.h5"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"candidate")

            with self.assertRaises(FileNotFoundError) as exc_info:
                evaluate.resolve_embeddings_path("missing/embeddings.h5", root_dir=tmp_path)

            message = str(exc_info.exception)
            self.assertIn("Available experiment embeddings", message)
            self.assertIn("output/experiments/exp_a/embeddings.h5", message)


class ResolveArtifactPathTests(unittest.TestCase):
    def test_resolves_relative_checkpoint_from_output_dir_parent_checkpoints(self):
        evaluate = load_evaluate_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "output" / "experiments" / "run" / "eval_manual_spiral"
            checkpoint = tmp_path / "output" / "experiments" / "run" / "checkpoints" / "checkpoint_best.pt"
            checkpoint.parent.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")

            resolved = evaluate.resolve_artifact_path(
                "checkpoint_best.pt",
                output_dir=output_dir,
                root_dir=tmp_path,
                kind="checkpoint",
            )

            self.assertEqual(resolved, checkpoint)

    def test_resolves_relative_labels_csv_from_output_dir_parent(self):
        evaluate = load_evaluate_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "output" / "experiments" / "run" / "eval_manual_spiral"
            labels_csv = tmp_path / "output" / "experiments" / "run" / "spiral_vs_not_spiral.csv"
            labels_csv.parent.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            labels_csv.write_text("file_path,cluster_id,label\n")

            resolved = evaluate.resolve_artifact_path(
                "spiral_vs_not_spiral.csv",
                output_dir=output_dir,
                root_dir=tmp_path,
                kind="labels CSV",
            )

            self.assertEqual(resolved, labels_csv)

    def test_missing_artifact_path_reports_search_locations(self):
        evaluate = load_evaluate_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "output" / "experiments" / "run" / "eval_manual_spiral"
            output_dir.mkdir(parents=True)

            with self.assertRaises(FileNotFoundError) as exc_info:
                evaluate.resolve_artifact_path(
                    "missing.pt",
                    output_dir=output_dir,
                    root_dir=tmp_path,
                    kind="checkpoint",
                )

            message = str(exc_info.exception)
            self.assertIn("Could not resolve checkpoint path", message)
            self.assertIn("output/experiments/run/checkpoints/missing.pt", message)

    def test_resolves_nested_stage_checkpoint_for_mae_dino_layout(self):
        evaluate = load_evaluate_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "output" / "experiments" / "pilot_mae_dino_timm_tiny" / "eval_manual_spiral"
            nested_checkpoint = (
                tmp_path
                / "output"
                / "experiments"
                / "pilot_mae_dino_timm_tiny"
                / "checkpoints"
                / "dino_stage2"
                / "checkpoint_best.pt"
            )
            nested_checkpoint.parent.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            nested_checkpoint.write_bytes(b"checkpoint")

            resolved = evaluate.resolve_artifact_path(
                "output/experiments/pilot_mae_dino_timm_tiny/checkpoints/checkpoint_best.pt",
                output_dir=output_dir,
                root_dir=tmp_path,
                kind="checkpoint",
            )

            self.assertEqual(resolved, nested_checkpoint)


if __name__ == "__main__":
    unittest.main()