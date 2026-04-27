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


if __name__ == "__main__":
    unittest.main()