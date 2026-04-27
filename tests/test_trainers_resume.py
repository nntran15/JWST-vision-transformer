import unittest

from src.trainers.dino_trainer import DINOTrainer
from src.trainers.mae_trainer import MAETrainer


def build_config(epochs: int = 30) -> dict:
    return {
        "ssl": {"framework": "timm"},
        "training": {
            "epochs": epochs,
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-5,
            "warmup_epochs": 5,
            "weight_decay": 0.05,
            "mixed_precision": False,
            "gradient_clip_norm": 1.0,
            "seed": 42,
        },
        "logging": {
            "log_every_n_steps": 10,
            "save_every_n_epochs": 1,
            "visualize_every_n_epochs": 1,
        },
        "data": {"image_size": 64},
        "dino": {},
    }


class DummyCheckpointManager:
    def __init__(self):
        self.save_calls = []

    def save_pytorch(self, *args, **kwargs):
        self.save_calls.append((args, kwargs))

    def save_jax(self, *args, **kwargs):
        self.save_calls.append((args, kwargs))


class TrainerResumeTests(unittest.TestCase):
    def test_mae_pytorch_skips_completed_resume_without_saving(self):
        trainer = MAETrainer(build_config())
        checkpoint_manager = DummyCheckpointManager()

        result = trainer.train_pytorch(
            model=object(),
            dataloader=[],
            device=object(),
            checkpoint_manager=checkpoint_manager,
            start_epoch=30,
        )

        self.assertIsNone(result)
        self.assertEqual(checkpoint_manager.save_calls, [])

    def test_mae_jax_skips_completed_resume_without_saving(self):
        trainer = MAETrainer(build_config())
        checkpoint_manager = DummyCheckpointManager()

        result = trainer.train_jax(
            model=object(),
            data_iterator=[],
            checkpoint_manager=checkpoint_manager,
            start_epoch=30,
        )

        self.assertIsNone(result)
        self.assertEqual(checkpoint_manager.save_calls, [])

    def test_dino_pytorch_skips_completed_resume_without_saving(self):
        trainer = DINOTrainer(build_config())
        checkpoint_manager = DummyCheckpointManager()

        result = trainer.train_pytorch(
            model=object(),
            dataloader=[],
            device=object(),
            checkpoint_manager=checkpoint_manager,
            start_epoch=30,
        )

        self.assertIsNone(result)
        self.assertEqual(checkpoint_manager.save_calls, [])

    def test_dino_jax_skips_completed_resume_without_saving(self):
        trainer = DINOTrainer(build_config())
        checkpoint_manager = DummyCheckpointManager()

        result = trainer.train_jax(
            dino_wrapper=object(),
            data_iterator=[],
            checkpoint_manager=checkpoint_manager,
            start_epoch=30,
        )

        self.assertIsNone(result)
        self.assertEqual(checkpoint_manager.save_calls, [])


if __name__ == "__main__":
    unittest.main()