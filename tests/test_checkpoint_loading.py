import unittest

import torch
import torch.nn as nn

from src.utils.checkpointing import load_downstream_encoder_weights


class DummyMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.decoder = nn.Linear(3, 4)

    def get_encoder(self):
        return self.encoder


class DummyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_backbone = nn.Linear(4, 3)
        self.student_head = nn.Linear(3, 2)

    def get_encoder(self):
        return self.student_backbone


class DownstreamCheckpointLoadingTests(unittest.TestCase):
    def test_loads_mae_encoder_prefix_from_model_state_dict(self):
        source = DummyMAE()
        target = DummyMAE()
        checkpoint = {
            "model_state_dict": {
                "encoder.weight": source.encoder.weight.detach().clone(),
                "encoder.bias": source.encoder.bias.detach().clone(),
                "decoder.weight": source.decoder.weight.detach().clone(),
                "decoder.bias": source.decoder.bias.detach().clone(),
            }
        }

        info = load_downstream_encoder_weights(target, checkpoint)

        self.assertEqual(info["prefix"], "encoder.")
        self.assertTrue(torch.equal(target.encoder.weight, source.encoder.weight))
        self.assertTrue(torch.equal(target.encoder.bias, source.encoder.bias))

    def test_loads_dino_student_backbone_prefix(self):
        source = DummyDINO()
        target = DummyDINO()
        checkpoint = {
            "model_state_dict": {
                "student_backbone.weight": source.student_backbone.weight.detach().clone(),
                "student_backbone.bias": source.student_backbone.bias.detach().clone(),
                "student_head.weight": source.student_head.weight.detach().clone(),
                "student_head.bias": source.student_head.bias.detach().clone(),
            }
        }

        info = load_downstream_encoder_weights(target, checkpoint)

        self.assertEqual(info["prefix"], "student_backbone.")
        self.assertTrue(torch.equal(target.student_backbone.weight, source.student_backbone.weight))
        self.assertTrue(torch.equal(target.student_backbone.bias, source.student_backbone.bias))

    def test_raises_when_checkpoint_has_no_matching_encoder_weights(self):
        target = DummyMAE()

        with self.assertRaises(ValueError) as exc_info:
            load_downstream_encoder_weights(
                target,
                {"model_state_dict": {"decoder.weight": torch.randn(2, 2)}},
            )

        self.assertIn("did not contain encoder weights", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()