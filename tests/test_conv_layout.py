"""CPU contract tests; NPU operator and graph checks are separate gates."""
import unittest
from types import SimpleNamespace

import torch

from drror_vllm_ascend.patches.conv_layout import prepare_conv_weight_layout


class ConvLayoutTests(unittest.TestCase):
    def test_values_identity_metadata_reload_and_idempotence(self):
        for channels in (1792, 2048, 2304, 2432, 2560):
            with self.subTest(channels=channels):
                original = torch.arange(channels * 4).reshape(channels, 1, 4).to(torch.bfloat16)
                parameter = torch.nn.Parameter(original.clone(), requires_grad=False)
                parameter.weight_loader = object()
                metadata = parameter.weight_loader
                module = SimpleNamespace(weight=parameter)
                result = prepare_conv_weight_layout(module)
                self.assertTrue(result["changed"])
                self.assertIs(module.weight, parameter)
                self.assertIs(module.weight.weight_loader, metadata)
                self.assertTrue(torch.equal(module.weight, original))
                self.assertTrue(module.weight.view(channels, 4).T.is_contiguous())
                pointer = module.weight.data_ptr()
                self.assertFalse(prepare_conv_weight_layout(module)["changed"])
                self.assertEqual(module.weight.data_ptr(), pointer)
                # An in-place reload respects the new strides and keeps values.
                replacement = torch.randn_like(original)
                module.weight.copy_(replacement)
                self.assertTrue(torch.equal(module.weight, replacement))
                self.assertFalse(prepare_conv_weight_layout(module)["changed"])

    def test_reject_training_and_unsupported_shape_dtype(self):
        for tensor, training in (
            (torch.zeros(2048, 1, 4, dtype=torch.bfloat16), True),
            (torch.zeros(2048, 1, 3, dtype=torch.bfloat16), False),
            (torch.zeros(2176, 1, 4, dtype=torch.bfloat16), False),
            (torch.zeros(2048, 1, 4), False),
        ):
            with self.subTest(shape=tensor.shape, training=training, dtype=tensor.dtype):
                parameter = torch.nn.Parameter(tensor, requires_grad=training)
                module = SimpleNamespace(weight=parameter)
                with self.assertRaises(ValueError):
                    prepare_conv_weight_layout(module)
                self.assertIs(module.weight, parameter)


if __name__ == "__main__":
    unittest.main()
