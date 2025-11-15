# Documentation: test_parametrization.py

## File Metadata
- **Path**: `test/ao/sparsity/test_parametrization.py`
- **Size**: 6630 bytes
- **Lines**: 172
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: sparse"]


import torch
from torch import nn
from torch.ao.pruning.sparsifier import utils
from torch.nn.utils import parametrize
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class ModelUnderTest(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.linear = nn.Linear(16, 16, bias=bias)
        self.seq = nn.Sequential(
            nn.Linear(16, 16, bias=bias), nn.Linear(16, 16, bias=bias)
        )

        # Make sure the weights are not random
        self.linear.weight = nn.Parameter(torch.zeros_like(self.linear.weight) + 1.0)
        self.seq[0].weight = nn.Parameter(torch.zeros_like(self.seq[0].weight) + 2.0)
        self.seq[1].weight = nn.Parameter(torch.zeros_like(self.seq[1].weight) + 3.0)
        if bias:
            self.linear = nn.Parameter(torch.zeros_like(self.linear.bias) + 10.0)
            self.seq[0] = nn.Parameter(torch.zeros_like(self.seq[0].bias) + 20.0)
            self.seq[0] = nn.Parameter(torch.zeros_like(self.seq[0].bias) + 30.0)

    def forward(self, x):
        x = self.linear(x)
        x = self.seq(x)
        return x


class TestFakeSparsity(TestCase):
    def test_masking_logic(self):
        model = nn.Linear(16, 16, bias=False)
        model.weight = nn.Parameter(torch.eye(16))
        x = torch.randn(3, 16)
        self.assertEqual(torch.mm(x, torch.eye(16)), model(x))

        mask = torch.zeros(16, 16)
        sparsity = utils.FakeSparsity(mask)
        parametrize.register_parametrization(model, "weight", sparsity)

        x = torch.randn(3, 16)
        self.assertEqual(torch.zeros(3, 16), model(x))

    def test_weights_parametrized(self):
        model = ModelUnderTest(bias=False)

        assert not hasattr(model.linear, "parametrizations")
        assert not hasattr(model.seq[0], "parametrizations")
        assert not hasattr(model.seq[1], "parametrizations")
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.linear, "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[0], "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[1], "weight", utils.FakeSparsity(mask)
        )

        assert hasattr(model.linear, "parametrizations")
        assert parametrize.is_parametrized(model.linear, "weight")
        assert hasattr(model.seq[0], "parametrizations")
        assert parametrize.is_parametrized(model.linear, "weight")
        assert hasattr(model.seq[1], "parametrizations")
        assert parametrize.is_parametrized(model.linear, "weight")

    def test_state_dict_preserved(self):
        model_save = ModelUnderTest(bias=False)

        mask = torch.eye(16)
        parametrize.register_parametrization(
            model_save.linear, "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model_save.seq[0], "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model_save.seq[1], "weight", utils.FakeSparsity(mask)
        )
        state_dict = model_save.state_dict()

        model_load = ModelUnderTest(bias=False)
        mask = torch.zeros(model_load.linear.weight.shape)
        parametrize.register_parametrization(
            model_load.linear, "weight", utils.FakeSparsity(mask)
        )
        mask = torch.zeros(model_load.seq[0].weight.shape)
        parametrize.register_parametrization(
            model_load.seq[0], "weight", utils.FakeSparsity(mask)
        )
        mask = torch.zeros(model_load.seq[1].weight.shape)
        parametrize.register_parametrization(
            model_load.seq[1], "weight", utils.FakeSparsity(mask)
        )
        # Keep this strict, as we are not loading the 'mask'
        model_load.load_state_dict(state_dict, strict=False)

        # Check the parametrizations are preserved
        assert hasattr(model_load.linear, "parametrizations")
        assert parametrize.is_parametrized(model_load.linear, "weight")
        assert hasattr(model_load.seq[0], "parametrizations")
        assert parametrize.is_parametrized(model_load.linear, "weight")
        assert hasattr(model_load.seq[1], "parametrizations")
        assert parametrize.is_parametrized(model_load.linear, "weight")

        # Check the weights are preserved
        self.assertEqual(
            model_save.linear.parametrizations["weight"].original,
            model_load.linear.parametrizations["weight"].original,
        )
        self.assertEqual(
            model_save.seq[0].parametrizations["weight"].original,
            model_load.seq[0].parametrizations["weight"].original,
        )
        self.assertEqual(
            model_save.seq[1].parametrizations["weight"].original,
            model_load.seq[1].parametrizations["weight"].original,
        )

        # Check the masks are not preserved in the state_dict
        # We store the state_dicts in the sparsifier, not in the model itself.
        # TODO: Need to find a clean way of exporting the parametrized model
        self.assertNotEqual(
            model_save.linear.parametrizations["weight"][0].mask,
            model_load.linear.parametrizations["weight"][0].mask,
        )
        self.assertNotEqual(
            model_save.seq[0].parametrizations["weight"][0].mask,
            model_load.seq[0].parametrizations["weight"][0].mask,
        )
        self.assertNotEqual(
            model_save.seq[1].parametrizations["weight"][0].mask,
            model_load.seq[1].parametrizations["weight"][0].mask,
        )

    def test_jit_trace(self):
        model = ModelUnderTest(bias=False)

        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.linear, "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[0], "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[1], "weight", utils.FakeSparsity(mask)
        )

        # Tracing
        example_x = torch.ones(3, 16)
        model_trace = torch.jit.trace_module(model, {"forward": example_x})

        x = torch.randn(3, 16)
        y = model(x)
        y_hat = model_trace(x)
        self.assertEqual(y_hat, y)


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 2 class(es): ModelUnderTest, TestFakeSparsity

### Functions
This file defines 6 function(s): __init__, forward, test_masking_logic, test_weights_parametrized, test_state_dict_preserved, test_jit_trace


## Key Components

The file contains 409 words across 172 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 6630 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
