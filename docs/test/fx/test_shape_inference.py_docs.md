# Documentation: test_shape_inference.py

## File Metadata
- **Path**: `test/fx/test_shape_inference.py`
- **Size**: 4171 bytes
- **Lines**: 117
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: fx"]

import copy
import unittest
from collections import defaultdict

import torch
import torch.fx as fx
from torch._dynamo.source import LocalSource
from torch.fx.experimental.shape_inference.infer_shape import infer_shape
from torch.fx.experimental.shape_inference.infer_symbol_values import (
    infer_symbol_values,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv


class TestShapeInference(unittest.TestCase):
    def test_infer_symbol_values(self):
        def mksym(shape_env, value, source, dynamic_dim) -> None:
            return shape_env.create_symintnode(
                shape_env.create_symbol(
                    value,
                    source=source,
                    dynamic_dim=dynamic_dim,
                ),
                hint=value,
                source=source,
            )

        shape_env = ShapeEnv()
        N = 8
        sample = {f"s{i}": 2 for i in range(N)}
        init_symints = [
            mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
            for k, v in sample.items()
        ]
        symints = copy.deepcopy(init_symints)
        symbol_to_idx_dict = {f"s{i}": i for i in range(N)}
        padding_constraints = defaultdict(list)

        # prepare constraints strings
        constraints = []
        constraints.append(
            "The size of tensor a (s1) must match the size of tensor b (1773) at non-singleton dimension 1)"
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, (s2//2) + 12] but got: [s0, 120]."
        )
        constraints.append("shape '[s0, -1, 32]' is invalid for input of size s0*s3")
        constraints.append(
            "a and b must have same reduction dim, but got [32*s0, s3] X [20, 15]."
        )
        constraints.append(
            "a and b must have same reduction dim, but got [s0, s4 + 1568] X [5728, 1024]."
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, 40] but got: [s0, s5]."
        )
        constraints.append(
            "shape '[s0, -1, 32]' is invalid for input of size s0*s6 + 1344*s0"
        )
        constraints.append(
            "shape '[-1, 47]' is invalid for input of size 32*s0*s6 + 1344*s0"
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, 47*s6] but got: [s0*s6, 47]."
        )
        constraints.append("Split sizes add up to 4258 but got the tensor's size of s7")

        for constraint in constraints:
            infer_symbol_values(
                symints,
                init_symints,
                symbol_to_idx_dict,
                padding_constraints,
                constraint,
            )

        self.assertEqual(symints[1], 1773)
        self.assertEqual(symints[2], 216)
        self.assertEqual(symints[3], 640)
        self.assertEqual(symints[4], 4160)
        self.assertEqual(symints[5], 40)
        self.assertEqual(symints[6], 160)
        self.assertEqual(symints[7], 4258)

    def test_infer_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w_1 = torch.empty([256, 328])
                self.b_1 = torch.empty([256])
                self.w_2 = torch.empty([328, 256])
                self.b_2 = torch.empty([328])

            def forward(self, x):
                l_1 = torch.nn.functional.linear(x, self.w_1, bias=self.b_1)
                s_1 = torch.sigmoid(l_1)
                l_2 = torch.nn.functional.linear(s_1, self.w_2, bias=self.b_2)
                t_1 = torch.tanh(l_2)
                return t_1

        def generate_graph_module(model):
            gm = fx.symbolic_trace(model)
            return gm

        m = TestModule()
        gm = generate_graph_module(m)
        input_tensors = [torch.randn(1, 1)]
        infer_shape(gm, input_tensors)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 2 class(es): TestShapeInference, TestModule

### Functions
This file defines 6 function(s): test_infer_symbol_values, mksym, test_infer_shape, __init__, forward, generate_graph_module


## Key Components

The file contains 384 words across 117 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4171 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
