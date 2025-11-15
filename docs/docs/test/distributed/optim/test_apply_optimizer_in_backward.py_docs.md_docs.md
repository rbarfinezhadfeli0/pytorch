# Documentation: `docs/test/distributed/optim/test_apply_optimizer_in_backward.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/optim/test_apply_optimizer_in_backward.py_docs.md`
- **Size**: 8,305 bytes (8.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/optim/test_apply_optimizer_in_backward.py`

## File Metadata

- **Path**: `test/distributed/optim/test_apply_optimizer_in_backward.py`
- **Size**: 5,662 bytes (5.53 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed.optim import (
    _apply_optimizer_in_backward,
    _get_in_backward_optimizers,
)


# TODO (rohan-varma): Add FSDP & DDP tests once supported


def _validate_params(params_list, fn):
    ref_params = params_list[0]
    for param_list in params_list[1:]:
        for p1, p2 in zip(ref_params, param_list):
            fn(p1, p2)


class ApplyOverlappedOptimizerTest(unittest.TestCase):
    def _run_training_loop_and_validate(self, inp, models, optimizers):
        for i in range(6):
            for model in models:
                model(inp).sum().backward()
            for opt in optimizers:
                opt.step()

            with self.subTest(i):
                _validate_params(
                    [model.parameters() for model in models],
                    torch.testing.assert_close,
                )

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

    def _test_apply_optimizer_in_backward(self, share_params) -> None:
        weight_optimizer_kwargs = {"lr": 1.0}
        bias_optimizer_kwargs = {"lr": 0.5}
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        if share_params:
            model[0].weight = model[1].weight

        # Use different optimizers for weights & biases.
        weights = [m.weight for m in model]
        biases = [m.bias for m in model]
        optim_weight = torch.optim.SGD(weights, **weight_optimizer_kwargs)
        optim_bias = torch.optim.SGD(biases, **bias_optimizer_kwargs)
        model_with_opt_in_bwd = deepcopy(model)

        # Apply different optimizer in backwards for weights and biases.
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            [m.weight for m in model_with_opt_in_bwd],
            optimizer_kwargs=weight_optimizer_kwargs,
        )

        _apply_optimizer_in_backward(
            torch.optim.SGD,
            [m.bias for m in model_with_opt_in_bwd],
            optimizer_kwargs=bias_optimizer_kwargs,
        )

        _validate_params(
            [
                model.parameters(),
                model_with_opt_in_bwd.parameters(),
            ],
            torch.testing.assert_close,
        )

        self._run_training_loop_and_validate(
            torch.randn(4, 10),
            [model, model_with_opt_in_bwd],
            [optim_weight, optim_bias],
        )

    def test_apply_optimizer_in_backward(self) -> None:
        self._test_apply_optimizer_in_backward(share_params=False)

    def test_apply_optimizer_in_backward_shared_params(self) -> None:
        self._test_apply_optimizer_in_backward(share_params=True)

    def test_no_register_hook(self):
        model_with_hook = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        initial_model = deepcopy(model_with_hook)
        model_no_hook = deepcopy(model_with_hook)
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_hook.parameters(),
            optimizer_kwargs={"lr": 0.03},
        )
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_no_hook.parameters(),
            optimizer_kwargs={"lr": 0.03},
            register_hook=False,
        )
        inp = torch.randn(4, 10)
        model_with_hook(inp).sum().backward()
        model_no_hook(inp).sum().backward()

        for p1, p2 in zip(model_with_hook.parameters(), initial_model.parameters()):
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(p1, p2)

        for p1, p2 in zip(model_no_hook.parameters(), initial_model.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_multiple_optim_for_params(self) -> None:
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        opt_0_kwargs = {"lr": 0.03}
        opt_1_kwargs = {"lr": 0.01}
        opt_0 = torch.optim.SGD(model.parameters(), **opt_0_kwargs)
        opt_1 = torch.optim.SGD(model.parameters(), **opt_1_kwargs)
        model_with_opt_in_bwd = deepcopy(model)
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_opt_in_bwd.parameters(),
            optimizer_kwargs=opt_0_kwargs,
        )
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_opt_in_bwd.parameters(),
            optimizer_kwargs=opt_1_kwargs,
        )
        self._run_training_loop_and_validate(
            torch.randn(4, 10),
            [model, model_with_opt_in_bwd],
            [opt_0, opt_1],
        )

    def test_get_optimizers_in_backward(self):
        # Create a simple test model
        class TestModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(5, 2)

        model = TestModel()

        # Apply optimizers in backward
        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {"lr": 0.01})
        in_backward_optims = _get_in_backward_optimizers(model)
        self.assertEqual(len(list(model.parameters())), len(in_backward_optims))
        result = set(in_backward_optims)
        expected = {
            optim for p in model.parameters() for optim in p._in_backward_optimizers
        }
        self.assertEqual(result, expected)

```



## High-Level Overview


This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ApplyOverlappedOptimizerTest`, `TestModel`

**Functions defined**: `_validate_params`, `_run_training_loop_and_validate`, `_test_apply_optimizer_in_backward`, `test_apply_optimizer_in_backward`, `test_apply_optimizer_in_backward_shared_params`, `test_no_register_hook`, `test_multiple_optim_for_params`, `test_get_optimizers_in_backward`, `__init__`

**Key imports**: unittest, deepcopy, torch, torch.nn as nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/optim`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `copy`: deepcopy
- `torch`
- `torch.nn as nn`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/optim/test_apply_optimizer_in_backward.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/optim`):

- [`test_zero_redundancy_optimizer.py_docs.md`](./test_zero_redundancy_optimizer.py_docs.md)
- [`test_named_optimizer.py_docs.md`](./test_named_optimizer.py_docs.md)


## Cross-References

- **File Documentation**: `test_apply_optimizer_in_backward.py_docs.md`
- **Keyword Index**: `test_apply_optimizer_in_backward.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/optim`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/optim/test_apply_optimizer_in_backward.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/optim`):

- [`test_named_optimizer.py_kw.md_docs.md`](./test_named_optimizer.py_kw.md_docs.md)
- [`test_apply_optimizer_in_backward.py_kw.md_docs.md`](./test_apply_optimizer_in_backward.py_kw.md_docs.md)
- [`test_named_optimizer.py_docs.md_docs.md`](./test_named_optimizer.py_docs.md_docs.md)
- [`test_zero_redundancy_optimizer.py_kw.md_docs.md`](./test_zero_redundancy_optimizer.py_kw.md_docs.md)
- [`test_zero_redundancy_optimizer.py_docs.md_docs.md`](./test_zero_redundancy_optimizer.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_apply_optimizer_in_backward.py_docs.md_docs.md`
- **Keyword Index**: `test_apply_optimizer_in_backward.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
