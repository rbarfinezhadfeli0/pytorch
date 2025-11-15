# Documentation: `docs/test/test_functional_optim.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_functional_optim.py_docs.md`
- **Size**: 9,667 bytes (9.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_functional_optim.py`

## File Metadata

- **Path**: `test/test_functional_optim.py`
- **Size**: 6,358 bytes (6.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import unittest
from typing import Optional

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, AdamW, SGD
from torch.testing._internal.common_utils import run_tests, TestCase


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(3, 3, bias=False)
        self.lin2 = nn.Linear(3, 3, bias=False)

    def forward(self, t1):
        return self.lin2(F.relu(self.lin1(t1)))


# dummy class to showcase custom optimizer registration with functional wrapper
class MyDummyFnOptimizer:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        _allow_empty_param_list: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 < weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        # call the custom optimizer step_param implementation
        with torch.no_grad():
            raise RuntimeError(
                "MyDummyFnOptimizer does not support step_param() as of now"
            )

    def step(self, gradients: list[Optional[Tensor]]):
        # call the custom optimizer step implementation
        with torch.no_grad():
            raise RuntimeError("MyDummyFnOptimizer does not support step() as of now")


if torch.distributed.is_available():
    from torch.distributed.optim.utils import (
        functional_optim_map,
        register_functional_optim,
    )


@unittest.skipIf(
    not torch.distributed.is_available(), "These are testing distributed functions"
)
class TestFunctionalOptimParity(TestCase):
    def _validate_parameters(self, params_1, params_2):
        for p1, p2 in zip(params_1, params_2):
            self.assertEqual(p1, p2)

    # Dynamo fails at compiling this for python 3.8/3.11
    # Since it passes while compiling the actual code under test
    # we disable dynamo here.
    @torch._disable_dynamo(recursive=False)
    def _test_functional_optim_parity(self, optim_cls, *args, **kwargs):
        module_optim = MyModule()
        module_functional = MyModule()
        optim_params = module_optim.parameters()
        optim = optim_cls(optim_params, *args, **kwargs)
        functional_optim_cls = functional_optim_map.get(optim_cls, None)
        if not functional_optim_cls:
            raise ValueError(f"Functional optimizer not implemented for {optim_cls}")
        optim_functional = functional_optim_cls(
            [], *args, **kwargs, _allow_empty_param_list=True
        )
        if not hasattr(optim_functional, "step_param"):
            raise ValueError(
                f"Functional optimizer class {optim_functional} must implement step_param method."
            )

        # Initial weights should match
        self._validate_parameters(
            module_optim.parameters(), module_functional.parameters()
        )
        # Save old parameters to verify optimizer modifies them.
        old_module_optim_params = [
            param.detach().clone() for param in module_optim.parameters()
        ]
        old_module_functional_params = [
            param.detach().clone() for param in module_functional.parameters()
        ]

        t1 = torch.randn(3, 3)
        for _ in range(10):
            module_optim.zero_grad()
            module_functional.zero_grad()
            # Forward + Backward
            optim_out = module_optim(t1).sum()
            functional_out = module_functional(t1).sum()
            optim_out.backward()
            functional_out.backward()
            # Optimizer step
            optim.step()
            # Functional optimizer step_param
            for param in module_functional.parameters():
                grad = param.grad
                optim_functional.step_param(param, grad)

            # Validate parameters are equal
            for optim_param, functional_param in zip(
                module_optim.parameters(), module_functional.parameters()
            ):
                self.assertEqual(optim_param, functional_param)
            # Validate parameters are modified.
            for i, (optim_param, functional_param) in enumerate(
                zip(module_optim.parameters(), module_functional.parameters())
            ):
                self.assertNotEqual(old_module_optim_params[i], optim_param)
                self.assertNotEqual(old_module_functional_params[i], functional_param)

    def _test_functional_optim_registration(self):
        fn_map_key = "MyDummyFnOptimizer"
        fn_optim = MyDummyFnOptimizer
        register_functional_optim(fn_map_key, fn_optim)
        functional_optim_cls = functional_optim_map.get(fn_map_key, None)
        if not functional_optim_cls:
            raise ValueError(f"Functional optimizer not registered for {fn_map_key}")

    def test_functional_optim_registration(self):
        self._test_functional_optim_registration()

    def test_functional_optim_parity_sgd(self):
        self._test_functional_optim_parity(SGD, 1e-2, momentum=0.9, weight_decay=0.01)

    def test_functional_optim_parity_adam(self):
        self._test_functional_optim_parity(Adam, 1e-2, betas=(0.9, 0.999), eps=1e-6)

    def test_functional_optim_parity_adam_w(self):
        self._test_functional_optim_parity(AdamW, 1e-2, betas=(0.9, 0.999), eps=1e-6)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 4 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyModule`, `MyDummyFnOptimizer`, `TestFunctionalOptimParity`

**Functions defined**: `__init__`, `forward`, `__init__`, `step_param`, `step`, `_validate_parameters`, `_test_functional_optim_parity`, `_test_functional_optim_registration`, `test_functional_optim_registration`, `test_functional_optim_parity_sgd`, `test_functional_optim_parity_adam`, `test_functional_optim_parity_adam_w`

**Key imports**: unittest, Optional, torch, torch.distributed, torch.nn as nn, torch.nn.functional as F, Tensor, Adam, AdamW, SGD, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `typing`: Optional
- `torch`
- `torch.distributed`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.optim`: Adam, AdamW, SGD
- `torch.testing._internal.common_utils`: run_tests, TestCase


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
python test/test_functional_optim.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_functional_optim.py_docs.md`
- **Keyword Index**: `test_functional_optim.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_functional_optim.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_functional_optim.py_docs.md_docs.md`
- **Keyword Index**: `test_functional_optim.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
