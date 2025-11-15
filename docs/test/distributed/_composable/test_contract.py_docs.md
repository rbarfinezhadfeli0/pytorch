# Documentation: `test/distributed/_composable/test_contract.py`

## File Metadata

- **Path**: `test/distributed/_composable/test_contract.py`
- **Size**: 6,454 bytes (6.30 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed._composable import _get_registry, contract
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.p = nn.Parameter(torch.randn(10, 10), requires_grad=True)
        self.b = torch.zeros(1)  # buffer

    def forward(self, x, y):
        with torch.no_grad():
            self.b += x.sum() + y.sum()

        return self.p + self.seq1(x) + self.seq2(y)


class TestContract(TestCase):
    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_add_hooks(self):
        def forward_pre_hook(
            module: nn.Module, inp: tuple[torch.Tensor]
        ) -> tuple[torch.Tensor]:
            return inp

        def forward_hook(
            module: nn.Module, inp: tuple[torch.Tensor], out: torch.Tensor
        ) -> torch.Tensor:
            return out

        def backward_pre_hook(
            module: nn.Module, grad_output: torch.Tensor
        ) -> torch.Tensor:
            return grad_output

        def backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor],
            grad_output: torch.Tensor,
        ) -> tuple[torch.Tensor]:
            return grad_input

        @contract()
        def noop_api(module: nn.Module) -> nn.Module:
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_hook)
            module.register_full_backward_pre_hook(backward_pre_hook)
            module.register_full_backward_hook(backward_hook)
            return module

        model = ToyModel()
        model_with_hooks = deepcopy(model)
        noop_api(model.seq1)
        noop_api(model.seq2)

        x, y = torch.randn(10, 10), torch.randn(10, 10)
        model(x, y).sum().backward()
        model_with_hooks(x, y).sum().backward()

        for p1, p2 in zip(model.parameters(), model_with_hooks.parameters()):
            self.assertEqual(p1, p2)

    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_modify_fqn(self):
        class ModelWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x):
                return self.module(x)

        @contract()
        def wrap_module(module: nn.Module) -> nn.Module:
            return ModelWrapper(module)

        model = ToyModel()

        regex = "Checking parameters: Composable distributed API implementations cannot modify FQNs."
        with self.assertRaisesRegex(RuntimeError, regex):
            wrap_module(model.seq1)

    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_state(self):
        def check_and_update_state_hook(
            module: nn.Module, inp: tuple[torch.Tensor]
        ) -> tuple[torch.Tensor]:
            self.assertEqual(api.state(module).dummy_state, 7)
            api.state(module).dummy_state = 8
            return inp

        # FIXME: circular reference looks a bit weird. Shall we make .state a
        # top-level API instead attached to contract API?
        @contract()
        def api(module: nn.Module) -> nn.Module:
            api.state(module).dummy_state = 7
            module.register_forward_pre_hook(check_and_update_state_hook)
            return module

        model = ToyModel()
        api(model.seq1)

        self.assertEqual(api.state(model.seq1).dummy_state, 7)
        model(torch.zeros(10, 10), torch.zeros(10, 10))
        self.assertEqual(api.state(model.seq1).dummy_state, 8)

    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_registry(self):
        @contract()
        def api1(module: nn.Module) -> nn.Module:
            return module

        @contract()
        def api2(module: nn.Module) -> nn.Module:
            return module

        model = ToyModel()
        model = api1(model)
        self.assertEqual(1, len(_get_registry(model)))
        self.assertTrue("api1" in _get_registry(model))
        model = api2(model)
        self.assertEqual(2, len(_get_registry(model)))
        self.assertTrue([_get_registry(model).keys()], ["api1", "api2"])
        self.assertEqual(None, _get_registry(model.seq1))
        self.assertEqual(None, _get_registry(model.seq2))

        with self.assertRaisesRegex(AssertionError, "api1 has already been applied"):
            model = api1(model)

    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_multi_module_api(self):
        @contract()
        def multi_module_api(modules: list[nn.Module]) -> nn.Module:
            return modules

        model = nn.Sequential(*[nn.Linear(3, 3) for _ in range(5)])
        multi_module_api([model[0], model[1]])
        multi_module_api([model[2], model[3]])
        multi_module_api([model[4]])
        # Check that modules have the same state and registry iff they shared
        # the same API call
        states = [multi_module_api.state(module) for module in model]
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[2], states[3])
        self.assertNotEqual(states[0], states[2])
        self.assertNotEqual(states[0], states[4])
        self.assertNotEqual(states[2], states[4])
        registries = [_get_registry(module) for module in model]
        self.assertEqual(registries[0], registries[1])
        self.assertEqual(registries[2], registries[3])
        self.assertNotEqual(registries[0], registries[2])
        self.assertNotEqual(registries[0], registries[4])
        self.assertNotEqual(registries[2], registries[4])
        # Check that applying an API to a module multiple times errors
        model = nn.Sequential(*[nn.Linear(3, 3) for _ in range(5)])
        multi_module_api([model[0], model[1]])
        with self.assertRaisesRegex(
            AssertionError,
            "Each distinct composable distributed API can only be applied to "
            r"a module once. multi_module_api has already been applied to the "
            "following module:",
        ):
            multi_module_api([model[0], model[2]])


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ToyModel`, `TestContract`, `ModelWrapper`

**Functions defined**: `__init__`, `forward`, `test_add_hooks`, `forward_pre_hook`, `forward_hook`, `backward_pre_hook`, `backward_hook`, `noop_api`, `test_modify_fqn`, `__init__`, `forward`, `wrap_module`, `test_state`, `check_and_update_state_hook`, `api`, `test_registry`, `api1`, `api2`, `test_multi_module_api`, `multi_module_api`

**Key imports**: deepcopy, torch, torch.nn as nn, _get_registry, contract, run_tests, skipIfTorchDynamo, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_composable`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`: deepcopy
- `torch`
- `torch.nn as nn`
- `torch.distributed._composable`: _get_registry, contract
- `torch.testing._internal.common_utils`: run_tests, skipIfTorchDynamo, TestCase


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
python test/distributed/_composable/test_contract.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_composable`):

- [`test_replicate.py_docs.md`](./test_replicate.py_docs.md)
- [`test_replicate_with_fsdp.py_docs.md`](./test_replicate_with_fsdp.py_docs.md)
- [`test_checkpoint.py_docs.md`](./test_checkpoint.py_docs.md)
- [`test_replicate_mixed_precision.py_docs.md`](./test_replicate_mixed_precision.py_docs.md)
- [`test_replicate_with_compiler.py_docs.md`](./test_replicate_with_compiler.py_docs.md)
- [`test_replicate_training.py_docs.md`](./test_replicate_training.py_docs.md)


## Cross-References

- **File Documentation**: `test_contract.py_docs.md`
- **Keyword Index**: `test_contract.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
