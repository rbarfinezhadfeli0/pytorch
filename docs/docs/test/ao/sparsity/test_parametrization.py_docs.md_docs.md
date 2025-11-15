# Documentation: `docs/test/ao/sparsity/test_parametrization.py_docs.md`

## File Metadata

- **Path**: `docs/test/ao/sparsity/test_parametrization.py_docs.md`
- **Size**: 9,664 bytes (9.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/ao/sparsity/test_parametrization.py`

## File Metadata

- **Path**: `test/ao/sparsity/test_parametrization.py`
- **Size**: 6,630 bytes (6.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
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


This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ModelUnderTest`, `TestFakeSparsity`

**Functions defined**: `__init__`, `forward`, `test_masking_logic`, `test_weights_parametrized`, `test_state_dict_preserved`, `test_jit_trace`

**Key imports**: torch, nn, utils, parametrize, raise_on_run_directly, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/ao/sparsity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.pruning.sparsifier`: utils
- `torch.nn.utils`: parametrize
- `torch.testing._internal.common_utils`: raise_on_run_directly, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/ao/sparsity/test_parametrization.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/ao/sparsity`):

- [`test_kernels.py_docs.md`](./test_kernels.py_docs.md)
- [`test_activation_sparsifier.py_docs.md`](./test_activation_sparsifier.py_docs.md)
- [`test_data_scheduler.py_docs.md`](./test_data_scheduler.py_docs.md)
- [`test_scheduler.py_docs.md`](./test_scheduler.py_docs.md)
- [`test_sparsity_utils.py_docs.md`](./test_sparsity_utils.py_docs.md)
- [`test_data_sparsifier.py_docs.md`](./test_data_sparsifier.py_docs.md)
- [`test_structured_sparsifier.py_docs.md`](./test_structured_sparsifier.py_docs.md)
- [`test_qlinear_packed_params.py_docs.md`](./test_qlinear_packed_params.py_docs.md)
- [`test_sparsifier.py_docs.md`](./test_sparsifier.py_docs.md)


## Cross-References

- **File Documentation**: `test_parametrization.py_docs.md`
- **Keyword Index**: `test_parametrization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/ao/sparsity`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/ao/sparsity`, which is part of the **testing infrastructure**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/ao/sparsity/test_parametrization.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/ao/sparsity`):

- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_data_sparsifier.py_kw.md_docs.md`](./test_data_sparsifier.py_kw.md_docs.md)
- [`test_activation_sparsifier.py_docs.md_docs.md`](./test_activation_sparsifier.py_docs.md_docs.md)
- [`test_data_scheduler.py_kw.md_docs.md`](./test_data_scheduler.py_kw.md_docs.md)
- [`test_sparsity_utils.py_kw.md_docs.md`](./test_sparsity_utils.py_kw.md_docs.md)
- [`test_structured_sparsifier.py_docs.md_docs.md`](./test_structured_sparsifier.py_docs.md_docs.md)
- [`test_composability.py_kw.md_docs.md`](./test_composability.py_kw.md_docs.md)
- [`test_kernels.py_kw.md_docs.md`](./test_kernels.py_kw.md_docs.md)
- [`test_structured_sparsifier.py_kw.md_docs.md`](./test_structured_sparsifier.py_kw.md_docs.md)
- [`test_data_sparsifier.py_docs.md_docs.md`](./test_data_sparsifier.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_parametrization.py_docs.md_docs.md`
- **Keyword Index**: `test_parametrization.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
