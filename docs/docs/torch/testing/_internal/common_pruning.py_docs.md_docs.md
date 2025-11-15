# Documentation: `docs/torch/testing/_internal/common_pruning.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/common_pruning.py_docs.md`
- **Size**: 17,013 bytes (16.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/common_pruning.py`

## File Metadata

- **Path**: `torch/testing/_internal/common_pruning.py`
- **Size**: 13,655 bytes (13.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# Owner(s): ["module: unknown"]

from typing import Any
from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn

class ImplementedSparsifier(BaseSparsifier):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(defaults=kwargs)

    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs: dict[str, Any]) -> None:
        module.parametrizations.weight[0].mask[0] = 0  # type: ignore[index, union-attr]
        linear_state = self.state['linear1.weight']
        linear_state['step_count'] = linear_state.get('step_count', 0) + 1


class MockSparseLinear(nn.Linear):
    """
    This class is a MockSparseLinear class to check convert functionality.
    It is the same as a normal Linear layer, except with a different type, as
    well as an additional from_dense method.
    """
    @classmethod
    def from_dense(cls, mod: nn.Linear) -> 'MockSparseLinear':
        """
        """
        linear = cls(mod.in_features,
                     mod.out_features)
        return linear


def rows_are_subset(subset_tensor: torch.Tensor, superset_tensor: torch.Tensor) -> bool:
    """
    Checks to see if all rows in subset tensor are present in the superset tensor
    """
    i = 0
    for row in subset_tensor:
        while i < len(superset_tensor):
            if not torch.equal(row, superset_tensor[i]):
                i += 1
            else:
                break
        else:
            return False
    return True


class SimpleLinear(nn.Module):
    r"""Model with only Linear layers without biases, some wrapped in a Sequential,
    some following the Sequential. Used to test basic pruned Linear-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=False),
            nn.Linear(5, 6, bias=False),
            nn.Linear(6, 4, bias=False),
        )
        self.linear1 = nn.Linear(4, 4, bias=False)
        self.linear2 = nn.Linear(4, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class LinearBias(nn.Module):
    r"""Model with only Linear layers, alternating layers with biases,
    wrapped in a Sequential. Used to test pruned Linear-Bias-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.Linear(5, 6, bias=False),
            nn.Linear(6, 3, bias=True),
            nn.Linear(3, 3, bias=True),
            nn.Linear(3, 10, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        return x


class LinearActivation(nn.Module):
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 6, bias=False),
            nn.Tanh(),
            nn.Linear(6, 4, bias=True),
        )
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(3, 10, bias=False)
        self.act2 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x


class LinearActivationFunctional(nn.Module):
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and functional
    activationals are called in between each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 6, bias=False),
            nn.ReLU(),
            nn.Linear(6, 4, bias=True),
        )
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.linear2 = nn.Linear(3, 8, bias=False)
        self.linear3 = nn.Linear(8, 10, bias=False)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        return x


class SimpleConv2d(nn.Module):
    r"""Model with only Conv2d layers, all without bias, some in a Sequential and some following.
    Used to test pruned Conv2d-Conv2d fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=False),
            nn.Conv2d(32, 64, 3, 1, bias=False),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x


class Conv2dBias(nn.Module):
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some outside.
    Used to test pruned Conv2d-Bias-Conv2d fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=True),
            nn.Conv2d(32, 32, 3, 1, bias=True),
            nn.Conv2d(32, 64, 3, 1, bias=False),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=True)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x


class Conv2dActivation(nn.Module):
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some following.
    Activation function modules in between each Sequential layer, functional activations called
    in-between each outside layer.
    Used to test pruned Conv2d-Bias-Activation-Conv2d fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, bias=True),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, 1, bias=False),
            nn.ReLU(),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.relu(x)
        x = self.conv2d2(x)
        x = F.hardtanh(x)
        return x


class Conv2dPadBias(nn.Module):
    r"""Model with only Conv2d layers, all with bias and some with padding > 0,
    some in a Sequential and some following. Activation function modules in between each layer.
    Used to test that bias is propagated correctly in the special case of
    pruned Conv2d-Bias-(Activation)Conv2d fusion, when the second Conv2d layer has padding > 0."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, bias=True),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, padding=1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, padding=1, bias=True)
        self.act2 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.act1(x)
        x = self.conv2d2(x)
        x = self.act2(x)
        return x


class Conv2dPool(nn.Module):
    r"""Model with only Conv2d layers, all with bias, some in a Sequential and some following.
    Activation function modules in between each layer, Pool2d modules in between each layer.
    Used to test pruned Conv2d-Pool2d-Conv2d fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(48, 52, kernel_size=3, padding=1, bias=True)
        self.conv2d3 = nn.Conv2d(52, 52, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.maxpool(x)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = F.relu(x)
        x = self.conv2d3(x)
        return x


class Conv2dPoolFlattenFunctional(nn.Module):
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a functional Flatten followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(11, 13, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # test functional flatten
        x = self.fc(x)
        return x


class Conv2dPoolFlatten(nn.Module):
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a Flatten module followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(44, 13, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class LSTMLinearModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a linear."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _hidden = self.lstm(input)
        decoded = self.linear(output)
        return decoded, output


class LSTMLayerNormLinearModel(nn.Module):
    """Container module with an LSTM, a LayerNorm, and a linear."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, state = self.lstm(x)
        x = self.norm(x)
        x = self.linear(x)
        return x, state

```



## High-Level Overview

"""    This class is a MockSparseLinear class to check convert functionality.    It is the same as a normal Linear layer, except with a different type, as    well as an additional from_dense method.

This Python file contains 17 class(es) and 30 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ImplementedSparsifier`, `MockSparseLinear`, `SimpleLinear`, `LinearBias`, `LinearActivation`, `LinearActivationFunctional`, `SimpleConv2d`, `Conv2dBias`, `Conv2dActivation`, `Conv2dPadBias`, `Conv2dPool`, `Conv2dPoolFlattenFunctional`, `Conv2dPoolFlatten`, `LSTMLinearModel`, `LSTMLayerNormLinearModel`

**Functions defined**: `__init__`, `update_mask`, `from_dense`, `rows_are_subset`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`

**Key imports**: Any, BaseSparsifier, torch, torch.nn.functional as F, nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch.ao.pruning`: BaseSparsifier
- `torch`
- `torch.nn.functional as F`


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
python torch/testing/_internal/common_pruning.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal`):

- [`common_jit.py_docs.md`](./common_jit.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_function_db.py_docs.md`](./autograd_function_db.py_docs.md)
- [`custom_op_db.py_docs.md`](./custom_op_db.py_docs.md)
- [`subclasses.py_docs.md`](./subclasses.py_docs.md)
- [`logging_utils.py_docs.md`](./logging_utils.py_docs.md)
- [`two_tensor.py_docs.md`](./two_tensor.py_docs.md)
- [`autocast_test_lists.py_docs.md`](./autocast_test_lists.py_docs.md)
- [`hypothesis_utils.py_docs.md`](./hypothesis_utils.py_docs.md)
- [`common_mkldnn.py_docs.md`](./common_mkldnn.py_docs.md)


## Cross-References

- **File Documentation**: `common_pruning.py_docs.md`
- **Keyword Index**: `common_pruning.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal`, which is part of the **core PyTorch library**.



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
python docs/torch/testing/_internal/common_pruning.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal`):

- [`hypothesis_utils.py_kw.md_docs.md`](./hypothesis_utils.py_kw.md_docs.md)
- [`fake_config_module2.py_kw.md_docs.md`](./fake_config_module2.py_kw.md_docs.md)
- [`static_module.py_kw.md_docs.md`](./static_module.py_kw.md_docs.md)
- [`common_pruning.py_kw.md_docs.md`](./common_pruning.py_kw.md_docs.md)
- [`composite_compliance.py_kw.md_docs.md`](./composite_compliance.py_kw.md_docs.md)
- [`common_mkldnn.py_docs.md_docs.md`](./common_mkldnn.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`common_dtype.py_docs.md_docs.md`](./common_dtype.py_docs.md_docs.md)
- [`common_methods_invocations.py_docs.md_docs.md`](./common_methods_invocations.py_docs.md_docs.md)
- [`hypothesis_utils.py_docs.md_docs.md`](./hypothesis_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `common_pruning.py_docs.md_docs.md`
- **Keyword Index**: `common_pruning.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
