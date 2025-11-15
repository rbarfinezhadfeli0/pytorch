# Documentation: `docs/test/distributed/pipelining/model_registry.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/pipelining/model_registry.py_docs.md`
- **Size**: 14,372 bytes (14.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/distributed/pipelining/model_registry.py`

## File Metadata

- **Path**: `test/distributed/pipelining/model_registry.py`
- **Size**: 11,258 bytes (10.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a model zoo for testing torch.distributed.pipelining.
import torch
from torch.autograd import Function
from torch.distributed.pipelining import pipe_split, SplitPoint


class ExampleCode(torch.nn.Module):
    def __init__(self, d_hid, splits=2):
        assert splits <= 8
        super().__init__()
        self.splits = splits
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.cval = torch.nn.Buffer(torch.randn((d_hid,), requires_grad=False))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)
        self.lin3 = torch.nn.Linear(d_hid, d_hid)
        self.lin4 = torch.nn.Linear(d_hid, d_hid)
        self.lin5 = torch.nn.Linear(d_hid, d_hid)
        self.lin6 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = torch.mm(x, self.mm_param0)
        x = torch.relu(x)
        # try passing a value that doesn't require_grad across skip boundaries
        a_constant = self.cval.clone()
        x = self.lin0(x)
        pipe_split()
        x = torch.relu(x) + a_constant
        x = torch.mm(x, self.mm_param1)
        if self.splits > 2:
            pipe_split()
            x = self.lin1(x)
            x = torch.relu(x)
        if self.splits > 3:
            pipe_split()
            x = self.lin2(x)
            x = torch.relu(x)
        if self.splits > 4:
            pipe_split()
            x = self.lin3(x)
            x = torch.relu(x)
        if self.splits > 5:
            pipe_split()
            x = self.lin4(x)
            x = torch.relu(x)
        if self.splits > 6:
            pipe_split()
            x = self.lin5(x)
            x = torch.relu(x)
        if self.splits > 7:
            pipe_split()
            x = self.lin6(x)
            x = torch.relu(x)
        return x


class ModelWithKwargs(torch.nn.Module):
    DEFAULT_DHID = 512
    DEFAULT_BATCH_SIZE = 256

    def __init__(self, d_hid: int = DEFAULT_DHID, splits=2):
        assert splits <= 8
        super().__init__()
        self.splits = splits
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)
        self.lin3 = torch.nn.Linear(d_hid, d_hid)
        self.lin4 = torch.nn.Linear(d_hid, d_hid)
        self.lin5 = torch.nn.Linear(d_hid, d_hid)
        self.lin6 = torch.nn.Linear(d_hid, d_hid)
        self.lin7 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y=torch.zeros(DEFAULT_BATCH_SIZE, DEFAULT_DHID)):
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = self.lin0(x)
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        x = torch.relu(x)
        if self.splits > 2:
            pipe_split()
            x = self.lin2(x)
            x = torch.relu(x)
        if self.splits > 3:
            pipe_split()
            x = self.lin3(x)
            x = torch.relu(x)
        if self.splits > 4:
            pipe_split()
            x = self.lin4(x)
            x = torch.relu(x)
        if self.splits > 5:
            pipe_split()
            x = self.lin5(x)
            x = torch.relu(x)
        if self.splits > 6:
            pipe_split()
            x = self.lin6(x)
            x = torch.relu(x)
        if self.splits > 7:
            pipe_split()
            x = self.lin7(x)
            x = torch.relu(x)
        return x


class ModelWithParamAlias(torch.nn.Module):
    default_dhid = 512
    default_batch_size = 256

    def __init__(self, d_hid: int = default_dhid):
        super().__init__()
        self.mm_param1 = self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = self.lin0 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = self.lin0(x)
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        x = torch.relu(x)
        return x


# MLP Layer
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class MLPKWargModule(torch.nn.Module):
    def __init__(self, d_hid: int, layer_num):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)
        self.layer_num = layer_num

    def forward(self, x, unused_kwarg: torch.Tensor = torch.zeros(1)):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        # Test when only 1 module has extra outputs
        # TODO: handle this case later
        # if self.layer_num == 0:
        #     return x, unused_kwarg
        # else:
        #     return x
        return x


# Multi-MLP model
class MultiMLP(torch.nn.Module):
    def __init__(self, d_hid: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPModule(d_hid) for _ in range(n_layers)])
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Multi-MLP with kwargs model
class MultiMLPKwargs(torch.nn.Module):
    def __init__(self, d_hid: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [MLPKWargModule(d_hid, i) for i in range(n_layers)]
        )
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x, unused_kwarg: torch.Tensor = torch.zeros(1)):
        for layer in self.layers:
            # TODO: handle this case later
            # if layer.layer_num == 0:
            #     x, _ = layer(x, unused_kwarg)
            # else:
            #     x = layer(x)
            x = layer(x)
        return x


class CustomLinearDx(Function):
    @staticmethod
    def forward(ctx, input_val, weight, bias, module, layer_idx):
        ctx.save_for_backward(input_val, weight, bias)
        ctx.module = module
        ctx.layer_idx = layer_idx
        return input_val.mm(weight.t()) + bias

    @staticmethod
    def backward(ctx, grad_output):
        input_val, weight, _ = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        ctx.module.cached_context[ctx.layer_idx].append(grad_output.clone())
        ctx.module.cached_context[str(ctx.layer_idx) + "_input"].append(
            input_val.clone()
        )
        return grad_input, None, None, None, None


class CustomLinearDxDw(Function):
    @staticmethod
    def forward(ctx, input_val, weight, bias):
        ctx.save_for_backward(input_val, weight, bias)
        return input_val.mm(weight.t()) + bias

    @staticmethod
    def backward(ctx, grad_output):
        input_val, weight, _ = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input_val)
        grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class MLPModuleWithDw(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.fc1_weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.fc1_bias = torch.nn.Parameter(torch.randn(d_hid))
        self.fc2_weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.fc2_bias = torch.nn.Parameter(torch.randn(d_hid))

        torch.nn.init.uniform_(self.fc1_weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc2_weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc1_bias, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc2_bias, -0.001, 0.001)

        self.cached_context = {}
        self.cached_context["fc1"] = []
        self.cached_context["fc2"] = []
        self.cached_context["fc1_input"] = []
        self.cached_context["fc2_input"] = []

        self.use_custom_logic = False

    def forward(self, x):
        if not self.use_custom_logic:
            self.hidden = CustomLinearDxDw.apply(x, self.fc1_weight, self.fc1_bias)
            self.hidden = torch.nn.functional.relu(self.hidden)
            output = CustomLinearDxDw.apply(self.hidden, self.fc2_weight, self.fc2_bias)
            return output

        self.hidden = CustomLinearDx.apply(
            x, self.fc1_weight, self.fc1_bias, self, "fc1"
        )
        self.hidden = torch.nn.functional.relu(self.hidden)
        output = CustomLinearDx.apply(
            self.hidden, self.fc2_weight, self.fc2_bias, self, "fc2"
        )
        return output

    def compute_dW(self):
        grad_output_fc1 = self.cached_context["fc1"].pop(0)
        grad_output_fc2 = self.cached_context["fc2"].pop(0)
        cached_input_fc1 = self.cached_context["fc1_input"].pop(0)
        cached_input_fc2 = self.cached_context["fc2_input"].pop(0)

        dW2 = grad_output_fc2.t().mm(cached_input_fc2)
        db2 = grad_output_fc2.sum(0)

        dW1 = grad_output_fc1.t().mm(cached_input_fc1)
        db1 = grad_output_fc1.sum(0)

        if self.fc1_weight.grad is not None:
            self.fc1_weight.grad += dW1
            self.fc1_bias.grad += db1
            self.fc2_weight.grad += dW2
            self.fc2_bias.grad += db2
        else:
            self.fc1_weight.grad = dW1
            self.fc1_bias.grad = db1
            self.fc2_weight.grad = dW2
            self.fc2_bias.grad = db2

    def toggle(self):
        self.use_custom_logic = not self.use_custom_logic


# Multi-MLP model With Dw
class MultiMLPWithDw(torch.nn.Module):
    def __init__(self, d_hid: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [MLPModuleWithDw(d_hid) for _ in range(n_layers)]
        )
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }
        self.use_custom_logic = False

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def toggle(self):
        self.use_custom_logic = not self.use_custom_logic
        for layer in self.layers:
            layer.toggle()

    def compute_dW(self):
        if not self.use_custom_logic:
            raise RuntimeError("Need to call toggle() to enable custom backward and dW")

        for i in reversed(range(len(self.layers))):
            self.layers[i].compute_dW()

```



## High-Level Overview


This Python file contains 11 class(es) and 26 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ExampleCode`, `ModelWithKwargs`, `ModelWithParamAlias`, `MLPModule`, `MLPKWargModule`, `MultiMLP`, `MultiMLPKwargs`, `CustomLinearDx`, `CustomLinearDxDw`, `MLPModuleWithDw`, `MultiMLPWithDw`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `forward`, `backward`, `forward`, `backward`, `__init__`, `forward`

**Key imports**: torch, Function, pipe_split, SplitPoint


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/pipelining`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.autograd`: Function
- `torch.distributed.pipelining`: pipe_split, SplitPoint


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/pipelining/model_registry.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/pipelining`):

- [`test_schedule_multiproc.py_docs.md`](./test_schedule_multiproc.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_microbatch.py_docs.md`](./test_microbatch.py_docs.md)
- [`test_schedule.py_docs.md`](./test_schedule.py_docs.md)
- [`test_pipe.py_docs.md`](./test_pipe.py_docs.md)
- [`test_transformer.py_docs.md`](./test_transformer.py_docs.md)
- [`test_stage.py_docs.md`](./test_stage.py_docs.md)
- [`schedule_registry.py_docs.md`](./schedule_registry.py_docs.md)
- [`test_unflatten.py_docs.md`](./test_unflatten.py_docs.md)


## Cross-References

- **File Documentation**: `model_registry.py_docs.md`
- **Keyword Index**: `model_registry.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/pipelining`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/pipelining`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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
python docs/test/distributed/pipelining/model_registry.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/pipelining`):

- [`test_transformer.py_kw.md_docs.md`](./test_transformer.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_docs.md_docs.md`](./test_schedule_multiproc.py_docs.md_docs.md)
- [`model_registry.py_kw.md_docs.md`](./model_registry.py_kw.md_docs.md)
- [`test_unflatten.py_docs.md_docs.md`](./test_unflatten.py_docs.md_docs.md)
- [`schedule_registry.py_docs.md_docs.md`](./schedule_registry.py_docs.md_docs.md)
- [`test_stage.py_docs.md_docs.md`](./test_stage.py_docs.md_docs.md)
- [`schedule_registry.py_kw.md_docs.md`](./schedule_registry.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_kw.md_docs.md`](./test_schedule_multiproc.py_kw.md_docs.md)
- [`test_unflatten.py_kw.md_docs.md`](./test_unflatten.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `model_registry.py_docs.md_docs.md`
- **Keyword Index**: `model_registry.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
