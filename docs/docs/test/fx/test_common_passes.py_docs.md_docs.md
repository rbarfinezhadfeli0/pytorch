# Documentation: `docs/test/fx/test_common_passes.py_docs.md`

## File Metadata

- **Path**: `docs/test/fx/test_common_passes.py_docs.md`
- **Size**: 5,782 bytes (5.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/fx/test_common_passes.py`

## File Metadata

- **Path**: `test/fx/test_common_passes.py`
- **Size**: 2,801 bytes (2.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: fx"]

import itertools

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph_module import GraphModule
from torch.fx.passes.dialect.common.cse_pass import CSEPass
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    raise_on_run_directly,
    TestCase,
)


def FactoryFunctionCall(x, device):
    y = torch.full(x.shape, 3, device=device)
    z = torch.add(y, x)
    return z


def TorchTensorCall(x):
    y = torch.tensor(3)
    return x + y


def TakeList(x):
    z = torch.cat([x, x])
    return z


def ReturnList(x):
    a = torch.arange(10).reshape(5, 2)
    z = torch.split(a, [1, 4])
    return z


def Mutation(x):
    y = x + 2
    y.add_(1)
    return x + y


def MutationInput(x):
    x.add_(1)
    y = x + 2
    return x + y


def MutationFactory(x, device):
    y = torch.full(x.shape, 3, device=device)
    y.add_(1)
    return x + y


def MutationTorchTensorCall(x):
    y = torch.tensor(3)
    y.add_(1)
    return x + y


def MutationMetadata(x):
    x.resize_(2)
    return x


Passes = [CSEPass]
Test_Cases = [
    TakeList,
    ReturnList,
    Mutation,
    MutationInput,
    MutationMetadata,
    MutationTorchTensorCall,
]
Factory_Test_Cases = [FactoryFunctionCall, MutationFactory]
Devices = ["cpu"]
if torch.cuda.is_available():
    Devices.append("cuda")


def name_fn(common_pass, f, device):
    """Names parameterized test cases."""
    return f"{type(common_pass()).__name__}_{f.__name__}_{device}"


@instantiate_parametrized_tests
class TestCommonPass(TestCase):
    @parametrize(
        "common_pass,f,device", itertools.product(Passes, Test_Cases, Devices), name_fn
    )
    def test_correctness(self, common_pass, f, device):
        inp = torch.randn(10, device=device)

        traced_m = make_fx(f)(inp)
        P = common_pass()

        res = P(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, GraphModule)

        inp_copy = inp.clone()
        expected = f(inp)
        result = modified_m(inp_copy)

        self.assertEqual(result, expected)

    @parametrize(
        "common_pass,f,device",
        itertools.product(Passes, Factory_Test_Cases, Devices),
        name_fn,
    )
    def test_correctness_factory(self, common_pass, f, device):
        inp = torch.randn(10, device=device)
        traced_m = make_fx(f)(inp, device)
        P = common_pass()

        res = P(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, GraphModule)

        inp_copy = inp.clone()
        expected = f(inp, device)
        result = modified_m(inp_copy, device)

        self.assertEqual(result, expected)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")

```



## High-Level Overview


This Python file contains 1 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCommonPass`

**Functions defined**: `FactoryFunctionCall`, `TorchTensorCall`, `TakeList`, `ReturnList`, `Mutation`, `MutationInput`, `MutationFactory`, `MutationTorchTensorCall`, `MutationMetadata`, `name_fn`, `test_correctness`, `test_correctness_factory`

**Key imports**: itertools, torch, make_fx, GraphModule, CSEPass


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `torch`
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.fx.graph_module`: GraphModule
- `torch.fx.passes.dialect.common.cse_pass`: CSEPass


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/fx/test_common_passes.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/fx`):

- [`test_dynamism.py_docs.md`](./test_dynamism.py_docs.md)
- [`test_dce_pass.py_docs.md`](./test_dce_pass.py_docs.md)
- [`test_source_matcher_utils.py_docs.md`](./test_source_matcher_utils.py_docs.md)
- [`test_graph_pickler.py_docs.md`](./test_graph_pickler.py_docs.md)
- [`named_tup.py_docs.md`](./named_tup.py_docs.md)
- [`test_cse_pass.py_docs.md`](./test_cse_pass.py_docs.md)
- [`test_fx_traceback.py_docs.md`](./test_fx_traceback.py_docs.md)
- [`test_gradual_type.py_docs.md`](./test_gradual_type.py_docs.md)
- [`test_pass_infra.py_docs.md`](./test_pass_infra.py_docs.md)
- [`test_lazy_graph_module.py_docs.md`](./test_lazy_graph_module.py_docs.md)


## Cross-References

- **File Documentation**: `test_common_passes.py_docs.md`
- **Keyword Index**: `test_common_passes.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/fx/test_common_passes.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/fx`):

- [`named_tup.py_kw.md_docs.md`](./named_tup.py_kw.md_docs.md)
- [`test_dynamism.py_kw.md_docs.md`](./test_dynamism.py_kw.md_docs.md)
- [`test_fx_traceback.py_docs.md_docs.md`](./test_fx_traceback.py_docs.md_docs.md)
- [`test_fx_xform_observer.py_docs.md_docs.md`](./test_fx_xform_observer.py_docs.md_docs.md)
- [`test_pass_infra.py_kw.md_docs.md`](./test_pass_infra.py_kw.md_docs.md)
- [`test_fx_xform_observer.py_kw.md_docs.md`](./test_fx_xform_observer.py_kw.md_docs.md)
- [`test_fx_node_hook.py_kw.md_docs.md`](./test_fx_node_hook.py_kw.md_docs.md)
- [`test_partitioner_order.py_docs.md_docs.md`](./test_partitioner_order.py_docs.md_docs.md)
- [`test_subgraph_rewriter.py_kw.md_docs.md`](./test_subgraph_rewriter.py_kw.md_docs.md)
- [`test_fx_split.py_docs.md_docs.md`](./test_fx_split.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_common_passes.py_docs.md_docs.md`
- **Keyword Index**: `test_common_passes.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
