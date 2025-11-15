# Documentation: `docs/test/fx/test_fx_param_shape_control_flow.py_docs.md`

## File Metadata

- **Path**: `docs/test/fx/test_fx_param_shape_control_flow.py_docs.md`
- **Size**: 8,375 bytes (8.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/fx/test_fx_param_shape_control_flow.py`

## File Metadata

- **Path**: `test/fx/test_fx_param_shape_control_flow.py`
- **Size**: 5,146 bytes (5.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]

import torch
import torch.fx
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class MyModuleBase(torch.nn.Module):
    def forward(self, x):
        matrx = self.get_mul_matrix()
        if self.no_relu():
            return torch.mm(x, matrx)
        else:
            return torch.relu(torch.mm(x, matrx))

    def get_mul_matrix(self):
        return self.param

    def no_relu(self):
        raise Exception("not implemented")  # noqa: TRY002


class MyModuleParamShape(MyModuleBase):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        return self.param.shape[0] < 10


class MyModuleParamSize(MyModuleBase):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        return self.param.size()[0] < 10


class MyModuleParamDim(MyModuleBase):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def get_mul_matrix(self):
        return self.param[0] if (self.param.dim() == 3) else self.param

    def no_relu(self):
        return self.param.dim() == 3


class MyModuleParamNDim(MyModuleBase):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def get_mul_matrix(self):
        return self.param[0] if (self.param.ndim == 3) else self.param

    def no_relu(self):
        return self.param.ndim == 3


class MyModuleParamNumEl(MyModuleBase):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        return self.param.numel() < 10 * 3


class MyModuleParamNElement(MyModuleBase):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        return self.param.nelement() < 10 * 3


class TestConstParamShapeInControlFlow(TestCase):
    def verify_mm_relu_mods(self, mm_only_mod, relu_mod):
        """
        Verify one module only does a mm op while the other
        performs both mm and relu ops in cascade
        """
        x = torch.randn(10, 5)
        torch.testing.assert_close(
            mm_only_mod(x), torch.mm(x, mm_only_mod.get_mul_matrix())
        )
        tracer = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph = tracer.trace(mm_only_mod)

        # verify the graph module calculates the same result
        graph_mod_mm = torch.fx.GraphModule(mm_only_mod, traced_graph)
        torch.testing.assert_close(
            graph_mod_mm(x), torch.mm(x, mm_only_mod.get_mul_matrix())
        )

        # Make a new module with different parameter shape to go down the different
        # code path
        x = torch.randn(10, 15)
        torch.testing.assert_close(
            relu_mod(x), torch.relu(torch.mm(x, relu_mod.get_mul_matrix()))
        )

        tracer2 = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph2 = tracer2.trace(relu_mod)

        # verify the graph module calculates the same result
        graph_mod_relu = torch.fx.GraphModule(relu_mod, traced_graph2)
        torch.testing.assert_close(
            graph_mod_relu(x), torch.relu(torch.mm(x, relu_mod.get_mul_matrix()))
        )

        graph1_node_targets = [n.target for n in traced_graph.nodes]
        graph2_node_targets = [n.target for n in traced_graph2.nodes]

        # the second graph has an extra relu function call node
        assert torch.mm in graph1_node_targets and torch.mm in graph2_node_targets
        assert (
            torch.relu not in graph1_node_targets and torch.relu in graph2_node_targets
        )

    def test_param_shape_const(self):
        mymod = MyModuleParamShape(in_channels=5)
        mymod2 = MyModuleParamShape(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_size_const(self):
        mymod = MyModuleParamSize(in_channels=5)
        mymod2 = MyModuleParamSize(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_dim_const(self):
        mymod = MyModuleParamDim(torch.nn.Parameter(torch.randn(2, 5, 3)))
        mymod2 = MyModuleParamDim(torch.nn.Parameter(torch.randn(15, 3)))
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_ndim_const(self):
        mymod = MyModuleParamNDim(torch.nn.Parameter(torch.randn(2, 5, 3)))
        mymod2 = MyModuleParamNDim(torch.nn.Parameter(torch.randn(15, 3)))
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_numel_const(self):
        mymod = MyModuleParamNumEl(in_channels=5)
        mymod2 = MyModuleParamNumEl(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_nelement_const(self):
        mymod = MyModuleParamNElement(in_channels=5)
        mymod2 = MyModuleParamNElement(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")

```



## High-Level Overview


This Python file contains 8 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyModuleBase`, `MyModuleParamShape`, `MyModuleParamSize`, `MyModuleParamDim`, `MyModuleParamNDim`, `MyModuleParamNumEl`, `MyModuleParamNElement`, `TestConstParamShapeInControlFlow`

**Functions defined**: `forward`, `get_mul_matrix`, `no_relu`, `__init__`, `no_relu`, `__init__`, `no_relu`, `__init__`, `get_mul_matrix`, `no_relu`, `__init__`, `get_mul_matrix`, `no_relu`, `__init__`, `no_relu`, `__init__`, `no_relu`, `verify_mm_relu_mods`, `test_param_shape_const`, `test_param_size_const`

**Key imports**: torch, torch.fx, raise_on_run_directly, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.fx`
- `torch.testing._internal.common_utils`: raise_on_run_directly, TestCase


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
python test/fx/test_fx_param_shape_control_flow.py
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

- **File Documentation**: `test_fx_param_shape_control_flow.py_docs.md`
- **Keyword Index**: `test_fx_param_shape_control_flow.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python docs/test/fx/test_fx_param_shape_control_flow.py_docs.md
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

- **File Documentation**: `test_fx_param_shape_control_flow.py_docs.md_docs.md`
- **Keyword Index**: `test_fx_param_shape_control_flow.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
