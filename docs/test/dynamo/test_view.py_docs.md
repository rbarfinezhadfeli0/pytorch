# Documentation: `test/dynamo/test_view.py`

## File Metadata

- **Path**: `test/dynamo/test_view.py`
- **Size**: 3,708 bytes (3.62 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.test_case


@torch._dynamo.config.patch("capture_scalar_outputs", True)
class ViewTests(torch._dynamo.test_case.TestCase):
    def test_view_to_2d(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(t, _u0):
            u0 = t[0].item()
            u1 = t[1].item()
            n = u0 * u1
            a = torch.randn(n)
            return a.view(-1, _u0)

        t = torch.tensor([2, 4], dtype=torch.int32)
        f(t, 2)

    def test_view_to_1d(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(t, _n):
            u0 = t[0].item()
            u1 = t[1].item()
            a = torch.randn(u0, u1)
            return a.view(_n)

        t = torch.tensor([2, 4], dtype=torch.int32)
        f(t, 8)

    def test_view_with_tensor_shape_params(self):
        # Test for issue #156720: aten.view.default with tensor shape parameters
        class TestModel(torch.nn.Module):
            def forward(self, x, shape_params):
                return torch.ops.aten.view.default(x, shape_params)

        x = torch.randn(24)
        shape_params = [
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
        ]

        model = TestModel()
        expected = model(x, shape_params)

        compiled_model = torch.compile(model, backend="eager")
        result = compiled_model(x, shape_params)

        torch.testing.assert_close(result, expected)

    def test_tensor_view_with_tensor_shape_params(self):
        # Test tensor.view() method with tensor shape parameters (list version)
        class TestModel(torch.nn.Module):
            def forward(self, x, shape_params):
                return x.view(shape_params)

        x = torch.randn(24)
        shape_params = (
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
        )

        model = TestModel()
        expected = model(x, shape_params)

        compiled_model = torch.compile(model, backend="eager")
        result = compiled_model(x, shape_params)

        torch.testing.assert_close(result, expected)

    def test_tensor_view_with_tensor_args(self):
        # Test tensor.view() method with individual tensor arguments
        class TestModel(torch.nn.Module):
            def forward(self, x, dim1, dim2, dim3):
                return x.view(dim1, dim2, dim3)

        x = torch.randn(24)
        dim1 = torch.tensor(2, dtype=torch.int32)
        dim2 = torch.tensor(3, dtype=torch.int32)
        dim3 = torch.tensor(4, dtype=torch.int32)

        model = TestModel()
        expected = model(x, dim1, dim2, dim3)

        compiled_model = torch.compile(model, backend="eager")
        result = compiled_model(x, dim1, dim2, dim3)

        torch.testing.assert_close(result, expected)

    def test_torch_reshape_with_tensor_shape_params(self):
        # Test torch.reshape() function with tensor shape parameters
        def test_fn(x, shape_params):
            return torch.reshape(x, shape_params)

        x = torch.randn(24)
        shape_params = [
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
        ]

        expected = test_fn(x, shape_params)

        compiled_fn = torch.compile(test_fn, backend="eager")
        result = compiled_fn(x, shape_params)

        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 4 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ViewTests`, `TestModel`, `TestModel`, `TestModel`

**Functions defined**: `test_view_to_2d`, `f`, `test_view_to_1d`, `f`, `test_view_with_tensor_shape_params`, `forward`, `test_tensor_view_with_tensor_shape_params`, `forward`, `test_tensor_view_with_tensor_args`, `forward`, `test_torch_reshape_with_tensor_shape_params`, `test_fn`

**Key imports**: torch, torch._dynamo, torch._dynamo.test_case, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dynamo`
- `torch._dynamo.test_case`


## Code Patterns & Idioms

### Common Patterns

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
python test/dynamo/test_view.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_view.py_docs.md`
- **Keyword Index**: `test_view.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
