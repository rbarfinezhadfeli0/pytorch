# Documentation: `docs/test/test_autograd_fallback.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_autograd_fallback.py_docs.md`
- **Size**: 19,856 bytes (19.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_autograd_fallback.py`

## File Metadata

- **Path**: `test/test_autograd_fallback.py`
- **Size**: 16,670 bytes (16.28 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: autograd"]

import contextlib
import warnings

import numpy as np

import torch
from torch._library.autograd import autograd_fallback_mode
from torch.library import _scoped_library
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestAutogradFallback(TestCase):
    test_ns = "_test_autograd_fallback"

    def setUp(self):
        super().setUp()
        self.libraries = []

    def tearDown(self):
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        for lib in self.libraries:
            lib._destroy()
        del self.libraries

    def get_op(self, name):
        return getattr(getattr(torch.ops, self.test_ns), name).default

    def get_lib(self):
        result = torch.library.Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        self.libraries.append(result)
        return result

    @parametrize("mode", ("nothing", "warn"))
    def test_no_grad(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor a, Tensor b, int c) -> Tensor")
            lib.impl("foo", lambda a, b, c: a + b + c, "CPU")
            op = self.get_op("foo")

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with torch.no_grad():
                    a = torch.randn([], requires_grad=True)
                    b = torch.randn([], requires_grad=True)
                    out = op(a, b, 1)
                self.assertFalse(out.requires_grad)

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                a = torch.randn([])
                b = torch.randn([])
                out = op(a, b, 1)
                self.assertFalse(out.requires_grad)

    @parametrize("mode", ("nothing", "warn"))
    def test_no_autograd_kernel(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor a, Tensor b, int c) -> Tensor")
            op = self.get_op("foo")

            def foo_impl(a, b, c):
                result = a.detach().numpy() + b.detach().numpy() + c
                return torch.tensor(result)

            lib.impl("foo", foo_impl, "CPU")

            # Some inputs requiring grad
            a = torch.randn([], requires_grad=False)
            b = torch.randn([], requires_grad=True)
            out = op(a, b, 1).sum()
            with self._check_ctx(mode, mode_nothing_raises=True):
                out.backward()
            self.assertIsNone(b.grad)

    def _check_ctx(self, mode, *, mode_nothing_raises=False):
        if mode == "warn":
            return self.assertWarnsRegex(
                UserWarning, "an autograd kernel was not registered"
            )
        assert mode == "nothing"
        if mode_nothing_raises:
            return self.assertRaisesRegex(RuntimeError, "does not require grad")
        return contextlib.nullcontext()

    @parametrize("mode", ("nothing", "warn"))
    def test_no_autograd_kernel_inplace(self, mode):
        with autograd_fallback_mode(mode):
            # input modified in-place gets returned as output
            lib = self.get_lib()
            lib.define("foo(Tensor(a!) self, Tensor(b!) y) -> (Tensor(a!), Tensor(b!))")
            op = self.get_op("foo")

            def foo_impl(x, y):
                with torch.no_grad():
                    x.sin_()
                    y.cos_()
                return x, y

            lib.impl("foo", foo_impl, "CPU")

            x = torch.randn(3, requires_grad=True)
            w = x.clone()
            v = x.clone()
            y0 = w[0]
            y1 = v[1]
            z0, z1 = op(y0, y1)
            for tensor in [w, v, z0, z1, y0, y1]:
                with self._check_ctx(mode):
                    tensor.sum().backward(retain_graph=True)

            # no outputs: we don't do anything. Maybe we should in the future.
            # This is not a common failure mode.
            lib.define("bar(Tensor(a!) self) -> ()")
            op = self.get_op("bar")

            def bar_impl(x):
                with torch.no_grad():
                    x.sin_()

            lib.impl("bar", bar_impl, "CPU")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                x = torch.randn([], requires_grad=True)
                y = x.clone()
                op(y)
                y.backward()
                self.assertEqual(x.grad, torch.ones_like(x))

    @parametrize("mode", ("nothing", "warn"))
    def test_cpu_return_self(self, mode):
        with autograd_fallback_mode(mode):
            # To be clear, none of these situations are OK and will lead
            # to other problems down the line. We're testing them because
            # it is fairly common to actually do these things.
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                lib.define("foo(Tensor self) -> Tensor")
                lib.impl("foo", lambda x: x, "CPU")
                op = self.get_op("foo")

                x = torch.randn(3, requires_grad=True)
                y = op(x).sum()
                with self._check_ctx(mode):
                    y.backward()
                    self.assertEqual(x.grad, torch.ones_like(x))

                lib.define("bar(Tensor(a!) self) -> Tensor(a!)")
                lib.impl("bar", lambda x: x, "CPU")
                op = self.get_op("bar")

                x = torch.randn(3, requires_grad=True)
                y = op(x).sum()
                with self._check_ctx(mode):
                    y.backward()
                    self.assertEqual(x.grad, torch.ones_like(x))

    @parametrize("mode", ("nothing", "warn"))
    def test_composite_registered_to_cpu(self, mode):
        with autograd_fallback_mode(mode):
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                lib.define("foo(Tensor self) -> Tensor")
                lib.impl("foo", lambda x: x.sin().sum(), "CPU")
                op = self.get_op("foo")

                x = torch.randn(3, requires_grad=True)
                y = op(x)
                with self._check_ctx(mode):
                    y.backward()
                    self.assertEqual(x.grad, x.cos())

    @parametrize("mode", ("nothing", "warn"))
    def test_autograd_function_registered_to_cpu(self, mode):
        with autograd_fallback_mode(mode):
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                lib.define("foo(Tensor self) -> Tensor")

                class NumpySin(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x):
                        ctx.save_for_backward(x)
                        return torch.tensor(np.sin(x.cpu().numpy()))

                    @staticmethod
                    def backward(ctx, gx):
                        (x,) = ctx.saved_tensors
                        return gx * x.cos()

                lib.impl("foo", NumpySin.apply, "CPU")
                op = self.get_op("foo")

                x = torch.randn(3, requires_grad=True)
                y = op(x).sum()
                with self._check_ctx(mode):
                    y.backward()
                    self.assertEqual(x.grad, x.cos())

    @parametrize("mode", ("nothing", "warn"))
    def test_inplace_autograd_function_registered_to_cpu(self, mode):
        with autograd_fallback_mode(mode):
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                lib.define("foo(Tensor(a!) self) -> Tensor(a!)")

                class NumpySin_(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x):
                        ctx.save_for_backward(x.clone())
                        x_np = x.detach().numpy()
                        np.sin(x_np, out=x_np)
                        ctx.mark_dirty(x)
                        return x

                    @staticmethod
                    def backward(ctx, gx):
                        (x,) = ctx.saved_tensors
                        return gx * x.cos()

                lib.impl("foo", NumpySin_.apply, "CPU")
                op = self.get_op("foo")

                x = torch.randn(3, requires_grad=True)
                z = x.clone()
                w = z[0]
                y = op(w)

                expected = torch.zeros_like(x)
                expected[0] = x[0].cos()
                with self._check_ctx(mode):
                    (gx,) = torch.autograd.grad(
                        y, x, torch.ones_like(y), retain_graph=True
                    )
                    self.assertEqual(gx, expected)

                expected = torch.ones_like(x)
                expected[0] = x[0].cos()
                with self._check_ctx(mode):
                    (gx,) = torch.autograd.grad(z, x, torch.ones_like(z))
                    self.assertEqual(gx, expected)

    @parametrize("mode", ("nothing", "warn"))
    def test_inplace_on_tensor_that_does_not_require_grad(self, mode):
        # We don't do anything special (that is, we don't rebase history).
        # See NOTE [autograd fallback and in-place operations] for why
        with autograd_fallback_mode(mode):
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                # Correct usage of (a!)
                lib.define("foo(Tensor(a!) self, Tensor other) -> Tensor(a!)")

                def foo_impl(x, y):
                    x_d = x.detach()
                    y = y.detach()
                    x_d.add_(y)
                    return x

                lib.impl("foo", foo_impl, "CPU")
                foo = self.get_op("foo")

                # Incorrect usage of (a!): user doesn't return tensor as-is
                lib.define("bar(Tensor(a!) self, Tensor other) -> Tensor(a!)")

                def bar_impl(x, y):
                    x_d = x.detach()
                    y = y.detach()
                    x_d.add_(y)
                    return x_d.clone()

                lib.impl("bar", bar_impl, "CPU")
                bar = self.get_op("bar")

                # User mutated input tensor but didn't return it.
                lib.define("baz(Tensor(a!) self, Tensor other) -> ()")

                def baz_impl(x, y):
                    x_d = x.detach()
                    y = y.detach()
                    x_d.add_(y)

                lib.impl("baz", baz_impl, "CPU")
                baz = self.get_op("baz")

                # Test in-place on non-view
                for op in (foo, bar, baz):
                    x = torch.randn(3)
                    y = torch.randn(3, requires_grad=True)
                    with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                        z = x.clone()
                        op(z, y)
                        torch.autograd.grad(z, y, torch.ones_like(z), allow_unused=True)

                # Test in-place on view
                for op in (foo, bar, baz):
                    x = torch.randn(3)
                    y = torch.randn(3, requires_grad=True)
                    with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                        z = x[:]
                        op(z, y)
                        torch.autograd.grad(z, x, torch.ones_like(z), allow_unused=True)

    @parametrize("mode", ("nothing", "warn"))
    def test_post_autograd_returns_leaf(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor a) -> (Tensor, Tensor)")
            op = self.get_op("foo")

            lib.impl(
                "foo", lambda a: (a.clone(), a.detach().clone().requires_grad_()), "CPU"
            )
            x = torch.randn(3, requires_grad=True)
            _, z = op(x)
            with self._check_ctx(mode):
                z.sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    def test_undefined_inputs_outputs(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor a, Tensor b) -> (Tensor, Tensor)")
            op = self.get_op("foo")

            def foo_impl(a, b):
                return None, b.clone()

            lib.impl("foo", foo_impl, "CPU")

            x = torch.randn(3, requires_grad=True)
            # NB: PyTorch dispatcher treats "None" as undefined Tensor.
            _, z = op(None, x)
            with self._check_ctx(mode):
                z.sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    def test_undefined_grads(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor a, Tensor b) -> (Tensor, Tensor)")
            op = self.get_op("foo")

            def foo_impl(a, b):
                return a.sin(), b.cos()

            lib.impl("foo", foo_impl, "CPU")

            x = torch.randn(3, requires_grad=True)
            y = torch.randn(3)
            w, z = op(x, y)
            w = torch._C._functions.UndefinedGrad()(w)
            z = torch._C._functions.UndefinedGrad()(z)
            with self._check_ctx(mode):
                (z + w).sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    def test_base_does_not_require_grad(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor(a!) x) -> Tensor(a!)")
            op = self.get_op("foo")

            def foo_impl(a):
                with torch.no_grad():
                    return a.zero_()

            lib.impl("foo", foo_impl, "CPU")
            x = torch.randn(3)
            y = x[:]
            y.requires_grad_()
            w = y[:]
            self.assertTrue(w._base is x)

            # Hook should be registered on w, but not w._base
            op(w)
            with self._check_ctx(mode):
                w.sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    def test_post_autograd_returns_mix_of_requires_grad_tensors(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)")
            op = self.get_op("foo")

            def foo_impl(a, b):
                with torch.no_grad():
                    x = a.clone()
                    z = b.clone()
                y = a * b
                return x, y, z

            lib.impl("foo", foo_impl, "CPU")
            a = torch.randn(3, requires_grad=True)
            b = torch.randn(3, requires_grad=True)
            x, y, z = op(a, b)

            with self._check_ctx(mode, mode_nothing_raises=True):
                torch.autograd.grad(
                    x, (a, b), torch.ones_like(x), allow_unused=True, retain_graph=True
                )

            with self._check_ctx(mode, mode_nothing_raises=False):
                torch.autograd.grad(
                    y, (a, b), torch.ones_like(y), allow_unused=True, retain_graph=True
                )

            with self._check_ctx(mode, mode_nothing_raises=True):
                torch.autograd.grad(
                    z, (a, b), torch.ones_like(z), allow_unused=True, retain_graph=True
                )

    @parametrize("mode", ("nothing", "warn"))
    def test_supports_tensor_lists(self, mode):
        with autograd_fallback_mode(mode):
            lib = self.get_lib()
            lib.define("foo(Tensor[] a) -> Tensor[]")
            op = self.get_op("foo")

            def foo_impl(a):
                x, y, z = a
                with torch.no_grad():
                    return x + y + z, x * y * z

            lib.impl("foo", foo_impl, "CPU")
            x = torch.randn(3, requires_grad=True)
            y = torch.randn(1, requires_grad=True)
            z = torch.randn(2, 1, requires_grad=True)
            a, b = op([x, y, z])
            with self._check_ctx(mode, mode_nothing_raises=True):
                torch.autograd.grad(
                    a,
                    (x, y, z),
                    torch.ones_like(a),
                    allow_unused=True,
                    retain_graph=True,
                )
            with self._check_ctx(mode, mode_nothing_raises=True):
                torch.autograd.grad(
                    b,
                    (x, y, z),
                    torch.ones_like(b),
                    allow_unused=True,
                    retain_graph=True,
                )


instantiate_parametrized_tests(TestAutogradFallback)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 34 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAutogradFallback`, `NumpySin`, `NumpySin_`

**Functions defined**: `setUp`, `tearDown`, `get_op`, `get_lib`, `test_no_grad`, `test_no_autograd_kernel`, `foo_impl`, `_check_ctx`, `test_no_autograd_kernel_inplace`, `foo_impl`, `bar_impl`, `test_cpu_return_self`, `test_composite_registered_to_cpu`, `test_autograd_function_registered_to_cpu`, `forward`, `backward`, `test_inplace_autograd_function_registered_to_cpu`, `forward`, `backward`, `test_inplace_on_tensor_that_does_not_require_grad`

**Key imports**: contextlib, warnings, numpy as np, torch, autograd_fallback_mode, _scoped_library


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `warnings`
- `numpy as np`
- `torch`
- `torch._library.autograd`: autograd_fallback_mode
- `torch.library`: _scoped_library


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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
python test/test_autograd_fallback.py
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

- **File Documentation**: `test_autograd_fallback.py_docs.md`
- **Keyword Index**: `test_autograd_fallback.py_kw.md`
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

- **Automatic Differentiation**: Uses autograd for gradient computation


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
python docs/test/test_autograd_fallback.py_docs.md
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

- **File Documentation**: `test_autograd_fallback.py_docs.md_docs.md`
- **Keyword Index**: `test_autograd_fallback.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
