# Documentation: `test/inductor/test_mps_basic.py`

## File Metadata

- **Path**: `test/inductor/test_mps_basic.py`
- **Size**: 8,627 bytes (8.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: mps"]
import importlib
import os
import sys

import numpy as np

import torch
from torch.testing import FileCheck, make_tensor
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MACOS_VERSION,
    parametrize,
)


MPS_UNSUPPORTED_TYPES = [torch.double, torch.cdouble] + (
    [torch.bfloat16] if MACOS_VERSION < 14.0 else []
)
MPS_DTYPES = [t for t in get_all_dtypes() if t not in MPS_UNSUPPORTED_TYPES]

importlib.import_module("filelock")

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    CommonTemplate,
    TestCase,
)


# TODO: Remove this file.
# This tests basic MPS compile functionality


@instantiate_parametrized_tests
class MPSBasicTests(TestCase):
    is_dtype_supported = CommonTemplate.is_dtype_supported
    common = check_model_gpu
    device = "mps"

    @parametrize("dtype", MPS_DTYPES)
    def test_add(self, dtype):
        self.common(
            lambda a, b: a + b,
            (
                make_tensor(1024, dtype=dtype, device=self.device),
                make_tensor(1024, dtype=dtype, device=self.device),
            ),
            check_lowp=False,
        )

    def test_log(self):
        self.common(lambda x: x.log(), (torch.rand(1024),))

    def test_acos(self):
        self.common(lambda x: x.acos(), (torch.rand(1024),))

    def test_atanh(self):
        self.common(lambda x: x.atanh(), (torch.rand(1024),))

    def test_floor(self):
        self.common(lambda x: x.floor(), (torch.rand(1024),))

    def test_sign(self):
        self.common(lambda x: x.sign(), (torch.rand(1024),))

    def test_sliced_input(self):
        self.common(
            lambda x: x[:, ::2].sin() + x[:, 1::2].cos(), (torch.rand(32, 1024),)
        )

    def test_where(self):
        def foo(x):
            rc = x.abs().sqrt()
            rc[x < 0] = -5
            return rc

        self.common(foo, (torch.rand(1024),))

    @parametrize("dtype", MPS_DTYPES)
    def test_cast(self, dtype):
        self.common(lambda a: a.to(dtype), (torch.rand(1024),))

    def test_broadcast(self):
        self.common(torch.add, (torch.rand(32, 1024), torch.rand(1024)))

    def test_inplace(self):
        def inc_(x):
            x += 1
            return x

        self.common(inc_, (torch.rand(1024),))

    def test_rms_norm_nograd(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/150629
        def fn(x, w):
            with torch.no_grad():
                return torch.nn.functional.rms_norm(x, x.shape, w)

        self.common(fn, (torch.rand(10), torch.ones(10)))

    def test_compile_numpy_scalar(self):
        def fn(x, y):
            return x / y

        self.common(fn, (torch.rand(10), np.exp(0.3)))

    def test_conv_transpose_channels_last(self):
        def fn(x, y):
            return torch.nn.functional.conv_transpose2d(x, y, stride=1, padding=1)

        self.common(
            fn,
            (
                torch.rand(1, 1, 16, 16).to(memory_format=torch.channels_last),
                torch.rand(1, 4, 8, 8),
            ),
        )

    def test_conv_train(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/161905
        def fn(x, y):
            return torch.nn.functional.conv2d(x, y, None, 1, 1, 1)

        self.common(
            fn,
            (
                torch.rand(4, 512, 7, 7, requires_grad=True),
                torch.rand(512, 512, 3, 3),
            ),
            check_gradient=True,
        )

    def test_cholesky(self):
        def fn(x):
            return (
                torch.linalg.cholesky(x, upper=False),
                torch.linalg.cholesky(x, upper=True),
            )

        self.common(fn, (torch.eye(64),), check_lowp=False)

    def test_reduced_max(self):
        # inductor test do not validate that max of say 16K half elements can be computed
        self.common(torch.max, (torch.rand(16384, dtype=torch.half),), check_lowp=False)

    def test_linalg_inv(self):
        def fn(x):
            return torch.linalg.inv(torch.linalg.cholesky(x))

        A = torch.diag(torch.tensor([20.0, 0.5, 5.0], dtype=torch.float32) ** 2)
        self.common(fn, (A,), check_lowp=False)


class MPSBasicTestsAOTI(TestCase):
    def check_model(self, m, inp, dynamic_shapes=None):
        res2 = m(*inp)
        ep = torch.export.export(m, inp, dynamic_shapes=dynamic_shapes)
        path = torch._inductor.aoti_compile_and_package(ep)
        m = torch._inductor.aoti_load_package(path)
        res = m(*inp)
        assert torch.allclose(res, res2)

    def test_add_mps(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inp = (torch.ones(3, 3, device="mps"), torch.ones(3, 3, device="mps"))
        m = M().to("mps")
        self.check_model(m, inp)

    def test_fallback_mps(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.nn.functional.linear(x, y)

        inp = (
            torch.randn(10, 10, device="mps"),
            torch.randn(10, 10, device="mps"),
        )
        m = M().to("mps")
        self.check_model(m, inp)

    def test_c10(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.cat(tensors=torch.split(x, 4, dim=1), dim=-2)

        inp = (torch.randn(2, 8, device="mps"),)
        m = M().to("mps")
        self.check_model(m, inp)

    def test_two_const(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.y = torch.ones(3, 3, device="mps")
                self.z = torch.full((3, 3), 2, device="mps")

            def forward(self, x):
                return x + self.y + self.z

        inp = (torch.ones(3, 3, device="mps"),)
        m = Model().to(device="mps")
        self.check_model(m, inp)

    def test_simple_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return torch.nn.functional.relu(input=add_0, inplace=False)

        x = torch.randn(128, 2048, device="mps")
        y = torch.randn(128, 2048, device="mps")
        inp = (x, y)

        m = Model().to(device="mps")
        dim0_x = torch.export.Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}

        self.check_model(m, inp, dynamic_shapes)

    def test_dynamic_cat(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.cat([a, b], dim=0)

        a = torch.randn(2, 4, device="mps")
        b = torch.randn(3, 4, device="mps")
        inp = (a, b)
        m = Model().to(device="mps")

        dim0_a = torch.export.Dim("dim0_a", min=1, max=10)
        dim0_b = torch.export.Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        self.check_model(m, inp, dynamic_shapes)

    def test_reuse_kernel(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.sin(b)
                d = torch.mm(b, c)
                return d

        example_inputs = (
            torch.randn(87, 87, device="mps"),
            torch.randn(87, 87, device="mps"),
        )
        model = Model()

        ep = torch.export.export(model, example_inputs)
        package_path = torch._export.aot_compile(ep.module(), example_inputs)

        target_str = "aoti_torch_mps_get_kernel_function("
        target_count = 1

        with open(os.path.splitext(package_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            FileCheck().check_count(
                target_str,
                target_count,
                exactly=True,
            ).run(src_code)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if torch.backends.mps.is_available():
        run_tests(needs="filelock")

```



## High-Level Overview


This Python file contains 9 class(es) and 46 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MPSBasicTests`, `MPSBasicTestsAOTI`, `M`, `M`, `M`, `Model`, `Model`, `Model`, `Model`

**Functions defined**: `test_add`, `test_log`, `test_acos`, `test_atanh`, `test_floor`, `test_sign`, `test_sliced_input`, `test_where`, `foo`, `test_cast`, `test_broadcast`, `test_inplace`, `inc_`, `test_rms_norm_nograd`, `fn`, `test_compile_numpy_scalar`, `fn`, `test_conv_transpose_channels_last`, `fn`, `test_conv_train`

**Key imports**: importlib, os, sys, numpy as np, torch, FileCheck, make_tensor, get_all_dtypes, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `importlib`
- `os`
- `sys`
- `numpy as np`
- `torch`
- `torch.testing`: FileCheck, make_tensor
- `torch.testing._internal.common_dtype`: get_all_dtypes
- `torch._dynamo.test_case`: run_tests


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python test/inductor/test_mps_basic.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_mps_basic.py_docs.md`
- **Keyword Index**: `test_mps_basic.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
