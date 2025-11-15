# Documentation: `test/dynamo/test_debug_utils.py`

## File Metadata

- **Path**: `test/dynamo/test_debug_utils.py`
- **Size**: 7,342 bytes (7.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import os
from unittest.mock import patch

import torch
from torch._dynamo import debug_utils
from torch._dynamo.debug_utils import aot_graph_input_parser, generate_env_vars_string
from torch._dynamo.test_case import TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests


f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


class TestDebugUtils(TestCase):
    def test_cast_model_to_fp64_dtype_args(self):
        # Test that dtype arguments are converted to fp64

        def fn(x):
            return (
                torch.ops.prims.convert_element_type(x, torch.float16),
                x.to(torch.float16),
                torch.full(x.shape, 2, dtype=torch.float32, device=x.device),
                x.new_empty(x.shape),
            )

        x = torch.randn(32, device="cpu")
        decomps = torch._decomp.core_aten_decompositions()
        fx = make_fx(fn, decomposition_table=decomps)(x)

        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float16)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float16);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,  # NOQA: B950
        )

        _, fp64_examples = debug_utils.cast_to_fp64(fx, (x,))
        self.assertEqual(fp64_examples, (x.to(torch.float64),))

        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float64)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float64);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float64, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,  # NOQA: B950
        )

    @patch.dict(os.environ, {"TORCHINDUCTOR_MAX_AUTOTUNE": "1", "TEST_ENV": "1"})
    def test_generate_env_vars_string(self):
        env_strings = generate_env_vars_string()
        self.assertIn(
            """os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '1'
""",
            env_strings,
        )
        self.assertIn(
            """import os
""",
            env_strings,
        )
        self.assertNotIn(
            """TEST_ENV
""",
            env_strings,
        )


class TestDebugUtilsDevice(TestCase):
    def test_aot_graph_parser(self, device):
        def forward(
            self,
            primals_1: "f32[1001, 6]",
            primals_2: "f32[1001]",
            primals_3: "f32[1001, 64]",
            primals_4: "f32[4190]",
            primals_5: "f32[4190]",
            primals_6: "f32[1739, 4190]",
            primals_48: "f32[6144, 4191]",
        ):
            _tensor_constant0: "i64[4190]" = self._tensor_constant0
            lift_fresh_copy: "i64[4190]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant0
            )
            _tensor_constant0 = None
            index: "f32[6144, 4190]" = torch.ops.aten.index.Tensor(  # noqa: F841
                primals_48, [None, lift_fresh_copy]
            )
            lift_fresh_copy = None

            _tensor_constant1: "i64[6]" = self._tensor_constant1
            lift_fresh_copy_1: "i64[6]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant1
            )
            _tensor_constant1 = None
            index_1: "f32[6144, 6]" = torch.ops.aten.index.Tensor(
                primals_48, [None, lift_fresh_copy_1]
            )
            primals_48 = lift_fresh_copy_1 = None
            permute: "f32[6, 1001]" = torch.ops.aten.permute.default(primals_1, [1, 0])
            primals_1 = None
            addmm: "f32[6144, 1001]" = torch.ops.aten.addmm.default(
                primals_2, index_1, permute
            )
            primals_2 = permute = None
            amax: "f32[6144, 1]" = torch.ops.aten.amax.default(addmm, [-1], True)
            sub: "f32[6144, 1001]" = torch.ops.aten.sub.Tensor(addmm, amax)
            exp: "f32[6144, 1001]" = torch.ops.aten.exp.default(sub)
            sub = None
            sum_1: "f32[6144, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
            div: "f32[6144, 1001]" = torch.ops.aten.div.Tensor(exp, sum_1)
            exp = None

            full_default: "i32[6144, 1001]" = torch.ops.aten.full.default(
                [6144, 1001],
                1,
                dtype=torch.int32,
                layout=torch.strided,
                device=device,
                pin_memory=False,
            )

            iota: "i32[1001]" = torch.ops.prims.iota.default(
                1001,
                start=0,
                step=1,
                dtype=torch.int32,
                device=device,
                requires_grad=False,
            )

            mul: "i32[6144, 1001]" = torch.ops.aten.mul.Tensor(full_default, iota)
            full_default = iota = None

            iota_1: "i32[6144]" = torch.ops.prims.iota.default(
                6144,
                start=0,
                step=1001,
                dtype=torch.int32,
                device=device,
                requires_grad=False,
            )
            view: "i32[6150144]" = torch.ops.aten.reshape.default(mul, [-1])
            mul = None
            view_1: "f32[6150144]" = torch.ops.aten.reshape.default(div, [-1])
            div = None
            _embedding_bag = torch.ops.aten._embedding_bag.default(
                primals_3, view, iota_1, False, 0, False, view_1
            )

            return _embedding_bag

        kwargs = aot_graph_input_parser(forward, device=device)
        # runs successfully
        forward(**kwargs)

    def test_sym_aot_graph_parser(self, device):
        def forward(
            self,
            primals_1: "f32[1001, 6]",  # noqa: F821
            primals_2: "f32[s0]",  # noqa: F821
            primals_3: "Sym(s0)",  # noqa: F821,
            primals_4: "f32[s1]",  # noqa: F821,
            primals_5: "Sym(s1)",  # noqa: F821,
        ):
            _tensor_constant0: "i64[4190]" = self._tensor_constant0

        kwargs = aot_graph_input_parser(
            forward, device=device, sym_shapes={"s0": 10}, default_sym_shape=5
        )

        self.assertEqual(list(kwargs["primals_2"].shape), [10])
        self.assertEqual(kwargs["primals_3"], 10)

        self.assertEqual(list(kwargs["primals_4"].shape), [5])
        self.assertEqual(kwargs["primals_5"], 5)


instantiate_device_type_tests(TestDebugUtils, globals())

devices = ["cuda", "hpu"]
instantiate_device_type_tests(TestDebugUtilsDevice, globals(), only_for=devices)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview

"""\def forward(self, x_1):    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float16)    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float16);  x_1 = None    full = torch.ops.aten.full.default([32], 2, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)    return (convert_element_type, _to_copy, full, empty)

This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDebugUtils`, `TestDebugUtilsDevice`

**Functions defined**: `test_cast_model_to_fp64_dtype_args`, `fn`, `forward`, `forward`, `test_generate_env_vars_string`, `test_aot_graph_parser`, `forward`, `test_sym_aot_graph_parser`, `forward`

**Key imports**: os, patch, torch, debug_utils, aot_graph_input_parser, generate_env_vars_string, TestCase, make_fx, instantiate_device_type_tests, os, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest.mock`: patch
- `torch`
- `torch._dynamo`: debug_utils
- `torch._dynamo.debug_utils`: aot_graph_input_parser, generate_env_vars_string
- `torch._dynamo.test_case`: TestCase
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests


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
python test/dynamo/test_debug_utils.py
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

- **File Documentation**: `test_debug_utils.py_docs.md`
- **Keyword Index**: `test_debug_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
