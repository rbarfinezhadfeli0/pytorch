# Documentation: `docs/test/test_jiterator.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_jiterator.py_docs.md`
- **Size**: 9,928 bytes (9.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_jiterator.py`

## File Metadata

- **Path**: `test/test_jiterator.py`
- **Size**: 6,673 bytes (6.52 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
# Owner(s): ["module: cuda"]

import torch
from torch.cuda.jiterator import _create_jit_fn as create_jit_fn
from torch.cuda.jiterator import _create_multi_output_jit_fn as create_multi_output_jit_fn
import sys
from itertools import product
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests, TEST_CUDA, NoTest
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_device_type import (
    skipCUDAIfVersionLessThan, instantiate_device_type_tests, dtypes, toleranceOverride, tol)

if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = NoTest  # noqa: F811


code_string = "template <typename T> T my_fused_kernel(T x, T y, T alpha, T beta) { return alpha * x + beta * y; }"
jitted_fn = create_jit_fn(code_string, alpha=1, beta=1)

def ref_fn(x, y, alpha=1, beta=1):
    return alpha * x + beta * y

class TestPythonJiterator(TestCase):
    @parametrize("shape_strides", [
        (([3, 3], [3, 1]), ([3, 3], [3, 1])),  # contiguous
    ])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16),
                     all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_contiguous(self, device, dtypes, shape_strides):
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])

        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])

        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        self.assertEqual(expected, result)

    # See https://github.com/pytorch/pytorch/pull/76394#issuecomment-1118018287 for details
    # On cuda 11.3, nvrtcCompileProgram is taking too long to
    # compile jiterator generated kernels for non-contiguous input that requires dynamic-casting.
    @skipCUDAIfVersionLessThan((11, 6))
    @parametrize("shape_strides", [
        (([3, 3], [1, 3]), ([3, 1], [1, 3])),  # non-contiguous
    ])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16),
                     all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_noncontiguous(self, device, dtypes, shape_strides):
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])

        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])

        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        self.assertEqual(expected, result)

    @dtypes(torch.float, torch.double, torch.float16, torch.bfloat16)
    @parametrize("alpha", [-1, 2.0, None])
    @parametrize("beta", [3, -4.2, None])
    @toleranceOverride({torch.float16 : tol(atol=1e-2, rtol=1e-3)})
    def test_extra_args(self, device, dtype, alpha, beta):
        a = torch.rand(3, device=device).mul(10).type(dtype)
        b = torch.rand(3, device=device).mul(10).type(dtype)

        extra_args = {}
        if alpha is not None:
            extra_args["alpha"] = alpha
        if beta is not None:
            extra_args["beta"] = beta

        expected = ref_fn(a, b, **extra_args)
        result = jitted_fn(a, b, **extra_args)

        self.assertEqual(expected, result)

    @parametrize("is_train", [True, False])
    def test_bool_extra_args(self, device, is_train):
        code_string = "template <typename T> T conditional(T x, T mask, bool is_train) { return is_train ? x * mask : x; }"
        jitted_fn = create_jit_fn(code_string, is_train=False)

        def ref_fn(x, mask, is_train):
            return x * mask if is_train else x

        a = torch.rand(3, device=device)
        b = torch.rand(3, device=device)

        expected = ref_fn(a, b, is_train=is_train)
        result = jitted_fn(a, b, is_train=is_train)
        self.assertEqual(expected, result)

    def test_multiple_functors(self, device):
        code_string = '''
        template <typename T> T fn(T x, T mask) { return x * mask; }
        template <typename T> T main_fn(T x, T mask, T y) { return fn(x, mask) + y; }
        '''
        jitted_fn = create_jit_fn(code_string)

        def ref_fn(x, mask, y):
            return x * mask + y

        a = torch.rand(3, device=device)
        b = torch.rand(3, device=device)
        c = torch.rand(3, device=device)

        expected = ref_fn(a, b, c)
        result = jitted_fn(a, b, c)
        self.assertEqual(expected, result)

    @parametrize("num_inputs", [1, 5, 8])
    def test_various_num_inputs(self, num_inputs):
        inputs = []
        for _ in range(num_inputs):
            inputs.append(torch.rand(3, device='cuda').mul(10))

        input_string = ",".join([f"T i{i}" for i in range(num_inputs)])
        function_body = "+".join([f"i{i}" for i in range(num_inputs)])
        code_string = f"template <typename T> T my_kernel({input_string}) {{ return {function_body}; }}"
        jitted_fn = create_jit_fn(code_string)

        def ref_fn(*inputs):
            return torch.sum(torch.stack(inputs), dim=0)

        expected = ref_fn(*inputs)
        result = jitted_fn(*inputs)

        self.assertEqual(expected, result)

    @parametrize("num_outputs", [1, 4, 8])
    def test_various_num_outputs(self, num_outputs):
        input = torch.rand(3, device='cuda')

        output_string = ", ".join([f"T& out{i}" for i in range(num_outputs)])
        function_body = ""
        for i in range(num_outputs):
            function_body += f"out{i} = input + {i};\n"
        # NB: return type must be void, otherwise ROCm silently fails
        code_string = f"template <typename T> void my_kernel(T input, {output_string}) {{ {function_body} }}"

        jitted_fn = create_multi_output_jit_fn(code_string, num_outputs)

        def ref_fn(input):
            outputs = []
            for i in range(num_outputs):
                outputs.append(input + i)

            if num_outputs == 1:
                return outputs[0]
            return tuple(outputs)

        expected = ref_fn(input)
        result = jitted_fn(input)

        for i in range(num_outputs):
            self.assertEqual(expected[i], result[i])

    @parametrize("code_string", [
        "template <typename T> T my _kernel(T x) { return x; }",
        "template <typename T> Tmy_kernel(T x) { return x; }",
    ])
    def test_invalid_function_name(self, code_string):
        with self.assertRaises(Exception):
            create_jit_fn(code_string)


instantiate_device_type_tests(TestPythonJiterator, globals(), only_for="cuda")

if __name__ == '__main__':
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPythonJiterator`

**Functions defined**: `ref_fn`, `test_all_dtype_contiguous`, `test_all_dtype_noncontiguous`, `test_extra_args`, `test_bool_extra_args`, `ref_fn`, `test_multiple_functors`, `ref_fn`, `test_various_num_inputs`, `ref_fn`, `test_various_num_outputs`, `ref_fn`, `test_invalid_function_name`

**Key imports**: torch, _create_jit_fn as create_jit_fn, _create_multi_output_jit_fn as create_multi_output_jit_fn, sys, product, TestCase, parametrize, run_tests, TEST_CUDA, NoTest, all_types_and_complex_and


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.cuda.jiterator`: _create_jit_fn as create_jit_fn
- `sys`
- `itertools`: product
- `torch.testing._internal.common_utils`: TestCase, parametrize, run_tests, TEST_CUDA, NoTest
- `torch.testing._internal.common_dtype`: all_types_and_complex_and


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/test_jiterator.py
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

- **File Documentation**: `test_jiterator.py_docs.md`
- **Keyword Index**: `test_jiterator.py_kw.md`
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

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/test_jiterator.py_docs.md
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

- **File Documentation**: `test_jiterator.py_docs.md_docs.md`
- **Keyword Index**: `test_jiterator.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
