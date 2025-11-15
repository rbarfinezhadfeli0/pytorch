# Documentation: `test/functorch/xfail_suggester.py`

## File Metadata

- **Path**: `test/functorch/xfail_suggester.py`
- **Size**: 3,709 bytes (3.62 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
import re

import torch


"""
Instructions:

1. pytest -n 8 test/test_vmap.py test/test_ops.py test/test_aotdispatch.py > result.txt
2. python test/xfail_suggester.py
"""

with open("result.txt") as f:
    lines = f.readlines()

failed = [line for line in lines if line.startswith("FAILED")]
p = re.compile("FAILED test/test_\w+.py::\w+::(\S+)")  # noqa: W605


def get_failed_test(line):
    m = p.match(line)
    if m is None:
        return None
    return m.group(1)


base_names = {
    "test_grad_",
    "test_vjp_",
    "test_vmapvjp_",
    "test_vmapvjp_has_batch_rule_",
    "test_vjpvmap_",
    "test_jvp_",
    "test_vmapjvp_",
    "test_vmapjvpall_has_batch_rule_",
    "test_vmapjvpall_",
    "test_jvpvjp_",
    "test_vjpvjp_",
    "test_decomposition_",
    "test_make_fx_exhaustive_",
    "test_vmap_exhaustive_",
    "test_op_has_batch_rule_",
    "test_vmap_autograd_grad_",
}

failed_tests = [get_failed_test(line) for line in lines]
failed_tests = [match for match in failed_tests if match is not None]
failed_tests = sorted(failed_tests)

suggested_xfails = {}


def remove_device_dtype(test):
    return "_".join(test.split("_")[:-2])


def belongs_to_base(test, base):
    if not test.startswith(base):
        return False
    candidates = [try_base for try_base in base_names if len(try_base) > len(base)]
    for candidate in candidates:
        if test.startswith(candidate):
            return False
    return True


def parse_namespace(base):
    mappings = {
        "nn_functional_": "nn.functional",
        "fft_": "fft",
        "linalg_": "linalg",
        "_masked_": "_masked",
        "sparse_": "sparse",
        "special_": "special",
    }
    for heading in mappings:
        if base.startswith(heading):
            return mappings[heading], base[len(heading) :]
    return None, base


def get_torch_module(namespace):
    if namespace is None:
        return torch
    if namespace == "nn.functional":
        return torch.nn.functional
    return getattr(torch, namespace)


def parse_base(base):
    namespace, rest = parse_namespace(base)

    apis = dir(get_torch_module(namespace))
    apis = sorted(apis, key=lambda x: -len(x))

    api = rest
    variant = ""
    for candidate in apis:
        if rest.startswith(candidate):
            api = candidate
            variant = rest[len(candidate) + 1 :]
            break
    print(base, namespace, api, variant)
    return namespace, api, variant


def any_starts_with(strs, thing):
    for s in strs:
        if s.startswith(thing):
            return True
    return False


def get_suggested_xfails(base, tests):
    result = []
    tests = [test[len(base) :] for test in tests if belongs_to_base(test, base)]

    base_tests = {remove_device_dtype(test) for test in tests}
    tests = set(tests)
    for base in base_tests:
        cpu_variant = base + "_cpu_float32"
        cuda_variant = base + "_cuda_float32"
        namespace, api, variant = parse_base(base)
        if namespace is not None:
            api = f"{namespace}.{api}"
        if cpu_variant in tests and cuda_variant in tests:
            result.append(f"xfail('{api}', '{variant}'),")
            continue
        if cpu_variant in tests:
            result.append(f"xfail('{api}', '{variant}', device_type='cpu'),")
            continue
        if cuda_variant in tests:
            result.append(f"xfail('{api}', '{variant}', device_type='cuda'),")
            continue
        result.append(f"skip('{api}', '{variant}',")
    return result


result = {base: get_suggested_xfails(base, failed_tests) for base in base_names}
for k, v in result.items():
    print("=" * 50)
    print(k)
    print("=" * 50)
    print("\n".join(v))

```



## High-Level Overview

"""Instructions:1. pytest -n 8 test/test_vmap.py test/test_ops.py test/test_aotdispatch.py > result.txt2. python test/xfail_suggester.py

This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_failed_test`, `remove_device_dtype`, `belongs_to_base`, `parse_namespace`, `get_torch_module`, `parse_base`, `any_starts_with`, `get_suggested_xfails`

**Key imports**: re, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `re`
- `torch`


## Code Patterns & Idioms

### Common Patterns

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
python test/functorch/xfail_suggester.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/functorch`):

- [`test_vmap.py_docs.md`](./test_vmap.py_docs.md)
- [`test_rearrange.py_docs.md`](./test_rearrange.py_docs.md)
- [`test_aot_joint_with_descriptors.py_docs.md`](./test_aot_joint_with_descriptors.py_docs.md)
- [`functorch_additional_op_db.py_docs.md`](./functorch_additional_op_db.py_docs.md)
- [`discover_coverage.py_docs.md`](./discover_coverage.py_docs.md)
- [`test_eager_transforms.py_docs.md`](./test_eager_transforms.py_docs.md)
- [`test_ac.py_docs.md`](./test_ac.py_docs.md)
- [`common_utils.py_docs.md`](./common_utils.py_docs.md)
- [`test_logging.py_docs.md`](./test_logging.py_docs.md)


## Cross-References

- **File Documentation**: `xfail_suggester.py_docs.md`
- **Keyword Index**: `xfail_suggester.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
