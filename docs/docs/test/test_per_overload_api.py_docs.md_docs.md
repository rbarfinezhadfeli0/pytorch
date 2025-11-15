# Documentation: `docs/test/test_per_overload_api.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_per_overload_api.py_docs.md`
- **Size**: 5,241 bytes (5.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_per_overload_api.py`

## File Metadata

- **Path**: `test/test_per_overload_api.py`
- **Size**: 2,573 bytes (2.51 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: unknown"]
import copy

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPerOverloadAPI(TestCase):
    def test_basics_opoverloadpacket(self):
        # add is only used as an example here. It is ok to update the test
        # if the semantics of add are modified in the future.
        add_packet = torch.ops.aten.add

        # class attributes
        self.assertEqual(add_packet.__name__, "add")
        self.assertEqual(str(add_packet), "aten.add")

        # callable
        self.assertEqual(add_packet(torch.tensor(2), torch.tensor(3)), torch.tensor(5))

        # correct module
        self.assertEqual(add_packet.__module__, add_packet.op.__module__)

        # caching
        another_add_packet = torch.ops.aten.add
        self.assertEqual(id(add_packet), id(another_add_packet))

        # deepcopy is a no-op
        self.assertEqual(id(add_packet), id(copy.deepcopy(add_packet)))

        # pretty print
        self.assertEqual(repr(add_packet), "<OpOverloadPacket(op='aten.add')>")

        self.assertRaises(AttributeError, lambda: add_packet.foo)

    def test_basics_opoverload(self):
        add_packet = torch.ops.aten.add
        add_tensoroverload = add_packet.Tensor

        # class attributes
        self.assertEqual(str(add_tensoroverload), "aten.add.Tensor")
        self.assertEqual(add_tensoroverload.__name__, "add.Tensor")
        self.assertEqual(add_tensoroverload.overloadpacket, add_packet)

        # deepcopy is a no-op
        self.assertEqual(id(add_tensoroverload), id(copy.deepcopy(add_tensoroverload)))

        # caching
        another_add_tensoroverload = torch.ops.aten.add.Tensor
        self.assertEqual(id(add_tensoroverload), id(another_add_tensoroverload))

        # pretty print
        self.assertEqual(
            repr(add_tensoroverload), "<OpOverload(op='aten.add', overload='Tensor')>"
        )

        # callable
        self.assertEqual(
            add_tensoroverload(torch.tensor(2), torch.tensor(3)), torch.tensor(5)
        )

        a = torch.tensor(2)
        b = torch.tensor(0)
        torch.ops.aten.add.out(a, a, out=b)
        self.assertEqual(b, torch.tensor(4))

        self.assertRaises(RuntimeError, lambda: add_tensoroverload(a, a, out=b))

    def test_decompose(self):
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        self.assertEqual(
            torch.ops.aten.linear.default.decompose(x, y),
            torch.ops.aten.linear.default(x, y),
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPerOverloadAPI`

**Functions defined**: `test_basics_opoverloadpacket`, `test_basics_opoverload`, `test_decompose`

**Key imports**: copy, torch, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `torch`
- `torch.testing._internal.common_utils`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/test_per_overload_api.py
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

- **File Documentation**: `test_per_overload_api.py_docs.md`
- **Keyword Index**: `test_per_overload_api.py_kw.md`
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
python docs/test/test_per_overload_api.py_docs.md
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

- **File Documentation**: `test_per_overload_api.py_docs.md_docs.md`
- **Keyword Index**: `test_per_overload_api.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
