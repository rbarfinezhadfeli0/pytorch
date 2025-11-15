# Documentation: `docs/test/test_content_store.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_content_store.py_docs.md`
- **Size**: 7,869 bytes (7.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_content_store.py`

## File Metadata

- **Path**: `test/test_content_store.py`
- **Size**: 4,902 bytes (4.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: pt2"]

import torch
from torch._prims.debug_prims import load_tensor_reader
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    run_tests,
    TemporaryDirectoryName,
    TestCase,
)
from torch.utils._content_store import (
    ContentStoreReader,
    ContentStoreWriter,
    hash_storage,
)


class TestContentStore(TestCase):
    def test_basic(self, device):
        # setup test data
        x = torch.randn(4, device=device)
        y = torch.randn(6, device=device)
        z = x.view(2, 2)
        # start writing
        with TemporaryDirectoryName() as loc:
            writer = ContentStoreWriter(loc)
            writer.write_tensor("x", x)
            writer.write_tensor("y", y)
            writer.write_tensor("z", z)
            # do some mutation that is VC UNTRACKED
            x.data.add_(1)
            writer.write_tensor("x2", x)
            writer.write_tensor("y2", y)
            writer.write_tensor("z2", z)
            del writer

            reader = ContentStoreReader(loc)
            n_x = reader.read_tensor("x")
            n_y = reader.read_tensor("y")
            n_z = reader.read_tensor("z")
            self.assertEqual(n_x + 1, x)
            self.assertEqual(n_y, y)
            self.assertEqual(n_z + 1, z)
            self.assertEqual(
                StorageWeakRef(n_x.untyped_storage()),
                StorageWeakRef(n_z.untyped_storage()),
            )
            n_x2 = reader.read_tensor("x2")
            n_y2 = reader.read_tensor("y2")
            n_z2 = reader.read_tensor("z2")
            self.assertEqual(n_x2, x)
            self.assertEqual(n_y2, y)
            self.assertEqual(n_z2, z)
            self.assertEqual(
                StorageWeakRef(n_y2.untyped_storage()),
                StorageWeakRef(n_y.untyped_storage()),
            )

    def test_scalar(self, device):
        # Should not raise an error
        hash_storage(torch.tensor(2, device=device).untyped_storage())

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_repeated_hash(self, device):
        # Test that repeated hashing doesn't trigger a recompile in dynamo
        # If it does, we will execute prims.xor_sum in eager which fails
        for _ in range(4):
            hash_storage(torch.tensor(2, device=device).untyped_storage())

    def test_load_tensor(self, device):
        with TemporaryDirectoryName() as loc:
            writer = ContentStoreWriter(loc)
            x = torch.randn(4, device=device)

            def same_meta_as_x(t):
                self.assertEqual(t.size(), x.size())
                self.assertEqual(t.stride(), x.stride())
                self.assertEqual(t.dtype, x.dtype)
                self.assertEqual(t.device, x.device)

            writer.write_tensor("x", x)

            with load_tensor_reader(loc):
                x2 = torch.ops.debugprims.load_tensor.default(
                    "x", (4,), (1,), dtype=torch.float32, device=device
                )
                self.assertEqual(x, x2)
                x3 = torch.ops.debugprims.load_tensor.default(
                    "x", (4,), (1,), dtype=torch.float32, device=device
                )
                self.assertEqual(x, x3)
                # Must not alias!
                self.assertNotEqual(
                    StorageWeakRef(x.untyped_storage()),
                    StorageWeakRef(x2.untyped_storage()),
                )
                self.assertNotEqual(
                    StorageWeakRef(x2.untyped_storage()),
                    StorageWeakRef(x3.untyped_storage()),
                )

                # Check fake tensor mode works too
                with FakeTensorMode():
                    x4 = torch.ops.debugprims.load_tensor.default(
                        "x", (4,), (1,), dtype=torch.float32, device=device
                    )
                    self.assertIsInstance(x4, FakeTensor)
                    same_meta_as_x(x4)

                # Check fp64 works on non-MPS platforms, since MPS doesn't currently
                # support fp64.
                if not device.startswith("mps"):
                    x5 = torch.ops.debugprims.load_tensor.default(
                        "x", (4,), (1,), dtype=torch.float64, device=device
                    )
                    self.assertEqual(x5.float(), x)
                    self.assertEqual(x5.dtype, torch.float64)

        x6 = torch.ops.debugprims.load_tensor.default(
            "x", (4,), (1,), dtype=torch.float32, device=device
        )
        same_meta_as_x(x6)


instantiate_device_type_tests(
    TestContentStore, globals(), allow_mps=True, allow_xpu=True
)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestContentStore`

**Functions defined**: `test_basic`, `test_scalar`, `test_repeated_hash`, `test_load_tensor`, `same_meta_as_x`

**Key imports**: torch, load_tensor_reader, FakeTensor, FakeTensorMode, StorageWeakRef, instantiate_device_type_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._prims.debug_prims`: load_tensor_reader
- `torch._subclasses.fake_tensor`: FakeTensor, FakeTensorMode
- `torch.multiprocessing.reductions`: StorageWeakRef
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/test_content_store.py
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

- **File Documentation**: `test_content_store.py_docs.md`
- **Keyword Index**: `test_content_store.py_kw.md`
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
python docs/test/test_content_store.py_docs.md
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

- **File Documentation**: `test_content_store.py_docs.md_docs.md`
- **Keyword Index**: `test_content_store.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
