# Documentation: `docs/test/distributed/tensor/test_placement_types.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/tensor/test_placement_types.py_docs.md`
- **Size**: 6,228 bytes (6.08 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/tensor/test_placement_types.py`

## File Metadata

- **Path**: `test/distributed/tensor/test_placement_types.py`
- **Size**: 3,201 bytes (3.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
import copy
import itertools
import sys
import unittest

from torch._dynamo.variables.distributed import PlacementClassVariable
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# Basic functionality test for Placement types.
class PlacementTypesTestCase(TestCase):
    def test_type_identification(self):
        shard = Shard(3)
        strided_shard = _StridedShard(dim=3, split_factor=7)
        partial_sum = Partial("sum")
        partial_max = Partial("max")
        replicate = Replicate()

        ident_tests = (
            (shard, True, False, False),
            (strided_shard, True, False, False),
            (partial_sum, False, True, False),
            (partial_max, False, True, False),
            (replicate, False, False, True),
        )
        for do_deepcopy in (False, True):
            for placement, is_shard, is_partial, is_replicate in ident_tests:
                if do_deepcopy:
                    placement = copy.deepcopy(placement)
                self.assertEqual(placement.is_shard(), is_shard)
                self.assertEqual(placement.is_partial(), is_partial)
                self.assertEqual(placement.is_replicate(), is_replicate)

    def test_equality(self):
        equivalence_classes = (
            (Shard(3), _StridedShard(dim=3, split_factor=7)),
            (Shard(4), _StridedShard(dim=4, split_factor=9)),
            (Replicate(),),
            (Partial("sum"),),
            (Partial("max"),),
        )
        for eq_class in equivalence_classes:
            # Each item in the equivalence class should be equal to every other item in
            # its class.
            for lhs, rhs in itertools.product(eq_class, eq_class):
                self.assertEqual(lhs, rhs)

            # Each item in the equivalence class should not be equal to any item in any
            # other class.
            for other_class in equivalence_classes:
                if other_class is eq_class:
                    continue
                for lhs, rhs in itertools.product(eq_class, other_class):
                    self.assertNotEqual(lhs, rhs)

        # Testing this case doesn't seem to fit neatly into the above equivalence class
        # framework.
        self.assertNotEqual(
            _StridedShard(dim=3, split_factor=1), _StridedShard(dim=3, split_factor=2)
        )

    @unittest.skipIf(
        sys.version_info < (3, 10), "kw_only is only available in python >= 3.10"
    )
    def test_strided_shard_kwonly_argument(self):
        with self.assertRaises(TypeError):
            _StridedShard(3, 4)
        _StridedShard(3, split_factor=4)

    def test_strided_shard_isinstance_shard(self):
        assert isinstance(_StridedShard(dim=3, split_factor=7), Shard)

    def test_dynamo_can_identify_placement_classes(self):
        for cls in (Replicate, Shard, _StridedShard, Partial):
            self.assertTrue(
                PlacementClassVariable.is_placement_type(cls), msg=f"failed on {cls}"
            )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 6 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PlacementTypesTestCase`

**Functions defined**: `test_type_identification`, `test_equality`, `test_strided_shard_kwonly_argument`, `test_strided_shard_isinstance_shard`, `test_dynamo_can_identify_placement_classes`

**Key imports**: copy, itertools, sys, unittest, PlacementClassVariable, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `itertools`
- `sys`
- `unittest`
- `torch._dynamo.variables.distributed`: PlacementClassVariable
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
python test/distributed/tensor/test_placement_types.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/tensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_dtensor.py_docs.md`](./test_dtensor.py_docs.md)
- [`test_dtensor_testbase.py_docs.md`](./test_dtensor_testbase.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_dtensor_dispatch_overhead.py_docs.md`](./test_dtensor_dispatch_overhead.py_docs.md)
- [`test_tensor_ops.py_docs.md`](./test_tensor_ops.py_docs.md)
- [`test_matrix_ops.py_docs.md`](./test_matrix_ops.py_docs.md)
- [`test_op_schema.py_docs.md`](./test_op_schema.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_attention.py_docs.md`](./test_attention.py_docs.md)


## Cross-References

- **File Documentation**: `test_placement_types.py_docs.md`
- **Keyword Index**: `test_placement_types.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/tensor/test_placement_types.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/tensor`):

- [`test_math_ops.py_docs.md_docs.md`](./test_math_ops.py_docs.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_dtensor_export.py_docs.md_docs.md`](./test_dtensor_export.py_docs.md_docs.md)
- [`test_convolution_ops.py_kw.md_docs.md`](./test_convolution_ops.py_kw.md_docs.md)
- [`test_placement_types.py_kw.md_docs.md`](./test_placement_types.py_kw.md_docs.md)
- [`test_common_rules.py_kw.md_docs.md`](./test_common_rules.py_kw.md_docs.md)
- [`test_dtensor_compile.py_kw.md_docs.md`](./test_dtensor_compile.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_api.py_docs.md_docs.md`](./test_api.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_placement_types.py_docs.md_docs.md`
- **Keyword Index**: `test_placement_types.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
