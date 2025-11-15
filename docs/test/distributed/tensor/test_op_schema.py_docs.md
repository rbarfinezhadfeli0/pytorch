# Documentation: `test/distributed/tensor/test_op_schema.py`

## File Metadata

- **Path**: `test/distributed/tensor/test_op_schema.py`
- **Size**: 4,641 bytes (4.53 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import random

from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, RuntimeSchemaInfo
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOpSchema(TestCase):
    def test_equality_checks_lists_of_dtensor_spec(self):
        """If x == y, then we must have h(x) == h(y)."""
        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        schema1 = OpSchema(op=None, args_schema=(dts, [dts]), kwargs_schema={})
        schema2 = OpSchema(op=None, args_schema=(dts, [dts, dts]), kwargs_schema={})
        # This is a regression test; these schemas used to compare equal.
        self.assertNotEqual(schema1, schema2)
        self.assertNotEqual(hash(schema1), hash(schema2))

    def test_equality_respects_static_attributes(self):
        def _get_sample_op_schemas(static_arg_val, static_kwarg_val):
            dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
            static_argnum = 2
            static_kwargkey = ["statickwarg"]
            annotated_schemas = [
                (False, False, None),
                (True, False, RuntimeSchemaInfo(static_argnum=static_argnum)),
                (False, True, RuntimeSchemaInfo(static_kwargkey=static_kwargkey)),
                (
                    True,
                    True,
                    RuntimeSchemaInfo(
                        static_argnum=static_argnum, static_kwargkey=static_kwargkey
                    ),
                ),
            ]

            # non-tensor args show up in hash iff the argnum is static/
            # kwargs show up in hash iff their name is in static_kwargkey.
            # random elements are random because they are not supposed to matter for
            # equality at all.
            args_schema = (dts, random.randint(1, 1000000), static_arg_val)
            kwargs_schema = {
                "ignoredkwarg": random.randint(1, 1000000),
                "statickwarg": static_kwarg_val,
            }
            return [
                (
                    has_static_arg,
                    has_static_kwarg,
                    OpSchema(
                        op=None,
                        args_schema=args_schema,
                        kwargs_schema=kwargs_schema,
                        schema_info=si,
                    ),
                )
                for (has_static_arg, has_static_kwarg, si) in annotated_schemas
            ]

        for lhs_has_static_arg, lhs_has_static_kwarg, lhs in _get_sample_op_schemas(
            1, 2
        ):
            # Static arg/kwarg both match
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                1, 2
            ):
                if (
                    lhs_has_static_arg == rhs_has_static_arg
                    and lhs_has_static_kwarg == rhs_has_static_kwarg
                ):
                    self.assertEqual(lhs, rhs)
                else:
                    self.assertNotEqual(lhs, rhs)

            # Static arg mismatch
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                3, 2
            ):
                if (
                    lhs_has_static_arg
                    or rhs_has_static_arg
                    or lhs_has_static_kwarg != rhs_has_static_kwarg
                ):
                    self.assertNotEqual(lhs, rhs)
                else:
                    self.assertEqual(lhs, rhs)

            # Static kwarg mismatch
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                1, 3
            ):
                if (
                    lhs_has_static_kwarg
                    or rhs_has_static_kwarg
                    or lhs_has_static_arg != rhs_has_static_arg
                ):
                    self.assertNotEqual(lhs, rhs)
                else:
                    self.assertEqual(lhs, rhs)

            # Static arg/kwarg both mismatch
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                3, 4
            ):
                if (
                    lhs_has_static_arg
                    or rhs_has_static_arg
                    or lhs_has_static_kwarg
                    or rhs_has_static_kwarg
                ):
                    self.assertNotEqual(lhs, rhs)
                else:
                    self.assertEqual(lhs, rhs)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""If x == y, then we must have h(x) == h(y)."""        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)        schema1 = OpSchema(op=None, args_schema=(dts, [dts]), kwargs_schema={})        schema2 = OpSchema(op=None, args_schema=(dts, [dts, dts]), kwargs_schema={})        # This is a regression test; these schemas used to compare equal.        self.assertNotEqual(schema1, schema2)        self.assertNotEqual(hash(schema1), hash(schema2))    def test_equality_respects_static_attributes(self):        def _get_sample_op_schemas(static_arg_val, static_kwarg_val):            dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)            static_argnum = 2            static_kwargkey = ["statickwarg"]            annotated_schemas = [                (False, False, None),                (True, False, RuntimeSchemaInfo(static_argnum=static_argnum)),                (False, True, RuntimeSchemaInfo(static_kwargkey=static_kwargkey)),                (                    True,                    True,                    RuntimeSchemaInfo(                        static_argnum=static_argnum, static_kwargkey=static_kwargkey                    ),                ),            ]            # non-tensor args show up in hash iff the argnum is static/            # kwargs show up in hash iff their name is in static_kwargkey.            # random elements are random because they are not supposed to matter for            # equality at all.            args_schema = (dts, random.randint(1, 1000000), static_arg_val)            kwargs_schema = {                "ignoredkwarg": random.randint(1, 1000000),                "statickwarg": static_kwarg_val,            }            return [                (                    has_static_arg,

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestOpSchema`

**Functions defined**: `test_equality_checks_lists_of_dtensor_spec`, `test_equality_respects_static_attributes`, `_get_sample_op_schemas`

**Key imports**: random, DTensorSpec, OpSchema, RuntimeSchemaInfo, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `random`
- `torch.distributed.tensor._dtensor_spec`: DTensorSpec
- `torch.distributed.tensor._op_schema`: OpSchema, RuntimeSchemaInfo
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
python test/distributed/tensor/test_op_schema.py
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
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_attention.py_docs.md`](./test_attention.py_docs.md)


## Cross-References

- **File Documentation**: `test_op_schema.py_docs.md`
- **Keyword Index**: `test_op_schema.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
