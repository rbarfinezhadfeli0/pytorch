# Documentation: `test/export/opinfo_schema.py`

## File Metadata

- **Path**: `test/export/opinfo_schema.py`
- **Size**: 4,020 bytes (3.93 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: export"]

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase
from torch.utils._pytree import tree_map


# Simplified naming for C++ classes
SchemaArgument = torch._C._SchemaArgument
SchemaArgType = torch._C._SchemaArgType
SchemaInfo = torch._C._SchemaInfo

test_classes = {}


class PreDispatchSchemaCheckMode(SchemaCheckMode):
    """
    Dispatch mode built on top of SchemaCheckMode that checks for incorrect op schemas
    for PreDispatch IR. This is meant to run ops in eager mode on concrete inputs, to
    see if they incorrectly claim to be functional (aliasing or mutating).

    If an op is claimed to be functional and either is detected, an error is raised.
    Errors will be silenced if the schema admits aliasing or mutation - the op may
    later decompose and become functional.
    """

    def __init__(self) -> None:
        self._dispatch_key = torch._C.DispatchKey.PreDispatch
        super().__init__()

    def _may_alias_or_mutate(self, func, types, args, kwargs):
        def unwrap(e):
            if isinstance(e, torch.Tensor) and type(e) is not torch.Tensor:
                try:
                    return e.elem
                except AttributeError:
                    return e
            return e

        # get arguments, outputs
        schema_info = SchemaInfo(func._schema)
        pre_arguments = normalize_function(
            func, args, kwargs, normalize_to_only_use_kwargs=True
        ).kwargs
        schema_info.add_argument_values(pre_arguments)
        out = func(*args, **kwargs)
        tuple_out = out if isinstance(out, tuple) else (out,)
        tuple_out = tree_map(unwrap, tuple_out)

        # check schema
        for i in range(len(func._schema.arguments)):
            for j in range(len(tuple_out)):
                if schema_info.may_contain_alias(
                    SchemaArgument(SchemaArgType.output, j),
                    SchemaArgument(SchemaArgType.input, i),
                ):
                    return True
            if schema_info.is_mutable(
                SchemaArgument(SchemaArgType.input, i),
            ):
                return True

        return False

    # creating this just so we have access to the offending op
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            return super().__torch_dispatch__(func, types, args=args, kwargs=kwargs)
        except RuntimeError as e:
            # check if schema claims to be either aliasing or mutating
            alias_or_mutate = self._may_alias_or_mutate(func, types, args, kwargs)
            if (
                not alias_or_mutate
            ):  # if schema is aliasing or mutating, will decompose further
                msg = e.args[0]
                e.args = (
                    f"""SchemaCheckMode failed with the following error on op <{func}>, meaning
    this op contains aliasing or mutations, despite claiming to be functional:\n\n"""
                    + msg,
                )
                raise e


class TestOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float, torch.int))
    def test_schema_check_op(self, device, dtype, op):
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        inputs = next(sample_inputs_itr)
        args = [inputs.input] + list(inputs.args)
        kwargs = inputs.kwargs
        with enable_python_dispatcher():
            with PreDispatchSchemaCheckMode():
                op.op(*args, **kwargs)


instantiate_device_type_tests(TestOpInfo, globals())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview

"""    Dispatch mode built on top of SchemaCheckMode that checks for incorrect op schemas    for PreDispatch IR. This is meant to run ops in eager mode on concrete inputs, to    see if they incorrectly claim to be functional (aliasing or mutating).    If an op is claimed to be functional and either is detected, an error is raised.    Errors will be silenced if the schema admits aliasing or mutation - the op may    later decompose and become functional.

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PreDispatchSchemaCheckMode`, `TestOpInfo`

**Functions defined**: `__init__`, `_may_alias_or_mutate`, `unwrap`, `__torch_dispatch__`, `test_schema_check_op`

**Key imports**: torch, enable_python_dispatcher, SchemaCheckMode, normalize_function, op_db, TestCase, tree_map, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch._subclasses.schema_check_mode`: SchemaCheckMode
- `torch.fx.operator_schemas`: normalize_function
- `torch.testing._internal.common_methods_invocations`: op_db
- `torch.testing._internal.common_utils`: TestCase
- `torch.utils._pytree`: tree_map
- `torch._dynamo.test_case`: run_tests


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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
python test/export/opinfo_schema.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_schema.py_docs.md`](./test_schema.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)
- [`test_export.py_docs.md`](./test_export.py_docs.md)


## Cross-References

- **File Documentation**: `opinfo_schema.py_docs.md`
- **Keyword Index**: `opinfo_schema.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
