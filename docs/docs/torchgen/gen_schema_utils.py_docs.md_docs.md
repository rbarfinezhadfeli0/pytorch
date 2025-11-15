# Documentation: `docs/torchgen/gen_schema_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/gen_schema_utils.py_docs.md`
- **Size**: 5,656 bytes (5.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/gen_schema_utils.py`

## File Metadata

- **Path**: `torchgen/gen_schema_utils.py`
- **Size**: 3,317 bytes (3.24 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any, Optional, Union

from torchgen.model import (
    Annotation,
    Argument,
    Arguments,
    BaseOperatorName,
    BaseTy,
    BaseType,
    CustomClassType,
    FunctionSchema,
    ListType,
    OperatorName,
    Return,
)


# Note: These aren't actually used in torchgen, they're some utilities for generating a schema
# from real arguments. For example, this is used to generate HigherOrderOperators' schema since
# their schemas can vary for different instances of the same HOP.


class TypeGen:
    convert_to_base_ty = {
        int: BaseTy.int,
        float: BaseTy.float,
        str: BaseTy.str,
        bool: BaseTy.bool,
    }

    @staticmethod
    def from_example(obj: Any) -> Union[BaseType, ListType, CustomClassType]:
        import torch

        if isinstance(obj, torch.fx.GraphModule):
            return BaseType(BaseTy.GraphModule)
        elif isinstance(obj, torch.Tensor):
            return BaseType(BaseTy.Tensor)
        elif isinstance(obj, torch.SymInt):
            return BaseType(BaseTy.SymInt)
        elif isinstance(obj, torch.SymBool):
            return BaseType(BaseTy.SymBool)
        elif isinstance(obj, torch.ScriptObject):
            return CustomClassType(obj._type().name())  # type: ignore[attr-defined]
        elif isinstance(obj, (list, tuple)):
            assert len(obj) > 0
            all_base_tys = [TypeGen.from_example(x) for x in obj]
            if len(set(all_base_tys)) > 1:
                raise RuntimeError(
                    f"Cannot generate schema for a sequence of args of heterogeneous types: {all_base_tys}. "
                    "Consider unpacking the argument and give proper names to them if possible "
                    "instead of using *args."
                )
            return ListType(all_base_tys[0], len(obj))
        tp = type(obj)
        if tp not in TypeGen.convert_to_base_ty:
            raise RuntimeError(f"unsupported type {tp}")
        return BaseType(TypeGen.convert_to_base_ty[tp])


class ReturnGen:
    @staticmethod
    def from_example(
        name: Optional[str], obj: Any, annotation: Optional[Annotation]
    ) -> Return:
        return Return(name, TypeGen.from_example(obj), annotation)


class ArgumentGen:
    @staticmethod
    def from_example(
        name: str, obj: Any, default: Optional[str], annotation: Optional[Annotation]
    ) -> Argument:
        return Argument(
            name, TypeGen.from_example(obj), default=default, annotation=annotation
        )


class FunctionSchemaGen:
    @staticmethod
    def from_example(
        op_name: str,
        example_inputs: tuple[tuple[str, Any], ...],
        example_outputs: tuple[Any, ...],
    ) -> FunctionSchema:
        args = []
        for name, inp in example_inputs:
            args.append(ArgumentGen.from_example(name, inp, None, None))
        # ignore the annotations and other attributes for now, we could add more when needed.
        arguments = Arguments(
            tuple(), None, tuple(args), tuple(), None, tuple(), tuple()
        )
        returns = tuple(
            ReturnGen.from_example(None, out, None) for out in example_outputs
        )
        op_name = OperatorName(BaseOperatorName(op_name, False, False, False), "")
        return FunctionSchema(op_name, arguments, returns)

```



## High-Level Overview


This Python file contains 4 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TypeGen`, `ReturnGen`, `ArgumentGen`, `FunctionSchemaGen`

**Functions defined**: `from_example`, `from_example`, `from_example`, `from_example`

**Key imports**: Any, Optional, Union, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, Optional, Union
- `torch`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torchgen`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`gen_backend_stubs.py_docs.md`](./gen_backend_stubs.py_docs.md)
- [`local.py_docs.md`](./local.py_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`model.py_docs.md`](./model.py_docs.md)
- [`yaml_utils.py_docs.md`](./yaml_utils.py_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`gen.py_docs.md`](./gen.py_docs.md)


## Cross-References

- **File Documentation**: `gen_schema_utils.py_docs.md`
- **Keyword Index**: `gen_schema_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torchgen`):

- [`gen_functionalization_type.py_docs.md_docs.md`](./gen_functionalization_type.py_docs.md_docs.md)
- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`context.py_kw.md_docs.md`](./context.py_kw.md_docs.md)
- [`native_function_generation.py_kw.md_docs.md`](./native_function_generation.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`gen_aoti_c_shim.py_docs.md_docs.md`](./gen_aoti_c_shim.py_docs.md_docs.md)
- [`local.py_docs.md_docs.md`](./local.py_docs.md_docs.md)
- [`gen.py_kw.md_docs.md`](./gen.py_kw.md_docs.md)
- [`gen_aoti_c_shim.py_kw.md_docs.md`](./gen_aoti_c_shim.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `gen_schema_utils.py_docs.md_docs.md`
- **Keyword Index**: `gen_schema_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
