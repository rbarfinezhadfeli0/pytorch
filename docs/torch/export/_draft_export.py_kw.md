# Keyword Index: `torch/export/_draft_export.py`

## File Information

- **Original File**: [torch/export/_draft_export.py](../../../torch/export/_draft_export.py)
- **Documentation**: [`_draft_export.py_docs.md`](./_draft_export.py_docs.md)
- **Folder**: `torch/export`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CaptureStructuredTrace`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`DraftExportReport`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`FailureReport`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`FailureType`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`LogRecord`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`class`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`from`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)

### Functions

- **`__enter__`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`__exit__`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`__init__`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`__repr__`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`__str__`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`_hash`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`_log_expression_created`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`apply_suggested_fixes`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`convert_dim_to_auto`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`draft_export`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`emit`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`get_loc`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`get_log_count`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`prettify_frame_locals`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`prettify_stack`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`print`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`successful`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`try_add`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)

### Imports

- **`._trace`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`.dynamic_shapes`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`.exported_program`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`Any`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`Callable`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`ExportedProgram`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`IntEnum`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`UserError`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`_DimHint`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`_export`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`collections.abc`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`dataclass`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`dataclasses`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`enum`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`getpass`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`json`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`log_draft_export_usage`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`logging`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`os`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`re`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`tempfile`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`time`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`torch`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`torch._dynamo.exc`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`torch._export.passes.insert_custom_op_guards`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`torch._logging._internal`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`torch._utils_internal`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`torch.utils._pytree`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)
- **`typing`**: [_draft_export.py_docs.md](./_draft_export.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
