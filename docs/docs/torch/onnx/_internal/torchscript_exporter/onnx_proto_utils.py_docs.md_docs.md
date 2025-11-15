# Documentation: `docs/torch/onnx/_internal/torchscript_exporter/onnx_proto_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/torchscript_exporter/onnx_proto_utils.py_docs.md`
- **Size**: 12,656 bytes (12.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/torchscript_exporter/onnx_proto_utils.py`

## File Metadata

- **Path**: `torch/onnx/_internal/torchscript_exporter/onnx_proto_utils.py`
- **Size**: 9,196 bytes (8.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# mypy: allow-untyped-defs
"""Utilities for manipulating the onnx and onnx-script dependencies and ONNX proto."""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any, TYPE_CHECKING

import torch
import torch.serialization
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import jit_utils, registration


if TYPE_CHECKING:
    import io
    from collections.abc import Mapping


def export_as_test_case(
    model_bytes: bytes, inputs_data, outputs_data, name: str, dir: str
) -> str:
    """Export an ONNX model as a self contained ONNX test case.

    The test case contains the model and the inputs/outputs data. The directory structure
    is as follows:

    dir
    \u251c\u2500\u2500 test_<name>
    \u2502   \u251c\u2500\u2500 model.onnx
    \u2502   \u2514\u2500\u2500 test_data_set_0
    \u2502       \u251c\u2500\u2500 input_0.pb
    \u2502       \u251c\u2500\u2500 input_1.pb
    \u2502       \u251c\u2500\u2500 output_0.pb
    \u2502       \u2514\u2500\u2500 output_1.pb

    Args:
        model_bytes: The ONNX model in bytes.
        inputs_data: The inputs data, nested data structure of numpy.ndarray.
        outputs_data: The outputs data, nested data structure of numpy.ndarray.

    Returns:
        The path to the test case directory.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "Export test case to ONNX format failed: Please install ONNX."
        ) from exc

    test_case_dir = os.path.join(dir, "test_" + name)
    os.makedirs(test_case_dir, exist_ok=True)
    _export_file(
        model_bytes,
        os.path.join(test_case_dir, "model.onnx"),
        {},
    )
    data_set_dir = os.path.join(test_case_dir, "test_data_set_0")
    if os.path.exists(data_set_dir):
        shutil.rmtree(data_set_dir)
    os.makedirs(data_set_dir)

    proto = onnx.load_model_from_string(model_bytes)  # type: ignore[attr-defined]

    for i, (input_proto, input) in enumerate(zip(proto.graph.input, inputs_data)):
        export_data(input, input_proto, os.path.join(data_set_dir, f"input_{i}.pb"))
    for i, (output_proto, output) in enumerate(zip(proto.graph.output, outputs_data)):
        export_data(output, output_proto, os.path.join(data_set_dir, f"output_{i}.pb"))

    return test_case_dir


def load_test_case(dir: str) -> tuple[bytes, Any, Any]:
    """Load a self contained ONNX test case from a directory.

    The test case must contain the model and the inputs/outputs data. The directory structure
    should be as follows:

    dir
    \u251c\u2500\u2500 test_<name>
    \u2502   \u251c\u2500\u2500 model.onnx
    \u2502   \u2514\u2500\u2500 test_data_set_0
    \u2502       \u251c\u2500\u2500 input_0.pb
    \u2502       \u251c\u2500\u2500 input_1.pb
    \u2502       \u251c\u2500\u2500 output_0.pb
    \u2502       \u2514\u2500\u2500 output_1.pb

    Args:
        dir: The directory containing the test case.

    Returns:
        model_bytes: The ONNX model in bytes.
        inputs: the inputs data, mapping from input name to numpy.ndarray.
        outputs: the outputs data, mapping from output name to numpy.ndarray.
    """
    try:
        import onnx
        from onnx import numpy_helper  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "Load test case from ONNX format failed: Please install ONNX."
        ) from exc

    with open(os.path.join(dir, "model.onnx"), "rb") as f:
        model_bytes = f.read()

    test_data_dir = os.path.join(dir, "test_data_set_0")

    inputs = {}
    input_files = glob.glob(os.path.join(test_data_dir, "input_*.pb"))
    for input_file in input_files:
        tensor = onnx.load_tensor(input_file)  # type: ignore[attr-defined]
        inputs[tensor.name] = numpy_helper.to_array(tensor)
    outputs = {}
    output_files = glob.glob(os.path.join(test_data_dir, "output_*.pb"))
    for output_file in output_files:
        tensor = onnx.load_tensor(output_file)  # type: ignore[attr-defined]
        outputs[tensor.name] = numpy_helper.to_array(tensor)

    return model_bytes, inputs, outputs


def export_data(data, value_info_proto, f: str) -> None:
    """Export data to ONNX protobuf format.

    Args:
        data: The data to export, nested data structure of numpy.ndarray.
        value_info_proto: The ValueInfoProto of the data. The type of the ValueInfoProto
            determines how the data is stored.
        f: The file to write the data to.
    """
    try:
        from onnx import numpy_helper  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "Export data to ONNX format failed: Please install ONNX."
        ) from exc

    with open(f, "wb") as opened_file:
        if value_info_proto.type.HasField("map_type"):
            opened_file.write(
                numpy_helper.from_dict(data, value_info_proto.name).SerializeToString()
            )
        elif value_info_proto.type.HasField("sequence_type"):
            opened_file.write(
                numpy_helper.from_list(data, value_info_proto.name).SerializeToString()
            )
        elif value_info_proto.type.HasField("optional_type"):
            opened_file.write(
                numpy_helper.from_optional(
                    data, value_info_proto.name
                ).SerializeToString()
            )
        else:
            assert value_info_proto.type.HasField("tensor_type")
            opened_file.write(
                numpy_helper.from_array(data, value_info_proto.name).SerializeToString()
            )


def _export_file(
    model_bytes: bytes,
    f: io.BytesIO | str,
    export_map: Mapping[str, bytes],
) -> None:
    """export/write model bytes into directory/protobuf/zip"""
    assert len(export_map) == 0
    with torch.serialization._open_file_like(f, "wb") as opened_file:
        opened_file.write(model_bytes)


def _add_onnxscript_fn(
    model_bytes: bytes,
    custom_opsets: Mapping[str, int],
) -> bytes:
    """Insert model-included custom onnx-script function into ModelProto"""
    try:
        import onnx
    except ImportError as e:
        raise errors.OnnxExporterError("Module onnx is not installed!") from e

    # For > 2GB model, onnx.load_fromstring would fail. However, because
    # in _export_onnx, the tensors should be saved separately if the proto
    # size > 2GB, and if it for some reason did not, the model would fail on
    # serialization anyway in terms of the protobuf limitation. So we don't
    # need to worry about > 2GB model getting here.
    model_proto = onnx.load_model_from_string(model_bytes)  # type: ignore[attr-defined]

    # Iterate graph nodes to insert only the included custom
    # function_proto into model_proto
    onnx_function_list = []  # type: ignore[var-annotated]
    included_node_func: set[str] = set()
    # onnx_function_list and included_node_func are expanded in-place
    _find_onnxscript_op(
        model_proto.graph, included_node_func, custom_opsets, onnx_function_list
    )

    if onnx_function_list:
        model_proto.functions.extend(onnx_function_list)
        model_bytes = model_proto.SerializeToString()
    return model_bytes


def _find_onnxscript_op(
    graph_proto,
    included_node_func: set[str],
    custom_opsets: Mapping[str, int],
    onnx_function_list: list,
):
    """Recursively iterate ModelProto to find ONNXFunction op as it may contain control flow Op."""
    for node in graph_proto.node:
        node_kind = node.domain + "::" + node.op_type
        # Recursive needed for control flow nodes: IF/Loop which has inner graph_proto
        for attr in node.attribute:
            if attr.g is not None:
                _find_onnxscript_op(
                    attr.g, included_node_func, custom_opsets, onnx_function_list
                )
        # Only custom Op with ONNX function and aten with symbolic_fn should be found in registry
        onnx_function_group = registration.registry.get_function_group(node_kind)
        # Ruled out corner cases: onnx/prim in registry
        if (
            node.domain
            and not jit_utils.is_aten(node.domain)
            and not jit_utils.is_prim(node.domain)
            and not jit_utils.is_onnx(node.domain)
            and onnx_function_group is not None
            and node_kind not in included_node_func
        ):
            specified_version = custom_opsets.get(node.domain, 1)
            onnx_fn = onnx_function_group.get(specified_version)
            if onnx_fn is not None:
                if hasattr(onnx_fn, "to_function_proto"):
                    onnx_function_proto = onnx_fn.to_function_proto()  # type: ignore[attr-defined]
                    onnx_function_list.append(onnx_function_proto)
                    included_node_func.add(node_kind)
                continue

            raise errors.UnsupportedOperatorError(
                node_kind,
                specified_version,
                onnx_function_group.get_min_supported()
                if onnx_function_group
                else None,
            )
    return onnx_function_list, included_node_func

```



## High-Level Overview

"""Utilities for manipulating the onnx and onnx-script dependencies and ONNX proto."""from __future__ import annotationsimport globimport osimport shutilfrom typing import Any, TYPE_CHECKINGimport torchimport torch.serializationfrom torch.onnx import errorsfrom torch.onnx._internal.torchscript_exporter import jit_utils, registrationif TYPE_CHECKING:    import io    from collections.abc import Mappingdef export_as_test_case(    model_bytes: bytes, inputs_data, outputs_data, name: str, dir: str) -> str:

This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `export_as_test_case`, `load_test_case`, `export_data`, `_export_file`, `_add_onnxscript_fn`, `_find_onnxscript_op`

**Key imports**: annotations, glob, os, shutil, Any, TYPE_CHECKING, torch, torch.serialization, errors, jit_utils, registration, io


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `glob`
- `os`
- `shutil`
- `typing`: Any, TYPE_CHECKING
- `torch`
- `torch.serialization`
- `torch.onnx`: errors
- `torch.onnx._internal.torchscript_exporter`: jit_utils, registration
- `io`
- `collections.abc`: Mapping
- `onnx`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/onnx/_internal/torchscript_exporter`):

- [`symbolic_opset7.py_docs.md`](./symbolic_opset7.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`symbolic_opset14.py_docs.md`](./symbolic_opset14.py_docs.md)
- [`symbolic_opset11.py_docs.md`](./symbolic_opset11.py_docs.md)
- [`verification.py_docs.md`](./verification.py_docs.md)
- [`symbolic_opset12.py_docs.md`](./symbolic_opset12.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`_experimental.py_docs.md`](./_experimental.py_docs.md)
- [`symbolic_opset20.py_docs.md`](./symbolic_opset20.py_docs.md)
- [`symbolic_opset9.py_docs.md`](./symbolic_opset9.py_docs.md)


## Cross-References

- **File Documentation**: `onnx_proto_utils.py_docs.md`
- **Keyword Index**: `onnx_proto_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/onnx/_internal/torchscript_exporter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/onnx/_internal/torchscript_exporter`):

- [`symbolic_opset14.py_docs.md_docs.md`](./symbolic_opset14.py_docs.md_docs.md)
- [`symbolic_opset18.py_kw.md_docs.md`](./symbolic_opset18.py_kw.md_docs.md)
- [`_experimental.py_kw.md_docs.md`](./_experimental.py_kw.md_docs.md)
- [`symbolic_opset13.py_kw.md_docs.md`](./symbolic_opset13.py_kw.md_docs.md)
- [`symbolic_opset12.py_docs.md_docs.md`](./symbolic_opset12.py_docs.md_docs.md)
- [`symbolic_opset16.py_docs.md_docs.md`](./symbolic_opset16.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`symbolic_helper.py_kw.md_docs.md`](./symbolic_helper.py_kw.md_docs.md)
- [`symbolic_opset8.py_docs.md_docs.md`](./symbolic_opset8.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `onnx_proto_utils.py_docs.md_docs.md`
- **Keyword Index**: `onnx_proto_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
