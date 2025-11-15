# Keyword Index: `test/onnx/torchlib/ops_test_common.py`

## File Information

- **Original File**: [test/onnx/torchlib/ops_test_common.py](../../../../test/onnx/torchlib/ops_test_common.py)
- **Documentation**: [`ops_test_common.py_docs.md`](./ops_test_common.py_docs.md)
- **Folder**: `test/onnx/torchlib`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`OrtAbortedError`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`class`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`for`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`name`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)

### Functions

- **`_capture_graph_and_evaluate_torch_script_evaluator`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`_format_model_and_input_information`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`_ort_session_run`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`_ort_session_run_return_dict`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`_safe_ort_session_run`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`add_decorate_info`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`convert_kwargs_for_onnx`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`convert_tensor_to_numpy`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`dtype_op_schema_compatible`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`duplicate_opinfo`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`duplicate_opinfo_for_prims`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`graph_executor`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`normal_xfail_skip_test_behaviors`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`skip`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`wrapped`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`xfail`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)

### Imports

- **`Any`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`Callable`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`__future__`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`_building`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`annotations`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`collections.abc`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`contextlib`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`copy`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`core`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`dataclasses`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`error_reproduction`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`ir`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`multiprocessing`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`numpy`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`onnx`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`onnxruntime`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`onnxruntime.capi.onnxruntime_pybind11_state`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`onnxscript`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`onnxscript.evaluator`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`os`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`pprint`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`pytest`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`sys`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`torch`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`torch.onnx._internal.exporter`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`torch.testing._internal.opinfo`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`typing`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`unittest`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)
- **`warnings`**: [ops_test_common.py_docs.md](./ops_test_common.py_docs.md)


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
