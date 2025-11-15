# Keyword Index: `torch/_inductor/codegen/cuda/cutlass_lib_extensions/evt_extensions.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cuda/cutlass_lib_extensions/evt_extensions.py](../../../../../../torch/_inductor/codegen/cuda/cutlass_lib_extensions/evt_extensions.py)
- **Documentation**: [`evt_extensions.py_docs.md`](./evt_extensions.py_docs.md)
- **Folder**: `torch/_inductor/codegen/cuda/cutlass_lib_extensions`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`EVTArgRenames`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`EpilogueFunctor`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`def`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`defined`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`type`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)

### Functions

- **`__init__`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`_get_arg_from_node`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`_render_argument_type`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`_trace`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`create_example_tensors`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cutlass_tensor_from_buffer`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`get`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`is_nested_visitor_type`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`new_name`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`parse`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`render_argument_type`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`render_stride`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`render_thread_type`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`trace`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)

### Imports

- **`..cuda_template`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`..cutlass_utils`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`Any`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`CUTLASSTemplate`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`Callable`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`Expr`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`IndentedBuffer`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`OrderedSet`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`Union`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`ast`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`collections.abc`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`ctypes`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cuda_env`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cutlass_cppgen.backend.c_types`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cutlass_cppgen.backend.epilogue`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cutlass_cppgen.backend.evt`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cutlass_cppgen.backend.evt.backend.emitter_base`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cutlass_cppgen.backend.evt.backend.sm90_emitter`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cutlass_cppgen.backend.evt.frontend`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cutlass_cppgen.backend.evt.ir.tensor`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`cutlass_library`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`sympy`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`textwrap`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`torch._inductor.codegen.cuda`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`torch._inductor.ir`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`torch._inductor.utils`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`torch.utils._ordered_set`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`torch_dtype_to_cutlass_type`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)
- **`typing`**: [evt_extensions.py_docs.md](./evt_extensions.py_docs.md)


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
