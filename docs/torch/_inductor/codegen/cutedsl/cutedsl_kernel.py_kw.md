# Keyword Index: `torch/_inductor/codegen/cutedsl/cutedsl_kernel.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cutedsl/cutedsl_kernel.py](../../../../../torch/_inductor/codegen/cutedsl/cutedsl_kernel.py)
- **Documentation**: [`cutedsl_kernel.py_docs.md`](./cutedsl_kernel.py_docs.md)
- **Folder**: `torch/_inductor/codegen/cutedsl`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CuteDSLKernelWrapper`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`CuteDSLTemplateKernel`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`ModificationWrapperCuteDSL`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`class`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`sits`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)

### Functions

- **`__init__`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`__post_init__`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`_add_kernel_input`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`_default`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`_emit_scalar_fragment`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`_get_input_dtype`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`_get_subgraph`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`_process_indexing`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`call_kernel`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`create_subgraph_body`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`def_kernel`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`gen_defines`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`gen_imports`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`get_output`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`get_tensor_buffers`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`hook`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`indirect_indexing`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`kexpr`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`load`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`modification`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`render`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`run`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`set_subgraph_body`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`store`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`to_dict`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`unpack_buffers`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)

### Imports

- **`...utils`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`.cutedsl_op_overrides`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`Any`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`Buffer`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`Callable`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`CuteDSLOpOverrides`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`OrderedSet`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`PartialRender`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`StoreMode`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`V`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`collections.abc`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`config`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`contextlib`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`cuda.bindings.driver`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`cutlass`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`cutlass._mlir.dialects`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`cutlass.cute`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`cutlass.cute.runtime`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`dataclasses`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`from_dlpack`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`logging`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`math`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`operator`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`ssa_to_indexable`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`sympy`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`sympy_index_symbol`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`textwrap`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`torch`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`torch._inductor`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`torch._inductor.codegen.common`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`torch._inductor.codegen.cutedsl._cutedsl_utils`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`torch._inductor.ir`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`torch._inductor.ops_handler`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`torch._inductor.select_algorithm`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`torch._inductor.utils`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`torch._inductor.virtualized`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)
- **`typing`**: [cutedsl_kernel.py_docs.md](./cutedsl_kernel.py_docs.md)


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
