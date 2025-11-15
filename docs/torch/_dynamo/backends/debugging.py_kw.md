# Keyword Index: `torch/_dynamo/backends/debugging.py`

## File Information

- **Original File**: [torch/_dynamo/backends/debugging.py](../../../../torch/_dynamo/backends/debugging.py)
- **Documentation**: [`debugging.py_docs.md`](./debugging.py_docs.md)
- **Folder**: `torch/_dynamo/backends`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ExplainWithBackend`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`ReluCompileError`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`TestingOnlyCompileError`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`and`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`class`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`directly`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`for`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`is`**: [debugging.py_docs.md](./debugging.py_docs.md)

### Functions

- **`__call__`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`__init__`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`__str__`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`_explain_graph_detail`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`aot_eager`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`aot_eager_decomp_partition`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`aot_eager_decomp_partition_crossref`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`aot_eager_decomp_partition_with_mode`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`boxed_nop`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`boxed_nop_with_mode`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`eager`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`eager_debug`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`eager_noexcept`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`fake_crossref_boxed_nop`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`fn`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`get_nop_func`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`ignore_builtins`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`inner`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`make_eager_backend_with_torch_function_mode`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`make_eager_backend_with_torch_function_modes`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`non_leaf_compile_error_TESTING_ONLY`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`output`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`pre_dispatch_eager`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`relu_accuracy_error_TESTING_ONLY`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`relu_compile_error_TESTING_ONLY`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`relu_runtime_error_TESTING_ONLY`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`run`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`runnable_gm`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torchscript`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`wrapper`**: [debugging.py_docs.md](./debugging.py_docs.md)

### Imports

- **`.common`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`.registry`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`Any`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`Callable`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`CompiledFn`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`CompilerBisector`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`ExitStack`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`GraphCompileReason`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`SchemaCheckMode`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`Target`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`_BoxedCodeGen`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`_guards`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`aot_autograd`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`collections.abc`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`config`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`contextlib`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`dataclasses`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`functools`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`functorch.compile`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`import_module`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`importlib`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`logging`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`lookup_backend`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`make_fx`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`min_cut_rematerialization_partition`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`of`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torch`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torch._dynamo.output_graph`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torch._functorch`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torch._functorch.compilers`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torch._inductor.compiler_bisector`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torch._subclasses.schema_check_mode`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torch.fx.graph`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`torch.fx.node`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`ts_compile`**: [debugging.py_docs.md](./debugging.py_docs.md)
- **`typing`**: [debugging.py_docs.md](./debugging.py_docs.md)


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
