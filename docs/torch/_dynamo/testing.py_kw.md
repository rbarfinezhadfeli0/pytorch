# Keyword Index: `torch/_dynamo/testing.py`

## File Information

- **Original File**: [torch/_dynamo/testing.py](../../../torch/_dynamo/testing.py)
- **Documentation**: [`testing.py_docs.md`](./testing.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AotEagerAndRecordGraphs`**: [testing.py_docs.md](./testing.py_docs.md)
- **`CompileCounter`**: [testing.py_docs.md](./testing.py_docs.md)
- **`CompileCounterWithBackend`**: [testing.py_docs.md](./testing.py_docs.md)
- **`EagerAndRecordGraphs`**: [testing.py_docs.md](./testing.py_docs.md)
- **`InductorAndRecordGraphs`**: [testing.py_docs.md](./testing.py_docs.md)

### Functions

- **`__call__`**: [testing.py_docs.md](./testing.py_docs.md)
- **`__init__`**: [testing.py_docs.md](./testing.py_docs.md)
- **`_fn`**: [testing.py_docs.md](./testing.py_docs.md)
- **`_make_fn_with_patches`**: [testing.py_docs.md](./testing.py_docs.md)
- **`_skipped_function_for_test_reconstruct`**: [testing.py_docs.md](./testing.py_docs.md)
- **`bw_compiler`**: [testing.py_docs.md](./testing.py_docs.md)
- **`check_dynamic_shape_capture`**: [testing.py_docs.md](./testing.py_docs.md)
- **`clear`**: [testing.py_docs.md](./testing.py_docs.md)
- **`clone_me`**: [testing.py_docs.md](./testing.py_docs.md)
- **`collect_results`**: [testing.py_docs.md](./testing.py_docs.md)
- **`debug_dir`**: [testing.py_docs.md](./testing.py_docs.md)
- **`debug_dump`**: [testing.py_docs.md](./testing.py_docs.md)
- **`debug_insert_nops`**: [testing.py_docs.md](./testing.py_docs.md)
- **`dummy_fx_compile`**: [testing.py_docs.md](./testing.py_docs.md)
- **`empty_line_normalizer`**: [testing.py_docs.md](./testing.py_docs.md)
- **`expectedFailureCodegenDynamic`**: [testing.py_docs.md](./testing.py_docs.md)
- **`expectedFailureDynamic`**: [testing.py_docs.md](./testing.py_docs.md)
- **`expectedFailureDynamicWrapper`**: [testing.py_docs.md](./testing.py_docs.md)
- **`extract_graph`**: [testing.py_docs.md](./testing.py_docs.md)
- **`extract_graph_and_tracker`**: [testing.py_docs.md](./testing.py_docs.md)
- **`extract_graph_backend`**: [testing.py_docs.md](./testing.py_docs.md)
- **`format_speedup`**: [testing.py_docs.md](./testing.py_docs.md)
- **`fw_compiler`**: [testing.py_docs.md](./testing.py_docs.md)
- **`insert_nops`**: [testing.py_docs.md](./testing.py_docs.md)
- **`make_test_cls_with_patches`**: [testing.py_docs.md](./testing.py_docs.md)
- **`normalize_gm`**: [testing.py_docs.md](./testing.py_docs.md)
- **`patched`**: [testing.py_docs.md](./testing.py_docs.md)
- **`rand_strided`**: [testing.py_docs.md](./testing.py_docs.md)
- **`reduce_to_scalar_loss`**: [testing.py_docs.md](./testing.py_docs.md)
- **`remove_optimized_module_prefix`**: [testing.py_docs.md](./testing.py_docs.md)
- **`remove_trailing_space`**: [testing.py_docs.md](./testing.py_docs.md)
- **`requires_bwd_pass`**: [testing.py_docs.md](./testing.py_docs.md)
- **`reset_rng_state`**: [testing.py_docs.md](./testing.py_docs.md)
- **`skipIfNotPy311`**: [testing.py_docs.md](./testing.py_docs.md)
- **`skipIfNotPy312`**: [testing.py_docs.md](./testing.py_docs.md)
- **`skipIfOnlyNotPy312`**: [testing.py_docs.md](./testing.py_docs.md)
- **`skipIfPy312`**: [testing.py_docs.md](./testing.py_docs.md)
- **`standard_test`**: [testing.py_docs.md](./testing.py_docs.md)
- **`strip_comment`**: [testing.py_docs.md](./testing.py_docs.md)
- **`xfailIfPy312`**: [testing.py_docs.md](./testing.py_docs.md)

### Imports

- **`.`**: [testing.py_docs.md](./testing.py_docs.md)
- **`.backends.registry`**: [testing.py_docs.md](./testing.py_docs.md)
- **`.bytecode_transformation`**: [testing.py_docs.md](./testing.py_docs.md)
- **`.guards`**: [testing.py_docs.md](./testing.py_docs.md)
- **`.types`**: [testing.py_docs.md](./testing.py_docs.md)
- **`.utils`**: [testing.py_docs.md](./testing.py_docs.md)
- **`Any`**: [testing.py_docs.md](./testing.py_docs.md)
- **`Callable`**: [testing.py_docs.md](./testing.py_docs.md)
- **`CheckFunctionManager`**: [testing.py_docs.md](./testing.py_docs.md)
- **`CompileCounterInt`**: [testing.py_docs.md](./testing.py_docs.md)
- **`ConvertFrameReturn`**: [testing.py_docs.md](./testing.py_docs.md)
- **`InstructionTranslator`**: [testing.py_docs.md](./testing.py_docs.md)
- **`OutputGraph`**: [testing.py_docs.md](./testing.py_docs.md)
- **`ParamSpec`**: [testing.py_docs.md](./testing.py_docs.md)
- **`aot_eager`**: [testing.py_docs.md](./testing.py_docs.md)
- **`collections.abc`**: [testing.py_docs.md](./testing.py_docs.md)
- **`config`**: [testing.py_docs.md](./testing.py_docs.md)
- **`contextlib`**: [testing.py_docs.md](./testing.py_docs.md)
- **`dis`**: [testing.py_docs.md](./testing.py_docs.md)
- **`functools`**: [testing.py_docs.md](./testing.py_docs.md)
- **`fx`**: [testing.py_docs.md](./testing.py_docs.md)
- **`logging`**: [testing.py_docs.md](./testing.py_docs.md)
- **`lookup_backend`**: [testing.py_docs.md](./testing.py_docs.md)
- **`numpy`**: [testing.py_docs.md](./testing.py_docs.md)
- **`os.path`**: [testing.py_docs.md](./testing.py_docs.md)
- **`patch`**: [testing.py_docs.md](./testing.py_docs.md)
- **`random`**: [testing.py_docs.md](./testing.py_docs.md)
- **`re`**: [testing.py_docs.md](./testing.py_docs.md)
- **`sys`**: [testing.py_docs.md](./testing.py_docs.md)
- **`torch`**: [testing.py_docs.md](./testing.py_docs.md)
- **`torch._dynamo.backends.debugging`**: [testing.py_docs.md](./testing.py_docs.md)
- **`torch._dynamo.output_graph`**: [testing.py_docs.md](./testing.py_docs.md)
- **`torch._dynamo.symbolic_convert`**: [testing.py_docs.md](./testing.py_docs.md)
- **`torch._inductor.compile_fx`**: [testing.py_docs.md](./testing.py_docs.md)
- **`torch_xla.core.xla_model`**: [testing.py_docs.md](./testing.py_docs.md)
- **`types`**: [testing.py_docs.md](./testing.py_docs.md)
- **`typing`**: [testing.py_docs.md](./testing.py_docs.md)
- **`typing_extensions`**: [testing.py_docs.md](./testing.py_docs.md)
- **`unittest`**: [testing.py_docs.md](./testing.py_docs.md)
- **`unittest.mock`**: [testing.py_docs.md](./testing.py_docs.md)


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
