# Keyword Index: `torch/_dynamo/debug_utils.py`

## File Information

- **Original File**: [torch/_dynamo/debug_utils.py](../../../torch/_dynamo/debug_utils.py)
- **Documentation**: [`debug_utils.py_docs.md`](./debug_utils.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AccuracyError`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`BuckTargetWriter`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`InputReader`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`InputWriter`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`NNModuleToString`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`NopInputReader`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`Repro`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`TensorContainer`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)

### Functions

- **`__init__`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`_cuda_system_info_comment`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`_mk_defaulter`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`_stride_or_default`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`aot_graph_input_parser`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`backend_accuracy_fails`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`build`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`can_convert_to_string`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`cast_dtype_args_to_fp64`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`cast_to`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`cast_to_fp64`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`clone_inputs_retaining_gradness`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`const`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`convert`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`decorator`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`filter`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`forward`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`gen_tensor`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`generate_config_string`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`generate_env_vars_string`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`get_minifier_repro_path`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`get_sym_int`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`helper_for_dump_minify`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`lines`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`load_args`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`minifier_dir`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`profile_to_file`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`run_fwd_maybe_bwd`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`same_two_models`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`save_it`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`storage`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`symint`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`tensor`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`unsupported`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`wrapper`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`write`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)

### Imports

- **`.`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`.testing`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`.utils`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`Any`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`Callable`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`ContentStoreReader`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`Counter`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`PRINT_OPTS`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`StorageWeakRef`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`Tensor`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`UntypedStorage`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`__future__`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`_addindent`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`annotations`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`atexit`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`cProfile`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`clone_inputs`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`collect_results`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`collections`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`collections.abc`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`config`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`copy`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`dtype_abbrs`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`functools`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`getpass`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`import_module`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`importlib`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`inspect`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`is_float_dtype`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`itertools`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`libfb.py.build_info`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`logging`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`normalize_path_separator`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`os`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`rand_strided`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`re`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`same`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`statically_known_true`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`subprocess`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`sys`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`tempfile`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`textwrap`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch._dynamo.config`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch._dynamo.testing`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch._functorch.config`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch._inductor.config`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch._inductor.cpp_builder`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch._prims_common`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch._subclasses.meta_utils`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch._tensor_str`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.fx.experimental._config`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.hub`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.multiprocessing.reductions`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.nn`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.nn.modules.module`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.storage`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.utils._content_store`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.utils._dtype_abbrs`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`torch.utils._pytree`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`tqdm`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`tree_map`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)
- **`typing`**: [debug_utils.py_docs.md](./debug_utils.py_docs.md)


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
