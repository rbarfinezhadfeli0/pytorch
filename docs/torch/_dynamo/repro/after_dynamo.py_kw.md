# Keyword Index: `torch/_dynamo/repro/after_dynamo.py`

## File Information

- **Original File**: [torch/_dynamo/repro/after_dynamo.py](../../../../torch/_dynamo/repro/after_dynamo.py)
- **Documentation**: [`after_dynamo.py_docs.md`](./after_dynamo.py_docs.md)
- **Folder**: `torch/_dynamo/repro`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`WrapBackendDebug`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)

### Functions

- **`__call__`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`__init__`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`_accuracy_fails`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`add_paths`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`backend_fails`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`common_flags`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`dump_backend_repro_as_file`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`dump_backend_state`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`dump_to_minify_after_dynamo`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`dynamo_accuracy_minifier_backend`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`dynamo_minifier_backend`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`generate_dynamo_fx_repro_string`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`repro_minify`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`repro_run`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`run_load_args`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`run_repro`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`wrap_backend_debug`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)

### Imports

- **`..`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`..backends.registry`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`..debug_utils`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`Any`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`Callable`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`CompilerFn`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`SequenceMatcher`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`argparse`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`clone_inputs_retaining_gradness`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`collections.abc`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`config`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`copy`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`difflib`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`functools`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`functorch.compile`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`fx_placeholder_targets`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`import_module`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`importlib`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`inf`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`logging`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`math`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`minifier`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`os`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`rand_strided`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`run_fwd_maybe_bwd`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`run_repro`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`shutil`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`sys`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`tensor`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`textwrap`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`torch`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`torch._dynamo`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`torch._dynamo.debug_utils`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`torch._dynamo.repro.after_dynamo`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`torch._dynamo.testing`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`torch.fx`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`torch.hub`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`tqdm`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`typing`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)
- **`your`**: [after_dynamo.py_docs.md](./after_dynamo.py_docs.md)


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
