# Keyword Index: `torch/_dynamo/pgo.py`

## File Information

- **Original File**: [torch/_dynamo/pgo.py](../../../torch/_dynamo/pgo.py)
- **Documentation**: [`pgo.py_docs.md`](./pgo.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AutoDynamic`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`AutoUnset`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`CodeId`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`InferStride`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`PGOCacheArtifact`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`ReservedWorkflowIdUserError`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`as`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`class`**: [pgo.py_docs.md](./pgo.py_docs.md)

### Functions

- **`__eq__`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`__hash__`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`__ior__`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`__post_init__`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`__str__`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`_collect_dynamic_sources`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`_collect_missing_sources`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`_hash_containing_file`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`_log_size_mismatch_recompile`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`_merge_atom`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`_merge_atom_tup`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`_munge_symint`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`_rewrite_cache_key_for_mega_cache`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`code_state_path`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`format_cache_key`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`get_cache_key`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`get_code_state`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`get_extra_cache_key`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`get_extra_remote_code_state`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`get_local_code_state`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`get_remote_cache`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`get_remote_code_state`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`hit`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`is_size_dynamic`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`is_stride_dynamic`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`log_frame_dynamic_whitelist`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`log_tup`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`lookup_remote_cache_entry`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`make`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`make_scalar`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`make_size`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`make_tensor`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`populate_cache`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`process_automatic_dynamic`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`put_code_state`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`put_local_code_state`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`put_remote_code_state`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`render`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`render_code_state`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`render_single`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`render_tuple`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`reset_code_state`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`should_use_remote_dynamo_pgo_cache`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`type`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`update_automatic_dynamic`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`write_local_impl`**: [pgo.py_docs.md](./pgo.py_docs.md)

### Imports

- **`FileLock`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`InstructionTranslator`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`JsonDataTy`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`Optional`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`OrderedSet`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`REMOTE_CACHE_VERSION`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`__future__`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`annotations`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`base64`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`cache_dir`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`collections`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`copy`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`create_cache`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`dataclasses`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`defaultdict`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`enum`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`functools`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`is_dynamic_source`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`is_fbcode`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`logging`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`os`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`override`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`pickle`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`re`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._dynamo.config`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._dynamo.symbolic_convert`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._dynamo.utils`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._dynamo.variables.builder`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._environment`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._inductor.fb.remote_cache`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._inductor.remote_cache`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._inductor.runtime.runtime_utils`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._logging._internal`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch._utils_internal`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch.compiler._cache`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch.compiler.config`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch.distributed`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch.utils._filelock`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`torch.utils._ordered_set`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`trace_structured_artifact`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`types`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`typing`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`typing_extensions`**: [pgo.py_docs.md](./pgo.py_docs.md)
- **`zlib`**: [pgo.py_docs.md](./pgo.py_docs.md)


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
