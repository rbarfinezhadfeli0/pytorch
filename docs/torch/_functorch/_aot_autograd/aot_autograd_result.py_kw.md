# Keyword Index: `torch/_functorch/_aot_autograd/aot_autograd_result.py`

## File Information

- **Original File**: [torch/_functorch/_aot_autograd/aot_autograd_result.py](../../../../torch/_functorch/_aot_autograd/aot_autograd_result.py)
- **Documentation**: [`aot_autograd_result.py_docs.md`](./aot_autograd_result.py_docs.md)
- **Folder**: `torch/_functorch/_aot_autograd`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AOTAutogradResult`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`BundledAOTAutogradResult`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`InductorOutput`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`class`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`from`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)

### Functions

- **`__init__`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`_is_backward`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`after_deserialization`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`check_exact_guard_match`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`deserialize`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`deserialize_bundled_cache_entry`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`forward`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`load`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`post_compile`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`pre_save`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`serialize_graph_module`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`wrap_post_compile`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)

### Imports

- **`.autograd_cache`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`.runtime_wrappers`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`.schemas`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`.utils`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`ABC`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`AOTAutogradCache`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`AOTAutogradCacheInfo`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`AOTConfig`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`Any`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`BackendCacheArtifact`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`BoxedBool`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`BoxedDeviceIndex`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`Callable`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`CompileEventLogger`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`FXGraphCacheMiss`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`FakeTensorMode`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`FxGraphCache`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`ShapeEnv`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`_CompileFxKwargs`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`__future__`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`abc`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`annotations`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`collections.abc`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`copy`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`dataclass`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`dataclass_repr`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`dataclasses`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`deepcopy`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`json`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`logging`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`should_use_remote_fx_graph_cache`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`simple_wraps`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch._dynamo.precompile_context`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch._dynamo.utils`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch._inductor.codecache`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch._inductor.compile_fx`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch._inductor.cudagraph_utils`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch._inductor.output_code`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch._inductor.utils`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch._subclasses`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`torchgen.utils`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)
- **`typing`**: [aot_autograd_result.py_docs.md](./aot_autograd_result.py_docs.md)


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
