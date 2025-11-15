# Documentation: `docs/test/jit/test_device_analysis.py_kw.md`

## File Metadata

- **Path**: `docs/test/jit/test_device_analysis.py_kw.md`
- **Size**: 5,439 bytes (5.31 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Keyword Index: `test/jit/test_device_analysis.py`

## File Information

- **Original File**: [test/jit/test_device_analysis.py](../../../test/jit/test_device_analysis.py)
- **Documentation**: [`test_device_analysis.py_docs.md`](./test_device_analysis.py_docs.md)
- **Folder**: `test/jit`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestDeviceAnalysis`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)

### Functions

- **`add`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`add_self`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`assert_device_equal`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`expand_as_fn`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`mul`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`node_output_device`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`prop_device_on_graph`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`relu_`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`reshape_as_fn`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`setUpClass`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`set_cpu`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`set_cuda`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`set_device`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`set_mkldnn`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_custom_device_op`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_device_apply`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_device_arg`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_device_if_propagation`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_fn`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_if_loop_mix`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_loop_device_change`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_loop_simple`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_mobilenet`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_nested_loops`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_set_dtype`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_simple`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_tensor_as_fns`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_while_change`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_zerodim_cpu`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_zerodim_gpu`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`test_zerodim_no_device`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`type_as_fn`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`view_as_fn`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`zerodim_test_core`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)

### Imports

- **`JitTestCase`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`apply_input_props_using_example`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`itertools`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`models`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`product`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`raise_on_run_directly`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`torch`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`torch.jit._passes._property_propagation`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`torch.testing._internal.jit_utils`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`torchvision`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)
- **`unittest`**: [test_device_analysis.py_docs.md](./test_device_analysis.py_docs.md)


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

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/jit/test_device_analysis.py_kw.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/jit`):

- [`test_attr.py_kw.md_docs.md`](./test_attr.py_kw.md_docs.md)
- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_dataclasses.py_docs.md_docs.md`](./test_dataclasses.py_docs.md_docs.md)
- [`test_aten_pow.py_kw.md_docs.md`](./test_aten_pow.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_graph_rewrite_passes.py_kw.md_docs.md`](./test_graph_rewrite_passes.py_kw.md_docs.md)
- [`test_module_containers.py_kw.md_docs.md`](./test_module_containers.py_kw.md_docs.md)
- [`test_complex.py_kw.md_docs.md`](./test_complex.py_kw.md_docs.md)
- [`test_types.py_kw.md_docs.md`](./test_types.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_device_analysis.py_kw.md_docs.md`
- **Keyword Index**: `test_device_analysis.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
