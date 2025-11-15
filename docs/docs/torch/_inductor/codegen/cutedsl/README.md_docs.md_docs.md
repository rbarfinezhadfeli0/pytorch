# Documentation: `docs/torch/_inductor/codegen/cutedsl/README.md_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/cutedsl/README.md_docs.md`
- **Size**: 6,219 bytes (6.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/cutedsl/README.md`

## File Metadata

- **Path**: `torch/_inductor/codegen/cutedsl/README.md`
- **Size**: 3,879 bytes (3.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This is a markdown documentation that is part of the PyTorch project.

## Original Source

```markdown
# CuteDSL Template System

## Quick Start

Writing a CuteDSL template:

```python
from torch._inductor.codegen.cutedsl import CuteDSLTemplate

template_source = """
@cute.kernel
def {{kernel_name}}_kernel(A, B, C):
    # Your CUTLASS kernel logic here
    pass

{{def_kernel("A", "B", "C")}}
    # Call the kernel
    {{kernel_name}}_kernel(A, B, C)
    return C
"""

my_template = CuteDSLTemplate(
    name="my_gemm",
    source=template_source,
)
```

## Architecture

- **[CuteDSLTemplate](cutedsl_template.py#L39)**: Template definition and registration. Generates ChoiceCallers for autotuning.
- **[CuteDSLTemplateKernel](cutedsl_kernel.py#L61)**: Handles code generation, provides template hooks (`def_kernel`), manages args.
- **[CuteDSLScheduling](cutedsl_scheduling.py#L28)**: Integrates with Inductor's scheduler, handles kernel compilation via [`async_compile.cutedsl()`](../../async_compile.py#L756).
- **[CuteDSLTemplateBuffer](../../ir.py)**: IR node representing a CuteDSL template operation in the graph.

### Compilation Process

CuteDSL requires source files for compilation (cannot compile from strings directly). The process:

1. **[CuteDSLScheduling](cutedsl_scheduling.py#L59)** generates the kernel code string and calls [`async_compile.cutedsl()`](../../async_compile.py#L756)
2. **[async_compile.cutedsl()](../../async_compile.py#L756)** uses [`PyCodeCache.write()`](../../codecache.py) to write source to a temporary `.py` file
3. **[PyCodeCache](../../codecache.py)** loads the module from disk, enabling CUTLASS compilation
4. The compiled kernel is wrapped in **[CuteDSLKernelWrapper](cutedsl_kernel.py#L22)** to provide a `.run()` interface
5. The generated Python file is cached via PyCodeCache, but CUTLASS compilation runs every time (no kernel-level caching yet)

**Debug tip**: Use `TORCH_LOGS="kernel_code"` to see the generated kernel source and file path during compilation.

## Writing Templates

Templates use Jinja2 syntax with these available hooks:

- `{{kernel_name}}` - Unique kernel identifier
- `{{def_kernel(args...)}}` - Generates kernel function signature and argument handling
- `{{input_nodes}}` - List of input buffers
- `{{output_node}}` - Output buffer
- `{{gen_defines()}}` - Generates autotunable parameter definitions with proper CuteDSL typing

## Autotunable Parameters

CuteDSL templates support autotunable parameters similar to Triton's `tl.constexpr` system:

```python
template_source = r"""
{{gen_defines()}}

@cute.kernel
def {{kernel_name}}_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    threads_per_block = THREADS_PER_BLOCK  # Uses autotuned value
    block_size = BLOCK_SIZE
    # ... kernel implementation
"""

# Pass parameters when generating template choices
template.maybe_append_choice(
    choices,
    input_nodes=[a, b],
    layout=layout,
    THREADS_PER_BLOCK=256,    # cutlass.Constexpr = 256
    BLOCK_SIZE=128,           # cutlass.Constexpr = 128
    SCALE_FACTOR=1.5,         # cutlass.Constexpr = 1.5
)
```

Templates must:
1. Define a `@cute.kernel` decorated function
2. Use `{{def_kernel()}}` to create the entry point
3. Return the output tensor
4. Use `{{gen_defines()}}` for autotunable parameters

See [test_cutedsl_template.py](../../../../test/inductor/test_cutedsl_template.py) for complete examples.

## Current Limitations / TODOs

- **No fusion support**: `can_fuse_vertical` and `can_fuse_horizontal` return False
- **Subgraph management**: Bodies and masks not fully implemented
- **File-based compilation**: Requires writing to disk (uses PyCodeCache)
- **Missing epilogue/prologue**: No support for fused operations yet
- **Fixed kernel suffix**: Uses hardcoded "_main" suffix
- **No CUTLASS kernel caching**: Only PyCodeCache works; CUTLASS compilation runs every time (major perf issue)


Note: Requires CUTLASS Python package (`pip install nvidia-cutlass`)
```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/_inductor/codegen/cutedsl`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen/cutedsl`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor/codegen/cutedsl`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`cutedsl_kernel.py_docs.md`](./cutedsl_kernel.py_docs.md)
- [`cutedsl_template.py_docs.md`](./cutedsl_template.py_docs.md)
- [`_cutedsl_utils.py_docs.md`](./_cutedsl_utils.py_docs.md)
- [`cutedsl_scheduling.py_docs.md`](./cutedsl_scheduling.py_docs.md)
- [`cutedsl_op_overrides.py_docs.md`](./cutedsl_op_overrides.py_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md`
- **Keyword Index**: `README.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen/cutedsl`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen/cutedsl`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_inductor/codegen/cutedsl`):

- [`cutedsl_scheduling.py_docs.md_docs.md`](./cutedsl_scheduling.py_docs.md_docs.md)
- [`cutedsl_scheduling.py_kw.md_docs.md`](./cutedsl_scheduling.py_kw.md_docs.md)
- [`cutedsl_kernel.py_docs.md_docs.md`](./cutedsl_kernel.py_docs.md_docs.md)
- [`cutedsl_template.py_docs.md_docs.md`](./cutedsl_template.py_docs.md_docs.md)
- [`_cutedsl_utils.py_docs.md_docs.md`](./_cutedsl_utils.py_docs.md_docs.md)
- [`cutedsl_template.py_kw.md_docs.md`](./cutedsl_template.py_kw.md_docs.md)
- [`cutedsl_op_overrides.py_kw.md_docs.md`](./cutedsl_op_overrides.py_kw.md_docs.md)
- [`_cutedsl_utils.py_kw.md_docs.md`](./_cutedsl_utils.py_kw.md_docs.md)
- [`cutedsl_op_overrides.py_docs.md_docs.md`](./cutedsl_op_overrides.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md_docs.md`
- **Keyword Index**: `README.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
