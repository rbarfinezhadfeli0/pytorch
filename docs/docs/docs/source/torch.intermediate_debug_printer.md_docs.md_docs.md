# Documentation: `docs/docs/source/torch.intermediate_debug_printer.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/torch.intermediate_debug_printer.md_docs.md`
- **Size**: 6,447 bytes (6.30 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/torch.intermediate_debug_printer.md`

## File Metadata

- **Path**: `docs/source/torch.intermediate_debug_printer.md`
- **Size**: 3,772 bytes (3.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
```{eval-rst}
:orphan:
```

# AOTInductor Intermediate Value Debug Printer

This is a user manual on how to use AOT Inductor Intermediate Value Debug Printer tool which is a utility tool that can help pinpoint CUDA IMA kernels / numerical discrepancies when uses AOT Inductor to compile a PyTorch model.

The main functionality of this tool is to automatically print out / or dump the value info of all intermediate tensor arguments before and after each kernel launch call in AOT Inductor.

## How to use

The debug printer can be configured via environment variable. The following flags are both supported to run with internal fbcode buck commands and OSS.

All configurations are defined here: [torch/_inductor/config.py](https://github.com/pytorch/pytorch/blob/768361e67f0eb36491d7b763ef38d7c928ebefe6/torch/_inductor/config.py#L1493-L1505)


```
    # options for debug printing/saving for intermediate tensor values for aot inductor

    0: disable debug dumping
    1: enable saving intermediate tensor values
    2: enable printing intermediate tensor values
    3: enable printing kernel names only (useful for pinpointing troublesome kernels)
```


1. To enable **default** mode debug printing:

    - Add flag `AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2` (PRINT_ONLY mode) for default printing all supported kernel tensor arg values.

    - Add flag `AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT={kernel_name_1, kernel_name_2,...}` for selectively printing tensor values associated with the specified kernels. (suggest to do a run with generating full printing logs first)

    Sample command:

    ```
    AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="aoti_torch_cuda_addmm_out" AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2 TORCH_LOGS="+inductor, output_code" python test/inductor/test_aot_inductor.py -k test_addmm_cuda
    ```


2. To enable **pinpoint** the problematic kernel name only: (Especially useful in CUDA IMA debugging)

   - Add flag `AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3` (PRINT_KERNEL_NAME_ONLY mode) no tensor numerical values will be dumped.

   Sample command:

   ```
   AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3 TORCH_LOGS="+inductor, output_code" python test/inductor/test_aot_inductor.py -k test_addmm_cuda
   ```

3. To enable **save** the intermediate tensor values:

    - Useful when you want to repro the error in a standalone kernel debugging repro. The saved intermediate tensor values can be used as debugging inputs to the problematic kernel.
    - Set `AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=1` (SAVE_ONLY mode)  for default saving all supported kernel tensor arg values to `.pt` in a tmp folder.
    - Similarly, add `AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT={kernel_name_1, kernel_name_2,...}` for selectively saving tensor values associated with the specified kernels.

    Sample command:
    ```
    AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="triton_poi_fused_0" AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=1 TORCH_LOGS="+inductor, output_code" python test/inductor/test_aot_inductor.py -k test_addmm_cuda
    ```

    The saved tensor values will be dumped in a format:  `<before/after_launch>_<kernel_name>_<arg_name>_<device>.pt`

    The dumped `.pt` tensors can be further loaded and used like this:
    ```
        def _load_tensor(path):
            return torch.load(path, weights_only=True)
        tensor = _load_tensor("../tmp/aoti_torch/before_launch_aoti_torch_cuda_addmm_out_buf1_cuda:0.pt")

        # Simply print tensor to view the full value
        print(tensor)
    ```

## Example Outputs

Before launch tensor stats:

![Sample image 1](_static/img/aoti_debug_printer/before_launch.png)


After launch tensor stats:

![Sample image 2](_static/img/aoti_debug_printer/after_launch.png)

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source`, which is part of the PyTorch project infrastructure.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/source`):

- [`torch.compiler_troubleshooting.md_docs.md`](./torch.compiler_troubleshooting.md_docs.md)
- [`torch.compiler_aot_inductor_debugging_guide.md_docs.md`](./torch.compiler_aot_inductor_debugging_guide.md_docs.md)
- [`mtia.memory.md_docs.md`](./mtia.memory.md_docs.md)
- [`torch.compiler_get_started.md_docs.md`](./torch.compiler_get_started.md_docs.md)
- [`torch.compiler_dynamo_deepdive.md_docs.md`](./torch.compiler_dynamo_deepdive.md_docs.md)
- [`mtia.mtia_graph.md_docs.md`](./mtia.mtia_graph.md_docs.md)
- [`hub.md_docs.md`](./hub.md_docs.md)
- [`torch_nccl_environment_variables.md_docs.md`](./torch_nccl_environment_variables.md_docs.md)
- [`optim.md_docs.md`](./optim.md_docs.md)
- [`torch.compiler_aot_inductor.md_docs.md`](./torch.compiler_aot_inductor.md_docs.md)


## Cross-References

- **File Documentation**: `torch.intermediate_debug_printer.md_docs.md`
- **Keyword Index**: `torch.intermediate_debug_printer.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`docs/docs/source`):

- [`distributions.md_docs.md_docs.md`](./distributions.md_docs.md_docs.md)
- [`distributed.optim.md_docs.md_docs.md`](./distributed.optim.md_docs.md_docs.md)
- [`torch.compiler_dynamic_shapes.md_kw.md_docs.md`](./torch.compiler_dynamic_shapes.md_kw.md_docs.md)
- [`tensor_attributes.rst_docs.md_docs.md`](./tensor_attributes.rst_docs.md_docs.md)
- [`tensor_attributes.rst_kw.md_docs.md`](./tensor_attributes.rst_kw.md_docs.md)
- [`torch.compiler_dynamo_overview.md_docs.md_docs.md`](./torch.compiler_dynamo_overview.md_docs.md_docs.md)
- [`mtia.memory.md_kw.md_docs.md`](./mtia.memory.md_kw.md_docs.md)
- [`nn.attention.varlen.md_kw.md_docs.md`](./nn.attention.varlen.md_kw.md_docs.md)
- [`cpu.rst_kw.md_docs.md`](./cpu.rst_kw.md_docs.md)
- [`torch.compiler_faq.md_docs.md_docs.md`](./torch.compiler_faq.md_docs.md_docs.md)


## Cross-References

- **File Documentation**: `torch.intermediate_debug_printer.md_docs.md_docs.md`
- **Keyword Index**: `torch.intermediate_debug_printer.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
