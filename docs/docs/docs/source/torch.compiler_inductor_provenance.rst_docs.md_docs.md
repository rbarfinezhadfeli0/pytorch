# Documentation: `docs/docs/source/torch.compiler_inductor_provenance.rst_docs.md`

## File Metadata

- **Path**: `docs/docs/source/torch.compiler_inductor_provenance.rst_docs.md`
- **Size**: 6,765 bytes (6.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/torch.compiler_inductor_provenance.rst`

## File Metadata

- **Path**: `docs/source/torch.compiler_inductor_provenance.rst`
- **Size**: 4,161 bytes (4.06 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
.. _torchinductor-provenance:

TorchInductor and AOTInductor Provenance Tracking
=================================================

This section describes how to use the provenance tracking feature for TorchInductor and AOTInductor in ``tlparse``.
Provenance tracking helps you visualize the relationships between the input GraphModule to (AOT)Inductor and the optimized code generated. This feature allows you to trace how your original operations are transformed during compilation.

Some example screenshots of the provenance tracking tool are shown below.
The tool visualizes the mapping between nodes in the input graph (panel 1), the post grad graph (panel 2), and the Inductor generated code (panel 3).

The **bolded** lines represent nodes/kernels covered by the current provenance tracing functionality.
We currently cover triton kernels, cpp kernels, and combo kernels.
The yellow highlighting shows the provenance of the nodes/kernels.


Example screenshot of the provenance tracking tool for TorchInductor:
 .. image:: _static/img/inductor_provenance/provenance_jit_inductor.png

Example screenshot of the provenance tracking tool for AOTInductor:
 .. image:: _static/img/inductor_provenance/provenance_aot_inductor.png


Using the Provenance Tracking Highlighter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow these steps to enable and use provenance tracking in your PyTorch project:

1. Install ``tlparse`` by ``cargo install tlparse``. If you don't have ``cargo``, see `The Cargo Book <https://doc.rust-lang.org/cargo/getting-started/installation.html>`__ for instructions to install.
2. Run your program with required flags:

   .. code-block:: bash

     TORCH_TRACE=~/my_trace_log_dir INDUCTOR_PROVENANCE=1 python your_program.py

   This will generate a log file in ``~/my_trace_log_dir``. The log file will be used by tlparse to generate the provenance tracking highlighter.
3. Run ``tlparse`` on the log with ``--inductor-provenance`` flag. For example:

   .. code-block:: bash

      tlparse log_file_name.log --inductor-provenance

   - Even if you don't add the ``--inductor-provenance`` flag, you should be able to see the mapping in json format in the ``inductor_provenance_tracking_node_mappings_<number>.json`` file in the ``index.html`` tlparse output.
   - Run ``tlpare`` directly on the log file. It might not work if you run "tlparse parse <folder_name>  --inductor-provenance".
   - The ``tlparse`` artifacts used by the provenance tracking highlighter are:

      * ``before_pre_grad_graph.txt``
      * ``after_post_grad_graph.txt``
      * ``inductor_aot_wrapper_code.txt``
      * ``inductor_output_code.txt``
      * ``inductor_provenance_tracking_node_mappings.json``

After running ``tlparse <file_name> --inductor-provenance``, you should see an additional "Provenance Tracking" section in the tlparse output. Clicking into the link(s) to access the provenance tracking tool.
For a demo, see: https://github.com/pytorch/tlparse/pull/93

 .. image:: _static/img/inductor_provenance/index.png


Source code corresponding to each Inductor kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``INDUCTOR_PROVENANCE=1``, you can also view the source code corresponding to each Inductor kernel in tlparse. To access it, click the "readable_html" link next to "inductor_provenance_tracking_kernel_stack_traces.json" in the tlparse output.

 .. image:: _static/img/inductor_provenance/index_2.png


Below are some example screenshots. The ``:1`` and ``:467`` suffixes at the end of the kernel names are used to distinguish different calls to the same kernel. We refer to these suffixes as debug handles.

 .. image:: _static/img/inductor_provenance/kernel_source_1.png
 .. image:: _static/img/inductor_provenance/kernel_source_2.png

You can also find the debug handle in the comments within the kernel source code.

 .. image:: _static/img/inductor_provenance/kernel_source_3.png


See Also
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tlparse`` is a tool written in Rust.

- Link to the tlparse GitHub repo: https://github.com/pytorch/tlparse
- Learn more about ``tlparse`` at :ref:`torch.compiler_troubleshooting`

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

- **File Documentation**: `torch.compiler_inductor_provenance.rst_docs.md`
- **Keyword Index**: `torch.compiler_inductor_provenance.rst_kw.md`
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

- **File Documentation**: `torch.compiler_inductor_provenance.rst_docs.md_docs.md`
- **Keyword Index**: `torch.compiler_inductor_provenance.rst_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
