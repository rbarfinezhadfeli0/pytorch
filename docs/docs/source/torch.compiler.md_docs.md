# Documentation: `docs/source/torch.compiler.md`

## File Metadata

- **Path**: `docs/source/torch.compiler.md`
- **Size**: 4,720 bytes (4.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
(torch.compiler_overview)=

# torch.compiler

`torch.compiler` is a namespace through which some of the internal compiler
methods are surfaced for user consumption. The main function and the feature in
this namespace is `torch.compile`.

`torch.compile` is a PyTorch function introduced in PyTorch 2.x that aims to
solve the problem of accurate graph capturing in PyTorch and ultimately enable
software engineers to run their PyTorch programs faster. `torch.compile` is
written in Python and it marks the transition of PyTorch from C++ to Python.

`torch.compile` leverages the following underlying technologies:

- **TorchDynamo (torch._dynamo)** is an internal API that uses a CPython
  feature called the Frame Evaluation API to safely capture PyTorch graphs.
  Methods that are available externally for PyTorch users are surfaced
  through the `torch.compiler` namespace.
- **TorchInductor** is the default `torch.compile` deep learning compiler
  that generates fast code for multiple accelerators and backends. You
  need to use a backend compiler to make speedups through `torch.compile`
  possible. For NVIDIA, AMD and Intel GPUs, it leverages OpenAI Triton as the key
  building block.
- **AOT Autograd** captures not only the user-level code, but also backpropagation,
  which results in capturing the backwards pass "ahead-of-time". This enables
  acceleration of both forwards and backwards pass using TorchInductor.

To better understand how `torch.compile` tracing behavior on your code, or to
learn more about the internals of `torch.compile`, please refer to the [`torch.compile` programming model](compile/programming_model.md).

:::{note}
In some cases, the terms `torch.compile`, TorchDynamo, `torch.compiler`
might be used interchangeably in this documentation.
:::

As mentioned above, to run your workflows faster, `torch.compile` through
TorchDynamo requires a backend that converts the captured graphs into a fast
machine code. Different backends can result in various optimization gains.
The default backend is called TorchInductor, also known as *inductor*,
TorchDynamo has a list of supported backends developed by our partners,
which can be seen by running `torch.compiler.list_backends()` each of which
with its optional dependencies.

Some of the most commonly used backends include:

**Training & inference backends**

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Backend
     - Description
   * - ``torch.compile(m, backend="inductor")``
     - Uses the TorchInductor backend. `Read more <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
   * - ``torch.compile(m, backend="cudagraphs")``
     - CUDA graphs with AOT Autograd. `Read more <https://github.com/pytorch/torchdynamo/pull/757>`__
   * - ``torch.compile(m, backend="ipex")``
     - Uses IPEX on CPU. `Read more <https://github.com/intel/intel-extension-for-pytorch>`__
```

**Inference-only backends**

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Backend
     - Description
   * - ``torch.compile(m, backend="tensorrt")``
     - Uses Torch-TensorRT for inference optimizations. Requires ``import torch_tensorrt`` in the calling script to register backend. `Read more <https://github.com/pytorch/TensorRT>`__
   * - ``torch.compile(m, backend="ipex")``
     - Uses IPEX for inference on CPU. `Read more <https://github.com/intel/intel-extension-for-pytorch>`__
   * - ``torch.compile(m, backend="tvm")``
     - Uses Apache TVM for inference optimizations. `Read more <https://tvm.apache.org/>`__
   * - ``torch.compile(m, backend="openvino")``
     - Uses OpenVINO for inference optimizations. `Read more <https://docs.openvino.ai/torchcompile>`__
```

## Read More

```{toctree}
:caption: Getting Started for PyTorch Users
:maxdepth: 2

torch.compiler_get_started
torch.compiler_api
torch.compiler.config
torch.compiler_dynamic_shapes
torch.compiler_fine_grain_apis
torch.compiler_backward
torch.compiler_aot_inductor
torch.compiler_inductor_profiling
torch.compiler_profiling_torch_compile
torch.compiler_faq
torch.compiler_troubleshooting
torch.compiler_performance_dashboard
torch.compiler_inductor_provenance
```

```{toctree}
:caption: torch.compile Programming Model
:maxdepth: 2

compile/programming_model
```

```{toctree}
:caption: Deep Dive for PyTorch Developers
:maxdepth: 1

torch.compiler_dynamo_overview
torch.compiler_dynamo_deepdive
torch.compiler_nn_module
torch.compiler_cudagraph_trees
torch.compiler_fake_tensor
```

```{toctree}
:caption: HowTo for PyTorch Backend Vendors
:maxdepth: 1

torch.compiler_custom_backends
torch.compiler_transformations
torch.compiler_ir
```

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

- **File Documentation**: `torch.compiler.md_docs.md`
- **Keyword Index**: `torch.compiler.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
