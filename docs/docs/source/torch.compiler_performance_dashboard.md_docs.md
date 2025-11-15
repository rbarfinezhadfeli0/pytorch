# Documentation: `docs/source/torch.compiler_performance_dashboard.md`

## File Metadata

- **Path**: `docs/source/torch.compiler_performance_dashboard.md`
- **Size**: 3,361 bytes (3.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# PyTorch 2.0 Performance Dashboard

**Author:** [Bin Bao](https://github.com/desertfire) and [Huy Do](https://github.com/huydhn)

PyTorch 2.0's performance is tracked nightly on this [dashboard](https://hud.pytorch.org/benchmark/compilers).
The performance collection runs on 12 GCP A100 nodes every night. Each node contains a 40GB A100 Nvidia GPU and
a 6-core 2.2GHz Intel Xeon CPU. The corresponding CI workflow file can be found
[here](https://github.com/pytorch/pytorch/blob/main/.github/workflows/inductor-perf-test-nightly.yml).

## How to read the dashboard?

The landing page shows tables for all three benchmark suites we measure, ``TorchBench``, ``Huggingface``, and ``TIMM``,
and graphs for one benchmark suite with the default setting. For example, the default graphs currently show the AMP
training performance trend in the past 7 days for ``TorchBench``. Droplists on the top of that page can be
selected to view tables and graphs with different options. In addition to the pass rate, there are 3 key
performance metrics reported there: ``Geometric mean speedup``, ``Mean compilation time``, and
``Peak memory footprint compression ratio``.
Both ``Geometric mean speedup`` and ``Peak memory footprint compression ratio`` are compared against
the PyTorch eager performance, and the larger the better. Each individual performance number on those tables can be clicked,
which will bring you to a view with detailed numbers for all the tests in that specific benchmark suite.

## What is measured on the dashboard?

All the dashboard tests are defined in this
[function](https://github.com/pytorch/pytorch/blob/3e18d3958be3dfcc36d3ef3c481f064f98ebeaf6/.ci/pytorch/test.sh#L305).
The exact test configurations are subject to change, but at the moment, we measure both inference and training
performance with AMP precision on the three benchmark suites. We also measure different settings of TorchInductor,
including ``default``, ``with_cudagraphs (default + cudagraphs)``, and ``dynamic (default + dynamic_shapes)``.

## Can I check if my PR affects TorchInductor's performance on the dashboard before merging?

Individual dashboard runs can be triggered manually by clicking the ``Run workflow`` button
[here](https://github.com/pytorch/pytorch/actions/workflows/inductor-perf-test-nightly.yml)
and submitting with your PR's branch selected. This will kick off a whole dashboard run with your PR's changes.
Once it is done, you can check the results by selecting the corresponding branch name and commit ID
on the performance dashboard UI. Be aware that this is an expensive CI run. With the limited
resources, please use this functionality wisely.

## How can I run any performance test locally?

The exact command lines used during a complete dashboard run can be found in any recent CI run logs.
The [workflow page](https://github.com/pytorch/pytorch/actions/workflows/inductor-perf-test-nightly.yml)
is a good place to look for logs from some of the recent runs.
In those logs, you can search for lines like
`python benchmarks/dynamo/huggingface.py --performance --cold-start-latency --inference --amp --backend inductor --disable-cudagraphs --device cuda`
and run them locally if you have a GPU working with PyTorch 2.0.
``python benchmarks/dynamo/huggingface.py -h`` will give you a detailed explanation on options of the benchmarking script.

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

- **File Documentation**: `torch.compiler_performance_dashboard.md_docs.md`
- **Keyword Index**: `torch.compiler_performance_dashboard.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
