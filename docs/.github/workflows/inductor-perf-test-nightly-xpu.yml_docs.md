# Documentation: `.github/workflows/inductor-perf-test-nightly-xpu.yml`

## File Metadata

- **Path**: `.github/workflows/inductor-perf-test-nightly-xpu.yml`
- **Size**: 6,697 bytes (6.54 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
name: inductor-perf-nightly-xpu

on:
  push:
    tags:
      - ciflow/inductor-perf-test-nightly-xpu/*
  schedule:
    - cron: 30 17 * * *
  workflow_dispatch:
    inputs:
      training:
        description: Run training (on by default)?
        required: false
        type: boolean
        default: true
      inference:
        description: Run inference (on by default)?
        required: false
        type: boolean
        default: true
      default:
        description: Run inductor_default?
        required: false
        type: boolean
        default: false
      dynamic:
        description: Run inductor_dynamic_shapes?
        required: false
        type: boolean
        default: false
      cppwrapper:
        description: Run inductor_cpp_wrapper?
        required: false
        type: boolean
        default: false
      cudagraphs:
        description: Run inductor_cudagraphs?
        required: false
        type: boolean
        default: false
      freezing_cudagraphs:
        description: Run inductor_cudagraphs with freezing for inference?
        required: false
        type: boolean
        default: false
      aotinductor:
        description: Run aot_inductor for inference?
        required: false
        type: boolean
        default: false
      maxautotune:
        description: Run inductor_max_autotune?
        required: false
        type: boolean
        default: false
      benchmark_configs:
        description: The list of configs used the benchmark
        required: false
        type: string
        default: inductor_huggingface_perf,inductor_timm_perf,inductor_torchbench_perf,cachebench

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}
  cancel-in-progress: true

permissions: read-all

jobs:
  get-label-type:
    name: get-label-type
    uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main
    if: ${{ (github.event_name != 'schedule' || github.repository == 'pytorch/pytorch') && github.repository_owner == 'pytorch' }}
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}
      curr_ref_type: ${{ github.ref_type }}
      opt_out_experiments: lf

  xpu-n-py3_10-inductor-benchmark-build:
    name: xpu-n-py3.10-inductor-benchmark
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-noble-xpu-n-py3.10
      docker-image-name: ci-image:pytorch-linux-noble-xpu-n-py3-inductor-benchmarks
      runner: linux.c7i.12xlarge
      test-matrix: |
        { include: [
          { config: "inductor_huggingface_perf_xpu", shard: 1, num_shards: 5, runner: "linux.idc.xpu" },
          { config: "inductor_huggingface_perf_xpu", shard: 2, num_shards: 5, runner: "linux.idc.xpu" },
          { config: "inductor_huggingface_perf_xpu", shard: 3, num_shards: 5, runner: "linux.idc.xpu" },
          { config: "inductor_huggingface_perf_xpu", shard: 4, num_shards: 5, runner: "linux.idc.xpu" },
          { config: "inductor_huggingface_perf_xpu", shard: 5, num_shards: 5, runner: "linux.idc.xpu" },
          { config: "inductor_timm_perf_xpu", shard: 1, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_timm_perf_xpu", shard: 2, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_timm_perf_xpu", shard: 3, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_timm_perf_xpu", shard: 4, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_timm_perf_xpu", shard: 5, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_timm_perf_xpu", shard: 6, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_torchbench_perf_xpu", shard: 1, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_torchbench_perf_xpu", shard: 2, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_torchbench_perf_xpu", shard: 3, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_torchbench_perf_xpu", shard: 4, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_torchbench_perf_xpu", shard: 5, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "inductor_torchbench_perf_xpu", shard: 6, num_shards: 6, runner: "linux.idc.xpu" },
        ]}
    secrets: inherit

  xpu-n-py3_10-inductor-benchmark-test-nightly:
    permissions:
      id-token: write
      contents: read
    if: github.event_name != 'workflow_dispatch'
    name: xpu-n-py3.10-inductor-benchmark
    uses: ./.github/workflows/_xpu-test.yml
    needs: xpu-n-py3_10-inductor-benchmark-build
    with:
      build-environment: linux-noble-xpu-n-py3.10
      dashboard-tag: training-true-inference-true-default-true-dynamic-true-cudagraphs-false-cppwrapper-true-aotinductor-true-freezing_cudagraphs-false-cudagraphs_low_precision-false
      docker-image: ${{ needs.xpu-n-py3_10-inductor-benchmark-build.outputs.docker-image }}
      test-matrix: ${{ needs.xpu-n-py3_10-inductor-benchmark-build.outputs.test-matrix }}
      timeout-minutes: 720
      # Disable monitor in perf tests for more investigation
      disable-monitor: true
      monitor-log-interval: 10
      monitor-data-collect-interval: 2
    secrets: inherit

  xpu-n-py3_10-inductor-benchmark-test:
    permissions:
      id-token: write
      contents: read
    if: github.event_name == 'workflow_dispatch'
    name: xpu-n-py3.10-inductor-test
    uses: ./.github/workflows/_xpu-test.yml
    needs: xpu-n-py3_10-inductor-benchmark-build
    with:
      build-environment: linux-noble-xpu-n-py3.10
      dashboard-tag: training-${{ inputs.training }}-inference-${{ inputs.inference }}-default-${{ inputs.default }}-dynamic-${{ inputs.dynamic }}-cudagraphs-${{ inputs.cudagraphs }}-cppwrapper-${{ inputs.cppwrapper }}-aotinductor-${{ inputs.aotinductor }}-maxautotune-${{ inputs.maxautotune }}-freezing_cudagraphs-${{ inputs.freezing_cudagraphs }}-cudagraphs_low_precision-${{ inputs.cudagraphs }}
      docker-image: ${{ needs.xpu-n-py3_10-inductor-benchmark-build.outputs.docker-image }}
      test-matrix: ${{ needs.xpu-n-py3_10-inductor-benchmark-build.outputs.test-matrix }}
      timeout-minutes: 720
      disable-monitor: false
      monitor-log-interval: 15
      monitor-data-collect-interval: 4
    secrets: inherit

```



## High-Level Overview

This file is part of the PyTorch framework located at `.github/workflows`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/workflows`, which is part of the PyTorch project infrastructure.



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
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python .github/workflows/inductor-perf-test-nightly-xpu.yml
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.github/workflows`):

- [`unstable-periodic.yml_docs.md`](./unstable-periodic.yml_docs.md)
- [`runner_determinator_script_sync.yaml_docs.md`](./runner_determinator_script_sync.yaml_docs.md)
- [`auto_request_review.yml_docs.md`](./auto_request_review.yml_docs.md)
- [`attention_op_microbenchmark.yml_docs.md`](./attention_op_microbenchmark.yml_docs.md)
- [`inductor-nightly.yml_docs.md`](./inductor-nightly.yml_docs.md)
- [`lint-autoformat.yml_docs.md`](./lint-autoformat.yml_docs.md)
- [`inductor-perf-test-b200.yml_docs.md`](./inductor-perf-test-b200.yml_docs.md)
- [`inductor-unittest.yml_docs.md`](./inductor-unittest.yml_docs.md)
- [`_linux-build.yml_docs.md`](./_linux-build.yml_docs.md)
- [`inductor-perf-test-nightly.yml_docs.md`](./inductor-perf-test-nightly.yml_docs.md)


## Cross-References

- **File Documentation**: `inductor-perf-test-nightly-xpu.yml_docs.md`
- **Keyword Index**: `inductor-perf-test-nightly-xpu.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
