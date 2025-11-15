# Documentation: `.github/workflows/inductor-perf-test-nightly-rocm-mi355.yml`

## File Metadata

- **Path**: `.github/workflows/inductor-perf-test-nightly-rocm-mi355.yml`
- **Size**: 6,477 bytes (6.33 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
name: inductor-perf-nightly-rocm-mi355

on:
  push:
    tags:
      - ciflow/inductor-perf-test-nightly-rocm-mi355/*
  schedule:
    - cron: 15 0 * * *
  # NB: GitHub has an upper limit of 10 inputs here, so before we can sort it
  # out, let try to run torchao cudagraphs_low_precision as part of cudagraphs
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
        default: true
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
        default: inductor_huggingface_perf_rocm_mi355,inductor_timm_perf_rocm_mi355,inductor_torchbench_perf_rocm_mi355

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

  linux-jammy-rocm-py3_10-inductor-benchmark-build:
    if: github.repository_owner == 'pytorch'
    name: rocm-py3_10-inductor-benchmark-build
    uses: ./.github/workflows/_linux-build.yml
    with:
      build-environment: linux-jammy-rocm-py3_10
      docker-image-name: ci-image:pytorch-linux-jammy-rocm-n-py3-benchmarks
      test-matrix: |
        { include: [
          { config: "inductor_huggingface_perf_rocm_mi355", shard: 1, num_shards: 5, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_huggingface_perf_rocm_mi355", shard: 2, num_shards: 5, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_huggingface_perf_rocm_mi355", shard: 3, num_shards: 5, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_huggingface_perf_rocm_mi355", shard: 4, num_shards: 5, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_huggingface_perf_rocm_mi355", shard: 5, num_shards: 5, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_timm_perf_rocm_mi355", shard: 1, num_shards: 7, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_timm_perf_rocm_mi355", shard: 2, num_shards: 7, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_timm_perf_rocm_mi355", shard: 3, num_shards: 7, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_timm_perf_rocm_mi355", shard: 4, num_shards: 7, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_timm_perf_rocm_mi355", shard: 5, num_shards: 7, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_timm_perf_rocm_mi355", shard: 6, num_shards: 7, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_timm_perf_rocm_mi355", shard: 7, num_shards: 7, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_torchbench_perf_rocm_mi355", shard: 1, num_shards: 9, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_torchbench_perf_rocm_mi355", shard: 2, num_shards: 9, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_torchbench_perf_rocm_mi355", shard: 3, num_shards: 9, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_torchbench_perf_rocm_mi355", shard: 4, num_shards: 9, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_torchbench_perf_rocm_mi355", shard: 5, num_shards: 9, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_torchbench_perf_rocm_mi355", shard: 6, num_shards: 9, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_torchbench_perf_rocm_mi355", shard: 7, num_shards: 9, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_torchbench_perf_rocm_mi355", shard: 8, num_shards: 9, runner: "linux.rocm.gpu.mi355.1" },
          { config: "inductor_torchbench_perf_rocm_mi355", shard: 9, num_shards: 9, runner: "linux.rocm.gpu.mi355.1" },
        ]}
    secrets: inherit

  linux-jammy-rocm-py3_10-inductor-benchmark-test:
    permissions:
      id-token: write
      contents: read
    name: rocm-py3_10-inductor-benchmark-test
    uses: ./.github/workflows/_rocm-test.yml
    needs: linux-jammy-rocm-py3_10-inductor-benchmark-build
    with:
      build-environment: linux-jammy-rocm-py3_10
      dashboard-tag: training-true-inference-true-default-true-dynamic-true-cudagraphs-true-cppwrapper-true-aotinductor-true-freezing_cudagraphs-true-cudagraphs_low_precision-true
      docker-image: ${{ needs.linux-jammy-rocm-py3_10-inductor-benchmark-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-rocm-py3_10-inductor-benchmark-build.outputs.test-matrix }}
      timeout-minutes: 720
      # Disable monitor in perf tests for more investigation
      disable-monitor: true
      monitor-log-interval: 10
      monitor-data-collect-interval: 2
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
python .github/workflows/inductor-perf-test-nightly-rocm-mi355.yml
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

- **File Documentation**: `inductor-perf-test-nightly-rocm-mi355.yml_docs.md`
- **Keyword Index**: `inductor-perf-test-nightly-rocm-mi355.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
