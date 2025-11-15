# Documentation: `.github/workflows/inductor-perf-test-nightly-aarch64.yml`

## File Metadata

- **Path**: `.github/workflows/inductor-perf-test-nightly-aarch64.yml`
- **Size**: 9,092 bytes (8.88 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
name: inductor-perf-nightly-aarch64

on:
  schedule:
    # Does not perform max_autotune on CPU, so skip the weekly run setup
    - cron: 0 7 * * *
  # NB: GitHub has an upper limit of 10 inputs here
  workflow_dispatch:
    inputs:
      training:
        # CPU for training is not typical, but leave the option open here
        description: Run training (off by default)?
        required: false
        type: boolean
        default: false
      inference:
        description: Run inference (on by default)?
        required: false
        type: boolean
        default: true
      default:
        description: Run inductor_default?
        required: false
        type: boolean
        default: true
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
      aotinductor:
        description: Run aot_inductor for inference?
        required: false
        type: boolean
        default: false
      benchmark_configs:
        description: The list of configs used the benchmark
        required: false
        type: string
        default: inductor_huggingface_perf_cpu_aarch64,inductor_timm_perf_cpu_aarch64,inductor_torchbench_perf_cpu_aarch64

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read

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

  linux-jammy-aarch64-py3_10-inductor-build:
    name: linux-jammy-aarch64-py3.10-inductor
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runner: linux.arm64.m7g.4xlarge
      build-environment: linux-jammy-aarch64-py3.10
      docker-image-name: ci-image:pytorch-linux-jammy-aarch64-py3.10-gcc13-inductor-benchmarks
      test-matrix: |
        { include: [
          { config: "inductor_huggingface_perf_cpu_aarch64", shard: 1, num_shards: 9, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_huggingface_perf_cpu_aarch64", shard: 2, num_shards: 9, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_huggingface_perf_cpu_aarch64", shard: 3, num_shards: 9, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_huggingface_perf_cpu_aarch64", shard: 4, num_shards: 9, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_huggingface_perf_cpu_aarch64", shard: 5, num_shards: 9, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_huggingface_perf_cpu_aarch64", shard: 6, num_shards: 9, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_huggingface_perf_cpu_aarch64", shard: 7, num_shards: 9, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_huggingface_perf_cpu_aarch64", shard: 8, num_shards: 9, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_huggingface_perf_cpu_aarch64", shard: 9, num_shards: 9, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard:  1, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard:  2, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard:  3, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard:  4, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard:  5, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard:  6, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard:  7, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard:  8, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard:  9, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard: 10, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard: 11, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard: 12, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard: 13, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard: 14, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_timm_perf_cpu_aarch64", shard: 15, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard:  1, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard:  2, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard:  3, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard:  4, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard:  5, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard:  6, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard:  7, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard:  8, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard:  9, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard: 10, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard: 11, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard: 12, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard: 13, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard: 14, num_shards: 15, runner: "linux.arm64.m7g.metal" },
          { config: "inductor_torchbench_perf_cpu_aarch64", shard: 15, num_shards: 15, runner: "linux.arm64.m7g.metal" },
        ]}
      selected-test-configs: ${{ inputs.benchmark_configs }}
      build-additional-packages: "vision audio torchao"
    secrets: inherit


  linux-jammy-aarch64-py3_10-inductor-test-nightly:
    name: linux-jammy-aarch64-py3.10-inductor
    uses: ./.github/workflows/_linux-test.yml
    needs: linux-jammy-aarch64-py3_10-inductor-build
    if: github.event.schedule == '0 7 * * *'
    with:
      build-environment: linux-jammy-aarch64-py3.10
      dashboard-tag: training-false-inference-true-default-true-dynamic-true-cppwrapper-true-aotinductor-true
      docker-image: ${{ needs.linux-jammy-aarch64-py3_10-inductor-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-aarch64-py3_10-inductor-build.outputs.test-matrix }}
      timeout-minutes: 720
      # disable monitor in perf tests for more investigation
      disable-monitor: false
      monitor-log-interval: 15
      monitor-data-collect-interval: 4
    secrets: inherit


  linux-jammy-aarch64-py3_10-inductor-test:
    name: linux-jammy-aarch64-py3.10-inductor
    uses: ./.github/workflows/_linux-test.yml
    needs: linux-jammy-aarch64-py3_10-inductor-build
    if: github.event_name == 'workflow_dispatch'
    with:
      build-environment: linux-jammy-aarch64-py3.10
      dashboard-tag: training-${{ inputs.training }}-inference-${{ inputs.inference }}-default-${{ inputs.default }}-dynamic-${{ inputs.dynamic }}-cppwrapper-${{ inputs.cppwrapper }}-aotinductor-${{ inputs.aotinductor }}
      docker-image: ${{ needs.linux-jammy-aarch64-py3_10-inductor-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-aarch64-py3_10-inductor-build.outputs.test-matrix }}
      timeout-minutes: 720
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
python .github/workflows/inductor-perf-test-nightly-aarch64.yml
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

- **File Documentation**: `inductor-perf-test-nightly-aarch64.yml_docs.md`
- **Keyword Index**: `inductor-perf-test-nightly-aarch64.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
