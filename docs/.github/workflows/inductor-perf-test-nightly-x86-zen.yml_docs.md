# Documentation: `.github/workflows/inductor-perf-test-nightly-x86-zen.yml`

## File Metadata

- **Path**: `.github/workflows/inductor-perf-test-nightly-x86-zen.yml`
- **Size**: 5,836 bytes (5.70 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
name: inductor-perf-nightly-x86-zen

on:
  push:
    tags:
      - ciflow/inductor-perf-test-nightly-x86-zen/*
  schedule:
    # - cron: 0 7 * * 1-6
    # - cron: 0 7 * * 0
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
      freezing:
        description: Run freezing?
        required: false
        type: boolean
        default: true
      benchmark_configs:
        description: The list of configs used the benchmark
        required: false
        type: string
        default: inductor_huggingface_perf_cpu_x86_zen,inductor_timm_perf_cpu_x86_zen,inductor_torchbench_perf_cpu_x86_zen

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

  inductor-build:
    name: inductor-build
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-gcc11-build
      docker-image-name: ci-image:pytorch-linux-jammy-py3-gcc11-inductor-benchmarks
      test-matrix: |
        { include: [
          { config: "inductor_huggingface_perf_cpu_x86_zen", shard: 1, num_shards: 3, runner: "linux.24xlarge.amd" },
          { config: "inductor_huggingface_perf_cpu_x86_zen", shard: 2, num_shards: 3, runner: "linux.24xlarge.amd" },
          { config: "inductor_huggingface_perf_cpu_x86_zen", shard: 3, num_shards: 3, runner: "linux.24xlarge.amd" },
          { config: "inductor_timm_perf_cpu_x86_zen", shard: 1, num_shards: 5, runner: "linux.24xlarge.amd" },
          { config: "inductor_timm_perf_cpu_x86_zen", shard: 2, num_shards: 5, runner: "linux.24xlarge.amd" },
          { config: "inductor_timm_perf_cpu_x86_zen", shard: 3, num_shards: 5, runner: "linux.24xlarge.amd" },
          { config: "inductor_timm_perf_cpu_x86_zen", shard: 4, num_shards: 5, runner: "linux.24xlarge.amd" },
          { config: "inductor_timm_perf_cpu_x86_zen", shard: 5, num_shards: 5, runner: "linux.24xlarge.amd" },
          { config: "inductor_torchbench_perf_cpu_x86_zen", shard: 1, num_shards: 4, runner: "linux.24xlarge.amd" },
          { config: "inductor_torchbench_perf_cpu_x86_zen", shard: 2, num_shards: 4, runner: "linux.24xlarge.amd" },
          { config: "inductor_torchbench_perf_cpu_x86_zen", shard: 3, num_shards: 4, runner: "linux.24xlarge.amd" },
          { config: "inductor_torchbench_perf_cpu_x86_zen", shard: 4, num_shards: 4, runner: "linux.24xlarge.amd" },
        ]}
      selected-test-configs: ${{ inputs.benchmark_configs }}
    secrets: inherit

  inductor-test-nightly:
    name: inductor-test-nightly
    uses: ./.github/workflows/_linux-test.yml
    needs: inductor-build
    if: github.event.schedule == '0 7 * * *'
    with:
      build-environment: linux-jammy-py3.10-gcc11-build
      dashboard-tag: training-false-inference-true-default-true-dynamic-true-cppwrapper-true-aotinductor-true-freezing-true
      docker-image: ${{ needs.inductor-build.outputs.docker-image }}
      test-matrix: ${{ needs.inductor-build.outputs.test-matrix }}
      timeout-minutes: 720
      # disable monitor in perf tests
      disable-monitor: false
      monitor-log-interval: 15
      monitor-data-collect-interval: 4
    secrets: inherit

  inductor-test:
    name: inductor-test
    uses: ./.github/workflows/_linux-test.yml
    needs: inductor-build
    with:
      build-environment: linux-jammy-py3.10-gcc11-build
      dashboard-tag: training-${{ inputs.training || 'false' }}-inference-${{ inputs.inference || 'true' }}-default-${{ inputs.default || 'true' }}-dynamic-${{ inputs.dynamic || 'true' }}-cppwrapper-${{ inputs.cppwrapper || 'true' }}-aotinductor-${{ inputs.aotinductor || 'true' }}-freezing-${{ inputs.freezing || 'true' }}
      docker-image: ${{ needs.inductor-build.outputs.docker-image }}
      test-matrix: ${{ needs.inductor-build.outputs.test-matrix }}
      timeout-minutes: 720
      # disable monitor in perf tests
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
python .github/workflows/inductor-perf-test-nightly-x86-zen.yml
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

- **File Documentation**: `inductor-perf-test-nightly-x86-zen.yml_docs.md`
- **Keyword Index**: `inductor-perf-test-nightly-x86-zen.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
