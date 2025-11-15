# Documentation: `.github/workflows/inductor-unittest.yml`

## File Metadata

- **Path**: `.github/workflows/inductor-unittest.yml`
- **Size**: 6,916 bytes (6.75 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
# Workflow: Inductor Unit Test
# 1. runs unit tests for inductor.
# 2. performs daily memory leak checks and reruns of disabled tests, scheduled at `29 8 * * *`.
name: inductor-unittest

on:
  workflow_call:
  schedule:
    - cron: 29 8 * * * # about 1:29am PDT, for mem leak check and rerun disabled tests.

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-unittest
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
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm86
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc9-inductor-benchmarks
      cuda-arch-list: '8.6'
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      test-matrix: |
        { include: [
          { config: "inductor", shard: 1, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.g5.4xlarge.nvidia.gpu" },
          { config: "inductor", shard: 2, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.g5.4xlarge.nvidia.gpu" },
          { config: "inductor_distributed", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.g5.12xlarge.nvidia.gpu" },
          { config: "inductor_cpp_wrapper", shard: 1, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.g5.4xlarge.nvidia.gpu" },
          { config: "inductor_cpp_wrapper", shard: 2, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.g5.4xlarge.nvidia.gpu" },
        ]}
    secrets: inherit

  inductor-test:
    name: inductor-test
    uses: ./.github/workflows/_linux-test.yml
    needs: inductor-build
    with:
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm86
      docker-image: ${{ needs.inductor-build.outputs.docker-image }}
      test-matrix: ${{ needs.inductor-build.outputs.test-matrix }}
    secrets: inherit

  inductor-halide-build:
    name: inductor-halide-build
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      build-environment: linux-jammy-py3.12-gcc11
      docker-image-name: ci-image:pytorch-linux-jammy-py3.12-halide
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      test-matrix: |
        { include: [
          { config: "inductor-halide", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.12xlarge" },
        ]}
    secrets: inherit

  inductor-halide-test:
    name: inductor-halide-test
    uses: ./.github/workflows/_linux-test.yml
    needs: inductor-halide-build
    with:
      build-environment: linux-jammy-py3.12-gcc11
      docker-image: ${{ needs.inductor-halide-build.outputs.docker-image }}
      test-matrix: ${{ needs.inductor-halide-build.outputs.test-matrix }}
    secrets: inherit

  inductor-pallas-build:
    name: inductor-pallas-build
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      build-environment: linux-jammy-cuda12.8-py3.12-gcc11
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-py3.12-pallas
      cuda-arch-list: '8.9'
      runner: linux.8xlarge.memory
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      test-matrix: |
        { include: [
          { config: "inductor-pallas", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.g5.12xlarge.nvidia.gpu" },
        ]}
    secrets: inherit

  inductor-pallas-test:
    name: inductor-pallas-test
    uses: ./.github/workflows/_linux-test.yml
    needs: inductor-pallas-build
    with:
      build-environment: linux-jammy-py3.12-gcc11
      docker-image: ${{ needs.inductor-pallas-build.outputs.docker-image }}
      test-matrix: ${{ needs.inductor-pallas-build.outputs.test-matrix }}
    secrets: inherit

  inductor-triton-cpu-build:
    name: inductor-triton-cpu-build
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      build-environment: linux-jammy-py3.12-gcc11
      docker-image-name: ci-image:pytorch-linux-jammy-py3.12-triton-cpu
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      test-matrix: |
        { include: [
          { config: "inductor-triton-cpu", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.12xlarge" },
        ]}
    secrets: inherit

  inductor-triton-cpu-test:
    name: linux-jammy-cpu-py3.12-gcc11-inductor-triton-cpu
    uses: ./.github/workflows/_linux-test.yml
    needs: inductor-triton-cpu-build
    with:
      build-environment: linux-jammy-py3.12-gcc11
      docker-image: ${{ needs.inductor-triton-cpu-build.outputs.docker-image }}
      test-matrix: ${{ needs.inductor-triton-cpu-build.outputs.test-matrix }}
    secrets: inherit

  inductor-cpu-build:
    name: inductor-cpu-build
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      build-environment: linux-jammy-py3.10-gcc11-build
      docker-image-name: ci-image:pytorch-linux-jammy-py3-gcc11-inductor-benchmarks
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      test-matrix: |
        { include: [
          { config: "inductor_amx", shard: 1, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge.amx" },
          { config: "inductor_amx", shard: 2, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge.amx" },
          { config: "inductor_avx2", shard: 1, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge.avx2" },
          { config: "inductor_avx2", shard: 2, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge.avx2" },
        ]}
    secrets: inherit

  inductor-cpu-test:
    name: inductor-cpu-test
    uses: ./.github/workflows/_linux-test.yml
    needs: inductor-cpu-build
    with:
      build-environment: linux-jammy-py3.10-gcc11-build
      docker-image: ${{ needs.inductor-cpu-build.outputs.docker-image }}
      test-matrix: ${{ needs.inductor-cpu-build.outputs.test-matrix }}
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
python .github/workflows/inductor-unittest.yml
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
- [`_linux-build.yml_docs.md`](./_linux-build.yml_docs.md)
- [`inductor-perf-test-nightly.yml_docs.md`](./inductor-perf-test-nightly.yml_docs.md)


## Cross-References

- **File Documentation**: `inductor-unittest.yml_docs.md`
- **Keyword Index**: `inductor-unittest.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
