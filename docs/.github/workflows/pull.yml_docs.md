# Documentation: `.github/workflows/pull.yml`

## File Metadata

- **Path**: `.github/workflows/pull.yml`
- **Size**: 17,136 bytes (16.73 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: pull

on:
  pull_request:
    branches-ignore:
      - nightly
  push:
    branches:
      - main
      - release/*
      - landchecks/*
    tags:
      - ciflow/pull/*
  workflow_dispatch:
  schedule:
    - cron: 29 8 * * *  # about 1:29am PDT

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read

jobs:
  llm-td:
    if: github.repository_owner == 'pytorch'
    name: before-test
    uses: ./.github/workflows/llm_td_retrieval.yml
    permissions:
      id-token: write
      contents: read

  target-determination:
    name: before-test
    uses: ./.github/workflows/target_determination.yml
    needs: llm-td
    permissions:
      id-token: write
      contents: read

  get-label-type:
    name: get-label-type
    uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main
    if: ${{ (github.event_name != 'schedule' || github.repository == 'pytorch/pytorch') && github.repository_owner == 'pytorch' }}
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}

  linux-jammy-py3_10-gcc11-build:
    name: linux-jammy-py3.10-gcc11
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-gcc11
      docker-image-name: ci-image:pytorch-linux-jammy-py3.10-gcc11
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "default", shard: 2, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "default", shard: 3, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "default", shard: 4, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "default", shard: 5, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "docs_test", shard: 1, num_shards: 1,  runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "jit_legacy", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "backwards_compat", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
          { config: "distributed", shard: 1, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "distributed", shard: 2, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "numpy_2_x", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
        ]}
    secrets: inherit

  linux-jammy-py3_10-gcc11-test:
    name: linux-jammy-py3.10-gcc11
    uses: ./.github/workflows/_linux-test.yml
    needs:
      - linux-jammy-py3_10-gcc11-build
      - target-determination
    with:
      build-environment: linux-jammy-py3.10-gcc11
      docker-image: ${{ needs.linux-jammy-py3_10-gcc11-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-py3_10-gcc11-build.outputs.test-matrix }}
    secrets: inherit

  linux-docs:
    name: linux-docs
    uses: ./.github/workflows/_docs.yml
    needs: linux-jammy-py3_10-gcc11-build
    with:
      build-environment: linux-jammy-py3.10-gcc11
      docker-image: ${{ needs.linux-jammy-py3_10-gcc11-build.outputs.docker-image }}
    secrets: inherit

  linux-jammy-py3_10-gcc11-no-ops:
    name: linux-jammy-py3.10-gcc11-no-ops
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-gcc11-no-ops
      docker-image-name: ci-image:pytorch-linux-jammy-py3.10-gcc11
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 1 },
        ]}
    secrets: inherit

  linux-jammy-py3_10-gcc11-pch:
    name: linux-jammy-py3.10-gcc11-pch
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-gcc11-pch
      docker-image-name: ci-image:pytorch-linux-jammy-py3.10-gcc11
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 1 },
        ]}
    secrets: inherit

  linux-jammy-py3_10-clang18-asan-build:
    name: linux-jammy-py3.10-clang18-asan
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner: linux.2xlarge.memory
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-clang18-asan
      docker-image-name: ci-image:pytorch-linux-jammy-py3-clang18-asan
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 7, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 2, num_shards: 7, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 3, num_shards: 7, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 4, num_shards: 7, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 5, num_shards: 7, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 6, num_shards: 7, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 7, num_shards: 7, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
        ]}
      sync-tag: asan-build
    secrets: inherit

  linux-jammy-py3_10-clang18-asan-test:
    name: linux-jammy-py3.10-clang18-asan
    uses: ./.github/workflows/_linux-test.yml
    needs:
      - linux-jammy-py3_10-clang18-asan-build
      - target-determination
    with:
      build-environment: linux-jammy-py3.10-clang18-asan
      docker-image: ${{ needs.linux-jammy-py3_10-clang18-asan-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-py3_10-clang18-asan-build.outputs.test-matrix }}
      sync-tag: asan-test
    secrets: inherit

  linux-jammy-py3_10-clang12-onnx-build:
    name: linux-jammy-py3.10-clang12-onnx
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-clang12-onnx
      docker-image-name: ci-image:pytorch-linux-jammy-py3-clang12-onnx
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
          { config: "default", shard: 2, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
        ]}
    secrets: inherit

  linux-jammy-py3_10-clang12-onnx-test:
    name: linux-jammy-py3.10-clang12-onnx
    uses: ./.github/workflows/_linux-test.yml
    needs:
      - linux-jammy-py3_10-clang12-onnx-build
      - target-determination
    with:
      build-environment: linux-jammy-py3.10-clang12-onnx
      docker-image: ${{ needs.linux-jammy-py3_10-clang12-onnx-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-py3_10-clang12-onnx-build.outputs.test-matrix }}
    secrets: inherit

  linux-jammy-py3_10-clang12-build:
    name: linux-jammy-py3.10-clang12
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-clang12
      docker-image-name: ci-image:pytorch-linux-jammy-py3.10-clang12
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 2, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 3, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 4, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 5, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "crossref", shard: 1, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "crossref", shard: 2, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "dynamo_wrapped", shard: 1, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "dynamo_wrapped", shard: 2, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "dynamo_wrapped", shard: 3, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "einops", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" }
        ]}
    secrets: inherit

  linux-jammy-py3_10-clang12-test:
    name: linux-jammy-py3.10-clang12
    uses: ./.github/workflows/_linux-test.yml
    needs:
      - linux-jammy-py3_10-clang12-build
      - target-determination
    with:
      build-environment: linux-jammy-py3.10-clang12
      docker-image: ${{ needs.linux-jammy-py3_10-clang12-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-py3_10-clang12-build.outputs.test-matrix }}
    secrets: inherit

  linux-jammy-py3_13-clang12-build:
    name: linux-jammy-py3.13-clang12
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.13-clang12
      docker-image-name: ci-image:pytorch-linux-jammy-py3.13-clang12
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 2, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 3, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 4, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "default", shard: 5, num_shards: 5, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "crossref", shard: 1, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "crossref", shard: 2, num_shards: 2, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "dynamo_wrapped", shard: 1, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "dynamo_wrapped", shard: 2, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "dynamo_wrapped", shard: 3, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" },
          { config: "einops", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge" }
        ]}
    secrets: inherit

  linux-jammy-py3_13-clang12-test:
    name: linux-jammy-py3.13-clang12
    uses: ./.github/workflows/_linux-test.yml
    needs: linux-jammy-py3_13-clang12-build
    with:
      build-environment: linux-jammy-py3.13-clang12
      docker-image: ${{ needs.linux-jammy-py3_13-clang12-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-py3_13-clang12-build.outputs.test-matrix }}
    secrets: inherit

  linux-jammy-cuda12_8-cudnn9-py3_10-clang12-build:
    name: linux-jammy-cuda12.8-cudnn9-py3.10-clang12
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-cuda12.8-cudnn9-py3.10-clang12
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3.10-clang12
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 1 },
        ]}
    secrets: inherit

  linux-jammy-cpu-py3_10-gcc11-bazel-test:
    name: linux-jammy-cpu-py3.10-gcc11-bazel-test
    uses: ./.github/workflows/_bazel-build-test.yml
    needs: get-label-type
    with:
      runner: "${{ needs.get-label-type.outputs.label-type }}linux.large"
      build-environment: linux-jammy-cuda12.8-py3.10-gcc11-bazel-test
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc11
      cuda-version: cpu
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
        ]}
    secrets: inherit

  linux-jammy-py3_10-gcc11-mobile-lightweight-dispatch-build:
    name: linux-jammy-py3.10-gcc11-mobile-lightweight-dispatch-build
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-gcc11-mobile-lightweight-dispatch-build
      docker-image-name: ci-image:pytorch-linux-jammy-py3.10-gcc11
      build-generates-artifacts: false
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 1 },
        ]}
    secrets: inherit

  linux-jammy-rocm-py3_10-build:
    # don't run build twice on main
    if: github.event_name == 'pull_request'
    name: linux-jammy-rocm-py3.10
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-rocm-py3.10
      docker-image-name: ci-image:pytorch-linux-jammy-rocm-n-py3
      sync-tag: rocm-build
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 3, runner: "linux.rocm.gpu.2" },
          { config: "default", shard: 2, num_shards: 3, runner: "linux.rocm.gpu.2" },
          { config: "default", shard: 3, num_shards: 3, runner: "linux.rocm.gpu.2" },
        ]}
    secrets: inherit

  linux-jammy-cuda12_8-py3_10-gcc9-inductor-build:
    name: cuda12.8-py3.10-gcc9-sm75
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm75
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc9-inductor-benchmarks
      cuda-arch-list: '7.5'
      test-matrix: |
        { include: [
          { config: "pr_time_benchmarks", shard: 1, num_shards: 1, runner: "linux.g4dn.metal.nvidia.gpu" },
        ]}
    secrets: inherit

  linux-jammy-cuda12_8-py3_10-gcc9-inductor-test:
    name: cuda12.8-py3.10-gcc9-sm75
    uses: ./.github/workflows/_linux-test.yml
    needs: linux-jammy-cuda12_8-py3_10-gcc9-inductor-build
    with:
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm75
      docker-image: ${{ needs.linux-jammy-cuda12_8-py3_10-gcc9-inductor-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-cuda12_8-py3_10-gcc9-inductor-build.outputs.test-matrix }}
    secrets: inherit

  linux-noble-xpu-n-py3_10-build:
    name: linux-noble-xpu-n-py3.10
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      # This should sync with the build in xpu.yml but xpu uses a larger runner
      # sync-tag: linux-xpu-n-build
      runner_prefix: ${{ needs.get-label-type.outputs.label-type }}
      build-environment: linux-noble-xpu-n-py3.10
      docker-image-name: ci-image:pytorch-linux-noble-xpu-n-py3
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 4, runner: "linux.idc.xpu" },
          { config: "default", shard: 2, num_shards: 4, runner: "linux.idc.xpu" },
          { config: "default", shard: 3, num_shards: 4, runner: "linux.idc.xpu" },
          { config: "default", shard: 4, num_shards: 4, runner: "linux.idc.xpu" },
        ]}
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

- **File Documentation**: `pull.yml_docs.md`
- **Keyword Index**: `pull.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
