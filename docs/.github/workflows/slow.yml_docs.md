# Documentation: `.github/workflows/slow.yml`

## File Metadata

- **Path**: `.github/workflows/slow.yml`
- **Size**: 5,442 bytes (5.31 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
# This workflow is dedicated to host slow jobs that are run only periodically because
# they are too slow to run in every commit.  The list of slow tests can be found in
# https://github.com/pytorch/test-infra/blob/generated-stats/stats/slow-tests.json
name: slow

on:
  push:
    branches:
      - main
      - release/*
    tags:
      - ciflow/slow/*
  schedule:
    - cron: 29 8 * * *  # about 1:29am PDT, for mem leak check and rerun disabled tests
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}-${{ github.event.schedule }}
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
      curr_ref_type: ${{ github.ref_type }}

  linux-jammy-cuda12_8-py3_10-gcc11-sm86-build:
    name: linux-jammy-cuda12.8-py3.10-gcc11-sm86
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-cuda12.8-py3.10-gcc11-sm86
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc11
      cuda-arch-list: 8.6
      test-matrix: |
        { include: [
          { config: "slow", shard: 1, num_shards: 3, runner: "linux.g5.4xlarge.nvidia.gpu" },
          { config: "slow", shard: 2, num_shards: 3, runner: "linux.g5.4xlarge.nvidia.gpu" },
          { config: "slow", shard: 3, num_shards: 3, runner: "linux.g5.4xlarge.nvidia.gpu" },
        ]}
    secrets: inherit

  linux-jammy-cuda12_8-py3_10-gcc11-sm86-test:
    name: linux-jammy-cuda12.8-py3.10-gcc11-sm86
    uses: ./.github/workflows/_linux-test.yml
    needs:
      - linux-jammy-cuda12_8-py3_10-gcc11-sm86-build
      - target-determination
    with:
      build-environment: linux-jammy-cuda12.8-py3.10-gcc11-sm86
      docker-image: ${{ needs.linux-jammy-cuda12_8-py3_10-gcc11-sm86-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-cuda12_8-py3_10-gcc11-sm86-build.outputs.test-matrix }}
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
          { config: "slow", shard: 1, num_shards: 2, runner: "linux.2xlarge" },
          { config: "slow", shard: 2, num_shards: 2, runner: "linux.2xlarge" },
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
          { config: "slow", shard: 1, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "slow", shard: 2, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
          { config: "slow", shard: 3, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge" },
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

- **File Documentation**: `slow.yml_docs.md`
- **Keyword Index**: `slow.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
