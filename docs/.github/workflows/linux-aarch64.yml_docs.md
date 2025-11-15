# Documentation: `.github/workflows/linux-aarch64.yml`

## File Metadata

- **Path**: `.github/workflows/linux-aarch64.yml`
- **Size**: 2,778 bytes (2.71 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: linux-aarch64

on:
  push:
    branches:
      - main
      - release/*
    tags:
      - ciflow/linux-aarch64/*
      - ciflow/trunk/*
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }} but found ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}
  cancel-in-progress: true

jobs:

  get-label-type:
    if: github.repository_owner == 'pytorch'
    name: get-label-type
    uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}
      curr_ref_type: ${{ github.ref_type }}

  linux-jammy-aarch64-py3_10-build:
    name: linux-jammy-aarch64-py3.10
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: ${{ needs.get-label-type.outputs.label-type }}
      build-environment: linux-jammy-aarch64-py3.10
      docker-image-name: ci-image:pytorch-linux-jammy-aarch64-py3.10-gcc13
      runner: linux.arm64.m7g.4xlarge
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.arm64.m7g.4xlarge" },
          { config: "default", shard: 2, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.arm64.m7g.4xlarge" },
          { config: "default", shard: 3, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.arm64.m7g.4xlarge" },
          { config: "default", shard: 1, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.arm64.m8g.4xlarge" },
          { config: "default", shard: 2, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.arm64.m8g.4xlarge" },
          { config: "default", shard: 3, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.arm64.m8g.4xlarge" },
        ]}
    secrets: inherit

  linux-jammy-aarch64-py3_10-test:
    name: linux-jammy-aarch64-py3.10
    uses: ./.github/workflows/_linux-test.yml
    needs: linux-jammy-aarch64-py3_10-build
    permissions:
      id-token: write
      contents: read
    with:
      build-environment: linux-jammy-aarch64-py3.10
      docker-image: ${{ needs.linux-jammy-aarch64-py3_10-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-aarch64-py3_10-build.outputs.test-matrix }}
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

- **File Documentation**: `linux-aarch64.yml_docs.md`
- **Keyword Index**: `linux-aarch64.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
