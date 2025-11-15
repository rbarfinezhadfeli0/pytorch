# Documentation: `.github/workflows/xpu.yml`

## File Metadata

- **Path**: `.github/workflows/xpu.yml`
- **Size**: 4,482 bytes (4.38 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: xpu

on:
  push:
    tags:
      - ciflow/xpu/*
  workflow_dispatch:
  schedule:
    # Run 3 times on weekdays and less frequently on weekends.
    - cron: 45 0,8,16 * * 1-5
    - cron: 45 4 * * 0,6

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}
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

  linux-jammy-xpu-n-1-py3_10-build:
    name: linux-jammy-xpu-n-1-py3.10
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      sync-tag: linux-xpu-n-1-build
      runner_prefix: ${{ needs.get-label-type.outputs.label-type }}
      build-environment: linux-jammy-xpu-n-1-py3.10
      docker-image-name: ci-image:pytorch-linux-jammy-xpu-n-1-py3
      runner: linux.c7i.12xlarge
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "default", shard: 2, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "default", shard: 3, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "default", shard: 4, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "default", shard: 5, num_shards: 6, runner: "linux.idc.xpu" },
          { config: "default", shard: 6, num_shards: 6, runner: "linux.idc.xpu" },
        ]}
    secrets: inherit

  linux-noble-xpu-n-py3_10-build:
    name: linux-noble-xpu-n-py3.10
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      sync-tag: linux-xpu-n-build
      runner_prefix: ${{ needs.get-label-type.outputs.label-type }}
      build-environment: linux-noble-xpu-n-py3.10
      docker-image-name: ci-image:pytorch-linux-noble-xpu-n-py3
      runner: linux.c7i.12xlarge
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 2, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 3, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 4, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 5, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 6, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 7, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 8, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 9, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 10, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 11, num_shards: 12, runner: "linux.idc.xpu" },
          { config: "default", shard: 12, num_shards: 12, runner: "linux.idc.xpu" },
        ]}
    secrets: inherit

  linux-noble-xpu-n-py3_10-test:
    name: linux-noble-xpu-n-py3.10
    uses: ./.github/workflows/_xpu-test.yml
    needs: linux-noble-xpu-n-py3_10-build
    permissions:
      id-token: write
      contents: read
    with:
      build-environment: linux-noble-xpu-n-py3.10
      docker-image: ${{ needs.linux-noble-xpu-n-py3_10-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-noble-xpu-n-py3_10-build.outputs.test-matrix }}
    secrets: inherit

  windows-xpu-n-1-build:
    if: github.repository_owner == 'pytorch'
    name: win-vs2022-xpu-n-1-py3
    uses: ./.github/workflows/_win-build.yml
    with:
      build-environment: win-vs2022-xpu-n-1-py3
      cuda-version: cpu
      use-xpu: true
      xpu-version: '2025.1'
      vc-year: '2022'
    secrets: inherit

  windows-xpu-n-build:
    if: github.repository_owner == 'pytorch'
    name: win-vs2022-xpu-n-py3
    uses: ./.github/workflows/_win-build.yml
    with:
      build-environment: win-vs2022-xpu-n-py3
      cuda-version: cpu
      use-xpu: true
      xpu-version: '2025.2'
      vc-year: '2022'
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

- **File Documentation**: `xpu.yml_docs.md`
- **Keyword Index**: `xpu.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
