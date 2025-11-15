# Documentation: `.github/workflows/s390x-periodic.yml`

## File Metadata

- **Path**: `.github/workflows/s390x-periodic.yml`
- **Size**: 2,908 bytes (2.84 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: s390x-periodic

on:
  schedule:
    # We have several schedules so jobs can check github.event.schedule to activate only for a fraction of the runs.
    # Also run less frequently on weekends.
    - cron: 29 8 * * *  # about 1:29am PDT, for mem leak check and rerun disabled tests
  push:
    tags:
      - ciflow/periodic/*
      - ciflow/s390/*
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

  linux-manylinux-2_28-py3-cpu-s390x-build:
    if: github.repository_owner == 'pytorch'
    name: linux-manylinux-2_28-py3-cpu-s390x
    uses: ./.github/workflows/_linux-build.yml
    with:
      build-environment: linux-s390x-binary-manywheel
      docker-image-name: pytorch/manylinuxs390x-builder:cpu-s390x
      runner: linux.s390x
      test-matrix: |
        { include: [
          { config: "default", shard: 1,  num_shards: 10, runner: "linux.s390x" },
          { config: "default", shard: 2,  num_shards: 10, runner: "linux.s390x" },
          { config: "default", shard: 3,  num_shards: 10, runner: "linux.s390x" },
          { config: "default", shard: 4,  num_shards: 10, runner: "linux.s390x" },
          { config: "default", shard: 5,  num_shards: 10, runner: "linux.s390x" },
          { config: "default", shard: 6,  num_shards: 10, runner: "linux.s390x" },
          { config: "default", shard: 7,  num_shards: 10, runner: "linux.s390x" },
          { config: "default", shard: 8,  num_shards: 10, runner: "linux.s390x" },
          { config: "default", shard: 9,  num_shards: 10, runner: "linux.s390x" },
          { config: "default", shard: 10, num_shards: 10, runner: "linux.s390x" },
        ]}
    secrets: inherit

  linux-manylinux-2_28-py3-cpu-s390x-test:
    permissions:
      id-token: write
      contents: read
    name: linux-manylinux-2_28-py3-cpu-s390x
    uses: ./.github/workflows/_linux-test.yml
    needs:
      - linux-manylinux-2_28-py3-cpu-s390x-build
      - target-determination
    with:
      build-environment: linux-s390x-binary-manywheel
      docker-image: pytorch/manylinuxs390x-builder:cpu-s390x
      test-matrix: ${{ needs.linux-manylinux-2_28-py3-cpu-s390x-build.outputs.test-matrix }}
      timeout-minutes: 600
      use-gha: "yes"
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

- **File Documentation**: `s390x-periodic.yml_docs.md`
- **Keyword Index**: `s390x-periodic.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
