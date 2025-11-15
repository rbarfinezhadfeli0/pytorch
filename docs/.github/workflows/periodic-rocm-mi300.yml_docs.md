# Documentation: `.github/workflows/periodic-rocm-mi300.yml`

## File Metadata

- **Path**: `.github/workflows/periodic-rocm-mi300.yml`
- **Size**: 3,102 bytes (3.03 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: periodic-rocm-mi300

on:
  schedule:
    # We have several schedules so jobs can check github.event.schedule to activate only for a fraction of the runs.
    # Also run less frequently on weekends.
    - cron: 45 0,8,16 * * 1-5
    - cron: 45 4 * * 0,6
    - cron: 45 4,12,20 * * 1-5
    - cron: 45 12 * * 0,6
    - cron: 29 8 * * *  # about 1:29am PDT, for mem leak check and rerun disabled tests
  push:
    tags:
      - ciflow/periodic/*
      - ciflow/periodic-rocm-mi300/*
    branches:
      - release/*
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}-${{ github.event.schedule }}
  cancel-in-progress: true

permissions: read-all

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
    if: (github.event_name != 'schedule' || github.repository == 'pytorch/pytorch') && github.repository_owner == 'pytorch'
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}
      curr_ref_type: ${{ github.ref_type }}

  linux-jammy-rocm-py3_10-build:
    name: linux-jammy-rocm-py3.10-mi300
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-rocm-py3.10-mi300
      docker-image-name: ci-image:pytorch-linux-jammy-rocm-n-py3
      test-matrix: |
        { include: [
          { config: "distributed", shard: 1, num_shards: 3, runner: "linux.rocm.gpu.gfx942.4", owners: ["module:rocm", "oncall:distributed"] },
          { config: "distributed", shard: 2, num_shards: 3, runner: "linux.rocm.gpu.gfx942.4", owners: ["module:rocm", "oncall:distributed"] },
          { config: "distributed", shard: 3, num_shards: 3, runner: "linux.rocm.gpu.gfx942.4", owners: ["module:rocm", "oncall:distributed"] },
        ]}
    secrets: inherit

  linux-jammy-rocm-py3_10-test:
    permissions:
      id-token: write
      contents: read
    name: linux-jammy-rocm-py3.10-mi300
    uses: ./.github/workflows/_rocm-test.yml
    needs:
      - linux-jammy-rocm-py3_10-build
      - target-determination
    with:
      build-environment: linux-jammy-rocm-py3.10-mi300
      docker-image: ${{ needs.linux-jammy-rocm-py3_10-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-rocm-py3_10-build.outputs.test-matrix }}
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

- **File Documentation**: `periodic-rocm-mi300.yml_docs.md`
- **Keyword Index**: `periodic-rocm-mi300.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
