# Documentation: `.github/workflows/rocm-mi355.yml`

## File Metadata

- **Path**: `.github/workflows/rocm-mi355.yml`
- **Size**: 3,079 bytes (3.01 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: rocm-mi355

on:
  push:
    tags:
      - ciflow/rocm-mi355/*
  workflow_dispatch:
  schedule:
    - cron: 30 11,1 * * *  # about 4:30am PDT and 6:30pm PDT

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}
  cancel-in-progress: true

permissions: read-all

jobs:
  target-determination:
    if: github.repository_owner == 'pytorch'
    name: before-test
    uses: ./.github/workflows/target_determination.yml
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

  linux-noble-rocm-py3_12-build:
    if: ${{ (github.event_name != 'schedule' || github.repository == 'pytorch/pytorch') && github.repository_owner == 'pytorch' }}
    name: linux-noble-rocm-py3.12-mi355
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-noble-rocm-py3.12-mi355
      docker-image-name: ci-image:pytorch-linux-noble-rocm-n-py3
      test-matrix: |
        { include: [
          { config: "default", shard: 1, num_shards: 6, runner: "linux.rocm.gpu.mi355.1" },
          { config: "default", shard: 2, num_shards: 6, runner: "linux.rocm.gpu.mi355.1" },
          { config: "default", shard: 3, num_shards: 6, runner: "linux.rocm.gpu.mi355.1" },
          { config: "default", shard: 4, num_shards: 6, runner: "linux.rocm.gpu.mi355.1" },
          { config: "default", shard: 5, num_shards: 6, runner: "linux.rocm.gpu.mi355.1" },
          { config: "default", shard: 6, num_shards: 6, runner: "linux.rocm.gpu.mi355.1" },
        ]}
    secrets: inherit

  linux-noble-rocm-py3_12-test:
    permissions:
      id-token: write
      contents: read
    name: linux-noble-rocm-py3.12-mi355
    uses: ./.github/workflows/_rocm-test.yml
    needs:
      - linux-noble-rocm-py3_12-build
      - target-determination
    with:
      build-environment: linux-noble-rocm-py3.12-mi355
      docker-image: ${{ needs.linux-noble-rocm-py3_12-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-noble-rocm-py3_12-build.outputs.test-matrix }}
      tests-to-include: >-
                        ${{ github.event_name == 'schedule' && 'test_nn test_torch test_cuda test_ops test_unary_ufuncs test_binary_ufuncs test_autograd inductor/test_torchinductor test_matmul_cuda test_scaled_matmul_cuda'
                           || '' }}
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

- **File Documentation**: `rocm-mi355.yml_docs.md`
- **Keyword Index**: `rocm-mi355.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
