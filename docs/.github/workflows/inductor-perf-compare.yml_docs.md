# Documentation: `.github/workflows/inductor-perf-compare.yml`

## File Metadata

- **Path**: `.github/workflows/inductor-perf-compare.yml`
- **Size**: 2,338 bytes (2.28 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: inductor-A100-perf-compare

on:
  push:
    tags:
      - ciflow/inductor-perf-compare/*
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read

jobs:
  get-default-label-prefix:
    if: github.repository_owner == 'pytorch'
    name: get-default-label-prefix
    uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}
      curr_ref_type: ${{ github.ref_type }}
      opt_out_experiments: lf

  build:
    name: cuda12.8-py3.10-gcc9-sm80
    uses: ./.github/workflows/_linux-build.yml
    needs:
      - get-default-label-prefix
    with:
      runner_prefix: "${{ needs.get-default-label-prefix.outputs.label-type }}"
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm80
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc9-inductor-benchmarks
      cuda-arch-list: '8.0'
      test-matrix: |
        { include: [
          { config: "inductor_huggingface_perf_compare", shard: 1, num_shards: 1, runner: "linux.aws.a100" },
          { config: "inductor_timm_perf_compare", shard: 1, num_shards: 2, runner: "linux.aws.a100" },
          { config: "inductor_timm_perf_compare", shard: 2, num_shards: 2, runner: "linux.aws.a100" },
          { config: "inductor_torchbench_perf_compare", shard: 1, num_shards: 1, runner: "linux.aws.a100" },
        ]}
      build-additional-packages: "vision audio fbgemm torchao"
    secrets: inherit

  test:
    name: cuda12.8-py3.10-gcc9-sm80
    uses: ./.github/workflows/_linux-test.yml
    needs: build
    with:
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm80
      docker-image: ${{ needs.build.outputs.docker-image }}
      test-matrix: ${{ needs.build.outputs.test-matrix }}
      # disable monitor in perf tests for more investigation
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `inductor-perf-compare.yml_docs.md`
- **Keyword Index**: `inductor-perf-compare.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
