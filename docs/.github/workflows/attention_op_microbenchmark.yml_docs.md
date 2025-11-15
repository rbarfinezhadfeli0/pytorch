# Documentation: `.github/workflows/attention_op_microbenchmark.yml`

## File Metadata

- **Path**: `.github/workflows/attention_op_microbenchmark.yml`
- **Size**: 2,656 bytes (2.59 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: attention_op_microbenchmark

on:
  push:
    tags:
      - ciflow/op-benchmark/*
  workflow_dispatch:
  schedule:
    # Run at 06:00 UTC everyday
    - cron: 0 7 * * *

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read

jobs:
  attn-microbenchmark-build:
    if: github.repository_owner == 'pytorch'
    uses: ./.github/workflows/_linux-build.yml
    with:
      runner: linux.12xlarge.memory
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm80
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc11
      cuda-arch-list: '8.0 9.0'
      test-matrix: |
        { include: [
          { config: "attention_microbenchmark_test", shard: 1, num_shards: 1, runner: "linux.aws.a100" },
          { config: "attention_microbenchmark_test", shard: 1, num_shards: 1, runner: "linux.aws.h100" },
        ]}
    secrets: inherit

  attn-microbenchmark-test:
    name: attn-microbenchmark-test
    uses: ./.github/workflows/_linux-test.yml
    needs: attn-microbenchmark-build
    with:
      timeout-minutes: 500
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm80
      docker-image: ${{ needs.attn-microbenchmark-build.outputs.docker-image }}
      test-matrix: ${{ needs.attn-microbenchmark-build.outputs.test-matrix }}
    secrets: inherit

  # B200 runner
  opmicrobenchmark-build-b200:
    if: github.repository_owner == 'pytorch'
    name: opmicrobenchmark-build-b200
    uses: ./.github/workflows/_linux-build.yml
    with:
      runner: linux.12xlarge.memory
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm100
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc11
      cuda-arch-list: '10.0'
      test-matrix: |
        { include: [
          { config: "operator_microbenchmark_test", shard: 1, num_shards: 1, runner: "linux.dgx.b200" },
        ]}
    secrets: inherit

  opmicrobenchmark-test-b200:
    name: opmicrobenchmark-test-b200
    uses: ./.github/workflows/_linux-test.yml
    needs: opmicrobenchmark-build-b200
    with:
      timeout-minutes: 500
      build-environment: linux-jammy-cuda12.8-py3.10-gcc9-sm100
      docker-image: ${{ needs.opmicrobenchmark-build-b200.outputs.docker-image }}
      test-matrix: ${{ needs.opmicrobenchmark-build-b200.outputs.test-matrix }}
      aws-role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_s3_and_ecr_read_only
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
- [`inductor-nightly.yml_docs.md`](./inductor-nightly.yml_docs.md)
- [`lint-autoformat.yml_docs.md`](./lint-autoformat.yml_docs.md)
- [`inductor-perf-test-b200.yml_docs.md`](./inductor-perf-test-b200.yml_docs.md)
- [`inductor-unittest.yml_docs.md`](./inductor-unittest.yml_docs.md)
- [`_linux-build.yml_docs.md`](./_linux-build.yml_docs.md)
- [`inductor-perf-test-nightly.yml_docs.md`](./inductor-perf-test-nightly.yml_docs.md)


## Cross-References

- **File Documentation**: `attention_op_microbenchmark.yml_docs.md`
- **Keyword Index**: `attention_op_microbenchmark.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
