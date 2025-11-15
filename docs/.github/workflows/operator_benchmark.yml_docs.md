# Documentation: `.github/workflows/operator_benchmark.yml`

## File Metadata

- **Path**: `.github/workflows/operator_benchmark.yml`
- **Size**: 2,516 bytes (2.46 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: operator_benchmark

on:
  push:
    tags:
      - ciflow/op-benchmark/*
  workflow_dispatch:
    inputs:
      test_mode:
        type: choice
        options:
          - 'short'
          - 'long'
          - 'all'
        description: tag filter for operator benchmarks, options from long, short, all
  schedule:
    # Run at 07:00 UTC every Sunday
    - cron: 0 7 * * 0
  pull_request:
    paths:
      - benchmarks/operator_benchmark/**
      - .github/workflows/operator_benchmark.yml

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read

jobs:
  x86-opbenchmark-build:
    if: github.repository_owner == 'pytorch'
    name: x86-opbenchmark-build
    uses: ./.github/workflows/_linux-build.yml
    with:
      build-environment: linux-jammy-py3.10-gcc11-build
      docker-image-name: ci-image:pytorch-linux-jammy-py3-gcc11-inductor-benchmarks
      test-matrix: |
        { include: [
          { config: "cpu_operator_benchmark_${{ inputs.test_mode || 'short' }}", shard: 1, num_shards: 1, runner: "linux.12xlarge" },
        ]}
    secrets: inherit

  x86-opbenchmark-test:
    name: x86-opbenchmark-test
    uses: ./.github/workflows/_linux-test.yml
    needs: x86-opbenchmark-build
    with:
      build-environment: linux-jammy-py3.10-gcc11-build
      docker-image: ${{ needs.x86-opbenchmark-build.outputs.docker-image }}
      test-matrix: ${{ needs.x86-opbenchmark-build.outputs.test-matrix }}
    secrets: inherit

  aarch64-opbenchmark-build:
    if: github.repository_owner == 'pytorch'
    name: aarch64-opbenchmark-build
    uses: ./.github/workflows/_linux-build.yml
    with:
      build-environment: linux-jammy-aarch64-py3.10
      runner: linux.arm64.m7g.4xlarge
      docker-image-name: ci-image:pytorch-linux-jammy-aarch64-py3.10-gcc13
      test-matrix: |
        { include: [
          { config: "cpu_operator_benchmark_short", shard: 1, num_shards: 1, runner: "linux.arm64.m8g.4xlarge" },
        ]}
    secrets: inherit

  aarch64-opbenchmark-test:
    name: aarch64-opbenchmark-test
    uses: ./.github/workflows/_linux-test.yml
    needs: aarch64-opbenchmark-build
    with:
      build-environment: linux-jammy-aarch64-py3.10
      docker-image: ${{ needs.aarch64-opbenchmark-build.outputs.docker-image }}
      test-matrix: ${{ needs.aarch64-opbenchmark-build.outputs.test-matrix }}
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

- **File Documentation**: `operator_benchmark.yml_docs.md`
- **Keyword Index**: `operator_benchmark.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
