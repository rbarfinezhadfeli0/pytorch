# Documentation: `.github/workflows/inductor-perf-test-nightly-macos.yml`

## File Metadata

- **Path**: `.github/workflows/inductor-perf-test-nightly-macos.yml`
- **Size**: 2,467 bytes (2.41 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
name: inductor-perf-nightly-macos

on:
  schedule:
    - cron: 0 7 * * *
  workflow_dispatch:
    inputs:
      training:
        description: Run training (on by default)?
        required: false
        type: boolean
        default: true
      inference:
        description: Run inference (on by default)?
        required: false
        type: boolean
        default: true
      benchmark_configs:
        description: The list of configs used the benchmark
        required: false
        type: string
        default: torchbench_perf_mps
  pull_request:
    paths:
      - .github/workflows/inductor-perf-test-nightly-macos.yml
      - .ci/pytorch/macos-test.sh

concurrency:
  group:  ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}
  cancel-in-progress: true

permissions: read-all

jobs:
  macos-perf-py3-arm64-build:
    if: github.repository_owner == 'pytorch'
    name: macos-perf-py3-arm64
    uses: ./.github/workflows/_mac-build.yml
    with:
      sync-tag: macos-perf-py3-arm64-build
      build-environment: macos-py3-arm64-distributed
      runner-type: macos-m1-stable
      build-generates-artifacts: true
      # To match the one pre-installed in the m1 runners
      python-version: 3.12.7
      test-matrix: |
        { include: [
          { config: "perf_smoketest", shard: 1, num_shards: 3, runner: "macos-m2-15" },
          { config: "perf_smoketest", shard: 2, num_shards: 3, runner: "macos-m2-15" },
          { config: "perf_smoketest", shard: 3, num_shards: 3, runner: "macos-m2-15" },
          { config: "aot_inductor_perf_smoketest", shard: 1, num_shards: 3, runner: "macos-m2-15" },
          { config: "aot_inductor_perf_smoketest", shard: 2, num_shards: 3, runner: "macos-m2-15" },
          { config: "aot_inductor_perf_smoketest", shard: 3, num_shards: 3, runner: "macos-m2-15" },
        ]}
    secrets: inherit

  macos-perf-py3-arm64-test:
    name: macos-perf-py3-arm64-mps
    uses: ./.github/workflows/_mac-test.yml
    needs: macos-perf-py3-arm64-build
    with:
      build-environment: macos-py3-arm64-distributed
      # Same as the build job
      python-version: 3.12.7
      test-matrix: ${{ needs.macos-perf-py3-arm64-build.outputs.test-matrix }}
      timeout-minutes: 300
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

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python .github/workflows/inductor-perf-test-nightly-macos.yml
```

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

- **File Documentation**: `inductor-perf-test-nightly-macos.yml_docs.md`
- **Keyword Index**: `inductor-perf-test-nightly-macos.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
