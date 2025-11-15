# Documentation: `.github/workflows/inductor-nightly.yml`

## File Metadata

- **Path**: `.github/workflows/inductor-nightly.yml`
- **Size**: 2,983 bytes (2.91 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: inductor-nightly

on:
  pull_request:
    paths:
      - .github/workflows/inductor-nightly.yml
      - benchmarks/dynamo/ci_expected_accuracy/dynamic_cpu_max_autotune_inductor_amp_freezing_huggingface_inference.csv
      - benchmarks/dynamo/ci_expected_accuracy/dynamic_cpu_max_autotune_inductor_amp_freezing_timm_inference.csv
      - benchmarks/dynamo/ci_expected_accuracy/dynamic_cpu_max_autotune_inductor_amp_freezing_torchbench_inference.csv
  workflow_dispatch:
  schedule:
    # Run every day at 7:00 AM UTC
    - cron: 0 7 * * *

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read

jobs:
  get-default-label-prefix:
    name: get-default-label-prefix
    uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main
    if: ${{ (github.event_name != 'schedule' || github.repository == 'pytorch/pytorch') && github.repository_owner == 'pytorch' }}
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}
      curr_ref_type: ${{ github.ref_type }}
      opt_out_experiments: lf

  nightly-dynamo-benchmarks-build:
    name: nightly-dynamo-benchmarks-build
    uses: ./.github/workflows/_linux-build.yml
    needs: get-default-label-prefix
    with:
      build-environment: linux-jammy-py3.10-gcc11-build
      docker-image-name: ci-image:pytorch-linux-jammy-py3-gcc11-inductor-benchmarks
      runner_prefix: "${{ needs.get-default-label-prefix.outputs.label-type }}"
      test-matrix: |
        { include: [
          { config: "dynamic_cpu_max_autotune_inductor_amp_freezing_huggingface", shard: 1, num_shards: 1, runner: "linux.8xlarge.amx" },
          { config: "dynamic_cpu_max_autotune_inductor_amp_freezing_timm", shard: 1, num_shards: 2, runner: "linux.8xlarge.amx" },
          { config: "dynamic_cpu_max_autotune_inductor_amp_freezing_timm", shard: 2, num_shards: 2, runner: "linux.8xlarge.amx" },
          { config: "dynamic_cpu_max_autotune_inductor_amp_freezing_torchbench", shard: 1, num_shards: 2, runner: "linux.8xlarge.amx" },
          { config: "dynamic_cpu_max_autotune_inductor_amp_freezing_torchbench", shard: 2, num_shards: 2, runner: "linux.8xlarge.amx" },
        ]}
      build-additional-packages: "vision audio torchao"
    secrets: inherit

  nightly-dynamo-benchmarks-test:
    name: nightly-dynamo-benchmarks-test
    uses: ./.github/workflows/_linux-test.yml
    needs: nightly-dynamo-benchmarks-build
    with:
      build-environment: linux-jammy-py3.10-gcc11-build
      docker-image: ${{ needs.nightly-dynamo-benchmarks-build.outputs.docker-image }}
      test-matrix: ${{ needs.nightly-dynamo-benchmarks-build.outputs.test-matrix }}
      timeout-minutes: 720
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
- [`lint-autoformat.yml_docs.md`](./lint-autoformat.yml_docs.md)
- [`inductor-perf-test-b200.yml_docs.md`](./inductor-perf-test-b200.yml_docs.md)
- [`inductor-unittest.yml_docs.md`](./inductor-unittest.yml_docs.md)
- [`_linux-build.yml_docs.md`](./_linux-build.yml_docs.md)
- [`inductor-perf-test-nightly.yml_docs.md`](./inductor-perf-test-nightly.yml_docs.md)


## Cross-References

- **File Documentation**: `inductor-nightly.yml_docs.md`
- **Keyword Index**: `inductor-nightly.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
