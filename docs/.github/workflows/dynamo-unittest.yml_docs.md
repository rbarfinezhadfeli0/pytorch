# Documentation: `.github/workflows/dynamo-unittest.yml`

## File Metadata

- **Path**: `.github/workflows/dynamo-unittest.yml`
- **Size**: 3,036 bytes (2.96 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
# Workflow: Dynamo Unit Test
# runs unit tests for dynamo.
name: dynamo-unittest

on:
  push:
    tags:
      - ciflow/dynamo/*
  workflow_call:
  schedule:
    - cron: 29 8 * * * # about 1:29am PDT

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read

jobs:
  get-label-type:
    name: get-label-type
    uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main
    if: ${{ (github.event_name != 'schedule' || github.repository == 'pytorch/pytorch') && github.repository_owner == 'pytorch' }}
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}
      curr_ref_type: ${{ github.ref_type }}
      opt_out_experiments: lf

  dynamo-build:
    name: dynamo-build
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py${{ matrix.python-version }}-clang12
      docker-image-name: ci-image:pytorch-linux-jammy-py${{ matrix.python-version }}-clang12
      test-matrix: |
        { include: [
          { config: "dynamo_core", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
          { config: "dynamo_wrapped", shard: 1, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
          { config: "dynamo_wrapped", shard: 2, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
          { config: "dynamo_wrapped", shard: 3, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
        ]}
    secrets: inherit

  dynamo-test:
    name: dynamo-test
    uses: ./.github/workflows/_linux-test.yml
    needs: [get-label-type, dynamo-build]
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    with:
      build-environment: linux-jammy-py${{ matrix.python-version }}-clang12
      docker-image: ci-image:pytorch-linux-jammy-py${{ matrix.python-version }}-clang12
      test-matrix: |
        { include: [
          { config: "dynamo_core", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
          { config: "dynamo_wrapped", shard: 1, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
          { config: "dynamo_wrapped", shard: 2, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
          { config: "dynamo_wrapped", shard: 3, num_shards: 3, runner: "${{ needs.get-label-type.outputs.label-type }}linux.c7i.2xlarge" },
        ]}
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

This is a test file. Run it with:

```bash
python .github/workflows/dynamo-unittest.yml
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

- **File Documentation**: `dynamo-unittest.yml_docs.md`
- **Keyword Index**: `dynamo-unittest.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
