# Documentation: `.github/workflows/test-b200.yml`

## File Metadata

- **Path**: `.github/workflows/test-b200.yml`
- **Size**: 2,842 bytes (2.78 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
# B200 Smoke Tests CI Workflow
#
# This workflow runs smoke tests on B200 hardware
#
# Flow:
# 1. Builds PyTorch with CUDA 12.8+ and sm100 architecture for B200
# 2. Runs smoke tests on linux.dgx.b200 runner
# 3. Tests executed are defined in .ci/pytorch/test.sh -> test_python_smoke_b200() function
#    - Includes matmul, scaled_matmul, FP8, and FlashAttention CuTe tests
#    - FlashAttention CuTe DSL is installed as part of test execution
#
# Triggered by:
# - Pull requests modifying this workflow file
# - Manual dispatch
# - Schedule (every 6 hours)
# - Adding ciflow/b200 label to a PR (creates ciflow/b200/* tag)

name: B200 Smoke Tests

on:
  pull_request:
    paths:
      - .github/workflows/test-b200.yml
  workflow_dispatch:
  schedule:
    - cron: 0 4,10,16,22 * * *  # every 6 hours
  push:
    tags:
      - ciflow/b200/*

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}-${{ github.event_name == 'schedule' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read

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

  linux-jammy-cuda12_8-py3_10-gcc11-sm100-build:
    name: linux-jammy-cuda12.8-py3.10-gcc11-sm100
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runner: linux.12xlarge.memory
      build-environment: linux-jammy-cuda12.8-py3.10-gcc11-sm100
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc11
      cuda-arch-list: '10.0'
      test-matrix: |
        { include: [
          { config: "smoke_b200", shard: 1, num_shards: 1, runner: "linux.dgx.b200" },
        ]}
      # config: "smoke_b200" maps to test_python_smoke_b200() in .ci/pytorch/test.sh
    secrets: inherit

  linux-jammy-cuda12_8-py3_10-gcc11-sm100-test:
    name: linux-jammy-cuda12.8-py3.10-gcc11-sm100
    uses: ./.github/workflows/_linux-test.yml
    needs:
      - linux-jammy-cuda12_8-py3_10-gcc11-sm100-build
    with:
      build-environment: linux-jammy-cuda12.8-py3.10-gcc11-sm100
      docker-image: ${{ needs.linux-jammy-cuda12_8-py3_10-gcc11-sm100-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-cuda12_8-py3_10-gcc11-sm100-build.outputs.test-matrix }}
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

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python .github/workflows/test-b200.yml
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

- **File Documentation**: `test-b200.yml_docs.md`
- **Keyword Index**: `test-b200.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
