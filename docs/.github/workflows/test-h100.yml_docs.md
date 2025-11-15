# Documentation: `.github/workflows/test-h100.yml`

## File Metadata

- **Path**: `.github/workflows/test-h100.yml`
- **Size**: 2,624 bytes (2.56 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
name: Limited CI on H100

on:
  pull_request:
    paths:
      - .github/workflows/test-h100.yml
      - test/inductor/test_max_autotune.py
      - torch/_inductor/kernel/mm.py
      - torch/_inductor/kernel/mm_grouped.py

  workflow_dispatch:
  schedule:
    - cron: 0 4,10,16,22 * * *  # every 6 hours
  push:
    tags:
      - ciflow/h100/*

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

  linux-jammy-cuda12_8-py3_10-gcc11-sm90-build:
    name: linux-jammy-cuda12.8-py3.10-gcc11-sm90
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-cuda12.8-py3.10-gcc11-sm90
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc11
      cuda-arch-list: '9.0'
      test-matrix: |
        { include: [
          { config: "smoke", shard: 1, num_shards: 1, runner: "linux.aws.h100" },
        ]}
    secrets: inherit

  linux-jammy-cuda12_8-py3_10-gcc11-sm90-test:
    name: linux-jammy-cuda12.8-py3.10-gcc11-sm90
    uses: ./.github/workflows/_linux-test.yml
    needs:
      - linux-jammy-cuda12_8-py3_10-gcc11-sm90-build
    with:
      build-environment: linux-jammy-cuda12.8-py3.10-gcc11-sm90
      docker-image: ${{ needs.linux-jammy-cuda12_8-py3_10-gcc11-sm90-build.outputs.docker-image }}
      test-matrix: ${{ needs.linux-jammy-cuda12_8-py3_10-gcc11-sm90-build.outputs.test-matrix }}
    secrets: inherit

  linux-jammy-cuda12_8-py3_10-gcc11-sm90-FA3-ABI-stable-test:
    name: linux-jammy-cuda12_8-py3_10-gcc11-sm90-FA3-ABI-stable-test
    uses: ./.github/workflows/_linux-test-stable-fa3.yml
    needs:
      - linux-jammy-cuda12_8-py3_10-gcc11-sm90-build
    with:
      build-environment: linux-jammy-cuda12.8-py3.10-gcc11-sm90
      docker-image: ${{ needs.linux-jammy-cuda12_8-py3_10-gcc11-sm90-build.outputs.docker-image }}
      timeout-minutes: 30
      s3-bucket: gha-artifacts
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
python .github/workflows/test-h100.yml
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

- **File Documentation**: `test-h100.yml_docs.md`
- **Keyword Index**: `test-h100.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
