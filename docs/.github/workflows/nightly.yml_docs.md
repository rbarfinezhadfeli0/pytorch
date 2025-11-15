# Documentation: `.github/workflows/nightly.yml`

## File Metadata

- **Path**: `.github/workflows/nightly.yml`
- **Size**: 3,801 bytes (3.71 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: nightly

on:
  schedule:
    - cron: 0 0 * * *
  push:
    tags:
      # NOTE: Doc build pipelines should only get triggered on:
      # Major or minor release candidates builds
      - v[0-9]+.[0-9]+.0+-rc[0-9]+
      # Final RC for major, minor and patch releases
      - v[0-9]+.[0-9]+.[0-9]+
      - ciflow/nightly/*
  workflow_dispatch:


concurrency:
  group: ${{ github.workflow }}--${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
  cancel-in-progress: true

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

  link-check:
    name: Link checks
    needs: get-label-type
    uses: ./.github/workflows/_link_check.yml
    with:
      runner: ${{ needs.get-label-type.outputs.label-type }}
      ref:    ${{ github.sha }}
    secrets: inherit

  docs-build:
    name: docs build
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-gcc11
      docker-image-name: ci-image:pytorch-linux-jammy-py3.10-gcc11
    secrets: inherit

  docs-push:
    name: docs push
    uses: ./.github/workflows/_docs.yml
    needs:
      - docs-build
      - get-label-type
    with:
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build-environment: linux-jammy-py3.10-gcc11
      docker-image: ${{ needs.docs-build.outputs.docker-image }}
      push: ${{ github.event_name == 'schedule' || github.event_name == 'workflow_dispatch' || startsWith(github.event.ref, 'refs/tags/v') }}
      run-doxygen: true
    secrets:
      GH_PYTORCHBOT_TOKEN: ${{ secrets.GH_PYTORCHBOT_TOKEN }}

  update-commit-hashes:
    runs-on: ubuntu-latest
    environment: update-commit-hash
    strategy:
      matrix:
        include:
          - repo-name: vision
            repo-owner: pytorch
            branch: main
            pin-folder: .github/ci_commit_pins
          - repo-name: audio
            repo-owner: pytorch
            branch: main
            pin-folder: .github/ci_commit_pins
          # executorch jobs are disabled since it needs some manual work for the hash update
          # - repo-name: executorch
          #   repo-owner: pytorch
          #   branch: main
          #   pin-folder: .ci/docker/ci_commit_pins
          - repo-name: triton
            repo-owner: triton-lang
            branch: main
            pin-folder: .ci/docker/ci_commit_pins
          - repo-name: vllm
            repo-owner: vllm-project
            branch: main
            pin-folder: .github/ci_commit_pins
    # Allow this to be triggered on either a schedule or on workflow_dispatch to allow for easier testing
    if: github.repository_owner == 'pytorch' && (github.event_name == 'schedule' || github.event_name == 'workflow_dispatch')
    steps:
      - name: "${{ matrix.repo-owner }}/${{ matrix.repo-name }} update-commit-hash"
        uses: pytorch/test-infra/.github/actions/update-commit-hash@main
        with:
          repo-owner: ${{ matrix.repo-owner }}
          repo-name: ${{ matrix.repo-name }}
          branch: ${{ matrix.branch }}
          pin-folder: ${{ matrix.pin-folder}}
          updatebot-token: ${{ secrets.UPDATEBOT_TOKEN }}
          pytorchbot-token: ${{ secrets.GH_PYTORCHBOT_TOKEN }}

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

- **File Documentation**: `nightly.yml_docs.md`
- **Keyword Index**: `nightly.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
