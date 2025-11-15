# Documentation: `.github/workflows/build-manywheel-images.yml`

## File Metadata

- **Path**: `.github/workflows/build-manywheel-images.yml`
- **Size**: 3,775 bytes (3.69 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: Build manywheel docker images

on:
  workflow_dispatch:
  push:
    branches:
      - main
      - release/*
    tags:
      # NOTE: Binary build pipelines should only get triggered on release candidate or nightly builds
      # Release candidate tags look like: v1.11.0-rc1
      - v[0-9]+.[0-9]+.[0-9]+-rc[0-9]+
    paths:
      - .ci/docker/**
      - .github/workflows/build-manywheel-images.yml
      - .github/actions/binary-docker-build/**
  pull_request:
    paths:
      - .ci/docker/**
      - .github/workflows/build-manywheel-images.yml
      - .github/actions/binary-docker-build/**

env:
  DOCKER_REGISTRY: "docker.io"
  DOCKER_BUILDKIT: 1
  WITH_PUSH: ${{ github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/heads/release') || startsWith(github.ref, 'refs/tags/v')) }}
concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
  cancel-in-progress: true

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

  build:
    environment: ${{ (github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/heads/release') || startsWith(github.ref, 'refs/tags/v')) && 'docker-build') || '' }}
    needs: get-label-type
    strategy:
      fail-fast: false
      matrix:
        include: [
          { name: "manylinux2_28-builder",          tag: "cuda13.0",          runner: "linux.9xlarge.ephemeral" },
          { name: "manylinux2_28-builder",          tag: "cuda12.8",          runner: "linux.9xlarge.ephemeral" },
          { name: "manylinux2_28-builder",          tag: "cuda12.9",          runner: "linux.9xlarge.ephemeral" },
          { name: "manylinux2_28-builder",          tag: "cuda12.6",          runner: "linux.9xlarge.ephemeral" },
          { name: "manylinuxaarch64-builder",       tag: "cuda13.0",          runner: "linux.arm64.2xlarge.ephemeral" },
          { name: "manylinuxaarch64-builder",       tag: "cuda12.9",          runner: "linux.arm64.2xlarge.ephemeral" },
          { name: "manylinuxaarch64-builder",       tag: "cuda12.8",          runner: "linux.arm64.2xlarge.ephemeral" },
          { name: "manylinuxaarch64-builder",       tag: "cuda12.6",          runner: "linux.arm64.2xlarge.ephemeral" },
          { name: "manylinux2_28-builder",          tag: "rocm7.0",           runner: "linux.9xlarge.ephemeral" },
          { name: "manylinux2_28-builder",          tag: "rocm7.1",           runner: "linux.9xlarge.ephemeral" },
          { name: "manylinux2_28-builder",          tag: "cpu",               runner: "linux.9xlarge.ephemeral" },
          { name: "manylinux2_28_aarch64-builder",  tag: "cpu-aarch64",       runner: "linux.arm64.2xlarge.ephemeral" },
          { name: "manylinux2_28-builder",          tag: "xpu",               runner: "linux.9xlarge.ephemeral" },
        ]
    runs-on: ${{ needs.get-label-type.outputs.label-type }}${{ matrix.runner }}
    name: ${{ matrix.name }}:${{ matrix.tag }}
    steps:
      - name: Build docker image
        uses: pytorch/pytorch/.github/actions/binary-docker-build@main
        with:
          docker-image-name: ${{ matrix.name }}
          custom-tag-prefix: ${{ matrix.tag }}
          docker-build-dir: manywheel
          DOCKER_TOKEN: ${{ secrets.DOCKER_TOKEN }}
          DOCKER_ID: ${{ secrets.DOCKER_ID }}

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

- **File Documentation**: `build-manywheel-images.yml_docs.md`
- **Keyword Index**: `build-manywheel-images.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
