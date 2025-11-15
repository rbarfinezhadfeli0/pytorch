# Documentation: `.github/workflows/generated-linux-s390x-binary-manywheel-nightly.yml`

## File Metadata

- **Path**: `.github/workflows/generated-linux-s390x-binary-manywheel-nightly.yml`
- **Size**: 18,196 bytes (17.77 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
# @generated DO NOT EDIT MANUALLY

# Template is at:    .github/templates/linux_binary_build_workflow.yml.j2
# Generation script: .github/scripts/generate_ci_workflows.py
name: linux-s390x-binary-manywheel


on:
  push:
    # NOTE: Meta Employees can trigger new nightlies using: https://fburl.com/trigger_pytorch_nightly_build
    branches:
      - nightly
    tags:
      # NOTE: Binary build pipelines should only get triggered on release candidate builds
      # Release candidate tags look like: v1.11.0-rc1
      - v[0-9]+.[0-9]+.[0-9]+-rc[0-9]+
      - 'ciflow/binaries/*'
      - 'ciflow/binaries_wheel/*'
  workflow_dispatch:

permissions:
  id-token: write

env:
  # Needed for conda builds
  ALPINE_IMAGE: "docker.io/s390x/alpine"
  AWS_DEFAULT_REGION: us-east-1
  BINARY_ENV_FILE: /tmp/env
  BUILD_ENVIRONMENT: linux-s390x-binary-manywheel
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  PR_NUMBER: ${{ github.event.pull_request.number }}
  PYTORCH_FINAL_PACKAGE_DIR: /artifacts
  PYTORCH_ROOT: /pytorch
  SHA1: ${{ github.event.pull_request.head.sha || github.sha }}
  SKIP_ALL_TESTS: 0
concurrency:
  group: linux-s390x-binary-manywheel-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
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
  manywheel-py3_10-cpu-s390x-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.10"
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
      timeout-minutes: 420
      build_name: manywheel-py3_10-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cpu-s390x-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_10-cpu-s390x-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cpu-s390x-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_10-cpu-s390x-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cpu-s390x
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_11-cpu-s390x-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.11"
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
      timeout-minutes: 420
      build_name: manywheel-py3_11-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cpu-s390x-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_11-cpu-s390x-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cpu-s390x-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_11-cpu-s390x-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cpu-s390x
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_12-cpu-s390x-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.12"
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
      timeout-minutes: 420
      build_name: manywheel-py3_12-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_12-cpu-s390x-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_12-cpu-s390x-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.12"
      build_name: manywheel-py3_12-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_12-cpu-s390x-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_12-cpu-s390x-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.12"
      build_name: manywheel-py3_12-cpu-s390x
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_13-cpu-s390x-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.13"
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
      timeout-minutes: 420
      build_name: manywheel-py3_13-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_13-cpu-s390x-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_13-cpu-s390x-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.13"
      build_name: manywheel-py3_13-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_13-cpu-s390x-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_13-cpu-s390x-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.13"
      build_name: manywheel-py3_13-cpu-s390x
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_13t-cpu-s390x-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.13t"
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
      timeout-minutes: 420
      build_name: manywheel-py3_13t-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_13t-cpu-s390x-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_13t-cpu-s390x-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.13t"
      build_name: manywheel-py3_13t-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_13t-cpu-s390x-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_13t-cpu-s390x-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.13t"
      build_name: manywheel-py3_13t-cpu-s390x
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_14-cpu-s390x-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.14"
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
      timeout-minutes: 420
      build_name: manywheel-py3_14-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_14-cpu-s390x-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_14-cpu-s390x-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.14"
      build_name: manywheel-py3_14-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_14-cpu-s390x-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_14-cpu-s390x-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.14"
      build_name: manywheel-py3_14-cpu-s390x
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_14t-cpu-s390x-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.14t"
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
      timeout-minutes: 420
      build_name: manywheel-py3_14t-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_14t-cpu-s390x-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_14t-cpu-s390x-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.14t"
      build_name: manywheel-py3_14t-cpu-s390x
      build_environment: linux-s390x-binary-manywheel
      runs_on: linux.s390x
      ALPINE_IMAGE: "docker.io/s390x/alpine"
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_14t-cpu-s390x-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_14t-cpu-s390x-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu-s390x
      DOCKER_IMAGE: pytorch/manylinuxs390x-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu-s390x
      DESIRED_PYTHON: "3.14t"
      build_name: manywheel-py3_14t-cpu-s390x
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

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

- **File Documentation**: `generated-linux-s390x-binary-manywheel-nightly.yml_docs.md`
- **Keyword Index**: `generated-linux-s390x-binary-manywheel-nightly.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
