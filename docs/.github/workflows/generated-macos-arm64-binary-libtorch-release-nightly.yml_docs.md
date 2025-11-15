# Documentation: `.github/workflows/generated-macos-arm64-binary-libtorch-release-nightly.yml`

## File Metadata

- **Path**: `.github/workflows/generated-macos-arm64-binary-libtorch-release-nightly.yml`
- **Size**: 5,064 bytes (4.95 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
# @generated DO NOT EDIT MANUALLY

# Template is at:    .github/templates/macos_binary_build_workflow.yml.j2
# Generation script: .github/scripts/generate_ci_workflows.py
name: macos-arm64-binary-libtorch-release

on:
# TODO: Migrate to new ciflow trigger, reference https://github.com/pytorch/pytorch/pull/70321
  push:
    # NOTE: Meta Employees can trigger new nightlies using: https://fburl.com/trigger_pytorch_nightly_build
    branches:
      - nightly
    tags:
      # NOTE: Binary build pipelines should only get triggered on release candidate builds
      # Release candidate tags look like: v1.11.0-rc1
      - v[0-9]+.[0-9]+.[0-9]+-rc[0-9]+
      - 'ciflow/binaries/*'
      - 'ciflow/binaries_libtorch/*'
  workflow_dispatch:

env:
  ALPINE_IMAGE: "308535385114.dkr.ecr.us-east-1.amazonaws.com/tool/alpine"
  AWS_DEFAULT_REGION: us-east-1
  BUILD_ENVIRONMENT: macos-arm64-binary-libtorch-release
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  PR_NUMBER: ${{ github.event.pull_request.number }}
  SKIP_ALL_TESTS: 0
concurrency:
  group: macos-arm64-binary-libtorch-release-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
  cancel-in-progress: true

jobs:
  libtorch-cpu-shared-with-deps-release-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    runs-on: macos-14-xlarge
    timeout-minutes: 240
    env:
      PYTORCH_ROOT: ${{ github.workspace }}/pytorch
      PACKAGE_TYPE: libtorch
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu
      SKIP_ALL_TESTS: 1
      LIBTORCH_CONFIG: release
      LIBTORCH_VARIANT: shared-with-deps
      # This is a dummy value for libtorch to work correctly with our batch scripts
      # without this value pip does not get installed for some reason
      DESIRED_PYTHON: "3.10"
    steps:
      # NOTE: These environment variables are put here so that they can be applied on every job equally
      #       They are also here because setting them at a workflow level doesn't give us access to the
      #       runner.temp variable, which we need.
      - name: Populate binary env
        shell: bash
        run: |
          # shellcheck disable=SC2129
          echo "BINARY_ENV_FILE=${RUNNER_TEMP}/env" >> "${GITHUB_ENV}"
          # shellcheck disable=SC2129
          echo "PYTORCH_FINAL_PACKAGE_DIR=${RUNNER_TEMP}/artifacts" >> "${GITHUB_ENV}"
          # shellcheck disable=SC2129
          echo "MAC_PACKAGE_WORK_DIR=${RUNNER_TEMP}" >> "${GITHUB_ENV}"
      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          # .4 version is min minor for 3.10, and also no-gil version of 3.13 needs at least 3.13.3
          python-version: "3.10.4"
          freethreaded: false
      - name: Checkout PyTorch
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha }}
          submodules: recursive
          path: pytorch
          show-progress: false
      - name: Clean PyTorch checkout
        run: |
          # Remove any artifacts from the previous checkouts
          git clean -fxd
        working-directory: pytorch
      - name: Populate binary env
        run: |
          "${PYTORCH_ROOT}/.circleci/scripts/binary_populate_env.sh"
      - name: Build PyTorch binary
        run: |
          set -eux -o pipefail
          # shellcheck disable=SC1090
          source "${BINARY_ENV_FILE:-/Users/distiller/project/env}"
          mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

          # Build
          USE_PYTORCH_METAL_EXPORT=1
          USE_COREML_DELEGATE=1
          TORCH_PACKAGE_NAME="${TORCH_PACKAGE_NAME//-/_}"
          export USE_PYTORCH_METAL_EXPORT
          export USE_COREML_DELEGATE
          export TORCH_PACKAGE_NAME
          "${PYTORCH_ROOT}/.ci/wheel/build_wheel.sh"
      - uses: actions/upload-artifact@v4.4.0
        if: always()
        with:
          name: libtorch-cpu-shared-with-deps-release
          retention-days: 14
          if-no-files-found: error
          path: "${{ env.PYTORCH_FINAL_PACKAGE_DIR }}"
  libtorch-cpu-shared-with-deps-release-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: libtorch-cpu-shared-with-deps-release-build
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: libtorch
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu
      DOCKER_IMAGE: libtorch-cxx11-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu
      LIBTORCH_CONFIG: release
      LIBTORCH_VARIANT: shared-with-deps
      build_name: libtorch-cpu-shared-with-deps-release
      use_s3: False
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

- **File Documentation**: `generated-macos-arm64-binary-libtorch-release-nightly.yml_docs.md`
- **Keyword Index**: `generated-macos-arm64-binary-libtorch-release-nightly.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
