# Documentation: `.github/workflows/_binary-upload.yml`

## File Metadata

- **Path**: `.github/workflows/_binary-upload.yml`
- **Size**: 5,265 bytes (5.14 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: upload

on:
  workflow_call:
    inputs:
      build_name:
        required: true
        type: string
        description: The build's name
      use_s3:
        type: boolean
        default: true
        description: If true, will download artifacts from s3. Otherwise will use the default GitHub artifact download action
      PYTORCH_ROOT:
        required: false
        type: string
        description: Root directory for the pytorch/pytorch repository. Not actually needed, but currently passing it in since we pass in the same inputs to the reusable workflows of all binary builds
      PACKAGE_TYPE:
        required: true
        type: string
        description: Package type
      DESIRED_CUDA:
        required: true
        type: string
        description: Desired CUDA version
      GPU_ARCH_VERSION:
        required: false
        type: string
        description: GPU Arch version
      GPU_ARCH_TYPE:
        required: true
        type: string
        description: GPU Arch type
      DOCKER_IMAGE:
        required: false
        type: string
        description: Docker image to use
      DOCKER_IMAGE_TAG_PREFIX:
        required: false
        type: string
        description: Docker image tag to use
      LIBTORCH_CONFIG:
        required: false
        type: string
        description: Desired libtorch config (for libtorch builds only)
      LIBTORCH_VARIANT:
        required: false
        type: string
        description: Desired libtorch variant (for libtorch builds only)
      DESIRED_PYTHON:
        required: false
        type: string
        description: Desired python version
    secrets:
      github-token:
        required: true
        description: Github Token

jobs:
  upload:
    runs-on: ubuntu-22.04
    container:
      image: continuumio/miniconda3:4.12.0
    env:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: ${{ inputs.PACKAGE_TYPE }}
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: ${{ inputs.DESIRED_CUDA }}
      GPU_ARCH_VERSION: ${{ inputs.GPU_ARCH_VERSION }}
      GPU_ARCH_TYPE: ${{ inputs.GPU_ARCH_TYPE }}
      DOCKER_IMAGE: ${{ inputs.DOCKER_IMAGE }}
      SKIP_ALL_TESTS: 1
      LIBTORCH_CONFIG: ${{ inputs.LIBTORCH_CONFIG }}
      LIBTORCH_VARIANT: ${{ inputs.LIBTORCH_VARIANT }}
      DESIRED_PYTHON: ${{ inputs.DESIRED_PYTHON }}
      BINARY_ENV_FILE: /tmp/env
      GITHUB_TOKEN: ${{ secrets.github-token }}
      PR_NUMBER: ${{ github.event.pull_request.number }}
      PYTORCH_FINAL_PACKAGE_DIR: /artifacts
      SHA1: ${{ github.event.pull_request.head.sha || github.sha }}
    steps:
      - name: Checkout PyTorch
        uses: pytorch/pytorch/.github/actions/checkout-pytorch@main
        with:
          no-sudo: true

      - name: Configure AWS credentials(PyTorch account) for nightly
        if: ${{ github.event_name == 'push' && github.event.ref == 'refs/heads/nightly' }}
        uses: aws-actions/configure-aws-credentials@ececac1a45f3b08a01d2dd070d28d111c5fe6722 # v4.1.0
        with:
          role-to-assume: arn:aws:iam::749337293305:role/gha_workflow_nightly_build_wheels
          aws-region: us-east-1

      - name: Configure AWS credentials(PyTorch account) for RC builds
        if: ${{ github.event_name == 'push' &&  (startsWith(github.event.ref, 'refs/tags/') && !startsWith(github.event.ref, 'refs/tags/ciflow/')) }}
        uses: aws-actions/configure-aws-credentials@ececac1a45f3b08a01d2dd070d28d111c5fe6722 # v4.1.0
        with:
          role-to-assume: arn:aws:iam::749337293305:role/gha_workflow_test_build_wheels
          aws-region: us-east-1

      - name: Download Build Artifacts
        id: download-artifacts
        # NB: When the previous build job is skipped, there won't be any artifacts and
        # this step will fail. Binary build jobs can only be skipped on CI, not nightly
        continue-on-error: true
        uses: actions/download-artifact@65a9edc5881444af0b9093a5e628f2fe47ea3b2e # v4.1.7
        with:
          name: ${{ inputs.build_name }}
          path: "${{ runner.temp }}/artifacts/"

      - name: Set DRY_RUN (only for tagged pushes)
        if: ${{ github.event_name == 'push' && (github.event.ref == 'refs/heads/nightly' || (startsWith(github.event.ref, 'refs/tags/') && !startsWith(github.event.ref, 'refs/tags/ciflow/'))) }}
        run: |
          echo "DRY_RUN=disabled" >> "$GITHUB_ENV"

      - name: Set UPLOAD_CHANNEL (only for tagged pushes)
        if: ${{ github.event_name == 'push' && startsWith(github.event.ref, 'refs/tags/') && !startsWith(github.event.ref, 'refs/tags/ciflow/') }}
        shell: bash -e -l {0}
        run: |
          # reference ends with an RC suffix
          if [[ "${GITHUB_REF_NAME}" = *-rc[0-9]* ]]; then
            echo "UPLOAD_CHANNEL=test" >> "$GITHUB_ENV"
          fi

      - name: Upload binaries
        if: steps.download-artifacts.outcome && steps.download-artifacts.outcome == 'success'
        shell: bash
        env:
          PKG_DIR: "${{ runner.temp }}/artifacts"
          UPLOAD_SUBFOLDER: "${{ env.DESIRED_CUDA }}"
          BUILD_NAME: ${{ inputs.build_name }}
        run: |
            set -ex
            bash .circleci/scripts/binary_upload.sh

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

- **File Documentation**: `_binary-upload.yml_docs.md`
- **Keyword Index**: `_binary-upload.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
