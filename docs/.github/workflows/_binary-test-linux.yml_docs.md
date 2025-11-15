# Documentation: `.github/workflows/_binary-test-linux.yml`

## File Metadata

- **Path**: `.github/workflows/_binary-test-linux.yml`
- **Size**: 9,199 bytes (8.98 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
name: linux-binary-test

on:
  workflow_call:
    inputs:
      build_name:
        required: true
        type: string
        description: The build's name
      build_environment:
        required: true
        type: string
        description: The build environment
      ALPINE_IMAGE:
        required: false
        type: string
        default: "308535385114.dkr.ecr.us-east-1.amazonaws.com/tool/alpine"
      PYTORCH_ROOT:
        required: true
        type: string
        description: Root directory for the pytorch/pytorch repository
      PACKAGE_TYPE:
        required: true
        type: string
        description: Package type
      DESIRED_CUDA:
        required: true
        type: string
        description: Desired Cuda version
      GPU_ARCH_VERSION:
        required: false
        type: string
        description: GPU Arch version
      GPU_ARCH_TYPE:
        required: true
        type: string
        description: GPU Arch type
      DOCKER_IMAGE:
        required: true
        type: string
        description: Docker image to use
      DOCKER_IMAGE_TAG_PREFIX:
        required: true
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
      runner_prefix:
        required: false
        default: ""
        type: string
        description: prefix for runner label
      runs_on:
        required: true
        type: string
        description: Hardware to run this job on. Valid values are linux.4xlarge, linux.4xlarge.nvidia.gpu, linux.arm64.2xlarge, and linux.rocm.gpu
    secrets:
      github-token:
        required: true
        description: Github Token

permissions:
  id-token: write

jobs:
  test:
    runs-on: ${{ inputs.runner_prefix}}${{ inputs.runs_on }}
    timeout-minutes: 240
    env:
      PYTORCH_ROOT: ${{ inputs.PYTORCH_ROOT }}
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
      ALPINE_IMAGE: ${{ inputs.ALPINE_IMAGE }}
      AWS_DEFAULT_REGION: us-east-1
      BINARY_ENV_FILE: /tmp/env
      BUILD_ENVIRONMENT: ${{ inputs.build_environment }}
      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      PR_NUMBER: ${{ github.event.pull_request.number }}
      PYTORCH_FINAL_PACKAGE_DIR: /artifacts
      SHA1: ${{ github.event.pull_request.head.sha || github.sha }}
    steps:
      - name: Make the env permanent during this workflow (but not the secrets)
        shell: bash
        run: |
          {
            echo "PYTORCH_ROOT=${{ env.PYTORCH_ROOT }}"
            echo "PACKAGE_TYPE=${{ env.PACKAGE_TYPE }}"

            echo "DESIRED_CUDA=${{ env.DESIRED_CUDA }}"
            echo "GPU_ARCH_VERSION=${{ env.GPU_ARCH_VERSION }}"
            echo "GPU_ARCH_TYPE=${{ env.GPU_ARCH_TYPE }}"
            echo "DOCKER_IMAGE=${{ env.DOCKER_IMAGE }}"
            echo "SKIP_ALL_TESTS=${{ env.SKIP_ALL_TESTS }}"
            echo "LIBTORCH_CONFIG=${{ env.LIBTORCH_CONFIG }}"
            echo "LIBTORCH_VARIANT=${{ env.LIBTORCH_VARIANT }}"
            echo "DESIRED_PYTHON=${{ env.DESIRED_PYTHON }}"

            echo "ALPINE_IMAGE=${{ env.ALPINE_IMAGE }}"
            echo "AWS_DEFAULT_REGION=${{ env.AWS_DEFAULT_REGION }}"
            echo "BINARY_ENV_FILE=${{ env.BINARY_ENV_FILE }}"
            echo "BUILD_ENVIRONMENT=${{ env.BUILD_ENVIRONMENT }}"
            echo "PR_NUMBER=${{ env.PR_NUMBER }}"
            echo "PYTORCH_FINAL_PACKAGE_DIR=${{ env.PYTORCH_FINAL_PACKAGE_DIR }}"
            echo "SHA1=${{ env.SHA1 }}"
          } >> "${GITHUB_ENV} }}"

      - name: "[FB EMPLOYEES] Enable SSH (Click me for login details)"
        if: inputs.build_environment != 'linux-s390x-binary-manywheel'
        uses: pytorch/test-infra/.github/actions/setup-ssh@main
        continue-on-error: true
        with:
          github-secret: ${{ secrets.github-token }}

        # Setup the environment
      - name: Checkout PyTorch
        uses: pytorch/pytorch/.github/actions/checkout-pytorch@main
        with:
          no-sudo: ${{ inputs.build_environment == 'linux-aarch64-binary-manywheel' || inputs.build_environment == 'linux-s390x-binary-manywheel' }}

      - name: Setup Linux
        if: inputs.build_environment != 'linux-s390x-binary-manywheel'
        uses: ./.github/actions/setup-linux

      - name: Chown workspace
        if: inputs.build_environment != 'linux-s390x-binary-manywheel'
        uses: ./.github/actions/chown-workspace
        with:
          ALPINE_IMAGE: ${{ inputs.ALPINE_IMAGE }}

      - name: Clean workspace
        shell: bash
        run: |
          rm -rf "${GITHUB_WORKSPACE}"
          mkdir "${GITHUB_WORKSPACE}"

      - name: Checkout PyTorch to pytorch dir
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          ref: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha }}
          submodules: recursive
          show-progress: false
          path: pytorch

      - name: Clean PyTorch checkout
        run: |
          # Remove any artifacts from the previous checkouts
          git clean -fxd
        working-directory: pytorch

      - name: Check if the job is disabled
        id: filter
        uses: ./pytorch/.github/actions/filter-test-configs
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          # NB: Use a mock test matrix with a default value here. After filtering, if the
          # returned matrix is empty, it means that the job is disabled
          test-matrix: |
            { include: [
              { config: "default" },
            ]}

      - name: Download Build Artifacts
        if: ${{ steps.filter.outputs.is-test-matrix-empty == 'False' }}
        uses: actions/download-artifact@65a9edc5881444af0b9093a5e628f2fe47ea3b2e # v4.1.7
        with:
          name: ${{ inputs.build_name }}
          path: "${{ runner.temp }}/artifacts/"

      - name: Install nvidia driver, nvidia-docker runtime, set GPU_FLAG
        uses: pytorch/test-infra/.github/actions/setup-nvidia@main
        if: ${{ inputs.GPU_ARCH_TYPE == 'cuda' && steps.filter.outputs.is-test-matrix-empty == 'False' }}

      - name: configure aws credentials
        id: aws_creds
        if: ${{ steps.filter.outputs.is-test-matrix-empty == 'False' && inputs.build_environment != 'linux-s390x-binary-manywheel' && startsWith(github.event.ref, 'refs/tags/ciflow/') }}
        uses: aws-actions/configure-aws-credentials@ececac1a45f3b08a01d2dd070d28d111c5fe6722 # v4.1.0
        with:
          role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_s3_and_ecr_read_only
          aws-region: us-east-1
          role-duration-seconds: 18000

      - name: Calculate docker image
        id: calculate-docker-image
        if: ${{ steps.filter.outputs.is-test-matrix-empty == 'False' && inputs.build_environment != 'linux-s390x-binary-manywheel' }}
        uses: pytorch/test-infra/.github/actions/calculate-docker-image@main
        with:
          docker-registry: ${{ startsWith(github.event.ref, 'refs/tags/ciflow/') && '308535385114.dkr.ecr.us-east-1.amazonaws.com' || 'docker.io' }}
          docker-image-name: ${{ inputs.DOCKER_IMAGE }}
          custom-tag-prefix: ${{ inputs.DOCKER_IMAGE_TAG_PREFIX }}
          docker-build-dir: .ci/docker
          working-directory: pytorch

      - name: Pull Docker image
        if: ${{ steps.filter.outputs.is-test-matrix-empty == 'False' && inputs.build_environment != 'linux-s390x-binary-manywheel' }}
        uses: pytorch/test-infra/.github/actions/pull-docker-image@main
        with:
          docker-image: ${{ steps.calculate-docker-image.outputs.docker-image }}

      - name: Test Pytorch binary
        if: ${{ steps.filter.outputs.is-test-matrix-empty == 'False' }}
        uses: ./pytorch/.github/actions/test-pytorch-binary
        env:
          DOCKER_IMAGE: ${{ steps.calculate-docker-image.outputs.docker-image || format('{0}:{1}', inputs.DOCKER_IMAGE, inputs.DOCKER_IMAGE_TAG_PREFIX) }}

      - name: Teardown Linux
        if: always() && inputs.build_environment != 'linux-s390x-binary-manywheel'
        uses: pytorch/test-infra/.github/actions/teardown-linux@main

      - name: Chown workspace
        if: always() && inputs.build_environment != 'linux-s390x-binary-manywheel'
        uses: ./pytorch/.github/actions/chown-workspace
        with:
          ALPINE_IMAGE: ${{ inputs.ALPINE_IMAGE }}

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
python .github/workflows/_binary-test-linux.yml
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

- **File Documentation**: `_binary-test-linux.yml_docs.md`
- **Keyword Index**: `_binary-test-linux.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
