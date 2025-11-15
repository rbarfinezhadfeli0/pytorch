# Documentation: `.github/workflows/generated-linux-binary-manywheel-nightly.yml`

## File Metadata

- **Path**: `.github/workflows/generated-linux-binary-manywheel-nightly.yml`
- **Size**: 206,848 bytes (202.00 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
# @generated DO NOT EDIT MANUALLY

# Template is at:    .github/templates/linux_binary_build_workflow.yml.j2
# Generation script: .github/scripts/generate_ci_workflows.py
name: linux-binary-manywheel


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
  ALPINE_IMAGE: "308535385114.dkr.ecr.us-east-1.amazonaws.com/tool/alpine"
  AWS_DEFAULT_REGION: us-east-1
  BINARY_ENV_FILE: /tmp/env
  BUILD_ENVIRONMENT: linux-binary-manywheel
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  PR_NUMBER: ${{ github.event.pull_request.number }}
  PYTORCH_FINAL_PACKAGE_DIR: /artifacts
  PYTORCH_ROOT: /pytorch
  SHA1: ${{ github.event.pull_request.head.sha || github.sha }}
  SKIP_ALL_TESTS: 0
concurrency:
  group: linux-binary-manywheel-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
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
  manywheel-py3_10-cpu-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu
      DESIRED_PYTHON: "3.10"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_10-cpu
      build_environment: linux-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cpu-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_10-cpu-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cpu
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.4xlarge
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cpu-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_10-cpu-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cpu
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_10-cuda12_6-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu126
      GPU_ARCH_VERSION: "12.6"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.6
      DESIRED_PYTHON: "3.10"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_10-cuda12_6
      build_environment: linux-binary-manywheel
      PYTORCH_EXTRA_INSTALL_REQUIREMENTS: nvidia-cuda-nvrtc-cu12==12.6.77; platform_system == 'Linux' | nvidia-cuda-runtime-cu12==12.6.77; platform_system == 'Linux' | nvidia-cuda-cupti-cu12==12.6.80; platform_system == 'Linux' | nvidia-cudnn-cu12==9.10.2.21; platform_system == 'Linux' | nvidia-cublas-cu12==12.6.4.1; platform_system == 'Linux' | nvidia-cufft-cu12==11.3.0.4; platform_system == 'Linux' | nvidia-curand-cu12==10.3.7.77; platform_system == 'Linux' | nvidia-cusolver-cu12==11.7.1.2; platform_system == 'Linux' | nvidia-cusparse-cu12==12.5.4.2; platform_system == 'Linux' | nvidia-cusparselt-cu12==0.7.1; platform_system == 'Linux' | nvidia-nccl-cu12==2.27.5; platform_system == 'Linux' | nvidia-nvshmem-cu12==3.4.5; platform_system == 'Linux' | nvidia-nvtx-cu12==12.6.77; platform_system == 'Linux' | nvidia-nvjitlink-cu12==12.6.85; platform_system == 'Linux' | nvidia-cufile-cu12==1.11.1.6; platform_system == 'Linux'
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cuda12_6-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_10-cuda12_6-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu126
      GPU_ARCH_VERSION: "12.6"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.6
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cuda12_6
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.4xlarge.nvidia.gpu  # 12.6 build can use maxwell (sm_50) runner
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cuda12_6-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_10-cuda12_6-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu126
      GPU_ARCH_VERSION: "12.6"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.6
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cuda12_6
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_10-cuda12_8-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu128
      GPU_ARCH_VERSION: "12.8"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.8
      DESIRED_PYTHON: "3.10"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_10-cuda12_8
      build_environment: linux-binary-manywheel
      PYTORCH_EXTRA_INSTALL_REQUIREMENTS: nvidia-cuda-nvrtc-cu12==12.8.93; platform_system == 'Linux' | nvidia-cuda-runtime-cu12==12.8.90; platform_system == 'Linux' | nvidia-cuda-cupti-cu12==12.8.90; platform_system == 'Linux' | nvidia-cudnn-cu12==9.10.2.21; platform_system == 'Linux' | nvidia-cublas-cu12==12.8.4.1; platform_system == 'Linux' | nvidia-cufft-cu12==11.3.3.83; platform_system == 'Linux' | nvidia-curand-cu12==10.3.9.90; platform_system == 'Linux' | nvidia-cusolver-cu12==11.7.3.90; platform_system == 'Linux' | nvidia-cusparse-cu12==12.5.8.93; platform_system == 'Linux' | nvidia-cusparselt-cu12==0.7.1; platform_system == 'Linux' | nvidia-nccl-cu12==2.27.5; platform_system == 'Linux' | nvidia-nvshmem-cu12==3.4.5; platform_system == 'Linux' | nvidia-nvtx-cu12==12.8.90; platform_system == 'Linux' | nvidia-nvjitlink-cu12==12.8.93; platform_system == 'Linux' | nvidia-cufile-cu12==1.13.1.3; platform_system == 'Linux'
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cuda12_8-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_10-cuda12_8-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu128
      GPU_ARCH_VERSION: "12.8"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.8
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cuda12_8
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.g4dn.4xlarge.nvidia.gpu # 12.8+ builds need sm_70+ runner
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cuda12_8-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_10-cuda12_8-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu128
      GPU_ARCH_VERSION: "12.8"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.8
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cuda12_8
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_10-cuda12_9-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu129
      GPU_ARCH_VERSION: "12.9"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.9
      DESIRED_PYTHON: "3.10"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_10-cuda12_9
      build_environment: linux-binary-manywheel
      PYTORCH_EXTRA_INSTALL_REQUIREMENTS: nvidia-cuda-nvrtc-cu12==12.9.86; platform_system == 'Linux' | nvidia-cuda-runtime-cu12==12.9.79; platform_system == 'Linux' | nvidia-cuda-cupti-cu12==12.9.79; platform_system == 'Linux' | nvidia-cudnn-cu12==9.10.2.21; platform_system == 'Linux' | nvidia-cublas-cu12==12.9.1.4; platform_system == 'Linux' | nvidia-cufft-cu12==11.4.1.4; platform_system == 'Linux' | nvidia-curand-cu12==10.3.10.19; platform_system == 'Linux' | nvidia-cusolver-cu12==11.7.5.82; platform_system == 'Linux' | nvidia-cusparse-cu12==12.5.10.65; platform_system == 'Linux' | nvidia-cusparselt-cu12==0.7.1; platform_system == 'Linux' | nvidia-nccl-cu12==2.27.5; platform_system == 'Linux' | nvidia-nvshmem-cu12==3.4.5; platform_system == 'Linux' | nvidia-nvtx-cu12==12.9.79; platform_system == 'Linux' | nvidia-nvjitlink-cu12==12.9.86; platform_system == 'Linux' | nvidia-cufile-cu12==1.14.1.1; platform_system == 'Linux'
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cuda12_9-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_10-cuda12_9-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu129
      GPU_ARCH_VERSION: "12.9"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.9
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cuda12_9
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.g4dn.4xlarge.nvidia.gpu # 12.8+ builds need sm_70+ runner
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cuda12_9-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_10-cuda12_9-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu129
      GPU_ARCH_VERSION: "12.9"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.9
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cuda12_9
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_10-cuda13_0-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu130
      GPU_ARCH_VERSION: "13.0"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda13.0
      DESIRED_PYTHON: "3.10"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_10-cuda13_0
      build_environment: linux-binary-manywheel
      PYTORCH_EXTRA_INSTALL_REQUIREMENTS: nvidia-cuda-nvrtc==13.0.88; platform_system == 'Linux' | nvidia-cuda-runtime==13.0.96; platform_system == 'Linux' | nvidia-cuda-cupti==13.0.85; platform_system == 'Linux' | nvidia-cudnn-cu13==9.13.0.50; platform_system == 'Linux' | nvidia-cublas==13.1.0.3; platform_system == 'Linux' | nvidia-cufft==12.0.0.61; platform_system == 'Linux' | nvidia-curand==10.4.0.35; platform_system == 'Linux' | nvidia-cusolver==12.0.4.66; platform_system == 'Linux' | nvidia-cusparse==12.6.3.3; platform_system == 'Linux' | nvidia-cusparselt-cu13==0.8.0; platform_system == 'Linux' | nvidia-nccl-cu13==2.27.7; platform_system == 'Linux' | nvidia-nvshmem-cu13==3.4.5; platform_system == 'Linux' | nvidia-nvtx==13.0.85; platform_system == 'Linux' | nvidia-nvjitlink==13.0.88; platform_system == 'Linux' | nvidia-cufile==1.15.1.6; platform_system == 'Linux'
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cuda13_0-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_10-cuda13_0-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu130
      GPU_ARCH_VERSION: "13.0"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda13.0
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cuda13_0
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.g4dn.4xlarge.nvidia.gpu # 12.8+ builds need sm_70+ runner
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-cuda13_0-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_10-cuda13_0-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu130
      GPU_ARCH_VERSION: "13.0"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda13.0
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-cuda13_0
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_10-rocm7_0-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: rocm7.0
      GPU_ARCH_VERSION: "7.0"
      GPU_ARCH_TYPE: rocm
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: rocm7.0
      DESIRED_PYTHON: "3.10"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      timeout-minutes: 300
      build_name: manywheel-py3_10-rocm7_0
      build_environment: linux-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-rocm7_0-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_10-rocm7_0-build
      - get-label-type
    runs-on: linux.rocm.gpu.mi250
    timeout-minutes: 240
    env:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: rocm7.0
      GPU_ARCH_VERSION: "7.0"
      GPU_ARCH_TYPE: rocm
      SKIP_ALL_TESTS: 1
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: rocm7.0
      DESIRED_PYTHON: "3.10"
    permissions:
      id-token: write
      contents: read
    steps:
      - name: Setup ROCm
        uses: ./.github/actions/setup-rocm
      - uses: actions/download-artifact@v4.1.7
        name: Download Build Artifacts
        with:
          name: manywheel-py3_10-rocm7_0
          path: "${{ runner.temp }}/artifacts/"
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
      - name: ROCm set GPU_FLAG
        run: |
          echo "GPU_FLAG=--device=/dev/mem --device=/dev/kfd --device=/dev/dri --group-add video --group-add daemon" >> "${GITHUB_ENV}"
      - name: configure aws credentials
        id: aws_creds
        if: ${{ startsWith(github.event.ref, 'refs/tags/ciflow/') }}
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_s3_and_ecr_read_only
          aws-region: us-east-1
          role-duration-seconds: 18000
      - name: Calculate docker image
        id: calculate-docker-image
        uses: pytorch/test-infra/.github/actions/calculate-docker-image@main
        with:
          docker-registry: ${{ startsWith(github.event.ref, 'refs/tags/ciflow/') && '308535385114.dkr.ecr.us-east-1.amazonaws.com' || 'docker.io' }}
          docker-image-name: manylinux2_28-builder
          custom-tag-prefix: rocm7.0
          docker-build-dir: .ci/docker
          working-directory: pytorch
      - name: Pull Docker image
        uses: pytorch/test-infra/.github/actions/pull-docker-image@main
        with:
          docker-image: ${{ steps.calculate-docker-image.outputs.docker-image }}
      - name: Test Pytorch binary
        uses: ./pytorch/.github/actions/test-pytorch-binary
        env:
          DOCKER_IMAGE: ${{ steps.calculate-docker-image.outputs.docker-image }}
      - name: Teardown ROCm
        uses: ./.github/actions/teardown-rocm
  manywheel-py3_10-rocm7_0-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_10-rocm7_0-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: rocm7.0
      GPU_ARCH_VERSION: "7.0"
      GPU_ARCH_TYPE: rocm
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: rocm7.0
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-rocm7_0
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_10-rocm7_1-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: rocm7.1
      GPU_ARCH_VERSION: "7.1"
      GPU_ARCH_TYPE: rocm
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: rocm7.1
      DESIRED_PYTHON: "3.10"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      timeout-minutes: 300
      build_name: manywheel-py3_10-rocm7_1
      build_environment: linux-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-rocm7_1-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_10-rocm7_1-build
      - get-label-type
    runs-on: linux.rocm.gpu.mi250
    timeout-minutes: 240
    env:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: rocm7.1
      GPU_ARCH_VERSION: "7.1"
      GPU_ARCH_TYPE: rocm
      SKIP_ALL_TESTS: 1
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: rocm7.1
      DESIRED_PYTHON: "3.10"
    permissions:
      id-token: write
      contents: read
    steps:
      - name: Setup ROCm
        uses: ./.github/actions/setup-rocm
      - uses: actions/download-artifact@v4.1.7
        name: Download Build Artifacts
        with:
          name: manywheel-py3_10-rocm7_1
          path: "${{ runner.temp }}/artifacts/"
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
      - name: ROCm set GPU_FLAG
        run: |
          echo "GPU_FLAG=--device=/dev/mem --device=/dev/kfd --device=/dev/dri --group-add video --group-add daemon" >> "${GITHUB_ENV}"
      - name: configure aws credentials
        id: aws_creds
        if: ${{ startsWith(github.event.ref, 'refs/tags/ciflow/') }}
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_s3_and_ecr_read_only
          aws-region: us-east-1
          role-duration-seconds: 18000
      - name: Calculate docker image
        id: calculate-docker-image
        uses: pytorch/test-infra/.github/actions/calculate-docker-image@main
        with:
          docker-registry: ${{ startsWith(github.event.ref, 'refs/tags/ciflow/') && '308535385114.dkr.ecr.us-east-1.amazonaws.com' || 'docker.io' }}
          docker-image-name: manylinux2_28-builder
          custom-tag-prefix: rocm7.1
          docker-build-dir: .ci/docker
          working-directory: pytorch
      - name: Pull Docker image
        uses: pytorch/test-infra/.github/actions/pull-docker-image@main
        with:
          docker-image: ${{ steps.calculate-docker-image.outputs.docker-image }}
      - name: Test Pytorch binary
        uses: ./pytorch/.github/actions/test-pytorch-binary
        env:
          DOCKER_IMAGE: ${{ steps.calculate-docker-image.outputs.docker-image }}
      - name: Teardown ROCm
        uses: ./.github/actions/teardown-rocm
  manywheel-py3_10-rocm7_1-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_10-rocm7_1-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: rocm7.1
      GPU_ARCH_VERSION: "7.1"
      GPU_ARCH_TYPE: rocm
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: rocm7.1
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-rocm7_1
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_10-xpu-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: xpu
      GPU_ARCH_TYPE: xpu
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: xpu
      DESIRED_PYTHON: "3.10"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_10-xpu
      build_environment: linux-binary-manywheel
      PYTORCH_EXTRA_INSTALL_REQUIREMENTS: intel-cmplr-lib-rt==2025.2.1 | intel-cmplr-lib-ur==2025.2.1 | intel-cmplr-lic-rt==2025.2.1 | intel-sycl-rt==2025.2.1 | oneccl-devel==2021.16.1; platform_system == 'Linux' and platform_machine == 'x86_64' | oneccl==2021.16.1; platform_system == 'Linux' and platform_machine == 'x86_64' | impi-rt==2021.16.1; platform_system == 'Linux' and platform_machine == 'x86_64' | onemkl-sycl-blas==2025.2.0 | onemkl-sycl-dft==2025.2.0 | onemkl-sycl-lapack==2025.2.0 | onemkl-sycl-rng==2025.2.0 | onemkl-sycl-sparse==2025.2.0 | dpcpp-cpp-rt==2025.2.1 | intel-opencl-rt==2025.2.1 | mkl==2025.2.0 | intel-openmp==2025.2.1 | tbb==2022.2.0 | tcmlib==1.4.0 | umf==0.11.0 | intel-pti==0.13.1
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_10-xpu-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_10-xpu-build
      - get-label-type
    runs-on: linux.idc.xpu
    timeout-minutes: 240
    env:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: xpu
      GPU_ARCH_TYPE: xpu
      SKIP_ALL_TESTS: 1
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: xpu
      DESIRED_PYTHON: "3.10"
    permissions:
      id-token: write
      contents: read
    steps:
      - name: Setup XPU
        uses: pytorch/pytorch/.github/actions/setup-xpu@main
      - name: configure aws credentials
        id: aws_creds
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_s3_and_ecr_read_only
          aws-region: us-east-1
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      - uses: actions/download-artifact@v4.1.7
        name: Download Build Artifacts
        with:
          name: manywheel-py3_10-xpu
          path: "${{ runner.temp }}/artifacts/"
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
      - name: Calculate docker image
        id: calculate-docker-image
        uses: pytorch/test-infra/.github/actions/calculate-docker-image@main
        with:
          docker-registry: ${{ startsWith(github.event.ref, 'refs/tags/ciflow/') && '308535385114.dkr.ecr.us-east-1.amazonaws.com' || 'docker.io' }}
          docker-image-name: manylinux2_28-builder
          custom-tag-prefix: xpu
          docker-build-dir: .ci/docker
          working-directory: pytorch
      - name: Pull Docker image
        uses: pytorch/test-infra/.github/actions/pull-docker-image@main
        with:
          docker-image: ${{ steps.calculate-docker-image.outputs.docker-image }}
      - name: Test Pytorch binary
        uses: ./pytorch/.github/actions/test-pytorch-binary
        env:
          DOCKER_IMAGE: ${{ steps.calculate-docker-image.outputs.docker-image }}
      - name: Teardown XPU
        uses: ./.github/actions/teardown-xpu
  manywheel-py3_10-xpu-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_10-xpu-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: xpu
      GPU_ARCH_TYPE: xpu
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: xpu
      DESIRED_PYTHON: "3.10"
      build_name: manywheel-py3_10-xpu
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_11-cpu-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu
      DESIRED_PYTHON: "3.11"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_11-cpu
      build_environment: linux-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cpu-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_11-cpu-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cpu
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.4xlarge
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cpu-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_11-cpu-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cpu
      GPU_ARCH_TYPE: cpu
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cpu
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cpu
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_11-cuda12_6-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu126
      GPU_ARCH_VERSION: "12.6"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.6
      DESIRED_PYTHON: "3.11"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_11-cuda12_6
      build_environment: linux-binary-manywheel
      PYTORCH_EXTRA_INSTALL_REQUIREMENTS: nvidia-cuda-nvrtc-cu12==12.6.77; platform_system == 'Linux' | nvidia-cuda-runtime-cu12==12.6.77; platform_system == 'Linux' | nvidia-cuda-cupti-cu12==12.6.80; platform_system == 'Linux' | nvidia-cudnn-cu12==9.10.2.21; platform_system == 'Linux' | nvidia-cublas-cu12==12.6.4.1; platform_system == 'Linux' | nvidia-cufft-cu12==11.3.0.4; platform_system == 'Linux' | nvidia-curand-cu12==10.3.7.77; platform_system == 'Linux' | nvidia-cusolver-cu12==11.7.1.2; platform_system == 'Linux' | nvidia-cusparse-cu12==12.5.4.2; platform_system == 'Linux' | nvidia-cusparselt-cu12==0.7.1; platform_system == 'Linux' | nvidia-nccl-cu12==2.27.5; platform_system == 'Linux' | nvidia-nvshmem-cu12==3.4.5; platform_system == 'Linux' | nvidia-nvtx-cu12==12.6.77; platform_system == 'Linux' | nvidia-nvjitlink-cu12==12.6.85; platform_system == 'Linux' | nvidia-cufile-cu12==1.11.1.6; platform_system == 'Linux'
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cuda12_6-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_11-cuda12_6-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu126
      GPU_ARCH_VERSION: "12.6"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.6
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cuda12_6
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.4xlarge.nvidia.gpu  # 12.6 build can use maxwell (sm_50) runner
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cuda12_6-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_11-cuda12_6-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu126
      GPU_ARCH_VERSION: "12.6"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.6
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cuda12_6
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_11-cuda12_8-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu128
      GPU_ARCH_VERSION: "12.8"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.8
      DESIRED_PYTHON: "3.11"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_11-cuda12_8
      build_environment: linux-binary-manywheel
      PYTORCH_EXTRA_INSTALL_REQUIREMENTS: nvidia-cuda-nvrtc-cu12==12.8.93; platform_system == 'Linux' | nvidia-cuda-runtime-cu12==12.8.90; platform_system == 'Linux' | nvidia-cuda-cupti-cu12==12.8.90; platform_system == 'Linux' | nvidia-cudnn-cu12==9.10.2.21; platform_system == 'Linux' | nvidia-cublas-cu12==12.8.4.1; platform_system == 'Linux' | nvidia-cufft-cu12==11.3.3.83; platform_system == 'Linux' | nvidia-curand-cu12==10.3.9.90; platform_system == 'Linux' | nvidia-cusolver-cu12==11.7.3.90; platform_system == 'Linux' | nvidia-cusparse-cu12==12.5.8.93; platform_system == 'Linux' | nvidia-cusparselt-cu12==0.7.1; platform_system == 'Linux' | nvidia-nccl-cu12==2.27.5; platform_system == 'Linux' | nvidia-nvshmem-cu12==3.4.5; platform_system == 'Linux' | nvidia-nvtx-cu12==12.8.90; platform_system == 'Linux' | nvidia-nvjitlink-cu12==12.8.93; platform_system == 'Linux' | nvidia-cufile-cu12==1.13.1.3; platform_system == 'Linux'
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cuda12_8-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_11-cuda12_8-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu128
      GPU_ARCH_VERSION: "12.8"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.8
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cuda12_8
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.g4dn.4xlarge.nvidia.gpu # 12.8+ builds need sm_70+ runner
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cuda12_8-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_11-cuda12_8-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu128
      GPU_ARCH_VERSION: "12.8"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.8
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cuda12_8
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_11-cuda12_9-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu129
      GPU_ARCH_VERSION: "12.9"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.9
      DESIRED_PYTHON: "3.11"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_11-cuda12_9
      build_environment: linux-binary-manywheel
      PYTORCH_EXTRA_INSTALL_REQUIREMENTS: nvidia-cuda-nvrtc-cu12==12.9.86; platform_system == 'Linux' | nvidia-cuda-runtime-cu12==12.9.79; platform_system == 'Linux' | nvidia-cuda-cupti-cu12==12.9.79; platform_system == 'Linux' | nvidia-cudnn-cu12==9.10.2.21; platform_system == 'Linux' | nvidia-cublas-cu12==12.9.1.4; platform_system == 'Linux' | nvidia-cufft-cu12==11.4.1.4; platform_system == 'Linux' | nvidia-curand-cu12==10.3.10.19; platform_system == 'Linux' | nvidia-cusolver-cu12==11.7.5.82; platform_system == 'Linux' | nvidia-cusparse-cu12==12.5.10.65; platform_system == 'Linux' | nvidia-cusparselt-cu12==0.7.1; platform_system == 'Linux' | nvidia-nccl-cu12==2.27.5; platform_system == 'Linux' | nvidia-nvshmem-cu12==3.4.5; platform_system == 'Linux' | nvidia-nvtx-cu12==12.9.79; platform_system == 'Linux' | nvidia-nvjitlink-cu12==12.9.86; platform_system == 'Linux' | nvidia-cufile-cu12==1.14.1.1; platform_system == 'Linux'
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cuda12_9-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_11-cuda12_9-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu129
      GPU_ARCH_VERSION: "12.9"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.9
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cuda12_9
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.g4dn.4xlarge.nvidia.gpu # 12.8+ builds need sm_70+ runner
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cuda12_9-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_11-cuda12_9-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu129
      GPU_ARCH_VERSION: "12.9"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda12.9
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cuda12_9
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_11-cuda13_0-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu130
      GPU_ARCH_VERSION: "13.0"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda13.0
      DESIRED_PYTHON: "3.11"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      build_name: manywheel-py3_11-cuda13_0
      build_environment: linux-binary-manywheel
      PYTORCH_EXTRA_INSTALL_REQUIREMENTS: nvidia-cuda-nvrtc==13.0.88; platform_system == 'Linux' | nvidia-cuda-runtime==13.0.96; platform_system == 'Linux' | nvidia-cuda-cupti==13.0.85; platform_system == 'Linux' | nvidia-cudnn-cu13==9.13.0.50; platform_system == 'Linux' | nvidia-cublas==13.1.0.3; platform_system == 'Linux' | nvidia-cufft==12.0.0.61; platform_system == 'Linux' | nvidia-curand==10.4.0.35; platform_system == 'Linux' | nvidia-cusolver==12.0.4.66; platform_system == 'Linux' | nvidia-cusparse==12.6.3.3; platform_system == 'Linux' | nvidia-cusparselt-cu13==0.8.0; platform_system == 'Linux' | nvidia-nccl-cu13==2.27.7; platform_system == 'Linux' | nvidia-nvshmem-cu13==3.4.5; platform_system == 'Linux' | nvidia-nvtx==13.0.85; platform_system == 'Linux' | nvidia-nvjitlink==13.0.88; platform_system == 'Linux' | nvidia-cufile==1.15.1.6; platform_system == 'Linux'
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cuda13_0-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_11-cuda13_0-build
      - get-label-type
    uses: ./.github/workflows/_binary-test-linux.yml
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu130
      GPU_ARCH_VERSION: "13.0"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda13.0
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cuda13_0
      build_environment: linux-binary-manywheel
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      runs_on: linux.g4dn.4xlarge.nvidia.gpu # 12.8+ builds need sm_70+ runner
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-cuda13_0-upload:  # Uploading
    if: ${{ github.repository_owner == 'pytorch' }}
    permissions:
      id-token: write
      contents: read
    needs: manywheel-py3_11-cuda13_0-test
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: cu130
      GPU_ARCH_VERSION: "13.0"
      GPU_ARCH_TYPE: cuda
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: cuda13.0
      DESIRED_PYTHON: "3.11"
      build_name: manywheel-py3_11-cuda13_0
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
    uses: ./.github/workflows/_binary-upload.yml

  manywheel-py3_11-rocm7_0-build:
    if: ${{ github.repository_owner == 'pytorch' }}
    uses: ./.github/workflows/_binary-build-linux.yml
    needs: get-label-type
    with:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: rocm7.0
      GPU_ARCH_VERSION: "7.0"
      GPU_ARCH_TYPE: rocm
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: rocm7.0
      DESIRED_PYTHON: "3.11"
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      timeout-minutes: 300
      build_name: manywheel-py3_11-rocm7_0
      build_environment: linux-binary-manywheel
    secrets:
      github-token: ${{ secrets.GITHUB_TOKEN }}
  manywheel-py3_11-rocm7_0-test:  # Testing
    if: ${{ github.repository_owner == 'pytorch' }}
    needs:
      - manywheel-py3_11-rocm7_0-build
      - get-label-type
    runs-on: linux.rocm.gpu.mi250
    timeout-minutes: 240
    env:
      PYTORCH_ROOT: /pytorch
      PACKAGE_TYPE: manywheel
      # TODO: This is a legacy variable that we eventually want to get rid of in
      #       favor of GPU_ARCH_VERSION
      DESIRED_CUDA: rocm7.0
      GPU_ARCH_VERSION: "7.0"
      GPU_ARCH_TYPE: rocm
      SKIP_ALL_TESTS: 1
      DOCKER_IMAGE: manylinux2_28-builder
      DOCKER_IMAGE_TAG_PREFIX: rocm7.0
      DESIRED_PYTHON: "3.11"
    permissions:
      id-token: write
      contents: read
    steps:
      - name: Setup ROCm
        uses: ./.github/actions/setup-rocm
      - uses: actions/download-artifact@v4.1.7
        name: Download Build Artifacts
        with:
          name: manywheel-py3_11-rocm7_0
          path: "${{ runner.temp }}/artifacts/"
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
      - name: ROCm set GPU_FLAG
        run: |
          echo "GPU_FLAG=--device=/dev/mem --device=/dev/kfd --device=/dev/dri --group-add video --group-add daemon" >> "${GITHUB_ENV}"
      - name: configure aws credentials
        id: aws_creds
        if: ${{ startsWith(github.event.ref, 'refs/tags/ciflow/') }}
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_s3_and_ecr_read_only
          aws-region: us-east-1
          role-duration-seconds: 18000
      - name: Calculate docker image
        id: calculate-docker-image
        uses: pytorch/test-infra/.github/actions/calculate-docker-image@main
        with:
          docker-registry: ${{ startsWith(github.event.ref, 'refs/tags/ciflow/') && '308535385114.dkr.ecr.us-east-1.amazonaws.com' || 'docker.io' }}
          docker-image-name: manylinux2_28-builder
          custom-tag-prefix: rocm7.0
          docker-build-dir: .ci/docker
          working-directory: pytorch
      - name: Pull Docker image
        uses: pytorch/test-infra/.github/actions/pull-docker-ima
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
- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `generated-linux-binary-manywheel-nightly.yml_docs.md`
- **Keyword Index**: `generated-linux-binary-manywheel-nightly.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
