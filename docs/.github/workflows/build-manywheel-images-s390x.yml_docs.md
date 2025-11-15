# Documentation: `.github/workflows/build-manywheel-images-s390x.yml`

## File Metadata

- **Path**: `.github/workflows/build-manywheel-images-s390x.yml`
- **Size**: 3,282 bytes (3.21 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: Build manywheel docker images for s390x

on:
  workflow_dispatch:
  push:
    tags:
      - ciflow/s390/*
    paths:
      - .github/workflows/build-manywheel-images-s390x.yml


env:
  DOCKER_REGISTRY: "docker.io"
  DOCKER_BUILDKIT: 1
  WITH_PUSH: ${{ github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/heads/release') || startsWith(github.ref, 'refs/tags/v')) }}

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
  cancel-in-progress: true

jobs:
  build-docker-cpu-s390x:
    if: github.repository_owner == 'pytorch'
    environment: ${{ (github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/heads/release') || startsWith(github.ref, 'refs/tags/v')) && 'docker-build') || '' }}
    runs-on: linux.s390x
    steps:
      - name: Checkout PyTorch
        uses: pytorch/pytorch/.github/actions/checkout-pytorch@main
        with:
          submodules: false
          no-sudo: true

      - name: Build Docker Image
        run: |
          .ci/docker/manywheel/build.sh manylinuxs390x-builder:cpu-s390x -t manylinuxs390x-builder:cpu-s390x

      - name: Tag and (if WITH_PUSH) push docker image to docker.io
        env:
          DOCKER_TOKEN: ${{ secrets.DOCKER_TOKEN }}
          DOCKER_ID: ${{ secrets.DOCKER_ID }}
          CREATED_FULL_DOCKER_IMAGE_NAME: manylinuxs390x-builder:cpu-s390x
        shell: bash
        run: |
          set -euox pipefail
          GITHUB_REF="${GITHUB_REF:-$(git symbolic-ref -q HEAD || git describe --tags --exact-match)}"
          GIT_BRANCH_NAME="${GITHUB_REF##*/}"
          GIT_COMMIT_SHA="${GITHUB_SHA:-$(git rev-parse HEAD)}"
          CI_FOLDER_SHA="$(git rev-parse HEAD:.ci/docker)"

          DOCKER_IMAGE_NAME_PREFIX="docker.io/pytorch/${CREATED_FULL_DOCKER_IMAGE_NAME}"

          docker tag "${CREATED_FULL_DOCKER_IMAGE_NAME}" "${DOCKER_IMAGE_NAME_PREFIX}-${GIT_BRANCH_NAME}"
          docker tag "${CREATED_FULL_DOCKER_IMAGE_NAME}" "${DOCKER_IMAGE_NAME_PREFIX}-${GIT_COMMIT_SHA}"
          docker tag "${CREATED_FULL_DOCKER_IMAGE_NAME}" "${DOCKER_IMAGE_NAME_PREFIX}-${CI_FOLDER_SHA}"

          # Pretty sure Github will mask tokens and I'm not sure if it will even be
          # printed due to pipe, but just in case
          set +x
          if [[ "${WITH_PUSH:-false}" == "true" ]]; then
            echo "${DOCKER_TOKEN}" | docker login -u "${DOCKER_ID}" --password-stdin
            docker push "${DOCKER_IMAGE_NAME_PREFIX}-${GIT_BRANCH_NAME}"
            docker push "${DOCKER_IMAGE_NAME_PREFIX}-${GIT_COMMIT_SHA}"
            docker push "${DOCKER_IMAGE_NAME_PREFIX}-${CI_FOLDER_SHA}"
          fi

      - name: Cleanup docker
        if: cancelled()
        shell: bash
        run: |
          # If podman build command is interrupted,
          # it can leave a couple of processes still running.
          # Order them to stop for clean shutdown.
          # It looks like sometimes some processes remain
          # after first cleanup.
          # Wait a bit and do cleanup again. It looks like it helps.
          docker system prune --build -f || true
          sleep 60
          docker system prune --build -f || true

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

- **File Documentation**: `build-manywheel-images-s390x.yml_docs.md`
- **Keyword Index**: `build-manywheel-images-s390x.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
