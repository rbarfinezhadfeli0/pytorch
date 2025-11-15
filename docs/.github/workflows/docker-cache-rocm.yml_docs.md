# Documentation: `.github/workflows/docker-cache-rocm.yml`

## File Metadata

- **Path**: `.github/workflows/docker-cache-rocm.yml`
- **Size**: 4,214 bytes (4.12 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: docker-cache-rocm

on:
  workflow_run:
    workflows: [docker-builds]
    branches: [main, release]
    types:
      - completed
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read
  actions: read

jobs:
  download-docker-builds-artifacts:
    if: github.repository_owner == 'pytorch'
    name: download-docker-builds-artifacts
    runs-on: ubuntu-latest
    outputs:
      pytorch-linux-jammy-rocm-n-py3: ${{ steps.process-artifacts.outputs.pytorch-linux-jammy-rocm-n-py3 }}
      pytorch-linux-noble-rocm-n-py3: ${{ steps.process-artifacts.outputs.pytorch-linux-noble-rocm-n-py3 }}
      pytorch-linux-jammy-rocm-n-py3-benchmarks: ${{ steps.process-artifacts.outputs.pytorch-linux-jammy-rocm-n-py3-benchmarks }}
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4.1.7
        with:
          run-id: ${{ github.event.workflow_run.id }}
          path: ./docker-builds-artifacts
          merge-multiple: true
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Process artifacts
        id: process-artifacts
        run: |
          ls -R ./docker-builds-artifacts
          cat ./docker-builds-artifacts/*txt >> "${GITHUB_OUTPUT}"
          cat "${GITHUB_OUTPUT}"

  docker-cache:
    if: github.repository_owner == 'pytorch'
    needs: download-docker-builds-artifacts
    strategy:
      fail-fast: false
      matrix:
        runner: [linux.rocm.gfx942.docker-cache]
        docker-image: [
          "${{ needs.download-docker-builds-artifacts.outputs.pytorch-linux-jammy-rocm-n-py3 }}",
          "${{ needs.download-docker-builds-artifacts.outputs.pytorch-linux-noble-rocm-n-py3 }}",
          "${{ needs.download-docker-builds-artifacts.outputs.pytorch-linux-jammy-rocm-n-py3-benchmarks }}"
        ]
    runs-on: "${{ matrix.runner }}"
    steps:
      - name: debug
        run: |
          JSON_STRINGIFIED="${{ toJSON(needs.download-docker-builds-artifacts.outputs) }}"
          echo "Outputs of download-docker-builds-artifacts job: ${JSON_STRINGIFIED}"

      - name: configure aws credentials
        id: aws_creds
        uses: aws-actions/configure-aws-credentials@ececac1a45f3b08a01d2dd070d28d111c5fe6722 # v4.1.0
        with:
          role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_s3_and_ecr_read_only
          aws-region: us-east-1
          role-duration-seconds: 18000

      - name: Login to Amazon ECR
        id: login-ecr
        continue-on-error: false
        uses: aws-actions/amazon-ecr-login@062b18b96a7aff071d4dc91bc00c4c1a7945b076 # v2.0.1

      - name: Generate ghrc.io tag
        id: ghcr-io-tag
        run: |
            ecr_image="${{ matrix.docker-image }}"
            ghcr_image="ghcr.io/pytorch/ci-image:${ecr_image##*:}"
            echo "ghcr_image=${ghcr_image}" >> "$GITHUB_OUTPUT"

      - name: Pull docker image
        uses: pytorch/test-infra/.github/actions/pull-docker-image@main
        with:
          docker-image: ${{ steps.ghcr-io-tag.outputs.ghcr_image }}

      - name: Save as tarball
        run: |
          docker_image_tag=${{ matrix.docker-image }}
          docker_image_tag="${docker_image_tag#*:}" # Remove everything before and including first ":"
          docker_image_tag="${docker_image_tag%-*}" # Remove everything after and including last "-"
          ref_name=${{ github.event.workflow_run.head_branch }}
          if [[ $ref_name =~ "release/" ]]; then
            ref_suffix="release"
          elif [[ $ref_name == "main" ]]; then
            ref_suffix="main"
          else
            echo "Unexpected branch in ref_name: ${ref_name}" && exit 1
          fi
          docker tag ${{ steps.ghcr-io-tag.outputs.ghcr_image }} ${{ matrix.docker-image }}
          # mv is atomic operation, so we use intermediate tar.tmp file to prevent read-write contention
          docker save -o ~/pytorch-data/docker/${docker_image_tag}.tar.tmp ${{ matrix.docker-image }}
          mv ~/pytorch-data/docker/${docker_image_tag}.tar.tmp ~/pytorch-data/docker/${docker_image_tag}_${ref_suffix}.tar

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

- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

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

- **File Documentation**: `docker-cache-rocm.yml_docs.md`
- **Keyword Index**: `docker-cache-rocm.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
