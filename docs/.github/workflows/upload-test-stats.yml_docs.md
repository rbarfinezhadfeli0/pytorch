# Documentation: `.github/workflows/upload-test-stats.yml`

## File Metadata

- **Path**: `.github/workflows/upload-test-stats.yml`
- **Size**: 6,197 bytes (6.05 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This appears to be a **test file**.

## Original Source

```yaml
name: Upload test stats

on:
  workflow_run:
    workflows:
      - pull
      - trunk
      - trunk-rocm-mi300
      - periodic
      - periodic-rocm-mi200
      - periodic-rocm-mi300
      - inductor
      - unstable
      - slow
      - slow-rocm-mi200
      - unstable-periodic
      - inductor-periodic
      - rocm-mi200
      - rocm-mi300
      - rocm-mi355
      - inductor-micro-benchmark
      - inductor-micro-benchmark-x86
      - inductor-cu124
      - inductor-rocm-mi200
      - inductor-rocm-mi300
      - mac-mps
      - linux-aarch64
    types:
      - completed

jobs:
  # the conclusion field in the github context is sometimes null
  # solution adapted from https://github.com/community/community/discussions/21090#discussioncomment-3226271
  get_workflow_conclusion:
    if: github.repository_owner == 'pytorch'
    runs-on: ubuntu-latest
    outputs:
      conclusion: ${{ fromJson(steps.get_conclusion.outputs.data).conclusion }}
    steps:
      - name: Get workflow run conclusion
        # TODO (huydhn): Pin this once https://github.com/octokit/request-action/issues/315 is resolved
        uses: octokit/request-action@05a2312de9f8207044c4c9e41fe19703986acc13 # v2.x
        id: get_conclusion
        with:
          route: GET /repos/${{ github.repository }}/actions/runs/${{ github.event.workflow_run.id }}/attempts/${{ github.event.workflow_run.run_attempt }}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  upload-test-stats:
    needs: get_workflow_conclusion
    if: github.repository_owner == 'pytorch'
    runs-on: ubuntu-22.04
    environment: upload-stats
    permissions:
      id-token: write
    name: Upload test stats for ${{ github.event.workflow_run.id }}, attempt ${{ github.event.workflow_run.run_attempt }}
    steps:
      - name: Print workflow information
        env:
          TRIGGERING_WORKFLOW: ${{ toJSON(github.event.workflow_run) }}
        run: echo "${TRIGGERING_WORKFLOW}"

      - name: Checkout PyTorch
        uses: pytorch/pytorch/.github/actions/checkout-pytorch@main

      - name: Configure aws credentials
        uses: aws-actions/configure-aws-credentials@ececac1a45f3b08a01d2dd070d28d111c5fe6722 # v4.1.0
        continue-on-error: true
        with:
          role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_upload-torch-test-stats
          aws-region: us-east-1

      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
        with:
          python-version: '3.11'
          cache: pip

      - run: |
          pip3 install requests==2.32.2 boto3==1.35.42

      - name: Upload test artifacts
        id: upload-s3
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          WORKFLOW_ARTIFACTS_URL: ${{ github.event.workflow_run.artifacts_url }}
          WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}
          WORKFLOW_RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}
          REPO_FULLNAME: ${{ github.event.workflow_run.repository.full_name }}
        run: |
          echo "${WORKFLOW_ARTIFACTS_URL}"

          # Note that in the case of Linux and Windows, their artifacts have already been uploaded to S3, so there simply won't be
          # anything on GitHub to upload. The command should return right away
          python3 -m tools.stats.upload_artifacts --workflow-run-id "${WORKFLOW_RUN_ID}" --workflow-run-attempt "${WORKFLOW_RUN_ATTEMPT}" --repo "${REPO_FULLNAME}"

      - name: Upload test stats
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}
          WORKFLOW_RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}
          WORKFLOW_URL: ${{ github.event.workflow_run.html_url }}
          HEAD_REPOSITORY: ${{ github.event.workflow_run.head_repository.full_name }}
          HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
        run: |
          echo "${WORKFLOW_URL}"
          python3 -m tools.stats.upload_test_stats --workflow-run-id "${WORKFLOW_RUN_ID}" --workflow-run-attempt "${WORKFLOW_RUN_ATTEMPT}" --head-branch "${HEAD_BRANCH}" --head-repository "${HEAD_REPOSITORY}"
          python3 -m tools.stats.upload_sccache_stats --workflow-run-id "${WORKFLOW_RUN_ID}" --workflow-run-attempt "${WORKFLOW_RUN_ATTEMPT}"

      - name: Analyze disabled tests rerun
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          WORKFLOW_ARTIFACTS_URL: ${{ github.event.workflow_run.artifacts_url }}
          WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}
          WORKFLOW_RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}
          REPO_FULLNAME: ${{ github.event.workflow_run.repository.full_name }}
        run: |
          # Analyze the results from disable tests rerun and upload them to S3
          python3 -m tools.stats.check_disabled_tests --workflow-run-id "${WORKFLOW_RUN_ID}" --workflow-run-attempt "${WORKFLOW_RUN_ATTEMPT}" --repo "${REPO_FULLNAME}"

      - name: Upload gpt-fast benchmark results to s3
        if: steps.upload-s3.outcome && steps.upload-s3.outcome == 'success' && contains(github.event.workflow_run.name, 'inductor-micro-benchmark')
        env:
          WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}
          WORKFLOW_RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}
          REPO_FULLNAME: ${{ github.event.workflow_run.repository.full_name }}
          HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
        run: |
          python3 -m tools.stats.upload_dynamo_perf_stats --workflow-run-id "${WORKFLOW_RUN_ID}" --workflow-run-attempt "${WORKFLOW_RUN_ATTEMPT}" --repo "${REPO_FULLNAME}" --head-branch "${HEAD_BRANCH}" --dynamodb-table torchci-oss-ci-benchmark --match-filename "^gpt_fast_benchmark"

  check-api-rate:
    if: ${{ always() && github.repository_owner == 'pytorch' }}
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - name: Get our GITHUB_TOKEN API limit usage
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          curl -H "Accept: application/vnd.github.v3+json" -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit

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

This is a test file. Run it with:

```bash
python .github/workflows/upload-test-stats.yml
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

- **File Documentation**: `upload-test-stats.yml_docs.md`
- **Keyword Index**: `upload-test-stats.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
