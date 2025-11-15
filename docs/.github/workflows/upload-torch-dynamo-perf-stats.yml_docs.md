# Documentation: `.github/workflows/upload-torch-dynamo-perf-stats.yml`

## File Metadata

- **Path**: `.github/workflows/upload-torch-dynamo-perf-stats.yml`
- **Size**: 3,636 bytes (3.55 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: Upload torch dynamo performance stats

on:
  workflow_run:
    workflows: [inductor-A100-perf-nightly, inductor-perf-nightly-A10g, inductor-perf-nightly-aarch64, inductor-perf-nightly-x86, inductor-perf-nightly-macos, inductor-perf-nightly-rocm, inductor-perf-nightly-h100]
    types:
      - completed

jobs:
  get-conclusion:
    runs-on: ubuntu-latest
    outputs:
      conclusion: ${{ fromJson(steps.get-conclusion.outputs.data).conclusion }}
    steps:
      - name: Get workflow run conclusion
        # TODO (huydhn): Pin this once https://github.com/octokit/request-action/issues/315 is resolved
        uses: octokit/request-action@05a2312de9f8207044c4c9e41fe19703986acc13 # v2.x
        id: get-conclusion
        with:
          route: GET /repos/${{ github.repository }}/actions/runs/${{ github.event.workflow_run.id }}/attempts/${{ github.event.workflow_run.run_attempt }}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  upload-perf-stats:
    needs: get-conclusion
    if: github.event.workflow_run.conclusion == 'success' || needs.get-conclusion.outputs.conclusion == 'success' ||
        github.event.workflow_run.conclusion == 'failure' || needs.get-conclusion.outputs.conclusion == 'failure'
    runs-on: ubuntu-22.04
    environment: upload-stats
    permissions:
      id-token: write
    name: Upload dynamo performance stats for ${{ github.event.workflow_run.id }}, attempt ${{ github.event.workflow_run.run_attempt }}
    steps:
      - name: Checkout PyTorch
        uses: pytorch/pytorch/.github/actions/checkout-pytorch@main
        with:
          submodules: false
          fetch-depth: 1

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

      - name: Upload torch dynamo performance stats to S3
        id: upload-s3
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          WORKFLOW_ARTIFACTS_URL: ${{ github.event.workflow_run.artifacts_url }}
          WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}
          WORKFLOW_RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}
          REPO_FULLNAME: ${{ github.event.workflow_run.repository.full_name }}
        run: |
          # Upload perf test reports from GHA to S3, which can now be downloaded
          # on HUD
          python3 -m tools.stats.upload_artifacts --workflow-run-id "${WORKFLOW_RUN_ID}" --workflow-run-attempt "${WORKFLOW_RUN_ATTEMPT}" --repo "${REPO_FULLNAME}"

      - name: Upload torch dynamo performance stats to s3
        if: steps.upload-s3.outcome && steps.upload-s3.outcome == 'success'
        env:
          WORKFLOW_RUN_ID: ${{ github.event.workflow_run.id }}
          WORKFLOW_RUN_ATTEMPT: ${{ github.event.workflow_run.run_attempt }}
          REPO_FULLNAME: ${{ github.event.workflow_run.repository.full_name }}
          HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
        run: |
          python3 -m tools.stats.upload_dynamo_perf_stats --workflow-run-id "${WORKFLOW_RUN_ID}" --workflow-run-attempt "${WORKFLOW_RUN_ATTEMPT}" --repo "${REPO_FULLNAME}" --head-branch "${HEAD_BRANCH}" --dynamodb-table torchci-dynamo-perf-stats --match-filename "^inductor_"

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

- **File Documentation**: `upload-torch-dynamo-perf-stats.yml_docs.md`
- **Keyword Index**: `upload-torch-dynamo-perf-stats.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
