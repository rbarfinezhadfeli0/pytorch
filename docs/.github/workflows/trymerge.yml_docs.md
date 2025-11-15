# Documentation: `.github/workflows/trymerge.yml`

## File Metadata

- **Path**: `.github/workflows/trymerge.yml`
- **Size**: 4,106 bytes (4.01 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: Validate and merge PR

on:
  repository_dispatch:
    types: [try-merge]

jobs:
  do_merge:
    name: try_merge_pr_${{ github.event.client_payload.pr_num }}
    runs-on: linux.24_04.4x
    environment: mergebot
    permissions:
      id-token: write
    env:
        GH_RUN_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
    steps:
      - name: Checkout repo
        id: checkout
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          fetch-depth: 0
          token: ${{ secrets.MERGEBOT_TOKEN }}

      - name: Setup Python
        uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
        with:
          python-version: '3.9'
          check-latest: false
          cache: pip
          architecture: x64
      - run: pip install pyyaml==6.0.2

      - name: Setup committer id
        run: |
          git config --global user.email "pytorchmergebot@users.noreply.github.com"
          git config --global user.name "PyTorch MergeBot"
      - name: Merge PR
        shell: bash
        env:
          GITHUB_TOKEN: ${{ secrets.MERGEBOT_TOKEN }}
          PR_NUM: ${{ github.event.client_payload.pr_num }}
          FORCE: ${{ github.event.client_payload.force}}
          COMMENT_ID: ${{ github.event.client_payload.comment_id }}
          REBASE: ${{ github.event.client_payload.rebase }}
          IGNORE_CURRENT: ${{ github.event.client_payload.ignore_current }}
          DRCI_BOT_KEY: ${{ secrets.DRCI_BOT_KEY }}
          GITHUB_RUN_ID: ${{ github.run_id }}
        run: |
          set -x
          if [ -n "${REBASE}" ]; then
            # attempt to rebase, if it fails then comment on the PR that it failed
            if ! python3 .github/scripts/tryrebase.py "${PR_NUM}" --branch "${REBASE}"; then
              python3 .github/scripts/comment_on_pr.py "${PR_NUM}" "merge"
              exit 0
            fi
            git checkout main
            git fetch -p
            # give github some time between the push and start workflows so that Github's messages
            # on the PR appear in chronological order (timing issues can shuffle them around)
            sleep 60
          fi

          # Require a comment id for merge operations
          if [ -z "${COMMENT_ID}" ]; then
            echo "Error: merge requires COMMENT_ID to be specified"
            exit 1
          fi

          if [ -n "${FORCE}" ]; then
            python3 .github/scripts/trymerge.py --force --comment-id "${COMMENT_ID}" "${PR_NUM}"
          elif [ -n "${IGNORE_CURRENT}" ]; then
            python3 .github/scripts/trymerge.py --ignore-current --comment-id "${COMMENT_ID}" "${PR_NUM}"
          else
            python3 .github/scripts/trymerge.py --comment-id "${COMMENT_ID}" "${PR_NUM}"
          fi
      - name: Comment on Canceled
        if: ${{ cancelled() && steps.checkout.outcome == 'success' }}
        continue-on-error: true
        env:
          GITHUB_TOKEN: ${{ secrets.MERGEBOT_TOKEN }}
          PR_NUM: ${{ github.event.client_payload.pr_num }}
        run: |
          set -x
          python3 .github/scripts/comment_on_pr.py "${PR_NUM}" "merge"

      - name: configure aws credentials
        uses: aws-actions/configure-aws-credentials@ececac1a45f3b08a01d2dd070d28d111c5fe6722 # v4.1.0
        continue-on-error: true
        with:
          role-to-assume: arn:aws:iam::308535385114:role/upload_to_ossci_raw_job_status
          aws-region: us-east-1

      - name: Upload merge record to s3
        if: always()
        continue-on-error: true
        uses: seemethere/upload-artifact-s3@baba72d0712b404f646cebe0730933554ebce96a # v5.1.0
        with:
          s3-bucket: ossci-raw-job-status
          s3-prefix: merges/${{ github.repository }}/${{ github.event.client_payload.pr_num }}/${{ github.event.client_payload.comment_id }}/${{ github.run_id }}
          path: merge_record.json

# We want newer merge commands to supersede old ones
concurrency:
  group: try-merge-${{ github.event.client_payload.pr_num }}
  cancel-in-progress: true

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

- **File Documentation**: `trymerge.yml_docs.md`
- **Keyword Index**: `trymerge.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
