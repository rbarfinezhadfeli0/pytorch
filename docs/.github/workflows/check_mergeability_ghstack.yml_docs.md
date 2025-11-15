# Documentation: `.github/workflows/check_mergeability_ghstack.yml`

## File Metadata

- **Path**: `.github/workflows/check_mergeability_ghstack.yml`
- **Size**: 2,669 bytes (2.61 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: Check mergeability of ghstack PR

on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches: [gh/**/base]

jobs:
  ghstack-mergeability-check:
    if: github.repository_owner == 'pytorch'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          fetch-depth: 0

      - name: Setup git
        shell: bash
        run: |
          git config --global user.email "pytorchmergebot@users.noreply.github.com"
          git config --global user.name "PyTorch MergeBot"
          git fetch origin main:main

      - name: Wait for orig branch
        shell: bash
        run: |
          BRANCH="${{ github.base_ref }}"
          echo "$BRANCH"
          BRANCH="${BRANCH%/base}/orig"
          echo "$BRANCH"

          WAIT_SECONDS=300
          END_WAIT=$((SECONDS+WAIT_SECONDS))
          BRANCH_EXISTS=0

          while [ $SECONDS -lt $END_WAIT ]; do
            git fetch --prune origin "${BRANCH}" || true
            if git rev-parse --verify "origin/${BRANCH}"; then
              BRANCH_EXISTS=1
              break
            fi
            echo "Waiting for branch ${BRANCH} to exist..."
            sleep 30  # Wait for 30 seconds before retrying
          done

          if [ $BRANCH_EXISTS -eq 0 ]; then
            echo "Branch ${BRANCH} not found after ${WAIT_SECONDS} seconds."
            echo "Mergeability check failed for infrastructure reasons."
            exit 1
          fi

      - name: Setup Python
        uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
        with:
          python-version: '3.9'
          cache: pip
          architecture: x64

      - run: pip install pyyaml==6.0.2
        shell: bash

      - name: Verify mergeability
        shell: bash
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          PR_NUM: ${{ github.event.pull_request.number }}
        run: |
          set -ex
          python3 .github/scripts/trymerge.py --check-mergeability "${PR_NUM}"

      - name: Print debug info
        if: failure()
        shell: bash
        env:
          PR_NUM: ${{ github.event.pull_request.number }}
        run: |
          {
            echo "# PR $PR_NUM is not mergeable into main"
            echo "To debug, run the diagnostic workflow:"
            echo "https://github.com/pytorch/test-infra/actions/workflows/pr-dependencies-check.yml"
          } >> "$GITHUB_STEP_SUMMARY"


concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
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

- **File Documentation**: `check_mergeability_ghstack.yml_docs.md`
- **Keyword Index**: `check_mergeability_ghstack.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
