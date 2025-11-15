# Documentation: `.github/workflows/target_determination.yml`

## File Metadata

- **Path**: `.github/workflows/target_determination.yml`
- **Size**: 3,384 bytes (3.30 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: target-determination

on:
  workflow_call:

jobs:

  get-label-type:
    name: get-label-type
    # Don't run on forked repos
    if: github.repository_owner == 'pytorch'
    uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}
      curr_ref_type: ${{ github.ref_type }}

  target-determination:
    # Don't run on forked repos
    if: github.repository_owner == 'pytorch'
    runs-on: "${{ needs.get-label-type.outputs.label-type }}linux.2xlarge"
    needs: get-label-type
    steps:
      # [pytorch repo ref]
      # Use a pytorch/pytorch reference instead of a reference to the local
      # checkout because when we run this action we don't *have* a local
      # checkout. In other cases you should prefer a local checkout.
      - name: Checkout PyTorch
        uses: pytorch/pytorch/.github/actions/checkout-pytorch@main
        with:
          submodules: false

      - name: Setup Linux
        uses: ./.github/actions/setup-linux

      - name: Get workflow job id
        id: get-job-id
        uses: ./.github/actions/get-workflow-job-id
        if: always()
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Download pytest cache
        uses: ./.github/actions/pytest-cache-download
        continue-on-error: true
        with:
          cache_dir: .pytest_cache
          job_identifier: ${{ github.workflow }}

      - name: Download LLM Artifacts from S3
        uses: seemethere/download-artifact-s3@1da556a7aa0a088e3153970611f6c432d58e80e6 # v4.2.0
        continue-on-error: true
        with:
          name: llm_results
          path: .additional_ci_files/llm_results

      - name: Do TD
        id: td
        continue-on-error: true
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          GITHUB_WORKFLOW: ${{ github.workflow }}
          GITHUB_JOB: ${{ github.job }}
          GITHUB_RUN_ID: ${{ github.run_id }}
          GITHUB_RUN_NUMBER: ${{ github.run_number }}
          GITHUB_RUN_ATTEMPT: ${{ github.run_attempt }}
          GITHUB_REF: ${{ github.ref }}
          JOB_ID: ${{ steps.get-job-id.outputs.job-id }}
          JOB_NAME: ${{ steps.get-job-id.outputs.job-name }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
        run: |
          unzip -o .additional_ci_files/llm_results/mappings.zip -d .additional_ci_files/llm_results || true
          python3 -m pip install boto3==1.35.42
          python3 tools/testing/do_target_determination_for_s3.py

      - name: Upload TD results to s3
        uses: seemethere/upload-artifact-s3@baba72d0712b404f646cebe0730933554ebce96a # v5.1.0
        if: steps.td.outcome == 'success'
        with:
          name: td_results
          retention-days: 14
          if-no-files-found: error
          path: td_results.json

      - name: Store TD results on GHA
        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2
        if: steps.td.outcome == 'success'
        with:
          name: td_results.json
          retention-days: 14
          if-no-files-found: error
          path: td_results.json

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

- **File Documentation**: `target_determination.yml_docs.md`
- **Keyword Index**: `target_determination.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
