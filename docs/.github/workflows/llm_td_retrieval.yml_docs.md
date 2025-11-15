# Documentation: `.github/workflows/llm_td_retrieval.yml`

## File Metadata

- **Path**: `.github/workflows/llm_td_retrieval.yml`
- **Size**: 3,976 bytes (3.88 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: Retrieval PyTorch Tests for Target Determination

on:
  workflow_call:

permissions:
  id-token: write
  contents: read

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

  llm-retrieval:
    # Don't run on forked repos
    if: github.repository_owner == 'pytorch'
    runs-on: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge"
    continue-on-error: true
    needs: get-label-type
    steps:
      - name: Clone PyTorch
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          repository: pytorch/pytorch
          fetch-depth: 0
          path: pytorch

      - name: Setup Linux
        uses: ./pytorch/.github/actions/setup-linux

      - name: Clone CodeLlama
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          repository: osalpekar/codellama
          ref: main
          path: codellama

      - name: Clone Target Determination Code
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          repository: osalpekar/llm-target-determinator
          ref: v0.0.2
          path: llm-target-determinator

      - name: Install requirements
        shell: bash
        run: |
          set -euxo pipefail
          python3 -m pip install -r llm-target-determinator/requirements.txt
          cd "${GITHUB_WORKSPACE}/codellama"
          python3 -m pip install -e .

      - name: Fetch CodeLlama Checkpoint
        shell: bash
        run: |
          set -euxo pipefail
          cd "${GITHUB_WORKSPACE}/codellama"
          mkdir "CodeLlama-7b-Python"
          aws s3 cp "s3://target-determinator-assets/CodeLlama-7b-Python" "CodeLlama-7b-Python" --recursive --no-progress

      - name: Fetch indexes
        uses: nick-fields/retry@7152eba30c6575329ac0576536151aca5a72780e # v3.0.0
        with:
          max_attempts: 3
          retry_wait_seconds: 10
          timeout_minutes: 5
          shell: bash
          command: |
            set -euxo pipefail
            python3 -m pip install awscli==1.29.40
            cd "${GITHUB_WORKSPACE}"/llm-target-determinator/assets
            aws s3 cp "s3://target-determinator-assets/indexes/latest" . --recursive

            unzip -o indexer-files\*.zip
            rm indexer-files*.zip

      - name: Run Retriever
        id: run_retriever
        continue-on-error: true  # ghstack not currently supported due to problems getting git diff
        shell: bash
        run: |
          set -euxo pipefail
          cd "${GITHUB_WORKSPACE}"/llm-target-determinator
          export PATH="$HOME/.local/bin:$PATH"
          torchrun \
            --standalone \
            --nnodes=1 \
            --nproc-per-node=1 \
            retriever.py \
            --experiment-name indexer-files \
            --pr-parse-format GITDIFF
          cd assets
          zip -r mappings.zip mappings

      - name: Upload results to s3
        uses: seemethere/upload-artifact-s3@baba72d0712b404f646cebe0730933554ebce96a # v5.1.0
        if: ${{ steps.run_retriever.outcome == 'success' }}
        with:
          name: llm_results
          retention-days: 14
          if-no-files-found: warn
          path: llm-target-determinator/assets/mappings.zip
        env:
          AWS_ACCESS_KEY_ID: ""
          AWS_SECRET_ACCESS_KEY: ""
          AWS_SESSION_TOKEN: ""
          AWS_DEFAULT_REGION: ""
          AWS_REGION: ""

      - name: Teardown Linux
        uses: pytorch/test-infra/.github/actions/teardown-linux@main
        if: always()

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

- **File Documentation**: `llm_td_retrieval.yml_docs.md`
- **Keyword Index**: `llm_td_retrieval.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
