# Documentation: `.github/workflows/vllm.yml`

## File Metadata

- **Path**: `.github/workflows/vllm.yml`
- **Size**: 4,093 bytes (4.00 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: vllm-test

on:
  push:
    branches:
      - main
      - release/*
    tags:
      - ciflow/vllm/*
  workflow_dispatch:
  schedule:
    - cron: '0 */8 * * *'  # every 8 hours at minute 0 (UTC)

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref_name }}-${{ github.ref_type == 'branch' && github.sha }}-${{ github.event_name == 'workflow_dispatch' }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read

jobs:
  get-label-type:
    name: get-label-type
    uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main
    if: ${{ (github.event_name != 'schedule' || github.repository == 'pytorch/pytorch') && github.repository_owner == 'pytorch' }}
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}
      curr_ref_type: ${{ github.ref_type }}
      opt_out_experiments: lf

  torch-build:
    name: ci-vllm-test
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      # When building vLLM, uv doesn't like that we rename wheel without changing the wheel metadata
      allow-reuse-old-whl: false
      build-additional-packages: "vision audio"
      build-external-packages: "vllm"
      build-environment: linux-jammy-cuda12.8-py3.12-gcc11
      docker-image-name: ci-image:pytorch-linux-jammy-cuda12.8-cudnn9-py3.12-gcc11-vllm
      cuda-arch-list: '8.0 8.9 9.0'
      runner: linux.24xlarge.memory
      test-matrix: |
        { include: [
          { config: "vllm_basic_correctness_test", shard: 1, num_shards: 1, runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_basic_models_test", shard: 1, num_shards: 1, runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_entrypoints_test", shard: 1, num_shards: 1,runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_regression_test", shard: 1, num_shards: 1, runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_multi_model_processor_test", shard: 1, num_shards: 1, runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_pytorch_compilation_unit_tests", shard: 1, num_shards: 1, runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_lora_28_failure_test", shard: 1, num_shards: 1, runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_multi_model_test_28_failure_test", shard: 1, num_shards: 1, runner: "linux.g6.4xlarge.experimental.nvidia.gpu"},
          { config: "vllm_language_model_test_extended_generation_28_failure_test", shard: 1, num_shards: 1, runner: "linux.g6.4xlarge.experimental.nvidia.gpu"},
          { config: "vllm_distributed_test_2_gpu_28_failure_test", shard: 1, num_shards: 1, runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_lora_test", shard: 0, num_shards: 4, runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_lora_test", shard: 1, num_shards: 4, runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_lora_test", shard: 2, num_shards: 4,  runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_lora_test", shard: 3, num_shards: 4,  runner: "linux.g6.4xlarge.experimental.nvidia.gpu" },
          { config: "vllm_lora_tp_test_distributed", shard: 1, num_shards: 1, runner: "linux.g6.12xlarge.nvidia.gpu"},
          { config: "vllm_distributed_test_28_failure_test", shard: 1, num_shards: 1, runner: "linux.g6.12xlarge.nvidia.gpu"}
        ]}
    secrets: inherit

  vllm-test-sm89:
      name: ci-vllm-test
      uses: ./.github/workflows/_linux-test.yml
      needs: [
        torch-build,
      ]
      with:
        build-environment: linux-jammy-cuda12.8-py3.12-gcc11
        docker-image: ${{ needs.torch-build.outputs.docker-image }}
        test-matrix: ${{ needs.torch-build.outputs.test-matrix }}
      secrets: inherit

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

- **File Documentation**: `vllm.yml_docs.md`
- **Keyword Index**: `vllm.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
