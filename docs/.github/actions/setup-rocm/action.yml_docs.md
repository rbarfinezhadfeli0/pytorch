# Documentation: `.github/actions/setup-rocm/action.yml`

## File Metadata

- **Path**: `.github/actions/setup-rocm/action.yml`
- **Size**: 4,843 bytes (4.73 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: Setup ROCm host

description: Set up ROCm host for CI

runs:
  using: composite
  steps:
    - name: Runner ROCm version
      if: always()
      shell: bash
      run: |
        dpkg -l | grep -E "  rocm"

    - name: Stop all running docker containers
      if: always()
      shell: bash
      run: |
        # ignore expansion of "docker ps -q" since it could be empty
        # shellcheck disable=SC2046
        docker stop $(docker ps -q) || true
        # Prune all stopped containers.
        docker container prune -f

    - name: Runner health check system info
      if: always()
      shell: bash
      run: |
        cat /etc/os-release || true
        cat /etc/apt/sources.list.d/rocm.list || true
        cat /opt/rocm/.info/version || true
        whoami

    - name: Runner health check amdgpu info
      if: always()
      shell: bash
      run: |
        dpkg -l | grep -E "  amdgpu"

    - name: Runner health check rocm-smi
      if: always()
      shell: bash
      run: |
        rocm-smi

    - name: Runner health check rocminfo
      if: always()
      shell: bash
      run: |
        rocminfo

    - name: Runner health check GPU count
      if: always()
      shell: bash
      run: |
        ngpu=$(rocminfo | grep -c -E 'Name:.*\sgfx')
        msg="Please file an issue on pytorch/pytorch reporting the faulty runner. Include a link to the runner logs so the runner can be identified"
        if [[ $ngpu -eq 0 ]]; then
            echo "Error: Failed to detect any GPUs on the runner"
            echo "$msg"
            exit 1
        fi

    - name: Runner diskspace health check
      uses: pytorch/pytorch/.github/actions/diskspace-cleanup@main
      if: always()

    - name: Runner health check disconnect on failure
      if: ${{ failure() }}
      shell: bash
      run: |
        killall runsvc.sh

    - name: Setup useful environment variables
      shell: bash
      run: |
        RUNNER_ARTIFACT_DIR="${RUNNER_TEMP}/artifacts"
        rm -rf "${RUNNER_ARTIFACT_DIR}"
        mkdir -p "${RUNNER_ARTIFACT_DIR}"
        echo "RUNNER_ARTIFACT_DIR=${RUNNER_ARTIFACT_DIR}" >> "${GITHUB_ENV}"

        RUNNER_TEST_RESULTS_DIR="${RUNNER_TEMP}/test-results"
        rm -rf "${RUNNER_TEST_RESULTS_DIR}"
        mkdir -p "${RUNNER_TEST_RESULTS_DIR}"
        echo "RUNNER_TEST_RESULTS_DIR=${RUNNER_TEST_RESULTS_DIR}" >> "${GITHUB_ENV}"

        RUNNER_DOCS_DIR="${RUNNER_TEMP}/docs"
        rm -rf "${RUNNER_DOCS_DIR}"
        mkdir -p "${RUNNER_DOCS_DIR}"
        echo "RUNNER_DOCS_DIR=${RUNNER_DOCS_DIR}" >> "${GITHUB_ENV}"

    - name: Preserve github env variables for use in docker
      shell: bash
      run: |
        env | grep '^GITHUB' >> "${RUNNER_TEMP}/github_env_${GITHUB_RUN_ID}"
        env | grep '^CI' >> "${RUNNER_TEMP}/github_env_${GITHUB_RUN_ID}"

    - name: ROCm set GPU_FLAG
      shell: bash
      run: |
        # All GPUs are visible to the runner; visibility, if needed, will be set by run_test.py.
        # Add render group for container creation.
        render_gid=`cat /etc/group | grep render | cut -d: -f3`
        # Ensure GPU isolation if pod is part of kubernetes setup with DEVICE_FLAG.
        if [ -f "/etc/podinfo/gha-render-devices" ]; then
          DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
        else
          DEVICE_FLAG="--device /dev/dri"
        fi
        # The --group-add daemon and --group-add bin are needed in the Ubuntu 24.04 and Almalinux OSs respectively.
        # This is due to the device files (/dev/kfd & /dev/dri) being owned by video group on bare metal.
        # This video group ID maps to subgid 1 inside the docker image due to the /etc/subgid entries.
        # The group name corresponding to group ID 1 can change depending on the OS, so both are necessary.
        echo "GPU_FLAG=--device=/dev/mem --device=/dev/kfd $DEVICE_FLAG --group-add video --group-add $render_gid --group-add daemon --group-add bin --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --network=host" >> "${GITHUB_ENV}"

    - name: configure aws credentials
      id: aws_creds
      uses: aws-actions/configure-aws-credentials@ececac1a45f3b08a01d2dd070d28d111c5fe6722 # v4.1.0
      with:
        role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_s3_and_ecr_read_only
        aws-region: us-east-1
        role-duration-seconds: 18000

    - name: Login to Amazon ECR
      id: login-ecr
      continue-on-error: true
      uses: aws-actions/amazon-ecr-login@062b18b96a7aff071d4dc91bc00c4c1a7945b076 # v2.0.1

    - name: Preserve github env variables for use in docker
      shell: bash
      run: |
        env | grep '^GITHUB' >> "${RUNNER_TEMP}/github_env_${GITHUB_RUN_ID}"
        env | grep '^CI' >> "${RUNNER_TEMP}/github_env_${GITHUB_RUN_ID}"
        env | grep '^RUNNER' >> "${RUNNER_TEMP}/github_env_${GITHUB_RUN_ID}"

```



## High-Level Overview

This file is part of the PyTorch framework located at `.github/actions/setup-rocm`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/actions/setup-rocm`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`.github/actions/setup-rocm`):



## Cross-References

- **File Documentation**: `action.yml_docs.md`
- **Keyword Index**: `action.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
