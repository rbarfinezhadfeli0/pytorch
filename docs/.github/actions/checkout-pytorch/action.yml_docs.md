# Documentation: `.github/actions/checkout-pytorch/action.yml`

## File Metadata

- **Path**: `.github/actions/checkout-pytorch/action.yml`
- **Size**: 3,502 bytes (3.42 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
name: Checkout PyTorch

description: Clean workspace and check out PyTorch

inputs:
  no-sudo:
    description: If set to any value, don't use sudo to clean the workspace
    required: false
  submodules:
    description: Works as stated in actions/checkout, but the default value is recursive
    required: false
    default: recursive
  fetch-depth:
    description: Works as stated in actions/checkout, but the default value is 0
    required: false
    default: "0"

runs:
  using: composite
  steps:
    - name: Check if in a container runner
      shell: bash
      id: check_container_runner
      run: echo "IN_CONTAINER_RUNNER=$(if [ -f /.inarc ] || [ -f /.incontainer ]; then echo true ; else echo false; fi)" >> "$GITHUB_OUTPUT"

    - name: Set up parallel fetch and clean workspace
      id: first-clean
      continue-on-error: true
      shell: bash
      if: ${{ steps.check_container_runner.outputs.IN_CONTAINER_RUNNER == 'false' }}
      env:
        NO_SUDO: ${{ inputs.no-sudo }}
      run: |
        # Use all available CPUs for fetching
        cd "${GITHUB_WORKSPACE}"
        git config --global fetch.parallel 0
        git config --global submodule.fetchJobs 0

        # Clean workspace. The default checkout action should also do this, but
        # do it here as well just in case
        if [[ -d .git ]]; then
          if [ -z "${NO_SUDO}" ]; then
            sudo git clean -ffdx
          else
            git clean -ffdx
          fi
        fi

    - name: Checkout PyTorch
      id: first-checkout-attempt
      continue-on-error: true
      uses: actions/checkout@v4
      with:
        ref: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha }}
        # --depth=1 for speed, manually fetch history and other refs as necessary
        fetch-depth: ${{ inputs.fetch-depth }}
        submodules: ${{ inputs.submodules }}
        show-progress: false

    - name: Clean submodules post checkout
      id: clean-submodules
      if: ${{ steps.check_container_runner.outputs.IN_CONTAINER_RUNNER == 'false' }}
      shell: bash
      env:
        NO_SUDO: ${{ inputs.no-sudo }}
      run: |
        cd "${GITHUB_WORKSPACE}"
        # Clean stale submodule dirs
        if [ -z "${NO_SUDO}" ]; then
          sudo git submodule foreach --recursive git clean -ffdx
        else
          git submodule foreach --recursive git clean -ffdx
        fi

    - name: Clean workspace (try again)
      if: ${{ steps.check_container_runner.outputs.IN_CONTAINER_RUNNER == 'false' &&
        (steps.first-clean.outcome != 'success' || steps.first-checkout-attempt.outcome != 'success') }}
      shell: bash
      env:
        NO_SUDO: ${{ inputs.no-sudo }}
      run: |
        retry () {
          $* || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
        }
        echo "${GITHUB_WORKSPACE}"
        if [ -z "${NO_SUDO}" ]; then
          retry sudo rm -rf "${GITHUB_WORKSPACE}"
        else
          retry rm -rf "${GITHUB_WORKSPACE}"
        fi
        mkdir "${GITHUB_WORKSPACE}"

    - name: Checkout PyTorch (try again)
      uses: actions/checkout@v4
      if: ${{ steps.first-clean.outcome != 'success' || steps.first-checkout-attempt.outcome != 'success' }}
      with:
        ref: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha }}
        fetch-depth: ${{ inputs.fetch-depth }}
        submodules: ${{ inputs.submodules }}
        show-progress: false

```



## High-Level Overview

This file is part of the PyTorch framework located at `.github/actions/checkout-pytorch`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/actions/checkout-pytorch`, which is part of the **core PyTorch library**.



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

Files in the same folder (`.github/actions/checkout-pytorch`):



## Cross-References

- **File Documentation**: `action.yml_docs.md`
- **Keyword Index**: `action.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
