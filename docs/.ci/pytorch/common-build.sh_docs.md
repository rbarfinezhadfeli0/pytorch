# Documentation: `.ci/pytorch/common-build.sh`

## File Metadata

- **Path**: `.ci/pytorch/common-build.sh`
- **Size**: 3,214 bytes (3.14 KB)
- **Type**: Shell Script
- **Extension**: `.sh`

## File Purpose

This is a shell script that is part of the PyTorch project.

## Original Source

```bash
#!/bin/bash
# Required environment variables:
#   $BUILD_ENVIRONMENT (should be set by your Docker image)

if [[ "$BUILD_ENVIRONMENT" != *win-* ]]; then
    # Save the absolute path in case later we chdir (as occurs in the gpu perf test)
    script_dir="$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )"

    if [[ "${BUILD_ENVIRONMENT}" == *-pch* ]]; then
        # This is really weird, but newer sccache somehow produces broken binary
        # see https://github.com/pytorch/pytorch/issues/139188
        sudo mv /opt/cache/bin/sccache-0.2.14a /opt/cache/bin/sccache
    fi

    if which sccache > /dev/null; then
        # Clear SCCACHE_BUCKET and SCCACHE_REGION if they are empty, otherwise
        # sccache will complain about invalid bucket configuration
        if [[ -z "${SCCACHE_BUCKET:-}" ]]; then
          unset SCCACHE_BUCKET
          unset SCCACHE_REGION
        fi

        # Save sccache logs to file
        sccache --stop-server > /dev/null  2>&1 || true
        rm -f ~/sccache_error.log || true

        function sccache_epilogue() {
            echo "::group::Sccache Compilation Log"
            echo '=================== sccache compilation log ==================='
            python "$script_dir/print_sccache_log.py" ~/sccache_error.log 2>/dev/null || true
            echo '=========== If your build fails, please take a look at the log above for possible reasons ==========='
            sccache --show-stats
            sccache --stop-server || true
            echo "::endgroup::"
        }

        # Register the function here so that the error log can be printed even when
        # sccache fails to start, i.e. timeout error
        trap_add sccache_epilogue EXIT

        if [[ -n "${SKIP_SCCACHE_INITIALIZATION:-}" ]]; then
            # sccache --start-server seems to hang forever on self hosted runners for GHA
            # so let's just go ahead and skip the --start-server altogether since it seems
            # as though sccache still gets used even when the sscache server isn't started
            # explicitly
            echo "Skipping sccache server initialization, setting environment variables"
            export SCCACHE_IDLE_TIMEOUT=0
            export SCCACHE_ERROR_LOG=~/sccache_error.log
            export RUST_LOG=sccache::server=error
        elif [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
            SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=0 sccache --start-server
        else
            # increasing SCCACHE_IDLE_TIMEOUT so that extension_backend_test.cpp can build after this PR:
            # https://github.com/pytorch/pytorch/pull/16645
            SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=0 RUST_LOG=sccache::server=error sccache --start-server
        fi

        # Report sccache stats for easier debugging. It's ok if this commands
        # timeouts and fails on MacOS
        sccache --zero-stats || true
    fi

    if which ccache > /dev/null; then
        # Report ccache stats for easier debugging
        ccache --zero-stats
        ccache --show-stats
        function ccache_epilogue() {
            ccache --show-stats
        }
        trap_add ccache_epilogue EXIT
    fi
fi

```



## High-Level Overview

This file is part of the PyTorch framework located at `.ci/pytorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/pytorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`.ci/pytorch`):

- [`codegen-test.sh_docs.md`](./codegen-test.sh_docs.md)
- [`common_utils.sh_docs.md`](./common_utils.sh_docs.md)
- [`python_doc_push_script.sh_docs.md`](./python_doc_push_script.sh_docs.md)
- [`build.sh_docs.md`](./build.sh_docs.md)
- [`install_cache_xla.sh_docs.md`](./install_cache_xla.sh_docs.md)
- [`docs-test.sh_docs.md`](./docs-test.sh_docs.md)
- [`multigpu-test.sh_docs.md`](./multigpu-test.sh_docs.md)
- [`docker-build-test.sh_docs.md`](./docker-build-test.sh_docs.md)
- [`macos-build-test.sh_docs.md`](./macos-build-test.sh_docs.md)


## Cross-References

- **File Documentation**: `common-build.sh_docs.md`
- **Keyword Index**: `common-build.sh_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
