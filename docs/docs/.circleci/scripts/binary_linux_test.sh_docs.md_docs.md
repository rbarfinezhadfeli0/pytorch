# Documentation: `docs/.circleci/scripts/binary_linux_test.sh_docs.md`

## File Metadata

- **Path**: `docs/.circleci/scripts/binary_linux_test.sh_docs.md`
- **Size**: 6,141 bytes (6.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `.circleci/scripts/binary_linux_test.sh`

## File Metadata

- **Path**: `.circleci/scripts/binary_linux_test.sh`
- **Size**: 3,827 bytes (3.74 KB)
- **Type**: Shell Script
- **Extension**: `.sh`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```bash
#!/bin/bash

OUTPUT_SCRIPT=${OUTPUT_SCRIPT:-/home/circleci/project/ci_test_script.sh}

# only source if file exists
if [[ -f /home/circleci/project/env ]]; then
  source /home/circleci/project/env
fi
cat >"${OUTPUT_SCRIPT}" <<EOL
# =================== The following code will be executed inside Docker container ===================
set -eux -o pipefail

retry () {
    "\$@"  || (sleep 1 && "\$@") || (sleep 2 && "\$@")
}

# Source binary env file here if exists
if [[ -e "${BINARY_ENV_FILE:-/nofile}" ]]; then
  source "${BINARY_ENV_FILE:-/nofile}"
fi

python_nodot="\$(echo $DESIRED_PYTHON | tr -d m.u)"

# Set up Python
if [[ "$PACKAGE_TYPE" != libtorch ]]; then
  python_path="/opt/python/cp\$python_nodot-cp\${python_nodot}"
  if [[ "\$python_nodot" = *t ]]; then
    python_digits="\$(echo $DESIRED_PYTHON | tr -cd [:digit:])"
    python_path="/opt/python/cp\$python_digits-cp\${python_digits}t"
  fi
  export PATH="\${python_path}/bin:\$PATH"
fi

EXTRA_CONDA_FLAGS=""
NUMPY_PIN=""
PROTOBUF_PACKAGE="defaults::protobuf"

if [[ "\$python_nodot" = *310* ]]; then
  # There's an issue with conda channel priority where it'll randomly pick 1.19 over 1.20
  # we set a lower boundary here just to be safe
  NUMPY_PIN=">=1.21.2"
  PROTOBUF_PACKAGE="protobuf>=3.19.0"
fi

if [[ "\$python_nodot" = *39* ]]; then
  # There's an issue with conda channel priority where it'll randomly pick 1.19 over 1.20
  # we set a lower boundary here just to be safe
  NUMPY_PIN=">=1.20"
fi

# Move debug wheels out of the package dir so they don't get installed
mkdir -p /tmp/debug_final_pkgs
mv /final_pkgs/debug-*.zip /tmp/debug_final_pkgs || echo "no debug packages to move"

# Install the package
# These network calls should not have 'retry's because they are installing
# locally and aren't actually network calls
# Pick only one package of multiple available (which happens as result of workflow re-runs)
pkg="/final_pkgs/\$(ls -1 /final_pkgs|sort|tail -1)"
if [[ "\$PYTORCH_BUILD_VERSION" == *dev* ]]; then
    CHANNEL="nightly"
else
    CHANNEL="test"
fi

if [[ "$PACKAGE_TYPE" != libtorch ]]; then
  if [[ "\$BUILD_ENVIRONMENT" != *s390x* ]]; then
    pip install "\$pkg" --index-url "https://download.pytorch.org/whl/\${CHANNEL}/${DESIRED_CUDA}"
    retry pip install -q numpy protobuf typing-extensions
  else
    pip install "\$pkg"
    retry pip install -q numpy protobuf typing-extensions
  fi
fi
if [[ "$PACKAGE_TYPE" == libtorch ]]; then
  pkg="\$(ls /final_pkgs/*-latest.zip)"
  unzip "\$pkg" -d /tmp
  cd /tmp/libtorch
fi

# Test the package
/pytorch/.ci/pytorch/check_binary.sh

if [[ "\$GPU_ARCH_TYPE" != *s390x* && "\$GPU_ARCH_TYPE" != *xpu* && "\$GPU_ARCH_TYPE" != *rocm*  && "$PACKAGE_TYPE" != libtorch ]]; then

  torch_pkg_size="$(ls -1 /final_pkgs/torch-* | sort |tail -1 |xargs wc -c |cut -d ' ' -f1)"
  # todo: implement check for large binaries
  # if the package is larger than 1.5GB, we disable the pypi check.
  # this package contains all libraries packaged in torch libs folder
  # example of such package is https://download.pytorch.org/whl/cu126_full/torch
  if [[ "\$torch_pkg_size" -gt  1500000000 ]]; then
    python /pytorch/.ci/pytorch/smoke_test/smoke_test.py --package=torchonly --torch-compile-check disabled --pypi-pkg-check disabled
  else
    python /pytorch/.ci/pytorch/smoke_test/smoke_test.py --package=torchonly --torch-compile-check disabled $extra_parameters
  fi

  if [[ "\$GPU_ARCH_TYPE" != *cpu-aarch64* ]]; then
    # https://github.com/pytorch/pytorch/issues/149422
    python /pytorch/.ci/pytorch/smoke_test/check_gomp.py
  fi
fi

# Clean temp files
cd /pytorch/.ci/pytorch/ && git clean -ffdx

# =================== The above code will be executed inside Docker container ===================
EOL
echo
echo
echo "The script that will run in the next step is:"
cat "${OUTPUT_SCRIPT}"

```



## High-Level Overview

This file is part of the PyTorch framework located at `.circleci/scripts`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.circleci/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python .circleci/scripts/binary_linux_test.sh
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.circleci/scripts`):

- [`publish_android_snapshot.sh_docs.md`](./publish_android_snapshot.sh_docs.md)
- [`binary_windows_test.sh_docs.md`](./binary_windows_test.sh_docs.md)
- [`binary_upload.sh_docs.md`](./binary_upload.sh_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`binary_windows_build.sh_docs.md`](./binary_windows_build.sh_docs.md)
- [`binary_populate_env.sh_docs.md`](./binary_populate_env.sh_docs.md)


## Cross-References

- **File Documentation**: `binary_linux_test.sh_docs.md`
- **Keyword Index**: `binary_linux_test.sh_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/.circleci/scripts`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/.circleci/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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
python docs/.circleci/scripts/binary_linux_test.sh_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/.circleci/scripts`):

- [`binary_populate_env.sh_docs.md_docs.md`](./binary_populate_env.sh_docs.md_docs.md)
- [`binary_linux_test.sh_kw.md_docs.md`](./binary_linux_test.sh_kw.md_docs.md)
- [`binary_windows_build.sh_docs.md_docs.md`](./binary_windows_build.sh_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`binary_windows_build.sh_kw.md_docs.md`](./binary_windows_build.sh_kw.md_docs.md)
- [`binary_populate_env.sh_kw.md_docs.md`](./binary_populate_env.sh_kw.md_docs.md)
- [`publish_android_snapshot.sh_kw.md_docs.md`](./publish_android_snapshot.sh_kw.md_docs.md)
- [`binary_upload.sh_kw.md_docs.md`](./binary_upload.sh_kw.md_docs.md)
- [`binary_upload.sh_docs.md_docs.md`](./binary_upload.sh_docs.md_docs.md)


## Cross-References

- **File Documentation**: `binary_linux_test.sh_docs.md_docs.md`
- **Keyword Index**: `binary_linux_test.sh_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
