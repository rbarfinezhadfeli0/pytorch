# Documentation: `benchmarks/functional_autograd_benchmark/README.md`

## File Metadata

- **Path**: `benchmarks/functional_autograd_benchmark/README.md`
- **Size**: 2,603 bytes (2.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```markdown
# Benchmarking tool for the autograd API

This folder contain a set of self-contained scripts that allows you to benchmark autograd with different common models.
It is designed to run the benchmark before and after your change and will generate a table to share on the PR.

To do so, you can use `functional_autograd_benchmark.py` to run the benchmarks before your change (using as output `before.txt`) and after your change (using as output `after.txt`).
You can then use `compare.py` to get a markdown table comparing the two runs.

The default arguments of `functional_autograd_benchmark.py` should be used in general. You can change them though to force a given device or force running even the (very) slow settings.

### Sample usage

```bash
# Make sure you compile pytorch in release mode and with the same flags before/after
export DEBUG=0
# When running on CPU, it might be required to limit the number of cores to avoid oversubscription
export OMP_NUM_THREADS=10

# Compile pytorch with the base revision
git checkout main
python -m pip install --no-build-isolation -v -e .

# Install dependencies:
# Scipy is required by detr
pip install scipy

# Run the benchmark for the base
# This will use the GPU if available.
pushd benchmarks/functional_autograd_benchmark
python functional_autograd_benchmark.py --output before.txt

# Compile pytorch with your change
popd
git checkout your_feature_branch
python -m pip install --no-build-isolation -v -e .

# Run the benchmark for the new version
pushd benchmarks/functional_autograd_benchmark
python functional_autograd_benchmark.py --output after.txt

# Get the markdown table that you can paste in your github PR
python compare.py

popd

```

### Files in this folder:
- `functional_autograd_benchmark.py` is the main entry point to run the benchmark.
- `compare.py` is the entry point to run the comparison script that generates a markdown table.
- `torchaudio_models.py` and `torchvision_models.py`  contains code extracted from torchaudio and torchvision to be able to run the models without having a specific version of these libraries installed.
- `ppl_models.py`, `vision_models.py` and `audio_text_models.py` contain all the getter functions used for the benchmark.


### Benchmarking against `functorch`

```bash
# Install stable functorch:
pip install functorch
# or install from source:
pip install git+https://github.com/pytorch/functorch

# Run the benchmark for the base
# This will use the GPU if available.
pushd benchmarks/functional_autograd_benchmark
python functional_autograd_benchmark.py --output bench-with-functorch.txt
```

```



## High-Level Overview

This file is part of the PyTorch framework located at `benchmarks/functional_autograd_benchmark`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/functional_autograd_benchmark`, which is part of the PyTorch project infrastructure.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/functional_autograd_benchmark`):

- [`ppl_models.py_docs.md`](./ppl_models.py_docs.md)
- [`torchaudio_models.py_docs.md`](./torchaudio_models.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`vision_models.py_docs.md`](./vision_models.py_docs.md)
- [`functional_autograd_benchmark.py_docs.md`](./functional_autograd_benchmark.py_docs.md)
- [`torchvision_models.py_docs.md`](./torchvision_models.py_docs.md)
- [`compare.py_docs.md`](./compare.py_docs.md)
- [`audio_text_models.py_docs.md`](./audio_text_models.py_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md`
- **Keyword Index**: `README.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
