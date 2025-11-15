# Documentation: `benchmarks/dynamo/Makefile`

## File Metadata

- **Path**: `benchmarks/dynamo/Makefile`
- **Size**: 3,085 bytes (3.01 KB)
- **Type**: Source File ()
- **Extension**: ``

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
# Usage:
#   make build-deps TORCHBENCH_MODELS=<model_names>
#   Support install a single torchbench model (e.g., "alexnet"),
#   or multiple torchbench model names (e.g., "alexnet basic_gnn_gcn BERT_pytorch"),
#   or empty (i.e., "") for installing all torchbench models.

clone-deps:
	(cd ../../.. \
		&& (test -e torchvision || git clone --recursive https://github.com/pytorch/vision torchvision) \
		&& (test -e torchdata || git clone --recursive https://github.com/pytorch/data.git torchdata) \
		&& (test -e torchtext || git clone --recursive https://github.com/pytorch/text torchtext) \
		&& (test -e torchaudio || git clone --recursive https://github.com/pytorch/audio torchaudio) \
		&& (test -e detectron2 || git clone --recursive https://github.com/facebookresearch/detectron2) \
		&& (test -e FBGEMM || git clone --recursive https://github.com/pytorch/FBGEMM) \
		&& (test -e torchrec || git clone --recursive https://github.com/pytorch/torchrec) \
		&& (test -e torchbenchmark || git clone --recursive https://github.com/pytorch/benchmark torchbenchmark) \
	)

pull-deps: clone-deps
	(cd ../../../torchvision    && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/vision.txt)" && git submodule update --init --recursive)
	(cd ../../../torchdata      && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/data.txt)" && git submodule update --init --recursive)
	(cd ../../../torchtext      && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/text.txt)" && git submodule update --init --recursive)
	(cd ../../../torchaudio     && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/audio.txt)" && git submodule update --init --recursive)
	(cd ../../../FBGEMM         && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/fbgemm.txt)" && git submodule update --init --recursive)
	(cd ../../../torchrec       && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/torchrec.txt)" && git submodule update --init --recursive)
	(cd ../../../detectron2     && git fetch && git checkout HEAD && git submodule update --init --recursive)
	(cd ../../../torchbenchmark && git fetch && git checkout "$$(cat ../pytorch/.github/ci_commit_pins/torchbench.txt)" && git submodule update --init --recursive)

build-deps: clone-deps
	uv pip install numpy scipy ninja pyyaml six mkl mkl-include setuptools wheel cmake \
		typing-extensions requests protobuf numba cython scikit-learn librosa
	(cd ../../../torchvision && uv pip install -e . --no-build-isolation)
	(cd ../../../torchdata && uv pip install -e .)
	(cd ../../../torchaudio && uv pip install -e . --no-build-isolation)
	(cd ../../../FBGEMM/fbgemm_gpu && uv pip install -r requirements.txt && uv pip install -e . --no-build-isolation)
	(cd ../../../torchrec && uv pip install -e .)
	(cd ../../../detectron2 && uv pip install -e . --no-build-isolation)
	(cd ../../../torchbenchmark && python install.py --continue_on_fail $(if $(TORCHBENCH_MODELS),models $(TORCHBENCH_MODELS)))
	uv pip uninstall torchrec-nightly fbgemm-gpu-nightly

```



## High-Level Overview

This file is part of the PyTorch framework located at `benchmarks/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`benchmarks/dynamo`):

- [`timm_models_list_cpu.txt_docs.md`](./timm_models_list_cpu.txt_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test.py_docs.md`](./test.py_docs.md)
- [`benchmarks.py_docs.md`](./benchmarks.py_docs.md)
- [`check_graph_breaks.py_docs.md`](./check_graph_breaks.py_docs.md)
- [`check_csv.py_docs.md`](./check_csv.py_docs.md)
- [`all_torchbench_models_list.txt_docs.md`](./all_torchbench_models_list.txt_docs.md)
- [`check_accuracy.py_docs.md`](./check_accuracy.py_docs.md)
- [`torchbench_models_list_cpu.txt_docs.md`](./torchbench_models_list_cpu.txt_docs.md)
- [`timm_models.py_docs.md`](./timm_models.py_docs.md)


## Cross-References

- **File Documentation**: `Makefile_docs.md`
- **Keyword Index**: `Makefile_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
