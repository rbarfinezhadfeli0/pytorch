# Documentation: `benchmarks/dynamo/check_graph_breaks.py`

## File Metadata

- **Path**: `benchmarks/dynamo/check_graph_breaks.py`
- **Size**: 4,084 bytes (3.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import os
import sys
import textwrap

import pandas as pd


# Hack to have something similar to DISABLED_TEST. These models are flaky.

flaky_models = {
    "yolov3",
    "detectron2_maskrcnn_r_101_c4",
    "XGLMForCausalLM",  # discovered in https://github.com/pytorch/pytorch/pull/128148
    "detectron2_fcos_r_50_fpn",
}


def get_field(csv, model_name: str, field: str):
    try:
        return csv.loc[csv["name"] == model_name][field].item()
    except Exception:
        return None


def check_graph_breaks(actual_csv, expected_csv, expected_filename):
    failed = []
    improved = []

    if "rocm" in expected_filename:
        flaky_models.update(
            {
                "alexnet",
                "demucs",
                "densenet121",
                "detectron2_fcos_r_50_fpn",
                "doctr_det_predictor",
                "doctr_reco_predictor",
                "levit_128",
                "llava",
                "microbench_unbacked_tolist_sum",
                "resnet50",
                "resnet152",
                "sam",
                "sam_fast",
                "stable_diffusion_text_encoder",
                "stable_diffusion_unet",
                "timm_efficientdet",
                "torchrec_dlrm",
                "vgg16",
                # LLM
                "meta-llama/Llama-3.2-1B",
                "google/gemma-2-2b",
                "google/gemma-3-4b-it",
                "openai/whisper-tiny",
                "Qwen/Qwen3-0.6B",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "openai/gpt-oss-20b",
            }
        )

    for model in actual_csv["name"]:
        graph_breaks = get_field(actual_csv, model, "graph_breaks")
        expected_graph_breaks = get_field(expected_csv, model, "graph_breaks")
        flaky = model in flaky_models

        if expected_graph_breaks is None:
            status = "MISSING:"
            improved.append(model)
        elif graph_breaks == expected_graph_breaks:
            status = "PASS_BUT_FLAKY" if flaky else "PASS"
            print(f"{model:34}  {status}")
            continue
        elif graph_breaks > expected_graph_breaks:
            if flaky:
                status = "FAIL_BUT_FLAKY:"
            else:
                status = "FAIL:"
                failed.append(model)
        elif graph_breaks < expected_graph_breaks:
            if flaky:
                status = "IMPROVED_BUT_FLAKY:"
            else:
                status = "IMPROVED:"
                improved.append(model)
        print(
            f"{model:34}  {status:19} graph_breaks={graph_breaks}, expected={expected_graph_breaks}"
        )

    msg = ""
    if failed or improved:
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have new dynamo graph breaks:
                {" ".join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have fixed dynamo graph breaks:
                {" ".join(improved)}

            """
            )
        sha = os.getenv("SHA1", "{your CI commit sha}")
        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        from pytorch/pytorch root, run
        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`
        and then `git add` the resulting local changes to expected CSVs to your commit.
        """
        )
    return failed or improved, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    args = parser.parse_args()

    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)

    failed, msg = check_graph_breaks(actual, expected, args.expected)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_field`, `check_graph_breaks`, `main`

**Key imports**: argparse, os, sys, textwrap, pandas as pd


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `os`
- `sys`
- `textwrap`
- `pandas as pd`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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
- [`check_csv.py_docs.md`](./check_csv.py_docs.md)
- [`all_torchbench_models_list.txt_docs.md`](./all_torchbench_models_list.txt_docs.md)
- [`check_accuracy.py_docs.md`](./check_accuracy.py_docs.md)
- [`torchbench_models_list_cpu.txt_docs.md`](./torchbench_models_list_cpu.txt_docs.md)
- [`timm_models.py_docs.md`](./timm_models.py_docs.md)


## Cross-References

- **File Documentation**: `check_graph_breaks.py_docs.md`
- **Keyword Index**: `check_graph_breaks.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
