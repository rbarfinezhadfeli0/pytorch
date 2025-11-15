# Documentation: `benchmarks/dynamo/huggingface_llm_models.py`

## File Metadata

- **Path**: `benchmarks/dynamo/huggingface_llm_models.py`
- **Size**: 3,295 bytes (3.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
import subprocess
import sys

import torch


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
except ModuleNotFoundError:
    print("Installing HuggingFace Transformers...")
    pip_install("git+https://github.com/huggingface/transformers.git#egg=transformers")
finally:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )


class Benchmark:
    @staticmethod
    def get_model_and_inputs(model_name, device):
        raise NotImplementedError("get_model_and_inputs() not implemented")


class WhisperBenchmark(Benchmark):
    SAMPLE_RATE = 16000
    DURATION = 30.0  # seconds

    @staticmethod
    def get_model_and_inputs(model_name, device):
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        model.config.forced_decoder_ids = None

        model.generation_config.do_sample = False
        model.generation_config.temperature = 0.0

        num_samples = int(WhisperBenchmark.DURATION * WhisperBenchmark.SAMPLE_RATE)
        audio = torch.randn(num_samples) * 0.1
        inputs = dict(
            processor(
                audio, sampling_rate=WhisperBenchmark.SAMPLE_RATE, return_tensors="pt"
            )
        )
        inputs["input_features"] = inputs["input_features"].to(device)

        decoder_start_token = model.config.decoder_start_token_id
        inputs["decoder_input_ids"] = torch.tensor(
            [[decoder_start_token]], device=device
        )

        return model, inputs


class TextGenerationBenchmark(Benchmark):
    INPUT_LENGTH = 1000
    OUTPUT_LENGTH = 2000

    @staticmethod
    def get_model_and_inputs(model_name, device):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        model.eval()

        model.generation_config.do_sample = False
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = TextGenerationBenchmark.OUTPUT_LENGTH
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.temperature = 0.0

        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(1, TextGenerationBenchmark.INPUT_LENGTH),
            device=device,
            dtype=torch.long,
        )
        example_inputs = {"input_ids": input_ids}

        return model, example_inputs


HF_LLM_MODELS: dict[str, Benchmark] = {
    "meta-llama/Llama-3.2-1B": TextGenerationBenchmark,
    "google/gemma-2-2b": TextGenerationBenchmark,
    "google/gemma-3-4b-it": TextGenerationBenchmark,
    "openai/whisper-tiny": WhisperBenchmark,
    "Qwen/Qwen3-0.6B": TextGenerationBenchmark,
    "mistralai/Mistral-7B-Instruct-v0.3": TextGenerationBenchmark,
    "openai/gpt-oss-20b": TextGenerationBenchmark,
}

```



## High-Level Overview


This Python file contains 3 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Benchmark`, `WhisperBenchmark`, `TextGenerationBenchmark`

**Functions defined**: `pip_install`, `get_model_and_inputs`, `get_model_and_inputs`, `get_model_and_inputs`

**Key imports**: subprocess, sys, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `subprocess`
- `sys`
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Command Execution**: Executes system commands - validate inputs

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

- **File Documentation**: `huggingface_llm_models.py_docs.md`
- **Keyword Index**: `huggingface_llm_models.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
