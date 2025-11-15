# Documentation: `benchmarks/dynamo/training_loss.py`

## File Metadata

- **Path**: `benchmarks/dynamo/training_loss.py`
- **Size**: 6,479 bytes (6.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import inspect
import os
import sys
import time
from datetime import timedelta

from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
import torch._dynamo
from torch.utils.data import DataLoader


torch.backends.cuda.matmul.allow_tf32 = True

# You will download around 84G dataset if you run this end to end training/evaluation example.

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def data_processing(num_samples, batch_size):
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].select(range(num_samples))
    small_eval_dataset = tokenized_datasets["test"].select(range(num_samples))

    train_dataloader = DataLoader(small_train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader


def training_iter_fn(batch, model, optimizer):
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def model_training_evaluation(
    backend, train_dataloader, eval_dataloader, model, optimizer, num_epochs, evaluation
):
    model.to(device)
    model.train()
    loss_history = []
    if not backend:
        # Run with native Pytorch
        opt_training_iter_fn = training_iter_fn
    else:
        # Support backends: eager, aot_eager, aot_nvfuser and inductor
        opt_training_iter_fn = torch._dynamo.optimize(backend)(training_iter_fn)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader, 0):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = opt_training_iter_fn(batch, model, optimizer)
            running_loss += loss.item()
            if i % 100 == 99:
                loss_history.append(running_loss / 100)
                running_loss = 0.0

    if evaluation:
        metric = load_metric("accuracy")
        model.eval()
        if not backend:
            opt_model = model
        else:
            opt_model = torch._dynamo.optimize(backend)(model)
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = opt_model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        return loss_history, metric.compute()
    else:
        return loss_history, None


def check_loss(ref_loss, res_loss):
    assert len(ref_loss) == len(res_loss)
    length = len(ref_loss)
    x = min(length, 10)
    return sum(res_loss[-x:]) / 10 <= sum(ref_loss[-x:]) / 10 + 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="TorchDynamo end to end training/evaluation benchmark"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="number of samples to train/eval (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--backend",
        choices=torch._dynamo.list_backends(exclude_tags=None),
        default="inductor",
        help="train/evaluate model with a given backend (default: inductor)",
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        help="train model using a given optimizer (default: Adam)",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="running evaluation after model training",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_dataloader, eval_dataloader = data_processing(
        args.num_samples, args.batch_size
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    optimizer_cls = getattr(sys.modules["torch.optim"], args.optimizer)
    if "capturable" in inspect.signature(optimizer_cls).parameters.keys():
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, capturable=True)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    native_start = time.time()
    ref_loss, accuracy = model_training_evaluation(
        None,
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        args.epochs,
        args.evaluation,
    )
    native_end = time.time()
    res_loss, accuracy = model_training_evaluation(
        args.backend,
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        args.epochs,
        args.evaluation,
    )
    dynamo_end = time.time()
    if check_loss(ref_loss, res_loss):
        print(
            "[PASSED] TorchDynamo end to end training loss is less than or equal to native PyTorch"
        )
    else:
        print(
            "[FAILED] TorchDynamo end to end training loss is greater than native Pytorch"
        )
    if args.evaluation:
        print(f"Model accuracy: {accuracy}")
    native_elapsed = native_end - native_start
    dynamo_elapsed = dynamo_end - native_end
    print(
        f"Train model on {args.epochs} epochs with backend {args.backend} and optimizer {args.optimizer}:"
    )
    print(f"PyTorch spent {timedelta(seconds=native_elapsed / args.epochs)} per epoch")
    print(
        f"TorchDynamo spent {timedelta(seconds=dynamo_elapsed / args.epochs)} per epoch"
    )


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `data_processing`, `tokenize_function`, `training_iter_fn`, `model_training_evaluation`, `check_loss`, `parse_args`, `main`

**Key imports**: argparse, inspect, os, sys, time, timedelta, load_dataset, load_metric, AutoModelForSequenceClassification, AutoTokenizer, torch, torch._dynamo


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `inspect`
- `os`
- `sys`
- `time`
- `datetime`: timedelta
- `datasets`: load_dataset, load_metric
- `transformers`: AutoModelForSequenceClassification, AutoTokenizer
- `torch`
- `torch._dynamo`
- `torch.utils.data`: DataLoader


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

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

- **File Documentation**: `training_loss.py_docs.md`
- **Keyword Index**: `training_loss.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
