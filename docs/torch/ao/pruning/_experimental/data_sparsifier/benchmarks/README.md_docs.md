# Documentation: `torch/ao/pruning/_experimental/data_sparsifier/benchmarks/README.md`

## File Metadata

- **Path**: `torch/ao/pruning/_experimental/data_sparsifier/benchmarks/README.md`
- **Size**: 5,322 bytes (5.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```markdown
# Data Sparsifier Benchmarking using the DLRM Model

## Introduction
The objective of this exercise is to use the data sparsifier to prune the embedding bags of the [DLRM Model](https://github.com/facebookresearch/dlrm) and observe the following -

1. **Disk usage savings**: Savings in model size after pruning.
2. **Model Quality**: How and by how much does performance deteriorate after pruning the embedding bags?
3. **Model forward time**: Can we speed up the model forward time by utilizing the sparsity? Specifically, can we introduce torch.sparse interim to reduce number of computations.

## Scope
The [DataNormSparsifier](https://github.com/pytorch/pytorch/blob/main/torch/ao/pruning/_experimental/data_sparsifier/data_norm_sparsifier.py) is used to sparsify the embeddings of the DLRM model. The model is sparsified for all the combinations of -
1. Sparsity Levels: [0.0, 0.1, 0.2, ... 0.9, 0.91, 0.92, ... 0.99, 1.0]
2. Sparse Block shapes: (1,1) and (1,4)
3. Norm: L1 and L2

## Dataset
The benchmarks are created for the dlrm model on the Kaggle CriteoDataset which can be downloaded from [here](https://ailab.criteo.com/ressources/) or [here](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310/1). <!-- codespell:ignore -->

## Results
1. **Disk Usage**: Introducing sparsity in the embeddings reduces file size after compression. The compressed model size goes down from 1.9 GB to 150 MB after 100% sparsity.

<img src="./images/disk_savings.png" align="center" height="250" width="400" ><img src="./images/accuracy.png" align="right" height="250" width="400" >


2. **Model Quality**: The model accuracy decreases slowly with sparsity levels. Even at 90% sparsity levels, the model accuracy decreases only by 2%.


3. **Model forward time**: Sparse coo tensors are introduced on the features before feeding into the top layer of the dlrm model. Post that, we perform a sparse ```torch.mm``` with the first linear weight of the top layer.
The takeaway is that the dlrm model with sparse coo tensor is slower (roughly 2x). This is because even though the sparsity levels are high in the embedding weights, the interaction step between the dense and sparse features increases the sparsity levels. Hence, creating sparse coo tensor on this not so sparse features actually slows down the model.

<img src="./images/forward_time.png" height="250" width="400" >


## Setup
The benchmark codes depend on the [DLRM codebase](https://github.com/facebookresearch/dlrm).
1. Clone the dlrm git repository
2. Download the dataset from [here](https://ailab.criteo.com/ressources/) or [here](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310/1) <!-- codespell:ignore -->
3. The DLRM model can be trained using the following script
```
# Make sure you go into the file and make sure that the path to dataset is correct.

./bench/dlrm_s_criteo_kaggle.sh --save-model=./models/criteo_model.ckpt [--use-gpu]

# This should also dump kaggleAdDisplayChallenge_processed.npz in the path where data is present
```

4. Copy the scripts data sparsifier benchmark scripts into to the dlrm directory.

## Scripts to run each experiment.

### **Disk savings**
```
python evaluate_disk_savings.py --model-path=<path_to_model_checkpoint> --sparsified-model-dump-path=<path_to_dump_sparsified_models>
```

Running this script should dump
* sparsified model checkpoints: model is sparsified for all the
    combinations of sparsity levels, block shapes and norms and dumped.

* ```sparse_model_metadata.csv```: This contains the compressed file size and path info for all the sparsified models. This file will be used for other experiments


### **Model Quality**
```
python evaluate_model_metrics.py --raw-data-file=<path_to_raw_data_txt_file> --processed-data-file=<path_to_kaggleAdDisplayChallenge_processed.npz> --sparse-model-metadata=<path_to_sparse_model_metadata_csv>
```
Running this script should dump ```sparse_model_metrics.csv``` that contains evaluation metrics for all sparsified models.

### **Model forward time**:
```
python evaluate_forward_time.py --raw-data-file=<path_to_raw_data_txt_file> --processed-data-file=<path_to_kaggleAdDisplayChallenge_processed.npz> --sparse-model-metadata=<path_to_sparse_model_metadata_csv>
```
Running this script should dump ```dlrm_forward_time_info.csv``` that contains forward time for all sparsified models with and without torch.sparse in the forward pass.

## Requirements
pytorch (latest)

scikit-learn

numpy

pandas

## Machine specs to create benchmark
AI AWS was used to run everything i.e. training the dlrm model and running data sparsifier benchmarks.

Machine: AI AWS

Instance Type: p4d.24xlarge

GPU: A100


## Future work
1. **Evaluate memory savings**: The idea is to use torch.sparse tensors to store weights of the embedding bags so that the model memory consumption improves. This will be possible once the embedding bags starts supporting torch.sparse backend.

2. **Sparsifying activations**: Use activation sparsifier to sparsify the activations of the dlrm model. The idea is to sparsify the features before feeding to the top dense layer (sparsify ```z``` [here](https://github.com/facebookresearch/dlrm/blob/11afc52120c5baaf0bfe418c610bc5cccb9c5777/dlrm_s_pytorch.py#L595)).

```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/ao/pruning/_experimental/data_sparsifier/benchmarks`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/pruning/_experimental/data_sparsifier/benchmarks`, which is part of the **core PyTorch library**.



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

Files in the same folder (`torch/ao/pruning/_experimental/data_sparsifier/benchmarks`):

- [`evaluate_model_metrics.py_docs.md`](./evaluate_model_metrics.py_docs.md)
- [`evaluate_disk_savings.py_docs.md`](./evaluate_disk_savings.py_docs.md)
- [`dlrm_utils.py_docs.md`](./dlrm_utils.py_docs.md)
- [`evaluate_forward_time.py_docs.md`](./evaluate_forward_time.py_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md`
- **Keyword Index**: `README.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
