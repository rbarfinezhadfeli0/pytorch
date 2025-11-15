# Keyword Index: `torch/profiler/_pattern_matcher.py`

## File Information

- **Original File**: [torch/profiler/_pattern_matcher.py](../../../torch/profiler/_pattern_matcher.py)
- **Documentation**: [`_pattern_matcher.py_docs.md`](./_pattern_matcher.py_docs.md)
- **Folder**: `torch/profiler`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Conv2dBiasFollowedByBatchNorm2dPattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`ExtraCUDACopyPattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`FP32MatMulPattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`ForLoopIndexingPattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`GradNotSetToNonePattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`MatMulDimInFP16Pattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`NamePattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`OptimizerSingleTensorPattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`Pattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`SynchronizedDataLoaderPattern`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`and`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`for`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`this`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`to`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)

### Functions

- **`__init__`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`benchmark`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`benchmark_summary`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`closest_multiple`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`eventTreeTraversal`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`format_time`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`go_up_until`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`input_dtypes`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`input_shapes`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`is_dataloader_function`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`match`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`matched_events`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`mutiple_of`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`next_of`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`prev_of`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`report`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`report_all_anti_patterns`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`root_of`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`same_ops`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`siblings_of`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`skip`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`source_code_location`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`summary`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)

### Imports

- **`Optional`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`index_of_first_match`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`json`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`math`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`os`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`profile`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`re`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`torch`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`torch._C._profiler`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`torch.profiler`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`torch.profiler._utils`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`torch.utils.benchmark`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)
- **`typing`**: [_pattern_matcher.py_docs.md](./_pattern_matcher.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
