# Keyword Index: `torch/csrc/profiler/collection.cpp`

## File Information

- **Original File**: [torch/csrc/profiler/collection.cpp](../../../../torch/csrc/profiler/collection.cpp)
- **Documentation**: [`collection.cpp_docs.md`](./collection.cpp_docs.md)
- **Folder**: `torch/csrc/profiler`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`ResultGreater`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`StealOrDefault`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`SubQueueThreadCache`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`TagToIOType`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`TransferEvents`**: [collection.cpp_docs.md](./collection.cpp_docs.md)

### Functions

- **`addKinetoEvents`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`adjust_durations_dfs`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`adjust_timestamps`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`adjust_timestamps_dfs`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`allTagsMapped`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`build_tree`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`extractEventsFromTrace`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`extractIndex`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`generateForwardBackwardLink`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`generateForwardBackwardLinks`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`getForwardThreadKey`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`get_cuda_sync_enabled`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`get_fwd_bwd_enabled`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`get_record_concrete_inputs_enabled`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`get_record_tensor_addrs_enabled`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`if`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`isHiddenEvent`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`kinetoEventCorrelationID`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`mark_finished`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`passEventsToKineto`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`reassociate`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`scopeToType`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`setKinetoTID`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`setParents`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`set_cuda_sync_enabled_val`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`set_fwd_bwd_enabled_val`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`set_in_tree_building`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`set_record_concrete_inputs_enabled_val`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`set_record_tensor_addrs_enabled_val`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`tagToIOType`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`toString`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`torchOpEndNS`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`while`**: [collection.cpp_docs.md](./collection.cpp_docs.md)

### Includes

- **`ATen/Context.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`ATen/record_function.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`algorithm`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`c10/util/Exception.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`c10/util/flat_hash_map.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`c10/util/overloaded.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`fmt/format.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`functional`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`libkineto.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`limits`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`memory`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`queue`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`torch/csrc/jit/runtime/interpreter.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`torch/csrc/profiler/collection.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`torch/csrc/profiler/data_flow.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`torch/csrc/profiler/kineto_shim.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`torch/csrc/profiler/orchestration/vulkan.h`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`type_traits`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`utility`**: [collection.cpp_docs.md](./collection.cpp_docs.md)

### Namespaces

- **`bool`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`static`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`std`**: [collection.cpp_docs.md](./collection.cpp_docs.md)
- **`torch`**: [collection.cpp_docs.md](./collection.cpp_docs.md)


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
