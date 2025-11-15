# Keyword Index: `torch/csrc/distributed/c10d/cuda/cutlass/gemm/kernel/persistent_async_input_scheduler.cuh`

## File Information

- **Original File**: [torch/csrc/distributed/c10d/cuda/cutlass/gemm/kernel/persistent_async_input_scheduler.cuh](../../../../../../../../../torch/csrc/distributed/c10d/cuda/cutlass/gemm/kernel/persistent_async_input_scheduler.cuh)
- **Documentation**: [`persistent_async_input_scheduler.cuh_docs.md`](./persistent_async_input_scheduler.cuh_docs.md)
- **Folder**: `torch/csrc/distributed/c10d/cuda/cutlass/gemm/kernel`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Arguments`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`AtomThrShape`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`BlockShape`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`CLCResponse`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`ClusterShape`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`ElementAccumulator`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`FrgTensorC`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`KernelSchedule`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`PersistentAsyncInputScheduler`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`PersistentTileSchedulerSm90AsyncInput`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`PersistentTileSchedulerSm90AsyncInputParams`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`ProblemShape`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`ProblemShapeMNKL`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`ProblemShapeType`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`Shape`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`SharedStorage`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`TileSchedulerPipeline`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`TileSchedulerPipelineState`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`TileSchedulerSelector`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`TileShape`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`WorkTileInfo`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`for`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)

### Functions

- **`PersistentTileSchedulerSm90AsyncInput`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`advance_to_next_work`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`can_implement`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`compute_epilogue`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`continue_current_work`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`fetch_next_work`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`fixup`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`get_current_work`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`get_current_work_for_linear_idx`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`get_grid_shape`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`get_k_tile_iterator`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`get_linear_idx_from_m_and_n`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`get_tiled_cta_shape_mnl`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`get_work_k_tile_count`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`get_work_k_tile_start`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`get_workspace_size`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`initial_work_tile_info`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`initialize_workspace`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`invalid_work_tile`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`is_final_split`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`is_last_tile`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`is_valid`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`is_work_tile_for_reduction`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`need_separate_reduction`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`pipeline`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`reduction_subtile_idx`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`requires_separate_reduction`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`separate_reduction`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`share`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`throttle_pipeline`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`to_underlying_arguments`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`valid_warpgroup_in_work_tile`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`wait_signal`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`work_tile_to_cluster_coord_mnkl`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)
- **`work_tile_to_cta_coord`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)

### Includes

- **`cutlass/gemm/kernel/static_tile_scheduler.hpp`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)

### Namespaces

- **`cutlass`**: [persistent_async_input_scheduler.cuh_docs.md](./persistent_async_input_scheduler.cuh_docs.md)


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
