# Keyword Index: `test/distributed/pipelining/test_schedule_multiproc.py`

## File Information

- **Original File**: [test/distributed/pipelining/test_schedule_multiproc.py](../../../../test/distributed/pipelining/test_schedule_multiproc.py)
- **Documentation**: [`test_schedule_multiproc.py_docs.md`](./test_schedule_multiproc.py_docs.md)
- **Folder**: `test/distributed/pipelining`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CustomSchedulesTest`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`CustomState`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`ScheduleTest`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`class`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`from`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)

### Functions

- **`__init__`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`backend_str`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`check_gradients`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`config`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`create_multi_stage_pipeline`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`create_single_stage_pipeline`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`device`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`dw_builder`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`dw_runner`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`forward_callback`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`grad_check`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`overlap_callback`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`run_reference_model`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`setup_models_and_data`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_custom_function_callback`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_eval_inference_mode`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_forward_only`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_grad_with_manual`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_grad_with_manual_interleaved`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_grad_with_tracer`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_kwargs_with_tracer`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_multi_iter`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_non_symmetric_stage_ids`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_pipeline_schedule_runtime_custom_sched`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_return_output`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_schedule_with_native_zero_bubble`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_schedule_with_weight_update_mlp_e2e`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_v_shape_schedules`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`test_zero_bubble_with_model_kwargs`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`zero_gradients`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)

### Imports

- **`MSELoss`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`ModelWithKwargs`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`_PipelineStageBase`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`_wait_batch_p2p`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`copy`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`dataclass`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`dataclasses`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`logging`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`model_registry`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`schedule_registry`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`torch`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`torch.distributed`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`torch.distributed.pipelining`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`torch.distributed.pipelining.schedules`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`torch.distributed.pipelining.stage`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`torch.nn.modules.loss`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_schedule_multiproc.py_docs.md](./test_schedule_multiproc.py_docs.md)


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
