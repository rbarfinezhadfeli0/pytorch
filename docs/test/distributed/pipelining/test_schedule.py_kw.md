# Keyword Index: `test/distributed/pipelining/test_schedule.py`

## File Information

- **Original File**: [test/distributed/pipelining/test_schedule.py](../../../../test/distributed/pipelining/test_schedule.py)
- **Documentation**: [`test_schedule.py_docs.md`](./test_schedule.py_docs.md)
- **Folder**: `test/distributed/pipelining`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MockPipelineStage`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`ScheduleTest`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`ScheduleUtilTests`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`TestScheduleCsv`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`TestScheduleLowering`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`TestSchedulePlan`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`TestValidateSchedule`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`of`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)

### Functions

- **`__init__`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`_create_grad_recv_info`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`_dump_csv`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`_parse_actions`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`_prepare_backward_infra`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`_prepare_forward_infra`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`loss_fn`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`setUp`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`stage_to_rank`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_action_parse`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_csv`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_csv_compare`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_generate_stage_to_rank_mapping`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_get_schedule_class`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_grad_with_split_b_w`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_grad_with_v_schedule`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_invalid_schedule_missing_action`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_invalid_schedule_missing_rank`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_merge_bw`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_pipeline_order`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_pipeline_order_flex_and_zero_bubble`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_pipeline_order_for_v_schedules`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_reduce_grad`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_schedule_eval_then_train`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_schedule_with_single_stage`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_send_recv`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_unshard_reshard`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_valid_schedule`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`test_zero_bubble_schedule_errors_with_compile`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)

### Imports

- **`FakeStore`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`MultiMLP`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`OptimizedModule`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`_PipelineStageBase`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`copy`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`csv`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`generate_stage_to_rank_mapping`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`logging`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`model_registry`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`os`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`requires_accelerator_dist_backend`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`torch`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`torch._dynamo`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`torch.distributed.pipelining`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`torch.distributed.pipelining._utils`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`torch.distributed.pipelining.schedules`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`torch.distributed.pipelining.stage`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [test_schedule.py_docs.md](./test_schedule.py_docs.md)


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
