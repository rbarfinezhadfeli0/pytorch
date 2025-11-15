# Keyword Index: `torch/csrc/jit/tensorexpr/stmt.h`

## File Information

- **Original File**: [torch/csrc/jit/tensorexpr/stmt.h](../../../../../torch/csrc/jit/tensorexpr/stmt.h)
- **Documentation**: [`stmt.h_docs.md`](./stmt.h_docs.md)
- **Folder**: `torch/csrc/jit/tensorexpr`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Op`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`StmtNode`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`TORCH_API`**: [stmt.h_docs.md](./stmt.h_docs.md)

### Functions

- **`Block`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`ToString`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`append_stmt`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`back`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`base_handle`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`begin`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`body`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`buf`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`buf_to_reuse`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`buffer_var`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`clear`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`cloneWithNewBodies`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`cloneWithNewBody`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`clone_and_replace`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`condition`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`dtype`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`empty`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`end`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`false_stmt`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`flat_index`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`front`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`func_name`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`getEnclosedRoot`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`getSharedParent`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`get_parent`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`gpu_block_index`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`gpu_block_index_str`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`gpu_thread_index`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`gpu_thread_index_str`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`if`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`init`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`insert_stmt_after`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`insert_stmt_before`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`isDefault`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`is_gpu_block_index`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`is_gpu_thread_index`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`is_parallel`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`loop_options`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`make`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`nstmts`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`prepend_stmt`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`removeBody`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`remove_stmt`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`replace_stmt`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_args`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_body`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_buf`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_buf_args`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_buf_out_args`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_buf_to_reuse`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_buffer_map`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_buffer_mapping`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_bufs`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_condition`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_false_stmt`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_gpu_block_index`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_gpu_thread_index`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_indices`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_parallel`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_parent`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_start`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_stmts`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_stop`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_true_stmt`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_val`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_value`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`set_var`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`splice`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`start`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`stop`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`true_stmt`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`value`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`var`**: [stmt.h_docs.md](./stmt.h_docs.md)

### Includes

- **`algorithm`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`list`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`string`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`torch/csrc/jit/tensorexpr/expr.h`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`unordered_set`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`utility`**: [stmt.h_docs.md](./stmt.h_docs.md)
- **`vector`**: [stmt.h_docs.md](./stmt.h_docs.md)

### Namespaces

- **`torch`**: [stmt.h_docs.md](./stmt.h_docs.md)


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
