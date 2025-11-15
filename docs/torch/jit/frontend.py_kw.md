# Keyword Index: `torch/jit/frontend.py`

## File Information

- **Original File**: [torch/jit/frontend.py](../../../torch/jit/frontend.py)
- **Documentation**: [`frontend.py_docs.md`](./frontend.py_docs.md)
- **Folder**: `torch/jit`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Builder`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`ExprBuilder`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`FrontendError`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`FrontendTypeError`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`NotSupportedError`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`StmtBuilder`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`UnsupportedNodeError`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`WithItemBuilder`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`and`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`assign_stmt`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`definitions`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`independently`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`inheritance`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`of`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`sourcelines`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`that`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`to`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`type`**: [frontend.py_docs.md](./frontend.py_docs.md)

### Functions

- **`__call__`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`__init__`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`__str__`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`_forward`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_AnnAssign`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Assert`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Assign`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Attribute`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_AugAssign`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_BinOp`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_BoolOp`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Break`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Call`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Compare`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Constant`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Continue`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Delete`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Dict`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_DictComp`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Ellipsis`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Expr`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_ExtSlice`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_For`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_GeneratorExp`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_If`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_IfExp`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Index`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_JoinedStr`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_List`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_ListComp`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Name`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_NameConstant`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Num`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Pass`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Print`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Raise`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Return`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_SliceExpr`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Starred`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Str`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Subscript`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_Tuple`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_UnaryOp`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_While`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_With`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_args`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_class_def`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_def`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_ignore_context_manager`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_param`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_param_list`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_return_ann_stmt`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_stmts`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_withitem`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`build_withitems`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`create_unique_name_ext`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`find_before`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`func`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`get_char`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`get_class_assigns`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`get_class_properties`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`get_default_args`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`get_default_args_for_class`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`get_jit_class_def`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`get_jit_def`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`is_classmethod`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`is_reserved_name`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`is_torch_jit_ignore_context_manager`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`maybe_build_assign`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`process_ins_outs`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`unused_fn`**: [frontend.py_docs.md](./frontend.py_docs.md)

### Imports

- **`DATACLASS_MAGIC_METHODS`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`List`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`_jit_internal`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`ast`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`collections`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`copy`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`dataclasses`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`dedent`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`division`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`get_qualified_name`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`inspect`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`namedtuple`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`re`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`statements`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`string`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`textwrap`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`torch`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`torch._C._jit_tree_views`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`torch._jit_internal`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`torch._sources`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`torch.jit._dataclass_impls`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`torch.jit._monkeytype_config`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`torch.jit.annotations`**: [frontend.py_docs.md](./frontend.py_docs.md)
- **`typing`**: [frontend.py_docs.md](./frontend.py_docs.md)


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
