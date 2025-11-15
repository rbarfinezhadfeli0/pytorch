# Documentation: `torch/_dynamo/variables/__init__.py`

## File Metadata

- **Path**: `torch/_dynamo/variables/__init__.py`
- **Size**: 6,894 bytes (6.73 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
"""
This package implements variable tracking and symbolic execution capabilities for Dynamo,
which are essential for converting Python code into FX graphs. It provides a comprehensive
set of variable types that handle different Python constructs during tracing.

Each variable type (like BuiltinVariable, TensorVariable, NNModuleVariable, etc.) is responsible
for tracking and symbolically executing operations on specific Python objects. This enables
Dynamo to:
- Track the flow of values through Python code
- Maintain correct semantics during graph conversion
- Handle complex Python features like context managers, iterators, and custom objects
- Support both eager and symbolic execution modes

The VariableTracker base class provides the foundation for all variable types, with each
subclass implementing specific behavior for different Python constructs. This modular design
allows Dynamo to accurately trace and optimize Python code while preserving its semantics.
"""

from .base import VariableTracker
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
    CatchWarningsCtxManagerVariable,
    ContextWrappingVariable,
    CUDADeviceVariable,
    DeterministicAlgorithmsVariable,
    DisabledSavedTensorsHooksVariable,
    DualLevelContextManager,
    DynamoConfigPatchVariable,
    ErrorOnGraphBreakVariable,
    FSDPParamGroupUseTrainingStateVariable,
    FxTracebackAnnotateVariable,
    GradIncrementNestingCtxManagerVariable,
    GradInplaceRequiresGradCtxManagerVariable,
    GradModeVariable,
    InferenceModeVariable,
    JvpIncrementNestingCtxManagerVariable,
    SDPAKernelVariable,
    SetFwdGradEnabledContextManager,
    TemporarilyPopInterpreterStackCtxManagerVariable,
    VmapIncrementNestingCtxManagerVariable,
    WithEnterFunctionVariable,
    WithExitFunctionVariable,
)
from .dicts import (
    ConstDictVariable,
    DefaultDictVariable,
    DictKeySetVariable,
    FrozensetVariable,
    MappingProxyVariable,
    NNModuleHooksDictVariable,
    SetVariable,
)
from .distributed import BackwardHookVariable, DistributedVariable, PlacementVariable
from .functions import (
    BuiltinMethodVariable,
    CollectionsNamedTupleFunction,
    CreateTMADescriptorExperimentalVariable,
    CreateTMADescriptorStableVariable,
    FunctionDecoratedByContextlibContextManagerVariable,
    FunctoolsPartialVariable,
    FunctoolsWrapsVariable,
    LocalGeneratorFunctionVariable,
    LocalGeneratorObjectVariable,
    NestedUserFunctionVariable,
    PolyfilledFunctionVariable,
    SkipFunctionVariable,
    TMADescriptorExperimentalVariable,
    TMADescriptorStableVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrapperUserFunctionVariable,
    WrapperUserMethodVariable,
)
from .higher_order_ops import (
    FunctionalCallVariable,
    FunctorchHigherOrderVariable,
    ReparametrizeModuleCallVariable,
    TorchHigherOrderOperatorVariable,
)
from .iter import (
    CountIteratorVariable,
    FilterVariable,
    IteratorVariable,
    ItertoolsVariable,
    MapVariable,
    ObjectIteratorVariable,
    RepeatIteratorVariable,
    ZipVariable,
)
from .lazy import LazyVariableTracker
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    NamedTupleVariable,
    RangeVariable,
    SliceVariable,
    TupleIteratorVariable,
    TupleVariable,
)
from .misc import (
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    CellVariable,
    DeletedVariable,
    ExceptionVariable,
    GetAttrVariable,
    LambdaVariable,
    MethodWrapperVariable,
    NewGlobalVariable,
    NumpyVariable,
    PythonModuleVariable,
    RandomClassVariable,
    RandomVariable,
    RegexPatternVariable,
    StringFormatVariable,
    SuperVariable,
    TorchVersionVariable,
    TypingVariable,
    UnknownVariable,
    WeakRefVariable,
)
from .nn_module import (
    FSDPManagedNNModuleVariable,
    NNModuleVariable,
    UnspecializedBuiltinNNModuleVariable,
    UnspecializedNNModuleVariable,
)
from .optimizer import OptimizerVariable
from .sdpa import SDPAParamsVariable
from .streams import EventVariable, StreamContextVariable, StreamVariable
from .tensor import (
    DataPtrVariable,
    FakeItemVariable,
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
    UntypedStorageVariable,
)
from .torch import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
from .user_defined import (
    FrozenDataClassVariable,
    MutableMappingVariable,
    RemovableHandleVariable,
    UserDefinedClassVariable,
    UserDefinedDictVariable,
    UserDefinedExceptionClassVariable,
    UserDefinedExceptionObjectVariable,
    UserDefinedListVariable,
    UserDefinedObjectVariable,
    UserDefinedSetVariable,
    UserDefinedTupleVariable,
)


__all__ = [
    "AutogradFunctionContextVariable",
    "AutogradFunctionVariable",
    "BackwardHookVariable",
    "BaseListVariable",
    "BuiltinVariable",
    "CatchWarningsCtxManagerVariable",
    "ConstantVariable",
    "ConstDictVariable",
    "ContextWrappingVariable",
    "CountIteratorVariable",
    "CreateTMADescriptorExperimentalVariable",
    "CreateTMADescriptorStableVariable",
    "CUDADeviceVariable",
    "DataPtrVariable",
    "DefaultDictVariable",
    "DeletedVariable",
    "DeterministicAlgorithmsVariable",
    "DictKeySetVariable",
    "DynamoConfigPatchVariable",
    "EnumVariable",
    "FakeItemVariable",
    "GetAttrVariable",
    "GradModeVariable",
    "IteratorVariable",
    "ItertoolsVariable",
    "LambdaVariable",
    "LazyVariableTracker",
    "ListIteratorVariable",
    "ListVariable",
    "NamedTupleVariable",
    "NestedUserFunctionVariable",
    "CellVariable",
    "NewGlobalVariable",
    "NNModuleVariable",
    "NumpyNdarrayVariable",
    "NumpyVariable",
    "OptimizerVariable",
    "PlacementVariable",
    "PolyfilledFunctionVariable",
    "PythonModuleVariable",
    "RangeVariable",
    "RegexPatternVariable",
    "RemovableHandleVariable",
    "RepeatIteratorVariable",
    "SDPAParamsVariable",
    "ErrorOnGraphBreakVariable",
    "SkipFunctionVariable",
    "SliceVariable",
    "StringFormatVariable",
    "SuperVariable",
    "TemporarilyPopInterpreterStackCtxManagerVariable",
    "TensorVariable",
    "TMADescriptorExperimentalVariable",
    "TMADescriptorStableVariable",
    "TorchCtxManagerClassVariable",
    "TorchInGraphFunctionVariable",
    "TorchVersionVariable",
    "TupleVariable",
    "UnknownVariable",
    "UnspecializedNNModuleVariable",
    "UnspecializedPythonVariable",
    "UntypedStorageVariable",
    "UserDefinedClassVariable",
    "UserDefinedTupleVariable",
    "UserDefinedObjectVariable",
    "UserFunctionVariable",
    "UserMethodVariable",
    "VariableTracker",
    "WithEnterFunctionVariable",
    "WithExitFunctionVariable",
    "MappingProxyVariable",
]

```



## High-Level Overview

"""This package implements variable tracking and symbolic execution capabilities for Dynamo,which are essential for converting Python code into FX graphs. It provides a comprehensiveset of variable types that handle different Python constructs during tracing.Each variable type (like BuiltinVariable, TensorVariable, NNModuleVariable, etc.) is responsiblefor tracking and symbolically executing operations on specific Python objects. This enablesDynamo to:- Track the flow of values through Python code- Maintain correct semantics during graph conversion- Handle complex Python features like context managers, iterators, and custom objects- Support both eager and symbolic execution modesThe VariableTracker base class provides the foundation for all variable types, with eachsubclass implementing specific behavior for different Python constructs. This modular designallows Dynamo to accurately trace and optimize Python code while preserving its semantics.

This Python file contains 2 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: VariableTracker, BuiltinVariable, ConstantVariable, EnumVariable, BackwardHookVariable, DistributedVariable, PlacementVariable, LazyVariableTracker, OptimizerVariable, SDPAParamsVariable, EventVariable, StreamContextVariable, StreamVariable, TorchCtxManagerClassVariable, TorchInGraphFunctionVariable


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/variables`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `.base`: VariableTracker
- `.builtin`: BuiltinVariable
- `.constant`: ConstantVariable, EnumVariable
- `.distributed`: BackwardHookVariable, DistributedVariable, PlacementVariable
- `.lazy`: LazyVariableTracker
- `.optimizer`: OptimizerVariable
- `.sdpa`: SDPAParamsVariable
- `.streams`: EventVariable, StreamContextVariable, StreamVariable
- `.torch`: TorchCtxManagerClassVariable, TorchInGraphFunctionVariable


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/_dynamo/variables`):

- [`streams.py_docs.md`](./streams.py_docs.md)
- [`nn_module.py_docs.md`](./nn_module.py_docs.md)
- [`higher_order_ops.py_docs.md`](./higher_order_ops.py_docs.md)
- [`tensor.py_docs.md`](./tensor.py_docs.md)
- [`torch.py_docs.md`](./torch.py_docs.md)
- [`constant.py_docs.md`](./constant.py_docs.md)
- [`dicts.py_docs.md`](./dicts.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`lists.py_docs.md`](./lists.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
