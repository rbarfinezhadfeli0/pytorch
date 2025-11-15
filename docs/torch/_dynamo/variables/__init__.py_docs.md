# Documentation: __init__.py

## File Metadata
- **Path**: `torch/_dynamo/variables/__init__.py`
- **Size**: 6894 bytes
- **Lines**: 230
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough


## Key Components

The file contains 410 words across 230 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 6894 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
