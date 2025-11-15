# Keyword Index: `torch/_dynamo/exc.py`

## File Information

- **Original File**: [torch/_dynamo/exc.py](../../../torch/_dynamo/exc.py)
- **Documentation**: [`exc.py_docs.md`](./exc.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ArgsMismatchError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`AttributeMutationError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`BackendCompilerFailed`**: [exc.py_docs.md](./exc.py_docs.md)
- **`CompileCollectiveRestartAnalysis`**: [exc.py_docs.md](./exc.py_docs.md)
- **`CondOpArgsMismatchError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`FailOnRecompileLimitHit`**: [exc.py_docs.md](./exc.py_docs.md)
- **`IncorrectUsage`**: [exc.py_docs.md](./exc.py_docs.md)
- **`InfiniteGeneratorError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`InternalTorchDynamoError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`InvalidBackend`**: [exc.py_docs.md](./exc.py_docs.md)
- **`KeyErrorMsg`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedAttributeError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedException`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedGeneratorExit`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedIndexError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedKeyError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedLookupError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedNotImplementedError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedRuntimeError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedTypeError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ObservedUserStopIteration`**: [exc.py_docs.md](./exc.py_docs.md)
- **`PackageError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`RecompileError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`RecompileLimitExceeded`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ResetRequired`**: [exc.py_docs.md](./exc.py_docs.md)
- **`RestartAnalysis`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ResumePrologueTracingError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`ShortenTraceback`**: [exc.py_docs.md](./exc.py_docs.md)
- **`SideEffectsError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`SkipCodeRecursiveException`**: [exc.py_docs.md](./exc.py_docs.md)
- **`SkipFrame`**: [exc.py_docs.md](./exc.py_docs.md)
- **`SpeculationRestartAnalysis`**: [exc.py_docs.md](./exc.py_docs.md)
- **`StepUnsupported`**: [exc.py_docs.md](./exc.py_docs.md)
- **`TensorifyScalarRestartAnalysis`**: [exc.py_docs.md](./exc.py_docs.md)
- **`TorchDynamoException`**: [exc.py_docs.md](./exc.py_docs.md)
- **`TorchRuntimeError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`UncapturedHigherOrderOpError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`UnknownPropertiesDuringBackwardTrace`**: [exc.py_docs.md](./exc.py_docs.md)
- **`UnsafeScriptObjectError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`UnspecializeRestartAnalysis`**: [exc.py_docs.md](./exc.py_docs.md)
- **`Unsupported`**: [exc.py_docs.md](./exc.py_docs.md)
- **`UserError`**: [exc.py_docs.md](./exc.py_docs.md)
- **`UserErrorType`**: [exc.py_docs.md](./exc.py_docs.md)
- **`for`**: [exc.py_docs.md](./exc.py_docs.md)

### Functions

- **`__init__`**: [exc.py_docs.md](./exc.py_docs.md)
- **`__repr__`**: [exc.py_docs.md](./exc.py_docs.md)
- **`__str__`**: [exc.py_docs.md](./exc.py_docs.md)
- **`_load_gb_type_to_gb_id_map`**: [exc.py_docs.md](./exc.py_docs.md)
- **`add_to_stats`**: [exc.py_docs.md](./exc.py_docs.md)
- **`augment_exc_message`**: [exc.py_docs.md](./exc.py_docs.md)
- **`collapse_resume_frames`**: [exc.py_docs.md](./exc.py_docs.md)
- **`exportdb_error_message`**: [exc.py_docs.md](./exc.py_docs.md)
- **`filter_stack`**: [exc.py_docs.md](./exc.py_docs.md)
- **`format_error_msg`**: [exc.py_docs.md](./exc.py_docs.md)
- **`format_error_msg_verbose`**: [exc.py_docs.md](./exc.py_docs.md)
- **`format_frame_info`**: [exc.py_docs.md](./exc.py_docs.md)
- **`format_graph_break_message`**: [exc.py_docs.md](./exc.py_docs.md)
- **`format_loop_skip_frame_message`**: [exc.py_docs.md](./exc.py_docs.md)
- **`format_skip_frame_message`**: [exc.py_docs.md](./exc.py_docs.md)
- **`get_dynamo_observed_exception`**: [exc.py_docs.md](./exc.py_docs.md)
- **`get_exc_message`**: [exc.py_docs.md](./exc.py_docs.md)
- **`get_gbid_documentation_link`**: [exc.py_docs.md](./exc.py_docs.md)
- **`get_real_stack`**: [exc.py_docs.md](./exc.py_docs.md)
- **`get_stack_above_dynamo`**: [exc.py_docs.md](./exc.py_docs.md)
- **`handle_observed_exception`**: [exc.py_docs.md](./exc.py_docs.md)
- **`raise_observed_exception`**: [exc.py_docs.md](./exc.py_docs.md)
- **`remove_dynamo_frames`**: [exc.py_docs.md](./exc.py_docs.md)
- **`remove_from_stats`**: [exc.py_docs.md](./exc.py_docs.md)
- **`remove_resume_prefix`**: [exc.py_docs.md](./exc.py_docs.md)
- **`unimplemented`**: [exc.py_docs.md](./exc.py_docs.md)
- **`unimplemented_with_warning`**: [exc.py_docs.md](./exc.py_docs.md)

### Imports

- **`.`**: [exc.py_docs.md](./exc.py_docs.md)
- **`.output_graph`**: [exc.py_docs.md](./exc.py_docs.md)
- **`.resume_execution`**: [exc.py_docs.md](./exc.py_docs.md)
- **`.symbolic_convert`**: [exc.py_docs.md](./exc.py_docs.md)
- **`.types`**: [exc.py_docs.md](./exc.py_docs.md)
- **`.utils`**: [exc.py_docs.md](./exc.py_docs.md)
- **`.variables`**: [exc.py_docs.md](./exc.py_docs.md)
- **`Any`**: [exc.py_docs.md](./exc.py_docs.md)
- **`BuiltinVariable`**: [exc.py_docs.md](./exc.py_docs.md)
- **`CompileId`**: [exc.py_docs.md](./exc.py_docs.md)
- **`DynamoFrameType`**: [exc.py_docs.md](./exc.py_docs.md)
- **`DynamoTracerOutput`**: [exc.py_docs.md](./exc.py_docs.md)
- **`InstructionTranslatorBase`**: [exc.py_docs.md](./exc.py_docs.md)
- **`Path`**: [exc.py_docs.md](./exc.py_docs.md)
- **`TORCH_DYNAMO_RESUME_IN_PREFIX`**: [exc.py_docs.md](./exc.py_docs.md)
- **`__future__`**: [exc.py_docs.md](./exc.py_docs.md)
- **`annotations`**: [exc.py_docs.md](./exc.py_docs.md)
- **`auto`**: [exc.py_docs.md](./exc.py_docs.md)
- **`config`**: [exc.py_docs.md](./exc.py_docs.md)
- **`counters`**: [exc.py_docs.md](./exc.py_docs.md)
- **`enum`**: [exc.py_docs.md](./exc.py_docs.md)
- **`extract_stack`**: [exc.py_docs.md](./exc.py_docs.md)
- **`functools`**: [exc.py_docs.md](./exc.py_docs.md)
- **`get_file_path_2`**: [exc.py_docs.md](./exc.py_docs.md)
- **`json`**: [exc.py_docs.md](./exc.py_docs.md)
- **`logging`**: [exc.py_docs.md](./exc.py_docs.md)
- **`lru_cache`**: [exc.py_docs.md](./exc.py_docs.md)
- **`pathlib`**: [exc.py_docs.md](./exc.py_docs.md)
- **`re`**: [exc.py_docs.md](./exc.py_docs.md)
- **`textwrap`**: [exc.py_docs.md](./exc.py_docs.md)
- **`torch._guards`**: [exc.py_docs.md](./exc.py_docs.md)
- **`torch._utils_internal`**: [exc.py_docs.md](./exc.py_docs.md)
- **`traceback`**: [exc.py_docs.md](./exc.py_docs.md)
- **`types`**: [exc.py_docs.md](./exc.py_docs.md)
- **`typing`**: [exc.py_docs.md](./exc.py_docs.md)


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
