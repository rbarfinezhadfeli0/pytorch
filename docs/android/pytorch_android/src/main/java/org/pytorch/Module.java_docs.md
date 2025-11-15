# Documentation: `android/pytorch_android/src/main/java/org/pytorch/Module.java`

## File Metadata

- **Path**: `android/pytorch_android/src/main/java/org/pytorch/Module.java`
- **Size**: 2,694 bytes (2.63 KB)
- **Type**: Source File (.java)
- **Extension**: `.java`

## File Purpose

This is a source file (.java) that is part of the PyTorch project.

## Original Source

```
// Copyright 2004-present Facebook. All Rights Reserved.

package org.pytorch;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.util.Map;

/** Java wrapper for torch::jit::Module. */
public class Module {

  private INativePeer mNativePeer;

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on specified
   * device.
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @param extraFiles map with extra files names as keys, content of them will be loaded to values.
   * @param device {@link org.pytorch.Device} to use for running specified module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::Module.
   */
  public static Module load(
      final String modelPath, final Map<String, String> extraFiles, final Device device) {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    return new Module(new NativePeer(modelPath, extraFiles, device));
  }

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on CPU.
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::Module.
   */
  public static Module load(final String modelPath) {
    return load(modelPath, null, Device.CPU);
  }

  Module(INativePeer nativePeer) {
    this.mNativePeer = nativePeer;
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the TorchScript module's 'forward' method.
   * @return return value from the 'forward' method.
   */
  public IValue forward(IValue... inputs) {
    return mNativePeer.forward(inputs);
  }

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the TorchScript method to run.
   * @param inputs arguments that will be passed to TorchScript method.
   * @return return value from the method.
   */
  public IValue runMethod(String methodName, IValue... inputs) {
    return mNativePeer.runMethod(methodName, inputs);
  }

  /**
   * Explicitly destroys the native torch::jit::Module. Calling this method is not required, as the
   * native object will be destroyed when this object is garbage-collected. However, the timing of
   * garbage collection is not guaranteed, so proactively calling {@code destroy} can free memory
   * more quickly. See {@link com.facebook.jni.HybridData#resetNative}.
   */
  public void destroy() {
    mNativePeer.resetNative();
  }
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `android/pytorch_android/src/main/java/org/pytorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `android/pytorch_android/src/main/java/org/pytorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`android/pytorch_android/src/main/java/org/pytorch`):

- [`PyTorchAndroid.java_docs.md`](./PyTorchAndroid.java_docs.md)
- [`PyTorchCodegenLoader.java_docs.md`](./PyTorchCodegenLoader.java_docs.md)
- [`MemoryFormat.java_docs.md`](./MemoryFormat.java_docs.md)
- [`LiteNativePeer.java_docs.md`](./LiteNativePeer.java_docs.md)
- [`IValue.java_docs.md`](./IValue.java_docs.md)
- [`LiteModuleLoader.java_docs.md`](./LiteModuleLoader.java_docs.md)
- [`NativePeer.java_docs.md`](./NativePeer.java_docs.md)
- [`INativePeer.java_docs.md`](./INativePeer.java_docs.md)
- [`DType.java_docs.md`](./DType.java_docs.md)


## Cross-References

- **File Documentation**: `Module.java_docs.md`
- **Keyword Index**: `Module.java_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
