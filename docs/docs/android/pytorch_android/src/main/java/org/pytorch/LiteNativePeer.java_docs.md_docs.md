# Documentation: `docs/android/pytorch_android/src/main/java/org/pytorch/LiteNativePeer.java_docs.md`

## File Metadata

- **Path**: `docs/android/pytorch_android/src/main/java/org/pytorch/LiteNativePeer.java_docs.md`
- **Size**: 4,781 bytes (4.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `android/pytorch_android/src/main/java/org/pytorch/LiteNativePeer.java`

## File Metadata

- **Path**: `android/pytorch_android/src/main/java/org/pytorch/LiteNativePeer.java`
- **Size**: 2,285 bytes (2.23 KB)
- **Type**: Source File (.java)
- **Extension**: `.java`

## File Purpose

This is a source file (.java) that is part of the PyTorch project.

## Original Source

```
package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.util.Map;

class LiteNativePeer implements INativePeer {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni_lite");
    PyTorchCodegenLoader.loadNativeLibs();
  }

  private final HybridData mHybridData;

  private static native HybridData initHybrid(
      String moduleAbsolutePath, Map<String, String> extraFiles, int deviceJniCode);

  private static native HybridData initHybridAndroidAsset(
      String assetName, /* android.content.res.AssetManager */
      Object androidAssetManager,
      int deviceJniCode);

  LiteNativePeer(String moduleAbsolutePath, Map<String, String> extraFiles, Device device) {
    mHybridData = initHybrid(moduleAbsolutePath, extraFiles, device.jniCode);
  }

  LiteNativePeer(
      String assetName, /* android.content.res.AssetManager */
      Object androidAssetManager,
      Device device) {
    mHybridData = initHybridAndroidAsset(assetName, androidAssetManager, device.jniCode);
  }

  /**
   * Explicitly destroys the native torch::jit::mobile::Module. Calling this method is not required,
   * as the native object will be destroyed when this object is garbage-collected. However, the
   * timing of garbage collection is not guaranteed, so proactively calling {@code resetNative} can
   * free memory more quickly. See {@link com.facebook.jni.HybridData#resetNative}.
   */
  public void resetNative() {
    mHybridData.resetNative();
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the TorchScript module's 'forward' method.
   * @return return value from the 'forward' method.
   */
  public native IValue forward(IValue... inputs);

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the TorchScript method to run.
   * @param inputs arguments that will be passed to TorchScript method.
   * @return return value from the method.
   */
  public native IValue runMethod(String methodName, IValue... inputs);
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
- [`IValue.java_docs.md`](./IValue.java_docs.md)
- [`LiteModuleLoader.java_docs.md`](./LiteModuleLoader.java_docs.md)
- [`NativePeer.java_docs.md`](./NativePeer.java_docs.md)
- [`INativePeer.java_docs.md`](./INativePeer.java_docs.md)
- [`DType.java_docs.md`](./DType.java_docs.md)
- [`Module.java_docs.md`](./Module.java_docs.md)


## Cross-References

- **File Documentation**: `LiteNativePeer.java_docs.md`
- **Keyword Index**: `LiteNativePeer.java_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/android/pytorch_android/src/main/java/org/pytorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/android/pytorch_android/src/main/java/org/pytorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/android/pytorch_android/src/main/java/org/pytorch`):

- [`NativePeer.java_kw.md_docs.md`](./NativePeer.java_kw.md_docs.md)
- [`LiteModuleLoader.java_docs.md_docs.md`](./LiteModuleLoader.java_docs.md_docs.md)
- [`PyTorchAndroid.java_docs.md_docs.md`](./PyTorchAndroid.java_docs.md_docs.md)
- [`LiteNativePeer.java_kw.md_docs.md`](./LiteNativePeer.java_kw.md_docs.md)
- [`IValue.java_kw.md_docs.md`](./IValue.java_kw.md_docs.md)
- [`PyTorchAndroid.java_kw.md_docs.md`](./PyTorchAndroid.java_kw.md_docs.md)
- [`NativePeer.java_docs.md_docs.md`](./NativePeer.java_docs.md_docs.md)
- [`INativePeer.java_docs.md_docs.md`](./INativePeer.java_docs.md_docs.md)
- [`Tensor.java_docs.md_docs.md`](./Tensor.java_docs.md_docs.md)
- [`MemoryFormat.java_docs.md_docs.md`](./MemoryFormat.java_docs.md_docs.md)


## Cross-References

- **File Documentation**: `LiteNativePeer.java_docs.md_docs.md`
- **Keyword Index**: `LiteNativePeer.java_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
