# Documentation: `docs/android/README.md_docs.md`

## File Metadata

- **Path**: `docs/android/README.md_docs.md`
- **Size**: 9,659 bytes (9.43 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `android/README.md`

## File Metadata

- **Path**: `android/README.md`
- **Size**: 7,819 bytes (7.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This is a markdown documentation that is part of the PyTorch project.

## Original Source

```markdown
# Android

## Demo applications and tutorials

Please refer to [meta-pytorch/executorch-examples](https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo) for the Android demo app based on [ExecuTorch](https://github.com/pytorch/executorch).

Please join our [Discord](https://discord.com/channels/1334270993966825602/1349854760299270284) for any questions.

## Publishing

##### Release
Release artifacts are published to jcenter:

```groovy
repositories {
    jcenter()
}

# lite interpreter build
dependencies {
    implementation 'org.pytorch:pytorch_android_lite:1.10.0'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.10.0'
}

# full jit build
dependencies {
    implementation 'org.pytorch:pytorch_android:1.10.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.10.0'
}
```

##### Nightly

Nightly(snapshots) builds are published every night from `master` branch to [nexus sonatype snapshots repository](https://oss.sonatype.org/#nexus-search;quick~pytorch_android)

To use them repository must be specified explicitly:
```groovy
repositories {
    maven {
        url "https://oss.sonatype.org/content/repositories/snapshots"
    }
}

# lite interpreter build
dependencies {
    ...
    implementation 'org.pytorch:pytorch_android_lite:1.12.0-SNAPSHOT'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.12.0-SNAPSHOT'
    ...
}

# full jit build
dependencies {
    ...
    implementation 'org.pytorch:pytorch_android:1.12.0-SNAPSHOT'
    implementation 'org.pytorch:pytorch_android_torchvision:1.12.0-SNAPSHOT'
    ...
}
```
The current nightly(snapshots) version is the value of `VERSION_NAME` in `gradle.properties` in current folder, at this moment it is `1.8.0-SNAPSHOT`.

## Building PyTorch Android from Source

In some cases you might want to use a local build of pytorch android, for example you may build custom libtorch binary with another set of operators or to make local changes.

For this you can use `./scripts/build_pytorch_android.sh` script.
```bash
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
bash ./scripts/build_pytorch_android.sh
```

The workflow contains several steps:

1\. Build libtorch for android for all 4 android abis (armeabi-v7a, arm64-v8a, x86, x86_64)

2\. Create symbolic links to the results of those builds:
`android/pytorch_android/src/main/jniLibs/${abi}` to the directory with output libraries
`android/pytorch_android/src/main/cpp/libtorch_include/${abi}` to the directory with headers. These directories are used to build `libpytorch.so` library that will be loaded on android device.

3\. And finally run `gradle` in `android/pytorch_android` directory with task `assembleRelease`

Script requires that Android SDK, Android NDK and gradle are installed.
They are specified as environment variables:

`ANDROID_HOME` - path to [Android SDK](https://developer.android.com/studio/command-line/sdkmanager.html)

`ANDROID_NDK` - path to [Android NDK](https://developer.android.com/studio/projects/install-ndk). It's recommended to use NDK 21.x.

`GRADLE_HOME` - path to [gradle](https://gradle.org/releases/)


After successful build you should see the result as aar file:

```bash
$ find pytorch_android/build/ -type f -name *aar
pytorch_android/build/outputs/aar/pytorch_android.aar
pytorch_android_torchvision/build/outputs/aar/pytorch_android.aar
```

It can be used directly in android projects, as a gradle dependency:
```groovy
allprojects {
    repositories {
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {
    implementation(name:'pytorch_android', ext:'aar')
    implementation(name:'pytorch_android_torchvision', ext:'aar')
    ...
    implementation 'com.facebook.soloader:nativeloader:0.10.5'
    implementation 'com.facebook.fbjni:fbjni-java-only:0.2.2'
}
```
We also have to add all transitive dependencies of our aars.
As `pytorch_android` [depends](https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/build.gradle#L76-L77) on `'com.facebook.soloader:nativeloader:0.10.5'` and `'com.facebook.fbjni:fbjni-java-only:0.2.2'`, we need to add them.
(In case of using maven dependencies they are added automatically from `pom.xml`).

## Linking to prebuilt libtorch library from gradle dependency

In some cases, you may want to use libtorch from your android native build.
You can do it without building libtorch android, using native libraries from PyTorch android gradle dependency.
For that, you will need to add the next lines to your gradle build.
```groovy
android {
...
    configurations {
       extractForNativeBuild
    }
...
    compileOptions {
        externalNativeBuild {
            cmake {
                arguments "-DANDROID_STL=c++_shared"
            }
        }
    }
...
    externalNativeBuild {
        cmake {
            path "CMakeLists.txt"
        }
    }
}

dependencies {
    extractForNativeBuild('org.pytorch:pytorch_android:1.10.0')
}

task extractAARForNativeBuild {
    doLast {
        configurations.extractForNativeBuild.files.each {
            def file = it.absoluteFile
            copy {
                from zipTree(file)
                into "$buildDir/$file.name"
                include "headers/**"
                include "jni/**"
            }
        }
    }
}

tasks.whenTaskAdded { task ->
  if (task.name.contains('externalNativeBuild')) {
    task.dependsOn(extractAARForNativeBuild)
  }
}
```

pytorch_android aar contains headers to link in `headers` folder and native libraries in `jni/$ANDROID_ABI/`.
As PyTorch native libraries use `ANDROID_STL` - we should use `ANDROID_STL=c++_shared` to have only one loaded binary of STL.

The added task will unpack them to gradle build directory.

In your native build you can link to them adding these lines to your CMakeLists.txt:


```cmake
# Relative path of gradle build directory to CMakeLists.txt
set(build_DIR ${CMAKE_SOURCE_DIR}/build)

file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

set(BUILD_SUBDIR ${ANDROID_ABI})
target_include_directories(${PROJECT_NAME} PRIVATE
  ${PYTORCH_INCLUDE_DIRS}
)

find_library(PYTORCH_LIBRARY pytorch_jni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)

find_library(FBJNI_LIBRARY fbjni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)

target_link_libraries(${PROJECT_NAME}
  ${PYTORCH_LIBRARY}
  ${FBJNI_LIBRARY})

```
If your CMakeLists.txt file is located in the same directory as your build.gradle, `set(build_DIR ${CMAKE_SOURCE_DIR}/build)` should work for you. But if you have another location of it, you may need to change it.

After that, you can use libtorch C++ API from your native code.
```cpp
#include <string>
#include <ATen/NativeFunctions.h>
#include <torch/script.h>
namespace pytorch_testapp_jni {
namespace {
    struct JITCallGuard {
      c10::InferenceMode guard;
      torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
    };
}

void loadAndForwardModel(const std::string& modelPath) {
  JITCallGuard guard;
  torch::jit::Module module = torch::jit::load(modelPath);
  module.eval();
  torch::Tensor t = torch::randn({1, 3, 224, 224});
  c10::IValue t_out = module.forward({t});
}
}
```

To load torchscript model for mobile we need some special setup which is placed in `struct JITCallGuard` in this example. It may change in future, you can track the latest changes keeping an eye in our [pytorch android jni code]([https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/cpp/pytorch_jni_jit.cpp#L28)

## PyTorch Android API Javadoc

You can find more details about the PyTorch Android API in the [Javadoc](https://pytorch.org/javadoc/).

```



## High-Level Overview

This file is part of the PyTorch framework located at `android`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `android`, which is part of the PyTorch project infrastructure.



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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`android`):

- [`common.sh_docs.md`](./common.sh_docs.md)
- [`run_tests.sh_docs.md`](./run_tests.sh_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md`
- **Keyword Index**: `README.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/android`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/android`, which is part of the PyTorch project infrastructure.



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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/android`):

- [`run_tests.sh_docs.md_docs.md`](./run_tests.sh_docs.md_docs.md)
- [`common.sh_kw.md_docs.md`](./common.sh_kw.md_docs.md)
- [`common.sh_docs.md_docs.md`](./common.sh_docs.md_docs.md)
- [`README.md_kw.md_docs.md`](./README.md_kw.md_docs.md)
- [`run_tests.sh_kw.md_docs.md`](./run_tests.sh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `README.md_docs.md_docs.md`
- **Keyword Index**: `README.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
