# Documentation: `docs/test/benchmark_utils/callgrind_artifacts.json_docs.md`

## File Metadata

- **Path**: `docs/test/benchmark_utils/callgrind_artifacts.json_docs.md`
- **Size**: 52,127 bytes (50.91 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file contains **examples or benchmarks**.

## Original Source

```markdown
# Documentation: `test/benchmark_utils/callgrind_artifacts.json`

## File Metadata

- **Path**: `test/benchmark_utils/callgrind_artifacts.json`
- **Size**: 232,454 bytes (227.01 KB)
- **Type**: JSON Configuration
- **Extension**: `.json`

## File Purpose

This file is part of the **testing infrastructure**. This file contains **examples or benchmarks**.

## Original Source

```json
{
    "baseline_inclusive": [
        "6746 /home/rdonnelly/mc/conda-bld/compilers_linux-64_1534865402226/work/.build/src/glibc-2.12.2/csu/../sysdeps/x86_64/elf/start.S:0x00000000001c3ce2 [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Modules/main.c:Py_Main [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Programs/python.c:main [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_EvalCode [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_EvalCodeEx [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_EvalFrameEx [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:_PyEval_EvalFrameDefault [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Python/pythonrun.c:PyRun_AnyFileExFlags [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Python/pythonrun.c:PyRun_FileExFlags [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Python/pythonrun.c:PyRun_SimpleFileExFlags [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 /tmp/build/80754af9/python_1599604603603/work/Python/pythonrun.c:run_mod [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "6746 ???:(below main) [/usr/lib64/libc-2.28.so]",
        "6746 ???:0x0000000000001050 [/usr/lib64/ld-2.28.so]",
        "2407 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:call_function [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "1206 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:_PyObject_FastCallKeywords [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "1196 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:_PyObject_FastCallDict [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "1180 /tmp/build/80754af9/python_1599604603603/work/Objects/dictobject.c:PyDict_SetItem [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "1019 /tmp/build/80754af9/python_1599604603603/work/Objects/methodobject.c:_PyCFunction_FastCallKeywords [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "1013 /tmp/build/80754af9/python_1599604603603/work/Objects/methodobject.c:_PyCFunction_FastCallDict [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "881 /tmp/build/80754af9/python_1599604603603/work/Objects/typeobject.c:type_call [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "867 /tmp/build/80754af9/python_1599604603603/work/Objects/dictobject.c:lookdict_unicode_nodummy [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "862 /tmp/build/80754af9/python_1599604603603/work/Objects/dictobject.c:PyDict_GetItem [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "789 /tmp/build/80754af9/python_1599604603603/work/Objects/rangeobject.c:range_new [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "686 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:PyObject_GetAttr [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "632 /tmp/build/80754af9/python_1599604603603/work/Objects/moduleobject.c:module_getattro [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "590 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:PyObject_GenericGetAttr [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "584 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:_PyObject_GenericGetAttrWithDict [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "561 /tmp/build/80754af9/python_1599604603603/work/Modules/timemodule.c:time_sleep [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "261 /home/test_user/miniconda3/envs/throwaway/include/pybind11/pybind11.h:pybind11::cpp_function::dispatcher(_object*, _object*, _object*) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "209 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_RestoreThread [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "207 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:PyObject_RichCompareBool [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "196 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:PyObject_GetIter [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "195 /tmp/build/80754af9/python_1599604603603/work/Objects/rangeobject.c:rangeiter_next [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "192 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:_PyStack_AsTuple [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "180 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:PyLong_FromLong [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "177 /tmp/build/80754af9/python_1599604603603/work/Objects/rangeobject.c:range_iter [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "167 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:PyObject_RichCompare [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "167 /tmp/build/80754af9/python_1599604603603/work/Python/ceval_gil.h:PyEval_RestoreThread",
        "157 /tmp/build/80754af9/python_1599604603603/work/Objects/tupleobject.c:PyTuple_New [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "129 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:PyNumber_Subtract [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "113 /tmp/build/80754af9/python_1599604603603/work/Objects/obmalloc.c:PyObject_Malloc [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "112 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_SaveThread [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "111 /tmp/build/80754af9/python_1599604603603/work/Objects/typeobject.c:_PyType_Lookup [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "100 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:_Py_CheckFunctionResult [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "98 /tmp/build/80754af9/python_1599604603603/work/Python/ceval_gil.h:PyEval_SaveThread",
        "98 /tmp/build/80754af9/python_1599604603603/work/Python/pytime.c:_PyTime_FromSecondsObject [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "94 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:PyNumber_FloorDivide [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "93 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:_PyObject_New [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "90 ???:pthread_mutex_lock [/usr/lib64/libpthread-2.28.so]",
        "87 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:PyLong_AsLong [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "81 /tmp/build/80754af9/python_1599604603603/work/Modules/gcmodule.c:_PyObject_GC_NewVar [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "80 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:binary_op1 [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "75 ???:pthread_mutex_unlock [/usr/lib64/libpthread-2.28.so]",
        "72 /tmp/build/80754af9/python_1599604603603/work/Objects/obmalloc.c:PyObject_Free [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "67 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:long_sub [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]"
    ],
    "baseline_exclusive": [
        "1394 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:_PyEval_EvalFrameDefault [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "867 /tmp/build/80754af9/python_1599604603603/work/Objects/dictobject.c:lookdict_unicode_nodummy [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "710 /tmp/build/80754af9/python_1599604603603/work/Objects/dictobject.c:PyDict_SetItem [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "338 /tmp/build/80754af9/python_1599604603603/work/Objects/dictobject.c:PyDict_GetItem [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "182 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:call_function [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "180 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:PyLong_FromLong [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "177 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:_PyObject_GenericGetAttrWithDict [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "134 /tmp/build/80754af9/python_1599604603603/work/Objects/rangeobject.c:range_new [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "113 /tmp/build/80754af9/python_1599604603603/work/Objects/obmalloc.c:PyObject_Malloc [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "111 /tmp/build/80754af9/python_1599604603603/work/Objects/typeobject.c:_PyType_Lookup [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "110 /tmp/build/80754af9/python_1599604603603/work/Objects/methodobject.c:_PyCFunction_FastCallDict [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "104 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:PyObject_RichCompare [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "95 /tmp/build/80754af9/python_1599604603603/work/Objects/rangeobject.c:rangeiter_next [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "90 ???:pthread_mutex_lock [/usr/lib64/libpthread-2.28.so]",
        "85 /home/test_user/miniconda3/envs/throwaway/include/pybind11/pybind11.h:pybind11::cpp_function::dispatcher(_object*, _object*, _object*) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "74 /tmp/build/80754af9/python_1599604603603/work/Python/pytime.c:_PyTime_FromSecondsObject [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "70 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:_PyObject_FastCallDict [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "66 /tmp/build/80754af9/python_1599604603603/work/Objects/obmalloc.c:_PyObject_Free [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "66 ???:__pthread_mutex_unlock_usercnt [/usr/lib64/libpthread-2.28.so]",
        "64 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:_Py_CheckFunctionResult [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "63 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:long_richcompare [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "62 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:PyNumber_Subtract [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "61 /tmp/build/80754af9/python_1599604603603/work/Objects/tupleobject.c:PyTuple_New [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "54 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:PyObject_GetAttr [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "54 /tmp/build/80754af9/python_1599604603603/work/Python/errors.c:PyErr_Restore [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "51 /tmp/build/80754af9/python_1599604603603/work/Modules/timemodule.c:time_sleep [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "49 /tmp/build/80754af9/python_1599604603603/work/Objects/dictobject.c:lookdict_unicode [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "47 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:long_sub [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "47 /tmp/build/80754af9/python_1599604603603/work/Objects/rangeobject.c:range_iter [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "47 /tmp/build/80754af9/python_1599604603603/work/Python/getargs.c:PyArg_UnpackTuple [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "45 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:PyLong_AsLongAndOverflow [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "44 /home/test_user/miniconda3/envs/throwaway/include/pybind11/detail/internals.h:pybind11::detail::get_internals() [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "44 /tmp/build/80754af9/python_1599604603603/work/Objects/tupleobject.c:_PyObject_FastCallDict",
        "44 /tmp/build/80754af9/python_1599604603603/work/Objects/typeobject.c:type_call [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "44 ???:pthread_cond_signal@@GLIBC_2.3.2 [/usr/lib64/libpthread-2.28.so]",
        "42 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:PyLong_AsLong [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "42 /tmp/build/80754af9/python_1599604603603/work/Objects/moduleobject.c:module_getattro [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "40 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:PyObject_RichCompareBool [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "39 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:long_div [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "35 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:_PyStack_AsTuple [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "35 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_RestoreThread [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "35 /tmp/build/80754af9/python_1599604603603/work/Python/ceval_gil.h:PyEval_RestoreThread",
        "35 /tmp/build/80754af9/python_1599604603603/work/Python/errors.c:PyErr_Occurred [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "34 /tmp/build/80754af9/python_1599604603603/work/Python/pytime.c:_PyTime_AsTimeval [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "31 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:PyNumber_Add [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "31 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:binary_op1 [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "26 /tmp/build/80754af9/python_1599604603603/work/Python/pystate.c:_PyThreadState_UncheckedGet [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "25 /tmp/build/80754af9/python_1599604603603/work/Objects/longobject.c:long_add [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "22 /tmp/build/80754af9/python_1599604603603/work/Modules/gcmodule.c:_PyObject_GC_NewVar [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "22 /tmp/build/80754af9/python_1599604603603/work/Python/pytime.c:_PyTime_GetMonotonicClock [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "21 /tmp/build/80754af9/python_1599604603603/work/Python/ceval_gil.h:PyEval_SaveThread",
        "21 /usr/include/c++/8/bits/stl_vector.h:pybind11::cpp_function::dispatcher(_object*, _object*, _object*)",
        "20 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:_PyObject_New [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "20 ???:clock_gettime [/usr/lib64/libc-2.28.so]",
        "19 /tmp/build/80754af9/python_1599604603603/work/Modules/gcmodule.c:_PyObject_GC_Malloc [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "19 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:PyObject_GetIter [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "18 /tmp/build/80754af9/python_1599604603603/work/Objects/capsule.c:PyCapsule_GetPointer [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "17 /home/test_user/miniconda3/envs/throwaway/include/pybind11/cast.h:pybind11::cpp_function::dispatcher(_object*, _object*, _object*)",
        "17 /tmp/build/80754af9/python_1599604603603/work/Objects/floatobject.c:PyFloat_AsDouble [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "17 /tmp/build/80754af9/python_1599604603603/work/Objects/rangeobject.c:range_dealloc [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "15 /tmp/build/80754af9/python_1599604603603/work/Modules/gcmodule.c:PyObject_GC_UnTrack [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "15 ???:__memset_avx2_unaligned_erms [/usr/lib64/libc-2.28.so]",
        "14 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:PyNumber_FloorDivide [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "13 /tmp/build/80754af9/python_1599604603603/work/Objects/frameobject.c:PyFrame_BlockSetup [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "13 /tmp/build/80754af9/python_1599604603603/work/Objects/typeobject.c:object_init [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "13 /usr/include/c++/8/bits/stl_bvector.h:pybind11::cpp_function::dispatcher(_object*, _object*, _object*)",
        "12 /home/test_user/miniconda3/envs/throwaway/include/pybind11/pytypes.h:pybind11::cpp_function::dispatcher(_object*, _object*, _object*)",
        "11 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:PyNumber_Index [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "11 /tmp/build/80754af9/python_1599604603603/work/Objects/frameobject.c:PyFrame_BlockPop [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "11 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_SaveThread [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "11 ???:select [/usr/lib64/libc-2.28.so]",
        "11 build/../torch/csrc/Module.cpp:void pybind11::cpp_function::initialize<initModule::{lambda()",
        "10 /tmp/build/80754af9/python_1599604603603/work/Objects/abstract.c:_PyObject_FastCallKeywords [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "10 /tmp/build/80754af9/python_1599604603603/work/Objects/typeobject.c:PyType_IsSubtype [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "9 ???:pthread_mutex_unlock [/usr/lib64/libpthread-2.28.so]",
        "8 /tmp/build/80754af9/python_1599604603603/work/Python/errors.c:PyErr_Clear [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "7 /tmp/build/80754af9/python_1599604603603/work/Objects/tupleobject.c:PyTuple_Size [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "7 ???:isnan [/usr/lib64/libc-2.28.so]",
        "6 /home/test_user/miniconda3/envs/throwaway/include/pybind11/pybind11.h:void pybind11::cpp_function::initialize<initModule::{lambda() [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "6 /home/test_user/miniconda3/envs/throwaway/include/pybind11/pytypes.h:pybind11::detail::function_call::~function_call()"
    ],
    "ones_no_data_inclusive": [
        "8959166 /home/rdonnelly/mc/conda-bld/compilers_linux-64_1534865402226/work/.build/src/glibc-2.12.2/csu/../sysdeps/x86_64/elf/start.S:0x00000000001c3ce2 [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Modules/main.c:Py_Main [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Programs/python.c:main [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_EvalCode [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_EvalCodeEx [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_EvalFrameEx [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:_PyEval_EvalFrameDefault [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Python/pythonrun.c:PyRun_AnyFileExFlags [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Python/pythonrun.c:PyRun_FileExFlags [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Python/pythonrun.c:PyRun_SimpleFileExFlags [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 /tmp/build/80754af9/python_1599604603603/work/Python/pythonrun.c:run_mod [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "8959166 ???:(below main) [/usr/lib64/libc-2.28.so]",
        "8959166 ???:0x0000000000001050 [/usr/lib64/ld-2.28.so]",
        "7418729 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:call_function [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "7356341 /tmp/build/80754af9/python_1599604603603/work/Objects/methodobject.c:_PyCFunction_FastCallKeywords [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "7353335 /tmp/build/80754af9/python_1599604603603/work/Objects/methodobject.c:_PyCFunction_FastCallDict [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "7114322 /data/users/test_user/repos/pytorch/build/../torch/csrc/autograd/generated/python_torch_functions.cpp:torch::autograd::THPVariable_ones(_object*, _object*, _object*)",
        "5411822 build/../torch/csrc/autograd/generated/variable_factories.h:torch::autograd::THPVariable_ones(_object*, _object*, _object*)",
        "5241822 /data/users/test_user/repos/pytorch/build/aten/src/ATen/Functions.cpp:at::ones(c10::ArrayRef<long>, c10::TensorOptions const&)",
        "5130822 build/aten/src/ATen/Functions.cpp:at::ones(c10::ArrayRef<long>, c10::TensorOptions const&) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "5114822 build/aten/src/ATen/Functions.cpp:at::ones(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "4964822 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::ones(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)",
        "4943822 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)> const&, c10::DispatchKey, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) const [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "4682822 build/../aten/src/ATen/core/boxing/KernelFunction_impl.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)> const&, c10::DispatchKey, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) const",
        "4660822 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor (*)(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>), at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> > >, at::Tensor (c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)>::call(c10::OperatorKernel*, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)",
        "4597822 build/../aten/src/ATen/core/boxing/impl/WrapFunctionIntoRuntimeFunctor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor (*)(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>), at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> > >, at::Tensor (c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)>::call(c10::OperatorKernel*, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)",
        "4586822 /data/users/test_user/repos/pytorch/build/aten/src/ATen/BackendSelectRegister.cpp:at::(anonymous namespace)::ones(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)",
        "4372822 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::(anonymous namespace)::ones(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)",
        "4352822 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)> const&, c10::DispatchKey, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) const'2 [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "4091822 build/../aten/src/ATen/core/boxing/KernelFunction_impl.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)> const&, c10::DispatchKey, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) const'2",
        "4069822 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor (*)(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>), at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> > >, at::Tensor (c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)>::call(c10::OperatorKernel*, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)'2",
        "4006822 build/../aten/src/ATen/core/boxing/impl/WrapFunctionIntoRuntimeFunctor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor (*)(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>), at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> > >, at::Tensor (c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)>::call(c10::OperatorKernel*, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)'2",
        "3995822 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/core/op_registration/hacky_wrapper_for_legacy_signatures.h:c10::impl::detail::with_scattered_tensor_options_impl_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&), &at::TypeDefault::ones>, c10::guts::typelist::typelist<c10::ArrayRef<long> >, c10::guts::typelist::typelist<> >::wrapper(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)",
        "3905822 build/../aten/src/ATen/core/op_registration/hacky_wrapper_for_legacy_signatures.h:c10::impl::detail::with_scattered_tensor_options_impl_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&), &at::TypeDefault::ones>, c10::guts::typelist::typelist<c10::ArrayRef<long> >, c10::guts::typelist::typelist<> >::wrapper(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "3831822 /data/users/test_user/repos/pytorch/build/aten/src/ATen/TypeDefault.cpp:at::TypeDefault::ones(c10::ArrayRef<long>, c10::TensorOptions const&)",
        "3742822 build/aten/src/ATen/TypeDefault.cpp:at::TypeDefault::ones(c10::ArrayRef<long>, c10::TensorOptions const&) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "3718822 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/native/TensorFactories.cpp:at::native::ones(c10::ArrayRef<long>, c10::TensorOptions const&)",
        "3715822 build/../aten/src/ATen/native/TensorFactories.cpp:at::native::ones(c10::ArrayRef<long>, c10::TensorOptions const&) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "3702822 build/../aten/src/ATen/native/TensorFactories.cpp:at::native::full(c10::ArrayRef<long>, c10::Scalar, c10::TensorOptions const&) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "2526822 /data/users/test_user/repos/pytorch/build/aten/src/ATen/Functions.cpp:at::empty(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)",
        "2438822 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::empty(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)",
        "2422822 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)> const&, c10::DispatchKey, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>) const [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "2209822 build/../aten/src/ATen/core/boxing/KernelFunction_impl.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)> const&, c10::DispatchKey, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>) const",
        "2198822 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor (*)(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>), at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat> > >, at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)",
        "2183822 build/../aten/src/ATen/core/boxing/impl/WrapFunctionIntoRuntimeFunctor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor (*)(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>), at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat> > >, at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)",
        "2178822 /data/users/test_user/repos/pytorch/build/aten/src/ATen/BackendSelectRegister.cpp:at::(anonymous namespace)::empty_memory_format(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)",
        "1934822 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::(anonymous namespace)::empty_memory_format(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)",
        "1917822 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)> const&, c10::DispatchKey, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>) const'2 [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "1704822 build/../aten/src/ATen/core/boxing/KernelFunction_impl.h:at::Tensor c10::Dispatcher::callWithDispatchKey<at::Tensor, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)> const&, c10::DispatchKey, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>) const'2",
        "1693822 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor (*)(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>), at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat> > >, at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)'2",
        "1678822 build/../aten/src/ATen/core/boxing/impl/WrapFunctionIntoRuntimeFunctor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor (*)(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>), at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat> > >, at::Tensor (c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)'2",
        "1673822 /data/users/test_user/repos/pytorch/build/aten/src/ATen/CPUType.cpp:at::CPUType::empty_memory_format(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)",
        "1669822 build/aten/src/ATen/CPUType.cpp:at::CPUType::empty_memory_format(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "1658822 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/native/TensorFactories.cpp:at::native::empty_cpu(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>)",
        "1433822 build/../aten/src/ATen/native/TensorFactories.cpp:at::native::empty_cpu(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "1112000 /data/users/test_user/repos/pytorch/build/aten/src/ATen/core/TensorMethods.cpp:at::Tensor::fill_(c10::Scalar) const",
        "1098500 build/../torch/csrc/autograd/generated/python_torch_functions.cpp:torch::autograd::THPVariable_ones(_object*, _object*, _object*) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "1062157 /tmp/build/80754af9/python_1599604603603/work/Objects/dictobject.c:PyDict_SetItem [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "1039000 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor::fill_(c10::Scalar) const",
        "1016000 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor& c10::Dispatcher::callWithDispatchKey<at::Tensor&, at::Tensor&, c10::Scalar>(c10::TypedOperatorHandle<at::Tensor& (at::Tensor&, c10::Scalar)> const&, c10::DispatchKey, at::Tensor&, c10::Scalar) const [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "939977 /tmp/build/80754af9/python_1599604603603/work/Objects/typeobject.c:subtype_dealloc [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "813000 build/../aten/src/ATen/core/boxing/KernelFunction_impl.h:at::Tensor& c10::Dispatcher::callWithDispatchKey<at::Tensor&, at::Tensor&, c10::Scalar>(c10::TypedOperatorHandle<at::Tensor& (at::Tensor&, c10::Scalar)> const&, c10::DispatchKey, at::Tensor&, c10::Scalar) const",
        "786000 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor& (*)(at::Tensor&, c10::Scalar), at::Tensor&, c10::guts::typelist::typelist<at::Tensor&, c10::Scalar> >, at::Tensor& (at::Tensor&, c10::Scalar)>::call(c10::OperatorKernel*, at::Tensor&, c10::Scalar)",
        "785000 build/../aten/src/ATen/core/boxing/impl/WrapFunctionIntoRuntimeFunctor.h:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoRuntimeFunctor_<at::Tensor& (*)(at::Tensor&, c10::Scalar), at::Tensor&, c10::guts::typelist::typelist<at::Tensor&, c10::Scalar> >, at::Tensor& (at::Tensor&, c10::Scalar)>::call(c10::OperatorKernel*, at::Tensor&, c10::Scalar)",
        "783000 /data/users/test_user/repos/pytorch/build/aten/src/ATen/TypeDefault.cpp:at::TypeDefault::fill__Scalar(at::Tensor&, c10::Scalar)",
        "767977 /data/users/test_user/repos/pytorch/build/../torch/csrc/autograd/python_variable.cpp:THPVariable_dealloc(THPVariable*)",
        "764977 build/../torch/csrc/autograd/python_variable.cpp:THPVariable_dealloc(THPVariable*) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "758000 build/aten/src/ATen/TypeDefault.cpp:at::TypeDefault::fill__Scalar(at::Tensor&, c10::Scalar) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "686977 build/../torch/csrc/autograd/python_variable.cpp:THPVariable_clear(THPVariable*) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "676822 /data/users/test_user/repos/pytorch/build/../c10/core/CPUAllocator.cpp:c10::DefaultCPUAllocator::allocate(unsigned long) const",
        "643822 build/../c10/core/CPUAllocator.cpp:c10::DefaultCPUAllocator::allocate(unsigned long) const [/data/users/test_user/repos/pytorch/torch/lib/libc10.so]",
        "643000 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/native/Fill.cpp:at::native::fill_(at::Tensor&, c10::Scalar)",
        "643000 build/../aten/src/ATen/native/Fill.cpp:at::native::fill_(at::Tensor&, c10::Scalar) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "642000 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/record_function.cpp:at::RecordFunction::RecordFunction(at::RecordScope)",
        "642000 build/../aten/src/ATen/native/Fill.cpp:at::native::fill_out(at::Tensor&, c10::Scalar) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "596977 build/../c10/util/intrusive_ptr.h:THPVariable_clear(THPVariable*)",
        "508822 build/../c10/core/CPUAllocator.cpp:c10::alloc_cpu(unsigned long) [/data/users/test_user/repos/pytorch/torch/lib/libc10.so]",
        "488000 build/../torch/csrc/utils/python_arg_parser.h:torch::autograd::THPVariable_ones(_object*, _object*, _object*)",
        "486000 build/../aten/src/ATen/record_function.cpp:at::RecordFunction::RecordFunction(at::RecordScope) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "461822 ???:posix_memalign [/usr/lib64/libc-2.28.so]",
        "434822 ???:_mid_memalign [/usr/lib64/libc-2.28.so]",
        "429000 /data/users/test_user/repos/pytorch/build/../torch/csrc/utils/python_arg_parser.cpp:torch::PythonArgParser::raw_parse(_object*, _object*, _object*, _object**)",
        "421000 build/../torch/csrc/utils/python_arg_parser.cpp:torch::PythonArgParser::raw_parse(_object*, _object*, _object*, _object**) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "408000 ???:__tls_get_addr [/usr/lib64/ld-2.28.so]",
        "389822 ???:_int_memalign [/usr/lib64/libc-2.28.so]",
        "388193 ???:_int_free [/usr/lib64/libc-2.28.so]",
        "386000 /data/users/test_user/repos/pytorch/build/aten/src/ATen/core/TensorMethods.cpp:at::Tensor::is_complex() const",
        "366000 /data/users/test_user/repos/pytorch/build/../aten/src/ATen/record_function.cpp:at::RecordFunction::~RecordFunction()",
        "361000 build/../torch/csrc/utils/python_arg_parser.cpp:torch::FunctionSignature::parse(_object*, _object*, _object*, _object**, bool) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "352977 ???:free [/usr/lib64/libc-2.28.so]",
        "350977 /data/users/test_user/repos/pytorch/build/../c10/core/TensorImpl.cpp:c10::TensorImpl::release_resources()",
        "315000 build/../aten/src/ATen/core/dispatch/Dispatcher.h:at::Tensor::is_complex() const",
        "302000 build/../aten/src/ATen/core/dispatch/Dispatcher.h:bool c10::Dispatcher::callWithDispatchKey<bool, at::Tensor const&>(c10::TypedOperatorHandle<bool (at::Tensor const&)> const&, c10::DispatchKey, at::Tensor const&) const [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "281000 /data/users/test_user/repos/pytorch/build/aten/src/ATen/core/TensorBody.h:at::Tensor at::detail::make_tensor<c10::TensorImpl, c10::intrusive_ptr<c10::StorageImpl, c10::detail::intrusive_target_default_null_type<c10::StorageImpl> >, c10::DispatchKey, caffe2::TypeMeta&>(c10::intrusive_ptr<c10::StorageImpl, c10::detail::intrusive_target_default_null_type<c10::StorageImpl> >&&, c10::DispatchKey&&, caffe2::TypeMeta&)",
        "273000 /data/users/test_user/repos/pytorch/build/../c10/core/impl/LocalDispatchKeySet.cpp:c10::impl::tls_local_dispatch_key_set()",
        "273000 build/../c10/core/impl/LocalDispatchKeySet.cpp:c10::impl::tls_local_dispatch_key_set() [/data/users/test_user/repos/pytorch/torch/lib/libc10.so]",
        "261000 /home/nwani/m3/conda-bld/compilers_linux-64_1560109574129/work/.build/x86_64-conda_cos6-linux-gnu/src/gcc/libstdc++-v3/libsupc++/del_ops.cc:operator delete(void*, unsigned long) [/home/test_user/miniconda3/envs/throwaway/lib/libstdc++.so.6.0.26]",
        "255000 /home/nwani/m3/conda-bld/compilers_linux-64_1560109574129/work/.build/x86_64-conda_cos6-linux-gnu/src/gcc/libstdc++-v3/libsupc++/del_op.cc:operator delete(void*) [/home/test_user/miniconda3/envs/throwaway/lib/libstdc++.so.6.0.26]",
        "236977 /usr/include/c++/8/bits/unique_ptr.h:c10::TensorImpl::release_resources()",
        "231000 /data/users/test_user/repos/pytorch/build/../c10/core/TensorImpl.h:c10::TensorImpl::~TensorImpl()",
        "228977 /data/users/test_user/repos/pytorch/build/../c10/core/CPUAllocator.cpp:c10::DefaultCPUAllocator::ReportAndDelete(void*)",
        "228977 build/../c10/core/CPUAllocator.cpp:c10::DefaultCPUAllocator::ReportAndDelete(void*) [/data/users/test_user/repos/pytorch/torch/lib/libc10.so]",
        "223663 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:PyObject_GetAttr [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "220000 /home/test_user/miniconda3/envs/throwaway/include/pybind11/pybind11.h:pybind11::gil_scoped_release::~gil_scoped_release() [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "209209 /tmp/build/80754af9/python_1599604603603/work/Python/ceval.c:PyEval_RestoreThread [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "205609 /tmp/build/80754af9/python_1599604603603/work/Objects/moduleobject.c:module_getattro [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "197500 /data/users/test_user/repos/pytorch/build/../torch/csrc/autograd/utils/wrap_outputs.h:torch::autograd::utils::wrap(at::Tensor)",
        "196000 build/aten/src/ATen/BackendSelectRegister.cpp:at::(anonymous namespace)::empty_memory_format(c10::ArrayRef<long>, c10::TensorOptions const&, std::optional<c10::MemoryFormat>) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "192000 build/../aten/src/ATen/record_function.cpp:at::RecordFunction::~RecordFunction() [/data/users/test_user/repos/pytorch/torch/lib/libtorch_cpu.so]",
        "192000 build/../c10/core/Device.h:c10::Device::validate() [/data/users/test_user/repos/pytorch/torch/lib/libc10.so]",
        "191567 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:PyObject_GenericGetAttr [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "190500 build/../torch/csrc/autograd/utils/wrap_outputs.h:torch::autograd::utils::wrap(at::Tensor) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "189561 /tmp/build/80754af9/python_1599604603603/work/Objects/object.c:_PyObject_GenericGetAttrWithDict [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "187000 build/../c10/core/TensorImpl.h:at::Tensor at::detail::make_tensor<c10::TensorImpl, c10::intrusive_ptr<c10::StorageImpl, c10::detail::intrusive_target_default_null_type<c10::StorageImpl> >, c10::DispatchKey, caffe2::TypeMeta&>(c10::intrusive_ptr<c10::StorageImpl, c10::detail::intrusive_target_default_null_type<c10::StorageImpl> >&&, c10::DispatchKey&&, caffe2::TypeMeta&)",
        "182816 /tmp/build/80754af9/python_1599604603603/work/Objects/dictobject.c:PyDict_GetItem [/home/test_user/miniconda3/envs/throwaway/bin/python3.6]",
        "181000 /data/users/test_user/repos/pytorch/build/../c10/core/TensorImpl.cpp:c10::TensorImpl::TensorImpl(c10::Storage&&, c10::DispatchKeySet, caffe2::TypeMeta const&)",
        "179500 /data/users/test_user/repos/pytorch/build/../torch/csrc/autograd/python_variable.cpp:THPVariable_Wrap(at::Tensor)",
        "178000 build/../c10/core/TensorImpl.cpp:c10::TensorImpl::TensorImpl(c10::Storage&&, c10::DispatchKeySet, caffe2::TypeMeta const&) [/data/users/test_user/repos/pytorch/torch/lib/libc10.so]",
        "173500 build/../torch/csrc/autograd/python_variable.cpp:THPVariable_Wrap(at::Tensor) [/data/users/test_user/repos/pytorch/torch/lib/libtorch_python.so]",
        "171000 build/../c10/core/TensorImpl.cpp:c10::TensorImpl::TensorImpl(c10::Storage&&, c10::DispatchKeySet, caffe2::TypeMeta const&, std::optional<c10::Device>) [/data/users/test_user/repos/pytorch/torch/lib/libc10.so]",
        "170175 ???:_int_malloc [/usr/lib64/libc-2.28.so]",
        "169000 /data/users/test_user/repos/pytorch/build/../c10/core/TensorImpl.h:c10::TensorImpl::empty_tensor_restride(c10::MemoryFormat)",
        "168000 /home/nwani/m3/conda-bld/compilers_linux-64_1560109574129/work/.build/x86_64-conda_cos6-linux-gnu/src/gcc/libstdc++-v3/libsupc++/new_op.cc:operator new(unsigned long) [/home/test_user/miniconda3/envs/throwaway/lib/libstdc++.so.6.0.26]",
        "167167 /tmp/build/80754af9/python_159960
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/benchmark_utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/benchmark_utils`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/benchmark_utils/callgrind_artifacts.json_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/benchmark_utils`):

- [`callgrind_artifacts.json_kw.md_docs.md`](./callgrind_artifacts.json_kw.md_docs.md)
- [`test_benchmark_utils.py_docs.md_docs.md`](./test_benchmark_utils.py_docs.md_docs.md)
- [`test_benchmark_utils.py_kw.md_docs.md`](./test_benchmark_utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `callgrind_artifacts.json_docs.md_docs.md`
- **Keyword Index**: `callgrind_artifacts.json_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
