# Documentation: `aten/src/ATen/test/vec_test_all_types.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/vec_test_all_types.cpp`
- **Size**: 105,082 bytes (102.62 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <ATen/test/vec_test_all_types.h>
#include <c10/util/irange.h>
namespace {
#if GTEST_HAS_TYPED_TEST
    template <typename T>
    class Memory : public ::testing::Test {};
    template <typename T>
    class Arithmetic : public ::testing::Test {};
    template <typename T>
    class Comparison : public ::testing::Test {};
    template <typename T>
    class Bitwise : public ::testing::Test {};
    template <typename T>
    class MinMax : public ::testing::Test {};
    template <typename T>
    class Nan : public ::testing::Test {};
    template <typename T>
    class Interleave : public ::testing::Test {};
    template <typename T>
    class SignManipulation : public ::testing::Test {};
    template <typename T>
    class SignManipulationHalfPrecision : public ::testing::Test {};
    template <typename T>
    class Rounding : public ::testing::Test {};
    template <typename T>
    class SqrtAndReciprocal : public ::testing::Test {};
    template <typename T>
    class SqrtAndReciprocalReal : public ::testing::Test {};
    template <typename T>
    class FractionAndRemainderReal : public ::testing::Test {};
    template <typename T>
    class Trigonometric : public ::testing::Test {};
    template <typename T>
    class ErrorFunctions : public ::testing::Test {};
    template <typename T>
    class Exponents : public ::testing::Test {};
    template <typename T>
    class Hyperbolic : public ::testing::Test {};
    template <typename T>
    class InverseTrigonometric : public ::testing::Test {};
    template <typename T>
    class InverseTrigonometricReal : public ::testing::Test {};
    template <typename T>
    class LGamma : public ::testing::Test {};
    template <typename T>
    class Logarithm : public ::testing::Test {};
    template <typename T>
    class LogarithmReals : public ::testing::Test {};
    template <typename T>
    class Pow : public ::testing::Test {};
    template <typename T>
    class RangeFactories : public ::testing::Test {};
    template <typename T>
    class BitwiseFloatsAdditional : public ::testing::Test {};
    template <typename T>
    class BitwiseFloatsAdditional2 : public ::testing::Test {};
    template <typename T>
    class RealTests : public ::testing::Test {};
    template <typename T>
    class ComplexTests : public ::testing::Test {};
    template <typename T>
    class QuantizationTests : public ::testing::Test {};
    template <typename T>
    class Quantization8BitTests : public ::testing::Test {};
    template <typename T>
    class Quantization8BitWithTailTests : public ::testing::Test {};
    template <typename T>
    class FunctionalTests : public ::testing::Test {};
    template <typename T>
    class FunctionalTestsReducedFloat : public ::testing::Test {};
    template <typename T>
    class InfiniteTests : public ::testing::Test {};
    template <typename T>
    class VecConvertTests : public ::testing::Test {};
    template <typename T>
    class VecConvertTestsReducedFloat : public ::testing::Test {};
    template <typename T>
    class VecMaskTests : public ::testing::Test {};
    using RealFloatTestedTypes = ::testing::Types<vfloat, vdouble>;
    using RealFloatReducedFloatTestedTypes = ::testing::Types<vfloat, vdouble, vBFloat16, vHalf>;
    using FloatTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl>;
    using ALLTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vlong, vint, vshort, vqint8, vquint8, vqint>;
    using QuantTestedTypes = ::testing::Types<vqint8, vquint8, vqint>;
    using Quantization8BitTestedTypes = ::testing::Types<vqint8, vquint8>;
#if (defined(CPU_CAPABILITY_AVX2) ||  defined(CPU_CAPABILITY_AVX512))  && !defined(_MSC_VER)
    using Quantization8BitWithTailTestedTypes =
        ::testing::Types<vqint8, vquint8>;
#endif
    using RealFloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vlong, vint, vshort>;
    using RealFloatIntReducedFloatTestedTypes = ::testing::Types<vfloat, vdouble, vlong, vint, vshort, vBFloat16, vHalf>;
    using FloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl, vlong, vint, vshort>;
    using ComplexTypes = ::testing::Types<vcomplex, vcomplexDbl>;
    using ReducedFloatTestedTypes = ::testing::Types<vBFloat16, vHalf>;
    TYPED_TEST_SUITE(Memory, ALLTestedTypes);
    TYPED_TEST_SUITE(Arithmetic, FloatIntTestedTypes);
    TYPED_TEST_SUITE(Comparison, RealFloatIntReducedFloatTestedTypes);
    TYPED_TEST_SUITE(Bitwise, FloatIntTestedTypes);
    TYPED_TEST_SUITE(MinMax, RealFloatIntTestedTypes);
    TYPED_TEST_SUITE(Nan, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Interleave, RealFloatIntTestedTypes);
    TYPED_TEST_SUITE(SignManipulation, FloatIntTestedTypes);
    TYPED_TEST_SUITE(SignManipulationHalfPrecision, ReducedFloatTestedTypes);
    TYPED_TEST_SUITE(Rounding, RealFloatTestedTypes);
    TYPED_TEST_SUITE(SqrtAndReciprocal, FloatTestedTypes);
    TYPED_TEST_SUITE(SqrtAndReciprocalReal, RealFloatTestedTypes);
    TYPED_TEST_SUITE(FractionAndRemainderReal, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Trigonometric, RealFloatTestedTypes);
    TYPED_TEST_SUITE(ErrorFunctions, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Exponents, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Hyperbolic, RealFloatTestedTypes);
    TYPED_TEST_SUITE(InverseTrigonometricReal, RealFloatTestedTypes);
    TYPED_TEST_SUITE(InverseTrigonometric, FloatTestedTypes);
    TYPED_TEST_SUITE(LGamma, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Logarithm, FloatTestedTypes);
    TYPED_TEST_SUITE(LogarithmReals, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Pow, RealFloatTestedTypes);
    TYPED_TEST_SUITE(RealTests, RealFloatTestedTypes);
    TYPED_TEST_SUITE(RangeFactories, FloatIntTestedTypes);
    TYPED_TEST_SUITE(BitwiseFloatsAdditional, RealFloatReducedFloatTestedTypes);
    TYPED_TEST_SUITE(BitwiseFloatsAdditional2, FloatTestedTypes);
    TYPED_TEST_SUITE(QuantizationTests, QuantTestedTypes);
    TYPED_TEST_SUITE(Quantization8BitTests, Quantization8BitTestedTypes);
    TYPED_TEST_SUITE(InfiniteTests, RealFloatTestedTypes);
#if (defined(CPU_CAPABILITY_AVX2) ||  defined(CPU_CAPABILITY_AVX512))  && !defined(_MSC_VER)
    TYPED_TEST_SUITE(
        Quantization8BitWithTailTests,
        Quantization8BitWithTailTestedTypes);
#endif
    TYPED_TEST_SUITE(FunctionalTests, RealFloatIntTestedTypes);
    TYPED_TEST_SUITE(FunctionalTestsReducedFloat, ReducedFloatTestedTypes);
    TYPED_TEST_SUITE(VecConvertTests, RealFloatIntTestedTypes);
    TYPED_TEST_SUITE(VecConvertTestsReducedFloat, ReducedFloatTestedTypes);
    TYPED_TEST_SUITE(VecMaskTests, RealFloatIntTestedTypes);
    TYPED_TEST(Memory, UnAlignedLoadStore) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr size_t b_size = vec::size() * sizeof(VT);
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN unsigned char ref_storage[128 * b_size];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN unsigned char storage[128 * b_size];
        auto seed = TestSeed();
        ValueGen<unsigned char> generator(seed);
        for (auto& x : ref_storage) {
            x = generator.get();
        }
        // test counted load stores
#if defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_ZVECTOR)
        for (int i = 1; i < 2 * vec::size(); i++) {
            vec v = vec::loadu(ref_storage, i);
            v.store(storage);
            size_t count = std::min(i * sizeof(VT), b_size);
            bool cmp = (std::memcmp(ref_storage, storage, count) == 0);
            ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nCount: " << i;
            if (::testing::Test::HasFailure()) {
                break;
            }
            // clear storage
            std::memset(storage, 0, b_size);
        }
#endif
        // testing unaligned load store
        for (size_t offset = 0; offset < b_size; offset += 1) {
            unsigned char* p1 = ref_storage + offset;
            unsigned char* p2 = storage + offset;
            for (; p1 + b_size <= std::end(ref_storage); p1 += b_size, p2 += b_size) {
                vec v = vec::loadu(p1);
                v.store(p2);
            }
            size_t written = p1 - ref_storage - offset;
            bool cmp = (std::memcmp(ref_storage + offset, storage + offset, written) == 0);
            ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nMismatch at unaligned offset: " << offset;
            if (::testing::Test::HasFailure()) {
                break;
            }
            // clear storage
            std::memset(storage, 0, sizeof storage);
        }
    }
    TYPED_TEST(SignManipulation, Absolute) {
        using vec = TypeParam;
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        test_unary<vec>(
            NAME_INFO(absolute), RESOLVE_OVERLOAD(local_abs),
            [](vec v) { return v.abs(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, checkRelativeErr),
            RESOLVE_OVERLOAD(filter_int_minimum));
    }
    TYPED_TEST(SignManipulation, Negate) {
        using vec = TypeParam;
        // negate overflows for minimum on int and long
        test_unary<vec>(
            NAME_INFO(negate), std::negate<ValueType<vec>>(),
            [](vec v) { return v.neg(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_int_minimum));
        test_unary<vec>(
            NAME_INFO(negate), std::negate<ValueType<vec>>(),
            [](vec v) { return -v; },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_int_minimum));
    }
    TYPED_TEST(SignManipulationHalfPrecision, AbsNegate) {
      typedef enum  {
        ABS,
        NEGATE
      } SignOpType;
      using vec = TypeParam;
      using VT = UholdType<TypeParam>;
      using RT = float; // reference
      float atol = 0.01f;
      float rtol = 0.01f;

      auto cmp = [&](RT ref, VT val) {
        return std::abs(ref - RT(val)) <= atol + rtol * std::abs(val);
      };

#define APPLY_FN_AND_STORE(VEC_TYPE)                            \
      [&](SignOpType op_type, VEC_TYPE& x_fp_vec, void *x_fp) { \
        if (op_type == SignOpType::NEGATE) {                    \
          x_fp_vec.neg().store(x_fp);                           \
        } else {                                                \
          x_fp_vec.abs().store(x_fp);                           \
        }                                                       \
      }

      auto apply_fn_and_store_ref = APPLY_FN_AND_STORE(vfloat);
      auto apply_fn_and_store_half = APPLY_FN_AND_STORE(vec);

      auto half_precision_ut = [&](SignOpType op_type) {
        constexpr auto N = vec::size();
        CACHE_ALIGN RT x_fp[N];
        CACHE_ALIGN VT x_hp[N];
        auto seed = TestSeed();
        ValueGen<RT> generator(RT(-1), RT(1), seed);
        for (const auto i : c10::irange(N)) {
            x_fp[i] = generator.get();
            x_hp[i] = VT(x_fp[i]);
        }
        auto x_fp_vec = vfloat::loadu(x_fp);
        apply_fn_and_store_ref(op_type, x_fp_vec, x_fp);
        x_fp_vec = vfloat::loadu(x_fp + vfloat::size());
        apply_fn_and_store_ref(op_type, x_fp_vec, x_fp + vfloat::size());

        auto x_hp_vec = vec::loadu(x_hp);
        apply_fn_and_store_half(op_type, x_hp_vec, x_hp);

        for (int64_t len = 0; len < N; len++) {
            ASSERT_TRUE(cmp(x_fp[len], x_hp[len])) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nabs/negate, Length: " << len << "; fp32: " << x_fp[len] << "; bf16/fp16: " << RT(x_hp[len]);
        }
      };

      half_precision_ut(SignOpType::ABS);
      half_precision_ut(SignOpType::NEGATE);
    }
    TYPED_TEST(Rounding, Round) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        UVT case1 = -658.5f;
        UVT exp1 = -658.f;
        UVT case2 = -657.5f;
        UVT exp2 = -658.f;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-1000, 1000}} })
            .addCustom({ {case1},exp1 })
            .addCustom({ {case2},exp2 })
            .setTrialCount(64000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(round),
            RESOLVE_OVERLOAD(at::native::round_impl),
            [](vec v) { return v.round(); },
            test_case);
    }
    TYPED_TEST(Rounding, Ceil) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(ceil),
            RESOLVE_OVERLOAD(std::ceil),
            [](vec v) { return v.ceil(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Rounding, Floor) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(floor),
            RESOLVE_OVERLOAD(std::floor),
            [](vec v) { return v.floor(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Rounding, Trunc) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(trunc),
            RESOLVE_OVERLOAD(std::trunc),
            [](vec v) { return v.trunc(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(SqrtAndReciprocal, Sqrt) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(sqrt),
            RESOLVE_OVERLOAD(local_sqrt),
            [](vec v) { return v.sqrt(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(SqrtAndReciprocalReal, RSqrt) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(rsqrt),
            rsqrt<ValueType<vec>>,
            [](vec v) { return v.rsqrt(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_zero));
    }
    TYPED_TEST(SqrtAndReciprocalReal, Reciprocal) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(reciprocal),
            reciprocal<ValueType<vec>>,
            [](vec v) { return v.reciprocal(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_zero));
    }
    TYPED_TEST(FractionAndRemainderReal, Frac) {
      using vec = TypeParam;
      test_unary<vec>(
          NAME_INFO(frac),
          RESOLVE_OVERLOAD(frac),
          [](vec v) { return v.frac(); },
          createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(FractionAndRemainderReal, Fmod) {
      using vec = TypeParam;
      test_binary<vec>(
          NAME_INFO(fmod),
          RESOLVE_OVERLOAD(std::fmod),
          [](const auto& v0, const auto& v1) { return vec(v0).fmod(v1); },
          createDefaultBinaryTestCase<vec>(TestSeed()),
          RESOLVE_OVERLOAD(filter_fmod));
    }
    TYPED_TEST(Trigonometric, Sin) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
            .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
            .setTrialCount(8000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(sin),
            RESOLVE_OVERLOAD(std::sin),
            [](vec v) { return v.sin(); },
            test_case);
    }
    TYPED_TEST(Trigonometric, Cos) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
            .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
            .setTrialCount(8000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(cos),
            RESOLVE_OVERLOAD(std::cos),
            [](vec v) { return v.cos(); },
            test_case);
    }
    TYPED_TEST(Trigonometric, Tan) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(tan),
            RESOLVE_OVERLOAD(std::tan),
            [](vec v) { return v.tan(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Hyperbolic, Tanh) {
        using vec = TypeParam;
// NOTE: Because SVE uses ACL logic, the precision changes, hence the adjusted tolerance.
#if defined(CPU_CAPABILITY_SVE)
        using UVT = UvalueType<vec>;
        UVT tolerance = getDefaultTolerance<UVT>();
        test_unary<vec>(
            NAME_INFO(tanH),
            RESOLVE_OVERLOAD(std::tanh),
            [](vec v) { return v.tanh(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), tolerance));
#else
        test_unary<vec>(
            NAME_INFO(tanH),
            RESOLVE_OVERLOAD(std::tanh),
            [](vec v) { return v.tanh(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
#endif
    }
    TYPED_TEST(Hyperbolic, Sinh) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(sinh),
            RESOLVE_OVERLOAD(std::sinh),
            [](vec v) { return v.sinh(); },
            test_case);
    }
    TYPED_TEST(Hyperbolic, Cosh) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(cosh),
            RESOLVE_OVERLOAD(std::cosh),
            [](vec v) { return v.cosh(); },
            test_case);
    }
    TYPED_TEST(InverseTrigonometric, Asin) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
            .setTrialCount(125536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(asin),
            RESOLVE_OVERLOAD(local_asin),
            [](vec v) { return v.asin(); },
            test_case);
    }
    TYPED_TEST(InverseTrigonometric, ACos) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
            .setTrialCount(125536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(acos),
            RESOLVE_OVERLOAD(local_acos),
            [](vec v) { return v.acos(); },
            test_case);
    }
    TYPED_TEST(InverseTrigonometric, ATan) {
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-100, 100}}, checkRelativeErr, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(atan),
            RESOLVE_OVERLOAD(std::atan),
            [](vec v) { return v.atan(); },
            test_case,
            RESOLVE_OVERLOAD(filter_zero));
    }
    TYPED_TEST(Logarithm, Log) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(log),
            RESOLVE_OVERLOAD(std::log),
            [](const vec& v) { return v.log(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(LogarithmReals, Log2) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(log2),
            RESOLVE_OVERLOAD(local_log2),
            [](const vec& v) { return v.log2(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Logarithm, Log10) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(log10),
            RESOLVE_OVERLOAD(std::log10),
            [](const vec& v) { return v.log10(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(LogarithmReals, Log1p) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-1, 1000}}, true, getDefaultTolerance<UVT>()})
            .addDomain(CheckWithinDomains<UVT>{ { {1000, 1.e+30}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(log1p),
            RESOLVE_OVERLOAD(std::log1p),
            [](const vec& v) { return v.log1p(); },
            test_case);
    }
    TYPED_TEST(Exponents, Exp) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(exp),
            RESOLVE_OVERLOAD(std::exp),
            [](const vec& v) { return v.exp(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Exponents, Expm1) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(expm1),
            RESOLVE_OVERLOAD(std::expm1),
            [](const vec& v) { return v.expm1(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(Exponents, ExpU20) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        using UVT = UvalueType<TypeParam>;

        // Explicit edge values
        VT v_too_small = VT(-100.0); // much less than -87.3
        VT exp_too_small = std::exp(v_too_small);
        VT v_neg_edge = VT(-0x1.5d5e2ap+6f);   // just at the edge
        VT exp_neg_edge = std::exp(v_neg_edge);
        VT v_zero = VT(0.0);         // middle, normal case
        VT exp_zero = std::exp(v_zero);
        VT v_pos_edge = VT(0x1.5d5e2ap+6f);    // just at the edge
        VT exp_pos_edge = std::exp(v_pos_edge);
        VT v_too_large = VT(100.0);  // much more than 87.3
        VT exp_too_large = std::exp(v_too_large);

        auto test_case = TestingCase<vec>::getBuilder()
            // Randoms in normal range, but the .addCustom() below guarantees we hit the special/fallback cases
            .addDomain(CheckWithinDomains<UVT>{{{-100, 100}}, false, getDefaultTolerance<UVT>()})
            .addCustom({ {v_too_small}, exp_too_small })
            .addCustom({ {v_neg_edge}, exp_neg_edge })
            .addCustom({ {v_zero}, exp_zero })
            .addCustom({ {v_pos_edge}, exp_pos_edge })
            .addCustom({ {v_too_large}, exp_too_large })
            .setTrialCount(65536)
            .setTestSeed(TestSeed());

        test_unary<vec>(
            NAME_INFO(exp_u20_edge_cases),
            RESOLVE_OVERLOAD(std::exp),
            [](const vec& v) { return v.exp_u20(); },
            test_case
        );
    }
    TYPED_TEST(ErrorFunctions, Erf) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(erf),
            RESOLVE_OVERLOAD(std::erf),
            [](const vec& v) { return v.erf(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(ErrorFunctions, Erfc) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(erfc),
            RESOLVE_OVERLOAD(std::erfc),
            [](const vec& v) { return v.erfc(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(ErrorFunctions, Erfinv) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(erfinv),
            RESOLVE_OVERLOAD(calc_erfinv),
            [](const vec& v) { return v.erfinv(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(Nan, IsNan) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT test_vals[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_vals[vec::size()];
        auto vals = 1 << (vec::size());
        for (const auto val : c10::irange(vals)) {
          for (int i = 0; i < vec::size(); ++i) {
            if (val & (1 << i)) {
              test_vals[i] = std::numeric_limits<VT>::quiet_NaN();
              // All bits are set to 1 if true, otherwise 0.
              // same rule as at::Vectorized<T>::binary_pred.
              std::memset(static_cast<void*>(&expected_vals[i]), 0xFF, sizeof(VT));
            } else {
              test_vals[i] = (VT)0.123;
              std::memset(static_cast<void*>(&expected_vals[i]), 0, sizeof(VT));
            }
          }
          vec actual = vec::loadu(test_vals).isnan();
          vec expected = vec::loadu(expected_vals);
          AssertVectorized<vec>(NAME_INFO(isnan), expected, actual).check();
        }
    }
    TEST(NanFloat16, IsNan) {
      for (unsigned int ii = 0; ii < 0xFFFF; ++ii) {
        c10::Half val(ii, c10::Half::from_bits());
        bool expected = std::isnan(val);
        CACHE_ALIGN c10::Half actual_vals[vHalf::size()];
        vHalf(val).isnan().store(actual_vals);
        for (auto actual_val : actual_vals) {
          EXPECT_EQ(expected, c10::bit_cast<uint16_t>(actual_val) != 0) << "fp16 isnan failure for bit pattern " << std::hex << ii << std::dec;
        }
      }
    }
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_BF16)
    TEST(NanBfloat16, IsNan) {
      for (unsigned int ii = 0; ii < 0xFFFF; ++ii) {
        c10::BFloat16 val(ii, c10::BFloat16::from_bits());
        bool expected = std::isnan(val);
        CACHE_ALIGN c10::BFloat16 actual_vals[at::vec::SVE256::Vectorized<c10::BFloat16>::size()];
        at::vec::SVE256::Vectorized<c10::BFloat16>(val).isnan().store(actual_vals);
        for (int jj = 0; jj < at::vec::SVE256::Vectorized<c10::BFloat16>::size(); ++jj) {
          EXPECT_EQ(expected, c10::bit_cast<uint16_t>(actual_vals[jj]) != 0) << "bf16 isnan failure for bit pattern " << std::hex << ii << std::dec;
        }
      }
    }
#endif
    TYPED_TEST(LGamma, LGamma) {
        using vec = TypeParam;
        using UVT = UvalueType<vec>;
        UVT tolerance = getDefaultTolerance<UVT>();
        // double: 2e+305  float: 4e+36 (https://sleef.org/purec.xhtml#eg)
        UVT maxCorrect = std::is_same_v<UVT, float> ? (UVT)4e+36 : (UVT)2e+305;
        TestingCase<vec> testCase = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)0}}, true, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)0, (UVT)1000 }}, true, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)1000, maxCorrect }}, true, tolerance})
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(lgamma),
            RESOLVE_OVERLOAD(std::lgamma),
            [](vec v) { return v.lgamma(); },
            testCase);
    }
    TYPED_TEST(InverseTrigonometricReal, ATan2) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(atan2),
            RESOLVE_OVERLOAD(std::atan2),
            [](const auto& v0, const auto& v1) {
              return vec(v0).atan2(v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Pow, Pow) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(pow),
            RESOLVE_OVERLOAD(std::pow),
            [](const auto& v0, const auto& v1) { return vec(v0).pow(v1); },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(RealTests, Hypot) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(hypot),
            RESOLVE_OVERLOAD(std::hypot),
            [](const auto& v0, const auto& v1) { return vec(v0).hypot(v1); },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(RealTests, NextAfter) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(nextafter),
            RESOLVE_OVERLOAD(std::nextafter),
            [](const auto& v0, const auto& v1) { return vec(v0).nextafter(v1); },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(Interleave, Interleave) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr auto N = vec::size() * 2LL;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT vals[N];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT interleaved[N];
        auto seed = TestSeed();
        ValueGen<VT> generator(seed);
        for (VT& v : vals) {
            v = generator.get();
        }
        copy_interleave(vals, interleaved);
        auto a = vec::loadu(vals);
        auto b = vec::loadu(vals + vec::size());
        auto cc = interleave2(a, b);
        AssertVectorized<vec>(NAME_INFO(Interleave FirstHalf), std::get<0>(cc), vec::loadu(interleaved)).check(true);
        AssertVectorized<vec>(NAME_INFO(Interleave SecondHalf), std::get<1>(cc), vec::loadu(interleaved + vec::size())).check(true);
    }
    TYPED_TEST(Interleave, DeInterleave) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr auto N = vec::size() * 2LL;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT vals[N];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT interleaved[N];
        auto seed = TestSeed();
        ValueGen<VT> generator(seed);
        for (VT& v : vals) {
            v = generator.get();
        }
        copy_interleave(vals, interleaved);
        // test interleaved with vals this time
        auto a = vec::loadu(interleaved);
        auto b = vec::loadu(interleaved + vec::size());
        auto cc = deinterleave2(a, b);
        AssertVectorized<vec>(NAME_INFO(DeInterleave FirstHalf), std::get<0>(cc), vec::loadu(vals)).check(true);
        AssertVectorized<vec>(NAME_INFO(DeInterleave SecondHalf), std::get<1>(cc), vec::loadu(vals + vec::size())).check(true);
    }
    TYPED_TEST(Arithmetic, Plus) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(plus),
            std::plus<VT>(),
            [](const auto& v0, const auto& v1) -> vec {
                return v0 + v1;
            },
            createDefaultBinaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_add_overflow));
    }
    TYPED_TEST(Arithmetic, Minus) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(minus),
            std::minus<VT>(),
            [](const auto& v0, const auto& v1) -> vec {
                return v0 - v1;
            },
            createDefaultBinaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_sub_overflow));
    }
    TYPED_TEST(Arithmetic, Multiplication) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(mult),
            RESOLVE_OVERLOAD(local_multiply),
            [](const auto& v0, const auto& v1) { return v0 * v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true),
            RESOLVE_OVERLOAD(filter_mult_overflow));
    }
    TYPED_TEST(Arithmetic, Division) {
        using vec = TypeParam;
        TestSeed seed;
        test_binary<vec>(
            NAME_INFO(division),
            RESOLVE_OVERLOAD(local_division),
            [](const auto& v0, const auto& v1) { return v0 / v1; },
            createDefaultBinaryTestCase<vec>(seed),
            RESOLVE_OVERLOAD(filter_div_ub));
    }
    TYPED_TEST(Bitwise, BitAnd) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(bit_and),
            RESOLVE_OVERLOAD(local_and),
            [](const auto& v0, const auto& v1) { return v0 & v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Bitwise, BitOr) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(bit_or),
            RESOLVE_OVERLOAD(local_or),
            [](const auto& v0, const auto& v1) { return v0 | v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Bitwise, BitXor) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(bit_xor),
            RESOLVE_OVERLOAD(local_xor),
            [](const auto& v0, const auto& v1) { return v0 ^ v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Equal) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(== ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::equal_to<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 == v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, NotEqual) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(!= ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::not_equal_to<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 != v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Greater) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(> ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::greater<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 > v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Less) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(< ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::less<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 < v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, GreaterEqual) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(>= ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::greater_equal<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 >= v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, LessEqual) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(<= ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::less_equal<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 <= v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(MinMax, Minimum) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(minimum),
            minimum<VT>,
            [](const auto& v0, const auto& v1) {
                return minimum(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, Maximum) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(maximum),
            maximum<VT>,
            [](const auto& v0, const auto& v1) {
                return maximum(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, ClampMin) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(clamp min),
            clamp_min<VT>,
            [](const auto& v0, const auto& v1) {
                return clamp_min(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, ClampMax) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(clamp max),
            clamp_max<VT>,
            [](const auto& v0, const auto& v1) {
                return clamp_max(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, Clamp) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_ternary<vec>(
            NAME_INFO(clamp), clamp<VT>,
            [](const vec& v0, const vec& v1, const vec& v2) {
                return clamp(v0, v1, v2);
            },
            createDefaultTernaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_clamp));
    }
    TYPED_TEST(MinMax, ClampVecN) {
        using VT = ValueType<TypeParam>;
        using vec = at::vec::VectorizedN<VT, 1>;
        test_ternary<vec>(
            NAME_INFO(clamp), clamp<VT>,
            [](const vec& v0, const vec& v1, const vec& v2) {
                return clamp(v0, v1, v2);
            },
            createDefaultTernaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_clamp));
    }
    TYPED_TEST(BitwiseFloatsAdditional, ZeroMask) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT test_vals[vec::size()];
        //all sets will be within 0  2^(n-1)
        auto power_sets = 1UL << (vec::size());
        for (const auto expected : c10::irange(power_sets)) {
            // generate test_val based on expected
            for (int i = 0; i < vec::size(); ++i)
            {
                if (expected & (1 << i)) {
                    test_vals[i] = (VT)0;
                }
                else {
                    test_vals[i] = (VT)0.897;
                }
            }
            int actual = vec::loadu(test_vals).zero_mask();
            ASSERT_EQ(expected, actual) << "Failure Details:\n"
                << std::hex << "Expected:\n#\t" << expected
                << "\nActual:\n#\t" << actual;
        }
    }
    TYPED_TEST(BitwiseFloatsAdditional, Convert) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        using IntVT = at::vec::int_same_size_t<VT>;

        // verify float to int
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT input1[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN IntVT expected_vals1[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN IntVT actual_vals1[vec::size()];
        for (int64_t i = 0; i < vec::size(); i++) {
            input1[i] = (VT)i * (VT)2.1 + (VT)0.5;
            expected_vals1[i] = static_cast<IntVT>(input1[i]);
        }
        at::vec::convert(input1, actual_vals1, vec::size());
        auto expected1 = VecType<IntVT>::loadu(expected_vals1);
        auto actual1 = VecType<IntVT>::loadu(actual_vals1);
        if (AssertVectorized<VecType<IntVT>>(NAME_INFO(test_convert_to_int), expected1, actual1).check()) {
          return;
        }

        // verify int to float
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN IntVT input2[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_vals2[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT actual_vals2[vec::size()];
        for (int64_t i = 0; i < vec::size(); i++) {
            input2[i] = (IntVT)i * (IntVT)2 + (IntVT)1;
            expected_vals2[i] = (VT)input2[i];
        }
        at::vec::convert(input2, actual_vals2, vec::size());
        auto expected2 = vec::loadu(expected_vals2);
        auto actual2 = vec::loadu(actual_vals2);
        AssertVectorized<vec>(NAME_INFO(test_convert_to_float), expected2, actual2).check();
    }
    TYPED_TEST(BitwiseFloatsAdditional, Fmadd) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;

        auto test_case = TestingCase<vec>::getBuilder()
          .addDomain(CheckWithinDomains<VT>{
              {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
              true, getDefaultTolerance<VT>()})
          .setTestSeed(TestSeed());

        test_ternary<vec>(
            NAME_INFO(fmadd), RESOLVE_OVERLOAD(local_fmadd),
            [](const vec& v0, const vec& v1, const vec& v2) {
                return at::vec::fmadd(v0, v1, v2);
            },
            test_case,
            RESOLVE_OVERLOAD(filter_fmadd));
    }
    TYPED_TEST(BitwiseFloatsAdditional, Fmsub) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;

        auto test_case = TestingCase<vec>::getBuilder()
          .addDomain(CheckWithinDomains<VT>{
              {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
              true, getDefaultTolerance<VT>()})
          .setTestSeed(TestSeed());

        test_ternary<vec>(
            NAME_INFO(fmsub), RESOLVE_OVERLOAD(local_fmsub),
            [](const vec& v0, const vec& v1, const vec& v2) {
                return at::vec::fmsub(v0, v1, v2);
            },
            test_case,
            RESOLVE_OVERLOAD(filter_fmadd));
    }
    TYPED_TEST(BitwiseFloatsAdditional, FmaddVecN) {
        using VT = ValueType<TypeParam>;
        using vec = at::vec::VectorizedN<VT, 1>;

        auto test_case = TestingCase<vec>::getBuilder()
          .addDomain(CheckWithinDomains<VT>{
              {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
              true, getDefaultTolerance<VT>()})
          .setTestSeed(TestSeed());

        test_ternary<vec>(
            NAME_INFO(fmadd), RESOLVE_OVERLOAD(local_fmadd),
            [](const vec& v0, const vec& v1, const vec& v2) {
                return at::vec::fmadd(v0, v1, v2);
            },
            test_case,
            RESOLVE_OVERLOAD(filter_fmadd));
    }
#if defined(CPU_CAPABILITY_NEON)
    TEST(BitwiseFloatsAdditional, HalfToFloatFmadd) {
        using vec = vhalf;
        using VT = ValueType<vec>;

        auto test_case = TestingCase<vec>::getBuilder()
          .addDomain(CheckWithinDomains<VT>{
              {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
              true, getDefaultTolerance<VT>()})
          .setTestSeed(TestSeed());

        test_ternary<vec>(
            NAME_INFO(half_to_float_fmadd), RESOLVE_OVERLOAD(local_fmadd),
            [](const vec& v0, const vec& v1, const vec& v2) {
              const auto [v2_float0, v2_float1] = convert_half_float(v2);
              const auto [result_float0, result_float1] = at::vec::fmadd(v0, v1, v2_float0, v2_float1);
              return convert_float_half(result_float0, result_float1);
            },
            test_case,
            RESOLVE_OVERLOAD(filter_fmadd));
    }
#endif
    template<typename vec, typename VT, int64_t mask>
    typename std::enable_if_t<(mask < 0 || mask> 255), void>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()])
    {
    }
    template<typename vec, typename VT, int64_t mask>
    typename std::enable_if_t<(mask >= 0 && mask <= 255), void>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()]) {
        // generate expected_val
        int64_t m = mask;
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = (m & 0x01) ? b[i] : a[i];
            m = m >> 1;
        }
        // test with blend
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::template blend<mask>(vec_a, vec_b);
        auto mask_str = std::string("\nblend mask: ") + std::to_string(mask);
        if (AssertVectorized<vec>(std::string(NAME_INFO(test_blend)) + mask_str, expected, actual).check()) return;
        test_blend<vec, VT, mask - 1>(expected_val, a, b);
    }
    template<typename vec, typename VT, int64_t idx, int64_t N>
    std::enable_if_t<(!is_complex<VT>::value && idx == N), bool>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
        using bit_rep = BitType<VT>;
        // generate expected_val
        for (int64_t i = 0; i < vec::size(); i++) {
            bit_rep hex_mask = 0;
            hex_mask=c10::bit_cast<bit_rep>(mask[i]);
            expected_val[i] = (hex_mask & 0x01) ? b[i] : a[i];
        }
        // test with blendv
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto vec_m = vec::loadu(mask);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::blendv(vec_a, vec_b, vec_m);
        auto mask_str = std::string("\nblendv mask: ");
        for (int64_t i = 0; i < vec::size(); i++) {
            mask_str += std::to_string(mask[i]) + " ";
        }
        if (AssertVectorized<vec>(std::string(NAME_INFO(test_blendv)) + mask_str, expected, actual).check()) {
            return false;
        }
        return true;
    }
    template<typename vec, typename VT, int64_t idx, int64_t N>
    std::enable_if_t<(!is_complex<VT>::value && idx != N), bool>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
        // shuffle mask and do blendv test
        VT m = mask[idx];
        if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
        if (m != (VT)0) {
          mask[idx] = (VT)0;
        }
        else {
          uint64_t hex_mask = 0xFFFFFFFFFFFFFFFF;
          std::memcpy(&mask[idx], &hex_mask, sizeof(VT));
        }
        if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
        mask[idx] = m;
        return true;
    }
    template<typename T, int N>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    void blend_init(T(&a)[N], T(&b)[N]) {
        a[0] = (T)1.0;
        b[0] = a[0] + (T)N;
        for (const auto i : c10::irange(1, N)) {
            a[i] = a[i - 1] + (T)(1.0);
            b[i] = b[i - 1] + (T)(1.0);
        }
    }
    TYPED_TEST(BitwiseFloatsAdditional, Blendv) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT a[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT b[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT mask[vec::size()] = {0};
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];
        blend_init(a, b);
        test_blendv<vec, VT, 0, vec::size()>(expected_val, a, b, mask);
    }
    TYPED_TEST(BitwiseFloatsAdditional2, Blend) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT a[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT b[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];
        blend_init(a, b);
        constexpr int64_t power_sets = 1LL << (vec::size());
        test_blend<vec, VT, power_sets - 1>(expected_val, a, b);
    }
    template<typename vec, typename VT>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    void test_set(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], int64_t count){
        if (count < 0) return;
        //generate expected_val
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = (i < count) ? b[i] : a[i];
        }
        // test with set
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::set(vec_a, vec_b, count);

        auto count_str = std::string("\ncount: ") + std::to_string(count);
        if (AssertVectorized<vec>(std::string(NAME_INFO(test_set)) + count_str, expected, actual).check()) {
          return;
        }
        test_set<vec, VT>(expected_val, a, b, (count == 0 ? -1 : cou
```



## High-Level Overview


This C++ file contains approximately 37 class(es)/struct(s) and 100 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `Memory`, `Arithmetic`, `Comparison`, `Bitwise`, `MinMax`, `Nan`, `Interleave`, `SignManipulation`, `SignManipulationHalfPrecision`, `Rounding`, `SqrtAndReciprocal`, `SqrtAndReciprocalReal`, `FractionAndRemainderReal`, `Trigonometric`, `ErrorFunctions`, `Exponents`, `Hyperbolic`, `InverseTrigonometric`, `InverseTrigonometricReal`, `LGamma`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/test/vec_test_all_types.h`
- `c10/util/irange.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python aten/src/ATen/test/vec_test_all_types.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `vec_test_all_types.cpp_docs.md`
- **Keyword Index**: `vec_test_all_types.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
