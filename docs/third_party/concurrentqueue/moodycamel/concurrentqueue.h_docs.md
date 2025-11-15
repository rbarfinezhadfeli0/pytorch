# Documentation: `third_party/concurrentqueue/moodycamel/concurrentqueue.h`

## File Metadata

- **Path**: `third_party/concurrentqueue/moodycamel/concurrentqueue.h`
- **Size**: 152,819 bytes (149.24 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Provides a C++11 implementation of a multi-producer, multi-consumer lock-free queue.
// An overview, including benchmark results, is provided here:
//     http://moodycamel.com/blog/2014/a-fast-general-purpose-lock-free-queue-for-c++
// The full design is also described in excruciating detail at:
//    http://moodycamel.com/blog/2014/detailed-design-of-a-lock-free-queue

// Simplified BSD license:
// Copyright (c) 2013-2020, Cameron Desrochers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Also dual-licensed under the Boost Software License (see LICENSE.md)

#pragma once

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
// Disable -Wconversion warnings (spuriously triggered when Traits::size_t and
// Traits::index_t are set to < 32 bits, causing integer promotion, causing warnings
// upon assigning any computed values)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"

#ifdef MCDBGQ_USE_RELACY
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
#endif
#endif

#if defined(_MSC_VER) && (!defined(_HAS_CXX17) || !_HAS_CXX17)
// VS2019 with /W4 warns about constant conditional expressions but unless /std=c++17 or higher
// does not support `if constexpr`, so we have no choice but to simply disable the warning
#pragma warning(push)
#pragma warning(disable: 4127)  // conditional expression is constant
#endif

#if defined(__APPLE__)
#include "TargetConditionals.h"
#endif

#ifdef MCDBGQ_USE_RELACY
#include "relacy/relacy_std.hpp"
#include "relacy_shims.h"
// We only use malloc/free anyway, and the delete macro messes up `= delete` method declarations.
// We'll override the default trait malloc ourselves without a macro.
#undef new
#undef delete
#undef malloc
#undef free
#else
#include <atomic>		// Requires C++11. Sorry VS2010.
#include <cassert>
#endif
#include <cstddef>              // for max_align_t
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <limits>
#include <climits>		// for CHAR_BIT
#include <array>
#include <thread>		// partly for __WINPTHREADS_VERSION if on MinGW-w64 w/ POSIX threading
#include <mutex>        // used for thread exit synchronization

// Platform-specific definitions of a numeric thread ID type and an invalid value
namespace moodycamel { namespace details {
	template<typename thread_id_t> struct thread_id_converter {
		typedef thread_id_t thread_id_numeric_size_t;
		typedef thread_id_t thread_id_hash_t;
		static thread_id_hash_t prehash(thread_id_t const& x) { return x; }
	};
} }
#if defined(MCDBGQ_USE_RELACY)
namespace moodycamel { namespace details {
	typedef std::uint32_t thread_id_t;
	static const thread_id_t invalid_thread_id  = 0xFFFFFFFFU;
	static const thread_id_t invalid_thread_id2 = 0xFFFFFFFEU;
	static inline thread_id_t thread_id() { return rl::thread_index(); }
} }
#elif defined(_WIN32) || defined(__WINDOWS__) || defined(__WIN32__)
// No sense pulling in windows.h in a header, we'll manually declare the function
// we use and rely on backwards-compatibility for this not to break
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
namespace moodycamel { namespace details {
	static_assert(sizeof(unsigned long) == sizeof(std::uint32_t), "Expected size of unsigned long to be 32 bits on Windows");
	typedef std::uint32_t thread_id_t;
	static const thread_id_t invalid_thread_id  = 0;			// See http://blogs.msdn.com/b/oldnewthing/archive/2004/02/23/78395.aspx
	static const thread_id_t invalid_thread_id2 = 0xFFFFFFFFU;	// Not technically guaranteed to be invalid, but is never used in practice. Note that all Win32 thread IDs are presently multiples of 4.
	static inline thread_id_t thread_id() { return static_cast<thread_id_t>(::GetCurrentThreadId()); }
} }
#elif defined(__arm__) || defined(_M_ARM) || defined(__aarch64__) || (defined(__APPLE__) && TARGET_OS_IPHONE) || defined(__MVS__) || defined(MOODYCAMEL_NO_THREAD_LOCAL)
namespace moodycamel { namespace details {
	static_assert(sizeof(std::thread::id) == 4 || sizeof(std::thread::id) == 8, "std::thread::id is expected to be either 4 or 8 bytes");
	
	typedef std::thread::id thread_id_t;
	static const thread_id_t invalid_thread_id;         // Default ctor creates invalid ID

	// Note we don't define a invalid_thread_id2 since std::thread::id doesn't have one; it's
	// only used if MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED is defined anyway, which it won't
	// be.
	static inline thread_id_t thread_id() { return std::this_thread::get_id(); }

	template<std::size_t> struct thread_id_size { };
	template<> struct thread_id_size<4> { typedef std::uint32_t numeric_t; };
	template<> struct thread_id_size<8> { typedef std::uint64_t numeric_t; };

	template<> struct thread_id_converter<thread_id_t> {
		typedef thread_id_size<sizeof(thread_id_t)>::numeric_t thread_id_numeric_size_t;
#ifndef __APPLE__
		typedef std::size_t thread_id_hash_t;
#else
		typedef thread_id_numeric_size_t thread_id_hash_t;
#endif

		static thread_id_hash_t prehash(thread_id_t const& x)
		{
#ifndef __APPLE__
			return std::hash<std::thread::id>()(x);
#else
			return *reinterpret_cast<thread_id_hash_t const*>(&x);
#endif
		}
	};
} }
#else
// Use a nice trick from this answer: http://stackoverflow.com/a/8438730/21475
// In order to get a numeric thread ID in a platform-independent way, we use a thread-local
// static variable's address as a thread identifier :-)
#if defined(__GNUC__) || defined(__INTEL_COMPILER)
#define MOODYCAMEL_THREADLOCAL __thread
#elif defined(_MSC_VER)
#define MOODYCAMEL_THREADLOCAL __declspec(thread)
#else
// Assume C++11 compliant compiler
#define MOODYCAMEL_THREADLOCAL thread_local
#endif
namespace moodycamel { namespace details {
	typedef std::uintptr_t thread_id_t;
	static const thread_id_t invalid_thread_id  = 0;		// Address can't be nullptr
	static const thread_id_t invalid_thread_id2 = 1;		// Member accesses off a null pointer are also generally invalid. Plus it's not aligned.
	inline thread_id_t thread_id() { static MOODYCAMEL_THREADLOCAL int x; return reinterpret_cast<thread_id_t>(&x); }
} }
#endif

// Constexpr if
#ifndef MOODYCAMEL_CONSTEXPR_IF
#if (defined(_MSC_VER) && defined(_HAS_CXX17) && _HAS_CXX17) || __cplusplus > 201402L
#define MOODYCAMEL_CONSTEXPR_IF if constexpr
#define MOODYCAMEL_MAYBE_UNUSED [[maybe_unused]]
#else
#define MOODYCAMEL_CONSTEXPR_IF if
#define MOODYCAMEL_MAYBE_UNUSED
#endif
#endif

// Exceptions
#ifndef MOODYCAMEL_EXCEPTIONS_ENABLED
#if (defined(_MSC_VER) && defined(_CPPUNWIND)) || (defined(__GNUC__) && defined(__EXCEPTIONS)) || (!defined(_MSC_VER) && !defined(__GNUC__))
#define MOODYCAMEL_EXCEPTIONS_ENABLED
#endif
#endif
#ifdef MOODYCAMEL_EXCEPTIONS_ENABLED
#define MOODYCAMEL_TRY try
#define MOODYCAMEL_CATCH(...) catch(__VA_ARGS__)
#define MOODYCAMEL_RETHROW throw
#define MOODYCAMEL_THROW(expr) throw (expr)
#else
#define MOODYCAMEL_TRY MOODYCAMEL_CONSTEXPR_IF (true)
#define MOODYCAMEL_CATCH(...) else MOODYCAMEL_CONSTEXPR_IF (false)
#define MOODYCAMEL_RETHROW
#define MOODYCAMEL_THROW(expr)
#endif

#ifndef MOODYCAMEL_NOEXCEPT
#if !defined(MOODYCAMEL_EXCEPTIONS_ENABLED)
#define MOODYCAMEL_NOEXCEPT
#define MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr) true
#define MOODYCAMEL_NOEXCEPT_ASSIGN(type, valueType, expr) true
#elif defined(_MSC_VER) && defined(_NOEXCEPT) && _MSC_VER < 1800
// VS2012's std::is_nothrow_[move_]constructible is broken and returns true when it shouldn't :-(
// We have to assume *all* non-trivial constructors may throw on VS2012!
#define MOODYCAMEL_NOEXCEPT _NOEXCEPT
#define MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr) (std::is_rvalue_reference<valueType>::value && std::is_move_constructible<type>::value ? std::is_trivially_move_constructible<type>::value : std::is_trivially_copy_constructible<type>::value)
#define MOODYCAMEL_NOEXCEPT_ASSIGN(type, valueType, expr) ((std::is_rvalue_reference<valueType>::value && std::is_move_assignable<type>::value ? std::is_trivially_move_assignable<type>::value || std::is_nothrow_move_assignable<type>::value : std::is_trivially_copy_assignable<type>::value || std::is_nothrow_copy_assignable<type>::value) && MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr))
#elif defined(_MSC_VER) && defined(_NOEXCEPT) && _MSC_VER < 1900
#define MOODYCAMEL_NOEXCEPT _NOEXCEPT
#define MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr) (std::is_rvalue_reference<valueType>::value && std::is_move_constructible<type>::value ? std::is_trivially_move_constructible<type>::value || std::is_nothrow_move_constructible<type>::value : std::is_trivially_copy_constructible<type>::value || std::is_nothrow_copy_constructible<type>::value)
#define MOODYCAMEL_NOEXCEPT_ASSIGN(type, valueType, expr) ((std::is_rvalue_reference<valueType>::value && std::is_move_assignable<type>::value ? std::is_trivially_move_assignable<type>::value || std::is_nothrow_move_assignable<type>::value : std::is_trivially_copy_assignable<type>::value || std::is_nothrow_copy_assignable<type>::value) && MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr))
#else
#define MOODYCAMEL_NOEXCEPT noexcept
#define MOODYCAMEL_NOEXCEPT_CTOR(type, valueType, expr) noexcept(expr)
#define MOODYCAMEL_NOEXCEPT_ASSIGN(type, valueType, expr) noexcept(expr)
#endif
#endif

#ifndef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
#ifdef MCDBGQ_USE_RELACY
#define MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
#else
// VS2013 doesn't support `thread_local`, and MinGW-w64 w/ POSIX threading has a crippling bug: http://sourceforge.net/p/mingw-w64/bugs/445
// g++ <=4.7 doesn't support thread_local either.
// Finally, iOS/ARM doesn't have support for it either, and g++/ARM allows it to compile but it's unconfirmed to actually work
#if (!defined(_MSC_VER) || _MSC_VER >= 1900) && (!defined(__MINGW32__) && !defined(__MINGW64__) || !defined(__WINPTHREADS_VERSION)) && (!defined(__GNUC__) || __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)) && (!defined(__APPLE__) || !TARGET_OS_IPHONE) && !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(__MVS__)
// Assume `thread_local` is fully supported in all other C++11 compilers/platforms
#define MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED    // tentatively enabled for now; years ago several users report having problems with it on
#endif
#endif
#endif

// VS2012 doesn't support deleted functions. 
// In this case, we declare the function normally but don't define it. A link error will be generated if the function is called.
#ifndef MOODYCAMEL_DELETE_FUNCTION
#if defined(_MSC_VER) && _MSC_VER < 1800
#define MOODYCAMEL_DELETE_FUNCTION
#else
#define MOODYCAMEL_DELETE_FUNCTION = delete
#endif
#endif

namespace moodycamel { namespace details {
#ifndef MOODYCAMEL_ALIGNAS
// VS2013 doesn't support alignas or alignof, and align() requires a constant literal
#if defined(_MSC_VER) && _MSC_VER <= 1800
#define MOODYCAMEL_ALIGNAS(alignment) __declspec(align(alignment))
#define MOODYCAMEL_ALIGNOF(obj) __alignof(obj)
#define MOODYCAMEL_ALIGNED_TYPE_LIKE(T, obj) typename details::Vs2013Aligned<std::alignment_of<obj>::value, T>::type
	template<int Align, typename T> struct Vs2013Aligned { };  // default, unsupported alignment
	template<typename T> struct Vs2013Aligned<1, T> { typedef __declspec(align(1)) T type; };
	template<typename T> struct Vs2013Aligned<2, T> { typedef __declspec(align(2)) T type; };
	template<typename T> struct Vs2013Aligned<4, T> { typedef __declspec(align(4)) T type; };
	template<typename T> struct Vs2013Aligned<8, T> { typedef __declspec(align(8)) T type; };
	template<typename T> struct Vs2013Aligned<16, T> { typedef __declspec(align(16)) T type; };
	template<typename T> struct Vs2013Aligned<32, T> { typedef __declspec(align(32)) T type; };
	template<typename T> struct Vs2013Aligned<64, T> { typedef __declspec(align(64)) T type; };
	template<typename T> struct Vs2013Aligned<128, T> { typedef __declspec(align(128)) T type; };
	template<typename T> struct Vs2013Aligned<256, T> { typedef __declspec(align(256)) T type; };
#else
	template<typename T> struct identity { typedef T type; };
#define MOODYCAMEL_ALIGNAS(alignment) alignas(alignment)
#define MOODYCAMEL_ALIGNOF(obj) alignof(obj)
#define MOODYCAMEL_ALIGNED_TYPE_LIKE(T, obj) alignas(alignof(obj)) typename details::identity<T>::type
#endif
#endif
} }


// TSAN can false report races in lock-free code.  To enable TSAN to be used from projects that use this one,
// we can apply per-function compile-time suppression.
// See https://clang.llvm.org/docs/ThreadSanitizer.html#has-feature-thread-sanitizer
#define MOODYCAMEL_NO_TSAN
#if defined(__has_feature)
 #if __has_feature(thread_sanitizer)
  #undef MOODYCAMEL_NO_TSAN
  #define MOODYCAMEL_NO_TSAN __attribute__((no_sanitize("thread")))
 #endif // TSAN
#endif // TSAN

// Compiler-specific likely/unlikely hints
namespace moodycamel { namespace details {
#if defined(__GNUC__)
	static inline bool (likely)(bool x) { return __builtin_expect((x), true); }
	static inline bool (unlikely)(bool x) { return __builtin_expect((x), false); }
#else
	static inline bool (likely)(bool x) { return x; }
	static inline bool (unlikely)(bool x) { return x; }
#endif
} }

#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
#include "internal/concurrentqueue_internal_debug.h"
#endif

namespace moodycamel {
namespace details {
	template<typename T>
	struct const_numeric_max {
		static_assert(std::is_integral<T>::value, "const_numeric_max can only be used with integers");
		static const T value = std::numeric_limits<T>::is_signed
			? (static_cast<T>(1) << (sizeof(T) * CHAR_BIT - 1)) - static_cast<T>(1)
			: static_cast<T>(-1);
	};

#if defined(__GLIBCXX__)
	typedef ::max_align_t std_max_align_t;      // libstdc++ forgot to add it to std:: for a while
#else
	typedef std::max_align_t std_max_align_t;   // Others (e.g. MSVC) insist it can *only* be accessed via std::
#endif

	// Some platforms have incorrectly set max_align_t to a type with <8 bytes alignment even while supporting
	// 8-byte aligned scalar values (*cough* 32-bit iOS). Work around this with our own union. See issue #64.
	typedef union {
		std_max_align_t x;
		long long y;
		void* z;
	} max_align_t;
}

// Default traits for the ConcurrentQueue. To change some of the
// traits without re-implementing all of them, inherit from this
// struct and shadow the declarations you wish to be different;
// since the traits are used as a template type parameter, the
// shadowed declarations will be used where defined, and the defaults
// otherwise.
struct ConcurrentQueueDefaultTraits
{
	// General-purpose size type. std::size_t is strongly recommended.
	typedef std::size_t size_t;
	
	// The type used for the enqueue and dequeue indices. Must be at least as
	// large as size_t. Should be significantly larger than the number of elements
	// you expect to hold at once, especially if you have a high turnover rate;
	// for example, on 32-bit x86, if you expect to have over a hundred million
	// elements or pump several million elements through your queue in a very
	// short space of time, using a 32-bit type *may* trigger a race condition.
	// A 64-bit int type is recommended in that case, and in practice will
	// prevent a race condition no matter the usage of the queue. Note that
	// whether the queue is lock-free with a 64-int type depends on the whether
	// std::atomic<std::uint64_t> is lock-free, which is platform-specific.
	typedef std::size_t index_t;
	
	// Internally, all elements are enqueued and dequeued from multi-element
	// blocks; this is the smallest controllable unit. If you expect few elements
	// but many producers, a smaller block size should be favoured. For few producers
	// and/or many elements, a larger block size is preferred. A sane default
	// is provided. Must be a power of 2.
	static const size_t BLOCK_SIZE = 32;
	
	// For explicit producers (i.e. when using a producer token), the block is
	// checked for being empty by iterating through a list of flags, one per element.
	// For large block sizes, this is too inefficient, and switching to an atomic
	// counter-based approach is faster. The switch is made for block sizes strictly
	// larger than this threshold.
	static const size_t EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD = 32;
	
	// How many full blocks can be expected for a single explicit producer? This should
	// reflect that number's maximum for optimal performance. Must be a power of 2.
	static const size_t EXPLICIT_INITIAL_INDEX_SIZE = 32;
	
	// How many full blocks can be expected for a single implicit producer? This should
	// reflect that number's maximum for optimal performance. Must be a power of 2.
	static const size_t IMPLICIT_INITIAL_INDEX_SIZE = 32;
	
	// The initial size of the hash table mapping thread IDs to implicit producers.
	// Note that the hash is resized every time it becomes half full.
	// Must be a power of two, and either 0 or at least 1. If 0, implicit production
	// (using the enqueue methods without an explicit producer token) is disabled.
	static const size_t INITIAL_IMPLICIT_PRODUCER_HASH_SIZE = 32;
	
	// Controls the number of items that an explicit consumer (i.e. one with a token)
	// must consume before it causes all consumers to rotate and move on to the next
	// internal queue.
	static const std::uint32_t EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE = 256;
	
	// The maximum number of elements (inclusive) that can be enqueued to a sub-queue.
	// Enqueue operations that would cause this limit to be surpassed will fail. Note
	// that this limit is enforced at the block level (for performance reasons), i.e.
	// it's rounded up to the nearest block size.
	static const size_t MAX_SUBQUEUE_SIZE = details::const_numeric_max<size_t>::value;

	// The number of times to spin before sleeping when waiting on a semaphore.
	// Recommended values are on the order of 1000-10000 unless the number of
	// consumer threads exceeds the number of idle cores (in which case try 0-100).
	// Only affects instances of the BlockingConcurrentQueue.
	static const int MAX_SEMA_SPINS = 10000;

	// Whether to recycle dynamically-allocated blocks into an internal free list or
	// not. If false, only pre-allocated blocks (controlled by the constructor
	// arguments) will be recycled, and all others will be `free`d back to the heap.
	// Note that blocks consumed by explicit producers are only freed on destruction
	// of the queue (not following destruction of the token) regardless of this trait.
	static const bool RECYCLE_ALLOCATED_BLOCKS = false;

	
#ifndef MCDBGQ_USE_RELACY
	// Memory allocation can be customized if needed.
	// malloc should return nullptr on failure, and handle alignment like std::malloc.
#if defined(malloc) || defined(free)
	// Gah, this is 2015, stop defining macros that break standard code already!
	// Work around malloc/free being special macros:
	static inline void* WORKAROUND_malloc(size_t size) { return malloc(size); }
	static inline void WORKAROUND_free(void* ptr) { return free(ptr); }
	static inline void* (malloc)(size_t size) { return WORKAROUND_malloc(size); }
	static inline void (free)(void* ptr) { return WORKAROUND_free(ptr); }
#else
	static inline void* malloc(size_t size) { return std::malloc(size); }
	static inline void free(void* ptr) { return std::free(ptr); }
#endif
#else
	// Debug versions when running under the Relacy race detector (ignore
	// these in user code)
	static inline void* malloc(size_t size) { return rl::rl_malloc(size, $); }
	static inline void free(void* ptr) { return rl::rl_free(ptr, $); }
#endif
};


// When producing or consuming many elements, the most efficient way is to:
//    1) Use one of the bulk-operation methods of the queue with a token
//    2) Failing that, use the bulk-operation methods without a token
//    3) Failing that, create a token and use that with the single-item methods
//    4) Failing that, use the single-parameter methods of the queue
// Having said that, don't create tokens willy-nilly -- ideally there should be
// a maximum of one token per thread (of each kind).
struct ProducerToken;
struct ConsumerToken;

template<typename T, typename Traits> class ConcurrentQueue;
template<typename T, typename Traits> class BlockingConcurrentQueue;
class ConcurrentQueueTests;


namespace details
{
	struct ConcurrentQueueProducerTypelessBase
	{
		ConcurrentQueueProducerTypelessBase* next;
		std::atomic<bool> inactive;
		ProducerToken* token;
		
		ConcurrentQueueProducerTypelessBase()
			: next(nullptr), inactive(false), token(nullptr)
		{
		}
	};
	
	template<bool use32> struct _hash_32_or_64 {
		static inline std::uint32_t hash(std::uint32_t h)
		{
			// MurmurHash3 finalizer -- see https://code.google.com/p/smhasher/source/browse/trunk/MurmurHash3.cpp
			// Since the thread ID is already unique, all we really want to do is propagate that
			// uniqueness evenly across all the bits, so that we can use a subset of the bits while
			// reducing collisions significantly
			h ^= h >> 16;
			h *= 0x85ebca6b;
			h ^= h >> 13;
			h *= 0xc2b2ae35;
			return h ^ (h >> 16);
		}
	};
	template<> struct _hash_32_or_64<1> {
		static inline std::uint64_t hash(std::uint64_t h)
		{
			h ^= h >> 33;
			h *= 0xff51afd7ed558ccd;
			h ^= h >> 33;
			h *= 0xc4ceb9fe1a85ec53;
			return h ^ (h >> 33);
		}
	};
	template<std::size_t size> struct hash_32_or_64 : public _hash_32_or_64<(size > 4)> {  };
	
	static inline size_t hash_thread_id(thread_id_t id)
	{
		static_assert(sizeof(thread_id_t) <= 8, "Expected a platform where thread IDs are at most 64-bit values");
		return static_cast<size_t>(hash_32_or_64<sizeof(thread_id_converter<thread_id_t>::thread_id_hash_t)>::hash(
			thread_id_converter<thread_id_t>::prehash(id)));
	}
	
	template<typename T>
	static inline bool circular_less_than(T a, T b)
	{
		static_assert(std::is_integral<T>::value && !std::numeric_limits<T>::is_signed, "circular_less_than is intended to be used only with unsigned integer types");
		return static_cast<T>(a - b) > static_cast<T>(static_cast<T>(1) << (static_cast<T>(sizeof(T) * CHAR_BIT - 1)));
		// Note: extra parens around rhs of operator<< is MSVC bug: https://developercommunity2.visualstudio.com/t/C4554-triggers-when-both-lhs-and-rhs-is/10034931
		//       silencing the bug requires #pragma warning(disable: 4554) around the calling code and has no effect when done here.
	}
	
	template<typename U>
	static inline char* align_for(char* ptr)
	{
		const std::size_t alignment = std::alignment_of<U>::value;
		return ptr + (alignment - (reinterpret_cast<std::uintptr_t>(ptr) % alignment)) % alignment;
	}

	template<typename T>
	static inline T ceil_to_pow_2(T x)
	{
		static_assert(std::is_integral<T>::value && !std::numeric_limits<T>::is_signed, "ceil_to_pow_2 is intended to be used only with unsigned integer types");

		// Adapted from http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		for (std::size_t i = 1; i < sizeof(T); i <<= 1) {
			x |= x >> (i << 3);
		}
		++x;
		return x;
	}
	
	template<typename T>
	static inline void swap_relaxed(std::atomic<T>& left, std::atomic<T>& right)
	{
		T temp = left.load(std::memory_order_relaxed);
		left.store(right.load(std::memory_order_relaxed), std::memory_order_relaxed);
		right.store(temp, std::memory_order_relaxed);
	}
	
	template<typename T>
	static inline T const& nomove(T const& x)
	{
		return x;
	}
	
	template<bool Enable>
	struct nomove_if
	{
		template<typename T>
		static inline T const& eval(T const& x)
		{
			return x;
		}
	};
	
	template<>
	struct nomove_if<false>
	{
		template<typename U>
		static inline auto eval(U&& x)
			-> decltype(std::forward<U>(x))
		{
			return std::forward<U>(x);
		}
	};
	
	template<typename It>
	static inline auto deref_noexcept(It& it) MOODYCAMEL_NOEXCEPT -> decltype(*it)
	{
		return *it;
	}
	
#if defined(__clang__) || !defined(__GNUC__) || __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
	template<typename T> struct is_trivially_destructible : std::is_trivially_destructible<T> { };
#else
	template<typename T> struct is_trivially_destructible : std::has_trivial_destructor<T> { };
#endif
	
#ifdef MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED
#ifdef MCDBGQ_USE_RELACY
	typedef RelacyThreadExitListener ThreadExitListener;
	typedef RelacyThreadExitNotifier ThreadExitNotifier;
#else
	class ThreadExitNotifier;

	struct ThreadExitListener
	{
		typedef void (*callback_t)(void*);
		callback_t callback;
		void* userData;
		
		ThreadExitListener* next;		// reserved for use by the ThreadExitNotifier
		ThreadExitNotifier* chain;		// reserved for use by the ThreadExitNotifier
	};

	class ThreadExitNotifier
	{
	public:
		static void subscribe(ThreadExitListener* listener)
		{
			auto& tlsInst = instance();
			std::lock_guard<std::mutex> guard(mutex());
			listener->next = tlsInst.tail;
			listener->chain = &tlsInst;
			tlsInst.tail = listener;
		}
		
		static void unsubscribe(ThreadExitListener* listener)
		{
			std::lock_guard<std::mutex> guard(mutex());
			if (!listener->chain) {
				return;  // race with ~ThreadExitNotifier
			}
			auto& tlsInst = *listener->chain;
			listener->chain = nullptr;
			ThreadExitListener** prev = &tlsInst.tail;
			for (auto ptr = tlsInst.tail; ptr != nullptr; ptr = ptr->next) {
				if (ptr == listener) {
					*prev = ptr->next;
					break;
				}
				prev = &ptr->next;
			}
		}
		
	private:
		ThreadExitNotifier() : tail(nullptr) { }
		ThreadExitNotifier(ThreadExitNotifier const&) MOODYCAMEL_DELETE_FUNCTION;
		ThreadExitNotifier& operator=(ThreadExitNotifier const&) MOODYCAMEL_DELETE_FUNCTION;
		
		~ThreadExitNotifier()
		{
			// This thread is about to exit, let everyone know!
			assert(this == &instance() && "If this assert fails, you likely have a buggy compiler! Change the preprocessor conditions such that MOODYCAMEL_CPP11_THREAD_LOCAL_SUPPORTED is no longer defined.");
			std::lock_guard<std::mutex> guard(mutex());
			for (auto ptr = tail; ptr != nullptr; ptr = ptr->next) {
				ptr->chain = nullptr;
				ptr->callback(ptr->userData);
			}
		}
		
		// Thread-local
		static inline ThreadExitNotifier& instance()
		{
			static thread_local ThreadExitNotifier notifier;
			return notifier;
		}

		static inline std::mutex& mutex()
		{
			// Must be static because the ThreadExitNotifier could be destroyed while unsubscribe is called
			static std::mutex mutex;
			return mutex;
		}
		
	private:
		ThreadExitListener* tail;
	};
#endif
#endif
	
	template<typename T> struct static_is_lock_free_num { enum { value = 0 }; };
	template<> struct static_is_lock_free_num<signed char> { enum { value = ATOMIC_CHAR_LOCK_FREE }; };
	template<> struct static_is_lock_free_num<short> { enum { value = ATOMIC_SHORT_LOCK_FREE }; };
	template<> struct static_is_lock_free_num<int> { enum { value = ATOMIC_INT_LOCK_FREE }; };
	template<> struct static_is_lock_free_num<long> { enum { value = ATOMIC_LONG_LOCK_FREE }; };
	template<> struct static_is_lock_free_num<long long> { enum { value = ATOMIC_LLONG_LOCK_FREE }; };
	template<typename T> struct static_is_lock_free : static_is_lock_free_num<typename std::make_signed<T>::type> {  };
	template<> struct static_is_lock_free<bool> { enum { value = ATOMIC_BOOL_LOCK_FREE }; };
	template<typename U> struct static_is_lock_free<U*> { enum { value = ATOMIC_POINTER_LOCK_FREE }; };
}


struct ProducerToken
{
	template<typename T, typename Traits>
	explicit ProducerToken(ConcurrentQueue<T, Traits>& queue);
	
	template<typename T, typename Traits>
	explicit ProducerToken(BlockingConcurrentQueue<T, Traits>& queue);
	
	ProducerToken(ProducerToken&& other) MOODYCAMEL_NOEXCEPT
		: producer(other.producer)
	{
		other.producer = nullptr;
		if (producer != nullptr) {
			producer->token = this;
		}
	}
	
	inline ProducerToken& operator=(ProducerToken&& other) MOODYCAMEL_NOEXCEPT
	{
		swap(other);
		return *this;
	}
	
	void swap(ProducerToken& other) MOODYCAMEL_NOEXCEPT
	{
		std::swap(producer, other.producer);
		if (producer != nullptr) {
			producer->token = this;
		}
		if (other.producer != nullptr) {
			other.producer->token = &other;
		}
	}
	
	// A token is always valid unless:
	//     1) Memory allocation failed during construction
	//     2) It was moved via the move constructor
	//        (Note: assignment does a swap, leaving both potentially valid)
	//     3) The associated queue was destroyed
	// Note that if valid() returns true, that only indicates
	// that the token is valid for use with a specific queue,
	// but not which one; that's up to the user to track.
	inline bool valid() const { return producer != nullptr; }
	
	~ProducerToken()
	{
		if (producer != nullptr) {
			producer->token = nullptr;
			producer->inactive.store(true, std::memory_order_release);
		}
	}
	
	// Disable copying and assignment
	ProducerToken(ProducerToken const&) MOODYCAMEL_DELETE_FUNCTION;
	ProducerToken& operator=(ProducerToken const&) MOODYCAMEL_DELETE_FUNCTION;
	
private:
	template<typename T, typename Traits> friend class ConcurrentQueue;
	friend class ConcurrentQueueTests;
	
protected:
	details::ConcurrentQueueProducerTypelessBase* producer;
};


struct ConsumerToken
{
	template<typename T, typename Traits>
	explicit ConsumerToken(ConcurrentQueue<T, Traits>& q);
	
	template<typename T, typename Traits>
	explicit ConsumerToken(BlockingConcurrentQueue<T, Traits>& q);
	
	ConsumerToken(ConsumerToken&& other) MOODYCAMEL_NOEXCEPT
		: initialOffset(other.initialOffset), lastKnownGlobalOffset(other.lastKnownGlobalOffset), itemsConsumedFromCurrent(other.itemsConsumedFromCurrent), currentProducer(other.currentProducer), desiredProducer(other.desiredProducer)
	{
	}
	
	inline ConsumerToken& operator=(ConsumerToken&& other) MOODYCAMEL_NOEXCEPT
	{
		swap(other);
		return *this;
	}
	
	void swap(ConsumerToken& other) MOODYCAMEL_NOEXCEPT
	{
		std::swap(initialOffset, other.initialOffset);
		std::swap(lastKnownGlobalOffset, other.lastKnownGlobalOffset);
		std::swap(itemsConsumedFromCurrent, other.itemsConsumedFromCurrent);
		std::swap(currentProducer, other.currentProducer);
		std::swap(desiredProducer, other.desiredProducer);
	}
	
	// Disable copying and assignment
	ConsumerToken(ConsumerToken const&) MOODYCAMEL_DELETE_FUNCTION;
	ConsumerToken& operator=(ConsumerToken const&) MOODYCAMEL_DELETE_FUNCTION;

private:
	template<typename T, typename Traits> friend class ConcurrentQueue;
	friend class ConcurrentQueueTests;
	
private: // but shared with ConcurrentQueue
	std::uint32_t initialOffset;
	std::uint32_t lastKnownGlobalOffset;
	std::uint32_t itemsConsumedFromCurrent;
	details::ConcurrentQueueProducerTypelessBase* currentProducer;
	details::ConcurrentQueueProducerTypelessBase* desiredProducer;
};

// Need to forward-declare this swap because it's in a namespace.
// See http://stackoverflow.com/questions/4492062/why-does-a-c-friend-class-need-a-forward-declaration-only-in-other-namespaces
template<typename T, typename Traits>
inline void swap(typename ConcurrentQueue<T, Traits>::ImplicitProducerKVP& a, typename ConcurrentQueue<T, Traits>::ImplicitProducerKVP& b) MOODYCAMEL_NOEXCEPT;


template<typename T, typename Traits = ConcurrentQueueDefaultTraits>
class ConcurrentQueue
{
public:
	typedef ::moodycamel::ProducerToken producer_token_t;
	typedef ::moodycamel::ConsumerToken consumer_token_t;
	
	typedef typename Traits::index_t index_t;
	typedef typename Traits::size_t size_t;
	
	static const size_t BLOCK_SIZE = static_cast<size_t>(Traits::BLOCK_SIZE);
	static const size_t EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD = static_cast<size_t>(Traits::EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD);
	static const size_t EXPLICIT_INITIAL_INDEX_SIZE = static_cast<size_t>(Traits::EXPLICIT_INITIAL_INDEX_SIZE);
	static const size_t IMPLICIT_INITIAL_INDEX_SIZE = static_cast<size_t>(Traits::IMPLICIT_INITIAL_INDEX_SIZE);
	static const size_t INITIAL_IMPLICIT_PRODUCER_HASH_SIZE = static_cast<size_t>(Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE);
	static const std::uint32_t EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE = static_cast<std::uint32_t>(Traits::EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4307)		// + integral constant overflow (that's what the ternary expression is for!)
#pragma warning(disable: 4309)		// static_cast: Truncation of constant value
#endif
	static const size_t MAX_SUBQUEUE_SIZE = (details::const_numeric_max<size_t>::value - static_cast<size_t>(Traits::MAX_SUBQUEUE_SIZE) < BLOCK_SIZE) ? details::const_numeric_max<size_t>::value : ((static_cast<size_t>(Traits::MAX_SUBQUEUE_SIZE) + (BLOCK_SIZE - 1)) / BLOCK_SIZE * BLOCK_SIZE);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

	static_assert(!std::numeric_limits<size_t>::is_signed && std::is_integral<size_t>::value, "Traits::size_t must be an unsigned integral type");
	static_assert(!std::numeric_limits<index_t>::is_signed && std::is_integral<index_t>::value, "Traits::index_t must be an unsigned integral type");
	static_assert(sizeof(index_t) >= sizeof(size_t), "Traits::index_t must be at least as wide as Traits::size_t");
	static_assert((BLOCK_SIZE > 1) && !(BLOCK_SIZE & (BLOCK_SIZE - 1)), "Traits::BLOCK_SIZE must be a power of 2 (and at least 2)");
	static_assert((EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD > 1) && !(EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD & (EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD - 1)), "Traits::EXPLICIT_BLOCK_EMPTY_COUNTER_THRESHOLD must be a power of 2 (and greater than 1)");
	static_assert((EXPLICIT_INITIAL_INDEX_SIZE > 1) && !(EXPLICIT_INITIAL_INDEX_SIZE & (EXPLICIT_INITIAL_INDEX_SIZE - 1)), "Traits::EXPLICIT_INITIAL_INDEX_SIZE must be a power of 2 (and greater than 1)");
	static_assert((IMPLICIT_INITIAL_INDEX_SIZE > 1) && !(IMPLICIT_INITIAL_INDEX_SIZE & (IMPLICIT_INITIAL_INDEX_SIZE - 1)), "Traits::IMPLICIT_INITIAL_INDEX_SIZE must be a power of 2 (and greater than 1)");
	static_assert((INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) || !(INITIAL_IMPLICIT_PRODUCER_HASH_SIZE & (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE - 1)), "Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE must be a power of 2");
	static_assert(INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0 || INITIAL_IMPLICIT_PRODUCER_HASH_SIZE >= 1, "Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE must be at least 1 (or 0 to disable implicit enqueueing)");

public:
	// Creates a queue with at least `capacity` element slots; note that the
	// actual number of elements that can be inserted without additional memory
	// allocation depends on the number of producers and the block size (e.g. if
	// the block size is equal to `capacity`, only a single block will be allocated
	// up-front, which means only a single producer will be able to enqueue elements
	// without an extra allocation -- blocks aren't shared between producers).
	// This method is not thread safe -- it is up to the user to ensure that the
	// queue is fully constructed before it starts being used by other threads (this
	// includes making the memory effects of construction visible, possibly with a
	// memory barrier).
	explicit ConcurrentQueue(size_t capacity = 32 * BLOCK_SIZE)
		: producerListTail(nullptr),
		producerCount(0),
		initialBlockPoolIndex(0),
		nextExplicitConsumerId(0),
		globalExplicitConsumerOffset(0)
	{
		implicitProducerHashResizeInProgress.clear(std::memory_order_relaxed);
		populate_initial_implicit_producer_hash();
		populate_initial_block_list(capacity / BLOCK_SIZE + ((capacity & (BLOCK_SIZE - 1)) == 0 ? 0 : 1));
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
		// Track all the producers using a fully-resolved typed list for
		// each kind; this makes it possible to debug them starting from
		// the root queue object (otherwise wacky casts are needed that
		// don't compile in the debugger's expression evaluator).
		explicitProducers.store(nullptr, std::memory_order_relaxed);
		implicitProducers.store(nullptr, std::memory_order_relaxed);
#endif
	}
	
	// Computes the correct amount of pre-allocated blocks for you based
	// on the minimum number of elements you want available at any given
	// time, and the maximum concurrent number of each type of producer.
	ConcurrentQueue(size_t minCapacity, size_t maxExplicitProducers, size_t maxImplicitProducers)
		: producerListTail(nullptr),
		producerCount(0),
		initialBlockPoolIndex(0),
		nextExplicitConsumerId(0),
		globalExplicitConsumerOffset(0)
	{
		implicitProducerHashResizeInProgress.clear(std::memory_order_relaxed);
		populate_initial_implicit_producer_hash();
		size_t blocks = (((minCapacity + BLOCK_SIZE - 1) / BLOCK_SIZE) - 1) * (maxExplicitProducers + 1) + 2 * (maxExplicitProducers + maxImplicitProducers);
		populate_initial_block_list(blocks);
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
		explicitProducers.store(nullptr, std::memory_order_relaxed);
		implicitProducers.store(nullptr, std::memory_order_relaxed);
#endif
	}
	
	// Note: The queue should not be accessed concurrently while it's
	// being deleted. It's up to the user to synchronize this.
	// This method is not thread safe.
	~ConcurrentQueue()
	{
		// Destroy producers
		auto ptr = producerListTail.load(std::memory_order_relaxed);
		while (ptr != nullptr) {
			auto next = ptr->next_prod();
			if (ptr->token != nullptr) {
				ptr->token->producer = nullptr;
			}
			destroy(ptr);
			ptr = next;
		}
		
		// Destroy implicit producer hash tables
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE != 0) {
			auto hash = implicitProducerHash.load(std::memory_order_relaxed);
			while (hash != nullptr) {
				auto prev = hash->prev;
				if (prev != nullptr) {		// The last hash is part of this object and was not allocated dynamically
					for (size_t i = 0; i != hash->capacity; ++i) {
						hash->entries[i].~ImplicitProducerKVP();
					}
					hash->~ImplicitProducerHash();
					(Traits::free)(hash);
				}
				hash = prev;
			}
		}
		
		// Destroy global free list
		auto block = freeList.head_unsafe();
		while (block != nullptr) {
			auto next = block->freeListNext.load(std::memory_order_relaxed);
			if (block->dynamicallyAllocated) {
				destroy(block);
			}
			block = next;
		}
		
		// Destroy initial free list
		destroy_array(initialBlockPool, initialBlockPoolSize);
	}

	// Disable copying and copy assignment
	ConcurrentQueue(ConcurrentQueue const&) MOODYCAMEL_DELETE_FUNCTION;
	ConcurrentQueue& operator=(ConcurrentQueue const&) MOODYCAMEL_DELETE_FUNCTION;
	
	// Moving is supported, but note that it is *not* a thread-safe operation.
	// Nobody can use the queue while it's being moved, and the memory effects
	// of that move must be propagated to other threads before they can use it.
	// Note: When a queue is moved, its tokens are still valid but can only be
	// used with the destination queue (i.e. semantically they are moved along
	// with the queue itself).
	ConcurrentQueue(ConcurrentQueue&& other) MOODYCAMEL_NOEXCEPT
		: producerListTail(other.producerListTail.load(std::memory_order_relaxed)),
		producerCount(other.producerCount.load(std::memory_order_relaxed)),
		initialBlockPoolIndex(other.initialBlockPoolIndex.load(std::memory_order_relaxed)),
		initialBlockPool(other.initialBlockPool),
		initialBlockPoolSize(other.initialBlockPoolSize),
		freeList(std::move(other.freeList)),
		nextExplicitConsumerId(other.nextExplicitConsumerId.load(std::memory_order_relaxed)),
		globalExplicitConsumerOffset(other.globalExplicitConsumerOffset.load(std::memory_order_relaxed))
	{
		// Move the other one into this, and leave the other one as an empty queue
		implicitProducerHashResizeInProgress.clear(std::memory_order_relaxed);
		populate_initial_implicit_producer_hash();
		swap_implicit_producer_hashes(other);
		
		other.producerListTail.store(nullptr, std::memory_order_relaxed);
		other.producerCount.store(0, std::memory_order_relaxed);
		other.nextExplicitConsumerId.store(0, std::memory_order_relaxed);
		other.globalExplicitConsumerOffset.store(0, std::memory_order_relaxed);
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
		explicitProducers.store(other.explicitProducers.load(std::memory_order_relaxed), std::memory_order_relaxed);
		other.explicitProducers.store(nullptr, std::memory_order_relaxed);
		implicitProducers.store(other.implicitProducers.load(std::memory_order_relaxed), std::memory_order_relaxed);
		other.implicitProducers.store(nullptr, std::memory_order_relaxed);
#endif
		
		other.initialBlockPoolIndex.store(0, std::memory_order_relaxed);
		other.initialBlockPoolSize = 0;
		other.initialBlockPool = nullptr;
		
		reown_producers();
	}
	
	inline ConcurrentQueue& operator=(ConcurrentQueue&& other) MOODYCAMEL_NOEXCEPT
	{
		return swap_internal(other);
	}
	
	// Swaps this queue's state with the other's. Not thread-safe.
	// Swapping two queues does not invalidate their tokens, however
	// the tokens that were created for one queue must be used with
	// only the swapped queue (i.e. the tokens are tied to the
	// queue's movable state, not the object itself).
	inline void swap(ConcurrentQueue& other) MOODYCAMEL_NOEXCEPT
	{
		swap_internal(other);
	}
	
private:
	ConcurrentQueue& swap_internal(ConcurrentQueue& other)
	{
		if (this == &other) {
			return *this;
		}
		
		details::swap_relaxed(producerListTail, other.producerListTail);
		details::swap_relaxed(producerCount, other.producerCount);
		details::swap_relaxed(initialBlockPoolIndex, other.initialBlockPoolIndex);
		std::swap(initialBlockPool, other.initialBlockPool);
		std::swap(initialBlockPoolSize, other.initialBlockPoolSize);
		freeList.swap(other.freeList);
		details::swap_relaxed(nextExplicitConsumerId, other.nextExplicitConsumerId);
		details::swap_relaxed(globalExplicitConsumerOffset, other.globalExplicitConsumerOffset);
		
		swap_implicit_producer_hashes(other);
		
		reown_producers();
		other.reown_producers();
		
#ifdef MOODYCAMEL_QUEUE_INTERNAL_DEBUG
		details::swap_relaxed(explicitProducers, other.explicitProducers);
		details::swap_relaxed(implicitProducers, other.implicitProducers);
#endif
		
		return *this;
	}
	
public:
	// Enqueues a single item (by copying it).
	// Allocates memory if required. Only fails if memory allocation fails (or implicit
	// production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0,
	// or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Thread-safe.
	inline bool enqueue(T const& item)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue<CanAlloc>(item);
	}
	
	// Enqueues a single item (by moving it, if possible).
	// Allocates memory if required. Only fails if memory allocation fails (or implicit
	// production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0,
	// or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Thread-safe.
	inline bool enqueue(T&& item)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue<CanAlloc>(std::move(item));
	}
	
	// Enqueues a single item (by copying it) using an explicit producer token.
	// Allocates memory if required. Only fails if memory allocation fails (or
	// Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Thread-safe.
	inline bool enqueue(producer_token_t const& token, T const& item)
	{
		return inner_enqueue<CanAlloc>(token, item);
	}
	
	// Enqueues a single item (by moving it, if possible) using an explicit producer token.
	// Allocates memory if required. Only fails if memory allocation fails (or
	// Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Thread-safe.
	inline bool enqueue(producer_token_t const& token, T&& item)
	{
		return inner_enqueue<CanAlloc>(token, std::move(item));
	}
	
	// Enqueues several items.
	// Allocates memory if required. Only fails if memory allocation fails (or
	// implicit production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE
	// is 0, or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Note: Use std::make_move_iterator if the elements should be moved instead of copied.
	// Thread-safe.
	template<typename It>
	bool enqueue_bulk(It itemFirst, size_t count)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue_bulk<CanAlloc>(itemFirst, count);
	}
	
	// Enqueues several items using an explicit producer token.
	// Allocates memory if required. Only fails if memory allocation fails
	// (or Traits::MAX_SUBQUEUE_SIZE has been defined and would be surpassed).
	// Note: Use std::make_move_iterator if the elements should be moved
	// instead of copied.
	// Thread-safe.
	template<typename It>
	bool enqueue_bulk(producer_token_t const& token, It itemFirst, size_t count)
	{
		return inner_enqueue_bulk<CanAlloc>(token, itemFirst, count);
	}
	
	// Enqueues a single item (by copying it).
	// Does not allocate memory. Fails if not enough room to enqueue (or implicit
	// production is disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE
	// is 0).
	// Thread-safe.
	inline bool try_enqueue(T const& item)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue<CannotAlloc>(item);
	}
	
	// Enqueues a single item (by moving it, if possible).
	// Does not allocate memory (except for one-time implicit producer).
	// Fails if not enough room to enqueue (or implicit production is
	// disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0).
	// Thread-safe.
	inline bool try_enqueue(T&& item)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue<CannotAlloc>(std::move(item));
	}
	
	// Enqueues a single item (by copying it) using an explicit producer token.
	// Does not allocate memory. Fails if not enough room to enqueue.
	// Thread-safe.
	inline bool try_enqueue(producer_token_t const& token, T const& item)
	{
		return inner_enqueue<CannotAlloc>(token, item);
	}
	
	// Enqueues a single item (by moving it, if possible) using an explicit producer token.
	// Does not allocate memory. Fails if not enough room to enqueue.
	// Thread-safe.
	inline bool try_enqueue(producer_token_t const& token, T&& item)
	{
		return inner_enqueue<CannotAlloc>(token, std::move(item));
	}
	
	// Enqueues several items.
	// Does not allocate memory (except for one-time implicit producer).
	// Fails if not enough room to enqueue (or implicit production is
	// disabled because Traits::INITIAL_IMPLICIT_PRODUCER_HASH_SIZE is 0).
	// Note: Use std::make_move_iterator if the elements should be moved
	// instead of copied.
	// Thread-safe.
	template<typename It>
	bool try_enqueue_bulk(It itemFirst, size_t count)
	{
		MOODYCAMEL_CONSTEXPR_IF (INITIAL_IMPLICIT_PRODUCER_HASH_SIZE == 0) return false;
		else return inner_enqueue_bulk<CannotAlloc>(itemFirst, count);
	}
	
	// Enqueues several items using an explicit producer token.
	// Does not allocate memory. Fails if not enough room to enqueue.
	// Note: Use std::make_move_iterator if the elements should be moved
	// instead of copied.
	// Thread-safe.
	template<typename It>
	bool try_enqueue_bulk(producer_token_t const& token, It itemFirst, size_t count)
	{
		return inner_enqueue_bulk<CannotAlloc>(token, itemFirst, count);
	}
	
	
	
	// Attempts to dequeue from the queue.
	// Returns false if all producer streams appeared empty at the time they
	// were checked (so, the queue is likely but not guaranteed to be empty).
	// Never allocates. Thread-safe.
	template<typename U>
	bool try_dequeue(U& item)
	{
		// Instead of simply trying each producer in turn (which could cause needless contention on the first
		// producer), we score them heuristically.
		size_t nonEmptyCount = 0;
		ProducerBase* best = nullptr;
		size_t bestSize = 0;
		for (auto ptr = producerListTail.load(std::memory_order_acquire); nonEmptyCount < 3 && ptr != nullptr; ptr = ptr->next_prod()) {
			auto size = ptr->size_approx();
			if (size > 0) {
				if (size > bestSize) {
					bestSize = size;
					best = ptr;
				}
				++nonEmptyCount;
			}
		}
		
		// If there was at least one non-empty queue but it appears empty at the time
		// we try to dequeue from it, we need to make sure every queue's been tried
		if (nonEmptyCount > 0) {
			if ((details::likely)(best->dequeue(item))) {
				return true;
			}
			for (auto ptr = producerListTail.load(std::memory_order_acquire); ptr != nullptr; ptr = ptr->next_prod()) {
				if (ptr != best && ptr->dequeue(item)) {
					return true;
				}
			}
		}
		return false;
	}
	
	// Attempts to dequeue from the queue.
	// Returns false if all producer streams appeared empty at the time they
	// were checked (so, the queue is likely but not guaranteed to be empty).
	// This differs from the try_dequeue(item) method in that this one does
	// not attempt to reduce contention by interleaving the order that producer
	// strea
```



## High-Level Overview


This C++ file contains approximately 12 class(es)/struct(s) and 332 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `details`, `moodycamel`

**Classes/Structs**: `thread_id_converter`, `thread_id_size`, `thread_id_size`, `thread_id_size`, `thread_id_converter`, `Vs2013Aligned`, `Vs2013Aligned`, `Vs2013Aligned`, `Vs2013Aligned`, `Vs2013Aligned`, `Vs2013Aligned`, `Vs2013Aligned`, `Vs2013Aligned`, `Vs2013Aligned`, `Vs2013Aligned`, `identity`, `const_numeric_max`, `and`, `ConcurrentQueueDefaultTraits`, `ProducerToken`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `third_party/concurrentqueue/moodycamel`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file includes:

- `TargetConditionals.h`
- `relacy/relacy_std.hpp`
- `relacy_shims.h`
- `atomic`
- `cassert`
- `cstddef`
- `cstdint`
- `cstdlib`
- `type_traits`
- `algorithm`
- `utility`
- `limits`
- `climits`
- `array`
- `thread`
- `mutex`
- `internal/concurrentqueue_internal_debug.h`


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`third_party/concurrentqueue/moodycamel`):

- [`LICENSE.md_docs.md`](./LICENSE.md_docs.md)
- [`lightweightsemaphore.h_docs.md`](./lightweightsemaphore.h_docs.md)


## Cross-References

- **File Documentation**: `concurrentqueue.h_docs.md`
- **Keyword Index**: `concurrentqueue.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
