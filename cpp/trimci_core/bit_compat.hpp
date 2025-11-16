#pragma once
// Portable bit operations to support MSVC and GCC/Clang
// Avoids dependency on cstdint/uint64_t by using unsigned long long.

#if defined(_MSC_VER)
  #include <intrin.h>
  inline int popcount64(unsigned long long x) {
  #if defined(_M_X64) || defined(__x86_64__)
    return static_cast<int>(__popcnt64(static_cast<unsigned __int64>(x)));
  #else
    unsigned int lo = static_cast<unsigned int>(x & 0xFFFFFFFFull);
    unsigned int hi = static_cast<unsigned int>(x >> 32);
    return static_cast<int>(__popcnt(lo) + __popcnt(hi));
  #endif
  }

  inline int ctz64(unsigned long long x) {
    if (x == 0) return 64;
    unsigned long idx = 0;
  #if defined(_M_X64) || defined(__x86_64__)
    (void)_BitScanForward64(&idx, static_cast<unsigned __int64>(x));
    return static_cast<int>(idx);
  #else
    unsigned long lo = static_cast<unsigned long>(x & 0xFFFFFFFFull);
    if (_BitScanForward(&idx, lo)) return static_cast<int>(idx);
    unsigned long hi = static_cast<unsigned long>(x >> 32);
    _BitScanForward(&idx, hi);
    return static_cast<int>(idx + 32);
  #endif
  }

  // Map GCC builtins to the portable wrappers on MSVC.
  #define __builtin_popcountll(x) popcount64((unsigned long long)(x))
  #define __builtin_ctzll(x)      ctz64((unsigned long long)(x))
#else
  // On GCC/Clang, defer to native builtins.
  inline int popcount64(unsigned long long x) { return __builtin_popcountll(x); }
  inline int ctz64(unsigned long long x)      { return __builtin_ctzll(x); }
#endif