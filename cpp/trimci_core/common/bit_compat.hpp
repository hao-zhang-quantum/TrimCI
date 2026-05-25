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

namespace trimci_core {
namespace detail {

template<typename StorageType>
struct HamiltonianBitOps {
    static int count_differences(const StorageType& a, const StorageType& b) {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            return popcount64(a ^ b);
        } else {
            int count = 0;
            for(size_t i=0; i<a.size(); ++i) {
                count += popcount64(a[i] ^ b[i]);
            }
            return count;
        }
    }

    static std::vector<int> storage_to_indices(const StorageType& s) {
        std::vector<int> indices;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            uint64_t temp = s;
            while(temp) {
                int p = ctz64(temp);
                indices.push_back(p);
                temp &= ~(1ULL << p);
            }
        } else {
            for(size_t i=0; i<s.size(); ++i) {
                uint64_t temp = s[i];
                while(temp) {
                    int p = ctz64(temp) + i * 64;
                    indices.push_back(p);
                    temp &= ~(1ULL << ctz64(temp));
                }
            }
        }
        return indices;
    }

    static int storage_to_indices_inline(const StorageType& s, int* buffer, int capacity) {
        int count = 0;
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
            uint64_t temp = s;
            while(temp) {
                if (count >= capacity) return count; // Safety break
                int p = ctz64(temp);
                buffer[count++] = p;
                temp &= ~(1ULL << p);
            }
        } else {
             for(size_t i=0; i<s.size(); ++i) {
                uint64_t temp = s[i];
                while(temp) {
                    if (count >= capacity) return count;
                    int p = ctz64(temp);
                    buffer[count++] = p + i * 64;
                    // Note: need to clear the bit we just found to advance
                    temp &= ~(1ULL << p);
                }
            }
        }
        return count;
    }
    
    static StorageType and_not(const StorageType& a, const StorageType& b) {
        if constexpr (std::is_same_v<StorageType, uint64_t>) {
             return a & (~b);
        } else {
             StorageType res;
             for(size_t i=0; i<a.size(); ++i) res[i] = a[i] & (~b[i]);
             return res;
        }
    }
};

} // namespace detail
} // namespace trimci_core