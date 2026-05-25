#pragma once
// Compatibility header for OpenMP: provides stubs when OpenMP is not enabled
// or when <omp.h> is not available (e.g. macOS arm64 without libomp).
//
// We require BOTH `_OPENMP` to be defined AND `<omp.h>` to be includable
// before using the real OpenMP runtime; otherwise fall back to no-op stubs.

#if defined(_OPENMP) && defined(__has_include)
  #if __has_include(<omp.h>)
    #include <omp.h>
    #define TRIMCI_OMP_NATIVE 1
  #endif
#elif defined(_OPENMP)
  // Compiler doesn't support __has_include; trust _OPENMP and try omp.h.
  #include <omp.h>
  #define TRIMCI_OMP_NATIVE 1
#endif

#ifndef TRIMCI_OMP_NATIVE
  inline int omp_get_max_threads() { return 1; }
  inline int omp_get_num_threads() { return 1; }
  inline int omp_get_thread_num() { return 0; }
  inline void omp_set_num_threads(int) {}
#endif