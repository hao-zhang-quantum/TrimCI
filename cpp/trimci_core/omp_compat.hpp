#pragma once
// Compatibility header for OpenMP: provides stubs when OpenMP is not enabled.
#ifdef _OPENMP
  #include <omp.h>
#else
  inline int omp_get_max_threads() { return 1; }
  inline int omp_get_thread_num() { return 0; }
  inline void omp_set_num_threads(int) {}
#endif