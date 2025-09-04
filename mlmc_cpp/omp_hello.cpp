#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp single
    std::cout << "OpenMP is ON. Version: " << _OPENMP
              << " | threads in this region: " << omp_get_num_threads()
              << std::endl;
  }
#else
  std::cout << "OpenMP is OFF (compiled without -fopenmp)." << std::endl;
#endif
  return 0;
}
