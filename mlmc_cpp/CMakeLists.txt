cmake_minimum_required(VERSION 3.15)
project(mlmc_cpp LANGUAGES C CXX)

# Language standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Common sources (NOTE: no GBM file here) ----
set(MLMC_SOURCES_COMMON
  main.cpp
  src/mlmc.cpp
  src/mlmc_test.cpp
  src/stoch_heat_eqn_fourier_modes.cpp
  src/stoch_heat_eqn_fourier_modes_var.cpp
  src/stoch_heat_eqn_fourier_modes_var_cc.cpp
  src/stoch_heat_eqn_fourier_modes_var_fe.cpp
  src/stoch_heat_eqn_energy_nn.cpp
  src/stoch_heat_eqn_energy_cc.cpp
  src/stoch_heat_eqn_energy_fe.cpp
  src/dean_kawasaki_nn.cpp
  src/dean_kawasaki_cc.cpp
)

# Headers
add_library(mlmc_headers INTERFACE)
target_include_directories(mlmc_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Serial target (uses serial GBM) ----
add_executable(mlmc_cpp_serial ${MLMC_SOURCES_COMMON} src/gbm_euler.cpp)
target_link_libraries(mlmc_cpp_serial PRIVATE mlmc_headers)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(mlmc_cpp_serial PRIVATE -O3 -march=native)
elseif (MSVC)
  target_compile_options(mlmc_cpp_serial PRIVATE /O2)
endif()

# ---- OpenMP target (uses OMP GBM) ----
add_executable(mlmc_cpp_omp ${MLMC_SOURCES_COMMON} src/gbm_euler_omp.cpp)
target_link_libraries(mlmc_cpp_omp PRIVATE mlmc_headers)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(mlmc_cpp_omp PRIVATE -O3 -march=native)
elseif (MSVC)
  target_compile_options(mlmc_cpp_omp PRIVATE /O2)
endif()

# OpenMP for C++ only (no need for the C component)
find_package(OpenMP COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
  target_link_libraries(mlmc_cpp_omp PRIVATE OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP CXX not found; mlmc_cpp_omp will build but run serially. \
If on macOS, use Homebrew LLVM (preferred) or install libomp.")
  # Optional AppleClang+libomp fallback (uncomment if you stick with AppleClang):
  # set(LIBOMP_ROOT "/opt/homebrew/opt/libomp")   # Intel Macs: "/usr/local/opt/libomp"
  # target_compile_options(mlmc_cpp_omp PRIVATE -Xpreprocessor -fopenmp)
  # target_include_directories(mlmc_cpp_omp PRIVATE ${LIBOMP_ROOT}/include)
  # target_link_directories(mlmc_cpp_omp PRIVATE ${LIBOMP_ROOT}/lib)
  # target_link_libraries(mlmc_cpp_omp PRIVATE omp)
endif()
