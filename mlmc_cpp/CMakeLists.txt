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

option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)

# ---- Common sources (NOTE: no GBM file here) ----
set(MLMC_SOURCES_COMMON
  src/mlmc.cpp
  src/mlmc_test.cpp
  src/stoch_heat_eqn_fourier_modes.cpp
  src/stoch_heat_eqn_fourier_modes_var.cpp
  src/stoch_heat_eqn_fourier_modes_var_cc.cpp
  src/stoch_heat_eqn_fourier_modes_var_fe.cpp
  src/dean_kawasaki_nn.cpp
  src/dean_kawasaki_cc.cpp
  src/dean_kawasaki_fe.cpp
)

# Headers
add_library(mlmc_headers INTERFACE)
target_include_directories(mlmc_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Serial target (uses serial GBM) ----
add_executable(mlmc_cpp_serial main.cpp ${MLMC_SOURCES_COMMON} src/gbm_euler.cpp
src/stoch_heat_eqn_energy_fe.cpp
src/stoch_heat_eqn_energy_cc.cpp
src/stoch_heat_eqn_energy_nn.cpp)
target_link_libraries(mlmc_cpp_serial PRIVATE mlmc_headers)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(mlmc_cpp_serial PRIVATE -O3 -march=native)
elseif (MSVC)
  target_compile_options(mlmc_cpp_serial PRIVATE /O2)
endif()

# --- Performance serial runs ---
add_executable(run_performance src/run_performance.cpp 
src/mlmc_test_performance.cpp
src/stoch_heat_eqn_energy_cc.cpp
${MLMC_SOURCES_COMMON} 
src/stoch_heat_eqn_energy_fe.cpp
src/stoch_heat_eqn_energy_nn.cpp)
target_link_libraries(run_performance PRIVATE mlmc_headers)


# -- Performance OpenMP runs -- 
add_executable(run_performance_omp
  src/run_performance_omp.cpp
  src/mlmc_test_performance.cpp
  ${MLMC_SOURCES_COMMON}
  src/gbm_euler_omp.cpp
  src/stoch_heat_eqn_energy_fe_omp.cpp
  src/stoch_heat_eqn_energy_cc_omp.cpp
  src/stoch_heat_eqn_energy_nn_omp.cpp
)
target_link_libraries(run_performance_omp PRIVATE mlmc_headers)

# ---- OpenMP target (uses OMP GBM) ----
add_executable(mlmc_cpp_omp
  ${MLMC_SOURCES_COMMON} 
  src/main_omp.cpp
  src/gbm_euler_omp.cpp 
  src/stoch_heat_eqn_energy_fe_omp.cpp
  src/stoch_heat_eqn_energy_cc_omp.cpp
  src/stoch_heat_eqn_energy_nn_omp.cpp
)
target_link_libraries(mlmc_cpp_omp PRIVATE mlmc_headers)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(mlmc_cpp_omp PRIVATE -O3 -march=native)
  target_compile_options(run_performance_omp PRIVATE -O3 -march=native)
elseif (MSVC)
  target_compile_options(mlmc_cpp_omp PRIVATE /O2)
  target_compile_options(run_performance_omp PRIVATE /O2)
endif()

# -------- OpenMP enablement (REQUIRED) --------
if (ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(mlmc_cpp_omp        PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(run_performance_omp PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP enabled for mlmc_cpp_omp and run_performance_omp")
else()
  message(STATUS "OpenMP explicitly disabled (ENABLE_OPENMP=OFF)")
endif()
