cmake_minimum_required(VERSION 3.15)
project(mlmc_cpp LANGUAGES C CXX)

# Set C language version
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set C++ language version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Tell CMake where to find your header files
include_directories(include)

# Define the executable target: mlmc_cpp
add_executable(mlmc_cpp
    main.cpp
    src/gbm_euler.cpp
    src/mlmc.cpp
    src/mlmc_test.cpp
    src/stoch_heat_eqn_qoi.cpp
)

