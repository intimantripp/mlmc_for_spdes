# MLMC C++ Implementation

This project implements a C/C++ version of the Multilevel Monte Carlo (MLMC) method, replicating the functionality of the existing Python implementation. The project includes the necessary components to run individual cases similar to the `mlmc_test` function in Python.

## Project Structure

The project is organized as follows:

```
mlmc_cpp
├── CMakeLists.txt          # CMake configuration file
├── README.md               # Project documentation
├── include                 # Header files
│   ├── mlmc_core.hpp       # Core MLMC functionality
│   ├── mlmc_test.hpp       # Testing functions for MLMC
│   └── gbm_euler.hpp       # Geometric Brownian Motion (GBM) functions
├── src                     # Source files
│   ├── mlmc_core.c         # Implementation of core MLMC functions
│   ├── mlmc_test.cpp       # Implementation of MLMC test functions
│   └── gbm_euler.cpp       # Implementation of GBM Euler method functions
├── main.cpp                # Entry point of the application
└── tests                   # Unit tests
    └── test_mlmc.cpp       # Unit tests for MLMC implementation
```

## Building the Project

To build the project, follow these steps:

1. Install CMake and a C/C++ compiler (like GCC or Clang) if you haven't already.
2. Navigate to the project directory in the terminal.
3. Create a build directory:
   ```
   mkdir build && cd build
   ```
4. Run CMake to configure the project:
   ```
   cmake ..
   ```
5. Compile the project using make:
   ```
   make
   ```
6. Execute the generated binary to run the MLMC simulations.

## Running Tests

The project includes unit tests to validate the correctness of the MLMC implementation. You can run the tests after building the project to ensure everything is functioning as expected.

## Usage

The main entry point of the application is `main.cpp`, which initializes the program, calls the MLMC functions, and runs tests. You can modify this file to run specific simulations or test cases as needed.

This setup provides a basic structure for your C/C++ implementation of the MLMC code, allowing you to run individual cases similar to the `mlmc_test` function in your Python implementation.