cmake_minimum_required(VERSION 3.10)
project(ToyLowering)
set(CMAKE_CXX_STANDARD 17) 

#============================================================================
# MLIR and LLVM
#============================================================================
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# LLVM and MLIR headers
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")
message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# !!! THIS IS IMPORTANT !!!
# Include the following files to make TableGen/LLVM/MLIR functions available
include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(CLANG_LIB_DIR ${LLVM_BINARY_DIR}/lib/clang/22/lib/x86_64-unknown-linux-gnu)
message(STATUS "CLANG_LIB_DIR: ${LLVM_BINARY_DIR}/lib/clang/22/lib/x86_64-unknown-linux-gnu")


#============================================================================
# Toy Compiler
#============================================================================
include_directories(.)
include_directories(${PROJECT_BINARY_DIR})
include_directories(${CUDA_ROOT}/include)

# frontend
add_subdirectory(frontend)

# middleend
add_subdirectory(middleend)

# backend
add_subdirectory(backend)

# runtime
add_subdirectory(runtime)

# Find the CUDA runtime library
find_library(CUDA_RUNTIME_LIBRARY
  NAMES cudart
  PATHS ${CUDA_ROOT}/lib64
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)
message(STATUS "CUDA_RUNTIME_LIBRARY: ${CUDA_RUNTIME_LIBRARY}")

# Find the CUDA driver library
find_library(CUDA_DRIVER_LIBRARY
  NAMES cuda
  PATHS /usr/lib/x86_64-linux-gnu
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)
message(STATUS "CUDA_DRIVER_LIBRARY: ${CUDA_DRIVER_LIBRARY}")

# main
add_executable(compiler
  main.cpp
)
target_compile_options(compiler PRIVATE -fno-rtti -g -O0)
target_link_libraries(compiler PRIVATE
  frontend
  middleend
  backend
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRNVVMDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRLinalgTransforms
  MLIRSCFToGPU
  MLIRGPUTransforms
  MLIRSCFToControlFlow
  MLIRGPUToNVVMTransforms
  MLIRConvertToLLVMPass
  MLIRNVVMToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRReconcileUnrealizedCasts
  MLIRParser
  MLIRSupport
  LLVMSupport
  LLVMX86Info
  LLVMNVPTXInfo
  ${CUDA_RUNTIME_LIBRARY}
)

# ${CLANG_LIB_DIR}/libclang_rt.ubsan_standalone.a

#===---------------------------------------------------------===#
#             Build a add_two_vectors client binary
#===---------------------------------------------------------===#
# runtime/CudaRuntimeWrappers.cpp
add_executable(add_two_vectors
  add_two_vectors.cpp
)
target_compile_options(add_two_vectors PRIVATE -fno-rtti -g -O0)
target_link_options(add_two_vectors PUBLIC -rdynamic)
target_link_libraries(add_two_vectors PUBLIC
  runtime
  ${CUDA_DRIVER_LIBRARY}
)
