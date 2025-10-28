cmake_minimum_required(VERSION 3.10)
project(ToyLowering)
set(CMAKE_CXX_STANDARD 17) 

set(LLVM_BINARY_DIR ${PROJECT_SOURCE_DIR}/llvm-project/build)
add_subdirectory(llvm-project/llvm ${LLVM_BINARY_DIR})

# LLVM and MLIR headers
include_directories(llvm-project/llvm/include)
include_directories(llvm-project/mlir/include)

# Some headers are generated automatically, which is why we need to include the following directories
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(.)
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
  MLIRRegisterAllExtensions
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
  MLIRBuiltinToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRParser
  MLIRSupport
  MLIRTableGen
  LLVMTableGen
  LLVMSupport
  clang_rt.ubsan_standalone-x86_64
  ${CUDA_RUNTIME_LIBRARY}
)


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
