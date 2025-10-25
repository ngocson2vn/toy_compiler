# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls)
add_public_tablegen_target(ToyBackendPassIncGen)

include_directories(${CUDA_ROOT}/include)

# cuda_utils
add_library(cuda_utils
  utils.cc
  cuda_utils.cc
)

add_library(backend
  backend.cc
  utils.cc
  cuda_utils.cc
  tile_loops_pass.cc
  passes.cc
)
add_dependencies(backend
  ToyBackendPassIncGen
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRLinalgTransforms
  MLIRGPUTransforms
  MLIRSCFToControlFlow
  MLIRParser
  MLIRSupport
  MLIRTableGen
  LLVMSupport
  intrinsics_gen
  clang_rt.ubsan_standalone-x86_64
)
target_compile_options(backend PRIVATE -fno-rtti -g -O0)
