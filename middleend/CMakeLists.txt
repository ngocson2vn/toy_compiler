# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls)
add_public_tablegen_target(ToyMiddleendPassIncGen)

add_library(middleend
  passes.cc
  middleend.cc
)
add_dependencies(middleend 
  ToyMiddleendPassIncGen
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
  clang_rt.ubsan_standalone-x86_64
)
target_compile_options(middleend PRIVATE -fno-rtti -g -O0)
