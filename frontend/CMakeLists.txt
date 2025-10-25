# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS toy_ops.td)
mlir_tablegen(toy_dialect.h.inc -gen-dialect-decls)
mlir_tablegen(toy_dialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(toy_ops.h.inc -gen-op-decls)
mlir_tablegen(toy_ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(ToyOpsIncGen)

# frontend library
add_library(frontend
  AST.cc
  frontend.cc
  toy_dialect.cc
)

add_dependencies(frontend 
  ToyOpsIncGen
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRParser
  MLIRSupport
  MLIRTableGen
  LLVMSupport
  clang_rt.ubsan_standalone-x86_64
)

target_compile_options(frontend PRIVATE -fno-rtti -g -O0)
