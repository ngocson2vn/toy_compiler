include "mlir/Pass/PassBase.td"

// Copy from tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Transforms/passes.td
// URL: https://github.com/tensorflow/tensorflow/blob/v2.12.0/tensorflow/compiler/xla/mlir_hlo/transforms/passes.td#L73
def TileLoopsPass : Pass<"tile-loops", "func::FuncOp"> {
  let summary = "Tiles parallel loops.";
  let description = [{ The pass converts an `scf.parallel` loop into a nested,
  "tiled", `scf.parallel` loop with 2 to 3 levels of nesting. The 3rd level of
  nesting represents operation unrolling within a tile and is only applied on
  simple memory access patterns (ones resulting from same shape, scalar, and/or
  constant operands).}];
  let constructor = "createTileLoopsPass()";
  let options = [
    ListOption<"tile_sizes_", "tile-sizes", "int64_t", "The size of the tile "
               "in each dimension, expressed as the number of "
               "`unroll_factors_` in that dimension.", "llvm::cl::ZeroOrMore">,
    ListOption<"unroll_factors_", "unroll-factors", "int64_t", "The unroll "
               "factor in each dimension, expressed as the number of elements "
               "in that dimension.", "llvm::cl::ZeroOrMore">,
  ];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}


def LowerMemRefToLLVMPass : Pass<"lower-memref-to-llvm", "ModuleOp"> {
  let summary = "Pass to lower memref to llvm";
}


def InjectRuntimeCtxPass : Pass<"inject-runtime-ctx", "ModuleOp"> {
  let summary = "Pass to inject runtime context";
}


def LowerLLVMToObjectPass : Pass<"lower-llvm-to-object", "ModuleOp"> {
  let summary = "Pass to lower LLVM to Object file";
  let options = [
    Option<"objectFilePath", "object-file-path", "std::string", "\"output.o\"", "The output object file path">
  ];
}
