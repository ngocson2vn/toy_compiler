include "mlir/Pass/PassBase.td"

def ConvertToyToStdPass: Pass<"convert-toy-to-std", "mlir::ModuleOp"> {
    let summary = "Convert Toy to standard dialects";
    let description = [{
      This pass converts the Toy dialect into the standard dialects.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::toy::ToyDialect"];
}

def ConvertTensorToMemRefPass: Pass<"convert-tensor-to-memref", "mlir::ModuleOp"> {
    let summary = "Convert Tensor to MemRef";
    let description = [{
      This pass converts the Tensor type into the MemRef type.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect"];
}
