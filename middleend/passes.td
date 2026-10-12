include "mlir/Pass/PassBase.td"

def ConvertToyToStdPass: Pass<"convert-toy-to-std", "mlir::ModuleOp"> {
    let summary = "Convert Toy to standard dialects";
    let description = [{
      This pass converts the Toy dialect into the standard dialects.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::toy::ToyDialect"];
}

def ConvertToyToLinalgPass: Pass<"convert-toy-to-linalg", "mlir::ModuleOp"> {
    let summary = "Convert Toy into Linalg dialect";
    let description = [{
      This pass converts the Toy dialect into the Linalg dialect.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::linalg::LinalgDialect",
                             "mlir::toy::ToyDialect"];
}

def FoldTieDimsOpPass: Pass<"fold-ties-dims-op", "mlir::ModuleOp"> {
    let summary = "Fold toy::TieDimsOp ops";
    let description = [{
      This pass folds toy::TieDimsOp ops.
    }];

    let dependentDialects = ["mlir::toy::ToyDialect"];
}

def FoldMaterializeInDestinationOpPass: Pass<"fold-materialize-in-destination-op", "mlir::ModuleOp"> {
    let summary = "Fold bufferization::MaterializeInDestinationOp ops";
    let description = [{
      This pass folds bufferization::MaterializeInDestinationOp ops.
    }];

    let dependentDialects = ["mlir::bufferization::BufferizationDialect"];
}

def ConvertTensorToMemRefPass: Pass<"convert-tensor-to-memref", "mlir::ModuleOp"> {
    let summary = "Convert Tensor to MemRef";
    let description = [{
      This pass converts the Tensor type into the MemRef type.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect"];
}

