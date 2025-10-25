#include "toy_dialect.h"
#include "frontend/toy_dialect.cpp.inc"

using namespace mlir;
using namespace mlir::toy;

// Register the dialect with the MLIR context
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "frontend/toy_ops.cpp.inc"
    >();
}

#define GET_OP_CLASSES
#include "frontend/toy_ops.cpp.inc"
