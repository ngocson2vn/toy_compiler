#pragma once

#include "llvm/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace toy {
namespace compiler {
namespace backend {

llvm::LogicalResult lower(mlir::ModuleOp& module);

} // namespace backend
} // namespace compiler
} // namespace toy