#pragma once

namespace llvm {
class LogicalResult;
} // namespace llvm

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace toy {
namespace compiler {
namespace middleend {

llvm::LogicalResult lower(mlir::ModuleOp& module);

} // namespace middleend
} // namespace compiler
} // namespace toy