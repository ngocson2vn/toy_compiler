#pragma once

#include "llvm/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace toy {
namespace compiler {
namespace backend {

/**
 * \brief Lower ModuleOp to CUBIN and host code
 *
 * \param module      - ModuleOp to be lowered
 * \param targetArch  - Target SM architecture, for example, sm_86, sm_90a
 */
llvm::LogicalResult lower(mlir::ModuleOp& module, const std::string& targetArch);

} // namespace backend
} // namespace compiler
} // namespace toy