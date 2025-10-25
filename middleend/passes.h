#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir::toy {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "middleend/passes.h.inc"

} // namespace mlir::toy
