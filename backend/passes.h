#pragma once

#include <memory>
#include "mlir/Pass/Pass.h"


namespace mlir {

class ModuleOp;

namespace func {
class FuncOp;
}

namespace toy {

#define GEN_PASS_DECL
#include "backend/passes.h.inc"

// Creates a TileLoopsPass with tiles sizes provided through `tile_sizes`
// and unroll factors provided through `unroll_factors`.
std::unique_ptr<OperationPass<func::FuncOp>> createTileLoopsPass(
    ArrayRef<int64_t> tileSizes = {}, ArrayRef<int64_t> unrollFactors = {});

}  // namespace toy
}  // namespace mlir
