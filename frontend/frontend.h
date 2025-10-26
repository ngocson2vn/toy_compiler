//===- MLIRGen.h - MLIR Generation from a Toy AST -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST for the Toy language.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

namespace mlir {
class MLIRContext;

template <typename T>
class OwningOpRef;

class ModuleOp;
} // namespace mlir

namespace toy {
namespace compiler {
namespace frontend {

mlir::OwningOpRef<mlir::ModuleOp> getModule(mlir::MLIRContext& context, const std::string& inputFilename);

} // namespace frontend
} // namespace compiler
} // namespace toy
