#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
// #include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "frontend/toy_dialect.h.inc"

#define GET_OP_CLASSES
#include "frontend/toy_ops.h.inc"