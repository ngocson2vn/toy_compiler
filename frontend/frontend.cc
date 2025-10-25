//===- MLIRGen.cpp - MLIR Generation from a Toy AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//
#include <numeric>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "AST.h"
#include "Parser.h"
#include "toy_dialect.h"
#include "frontend.h"

using namespace mlir::toy;
using namespace toy::compiler::frontend;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST& f : moduleAST)
      mlirGen(f);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (auto& arg : proto.getArgs()) {
      argTypes.push_back(getType(arg->getType()));
    }
    auto funcType = builder.getFunctionType(argTypes, {});
    return builder.create<FuncOp>(location, proto.getName(), funcType);
  }

  /// Emit a new function and add it to the MLIR module.
  FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    FuncOp function = mlirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block& entryBlock = *function.addEntryBlock();

    // Declare all the function arguments in the symbol table.
    auto protoArgs = funcAST.getProto()->getArgs();
    for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    mlir::toy::ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<mlir::toy::ReturnOp>(entryBlock.back());

    if (!returnOp) {
      builder.create<mlir::toy::ReturnOp>(loc(funcAST.getProto()->loc()));
    } else if (returnOp.getOperands().size() > 0) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(VarType{})));
    }

    return function;
  }

  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;

    auto location = loc(binop.loc());

    if (lhs.getType() != rhs.getType()) {
      emitError(location, "binary operator '") << binop.getOp() << "'" << " argument types mismatch";
      return nullptr;
    }

    int64_t size = -1;
    auto lhsType = cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = cast<mlir::RankedTensorType>(rhs.getType());
    if (lhsType.hasStaticShape() && rhsType.hasStaticShape()) {
      assert(lhsType.getNumElements() == rhsType.getNumElements() && "argument types mismatch");
      size = lhsType.getNumElements();
    }

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      if (size > 0) {
        // auto sizeVal = builder.create<ConstantOp>(location, size);
        return builder.create<AddOp>(location, lhs, rhs, size);
      }
      emitError(location, "binary operator '") << binop.getOp() << "'" << " argument types mismatch";
    default:
      emitError(location, "invalid binary operator '") << binop.getOp() << "'";
      return nullptr; 
    }
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(*ret.getExpr().value())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    // builder.create<ReturnOp>(location, expr ? llvm::ArrayRef(expr)
    //                                         : llvm::ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  mlir::Value mlirGen(AssignExprAST &expr) {
    mlir::Value srcVal = mlirGen(*expr.getSrc());
    if (!srcVal) {
      emitError(loc(expr.loc()), "error: unknown source value");
      return nullptr;
    }

    auto& dst = *cast<VariableExprAST>(expr.getDst());
    if (auto dstVal = symbolTable.lookup(dst.getName())) {
      // auto dataType = cast<mlir::RankedTensorType>(dstVal.getType());
      builder.create<mlir::toy::StoreOp>(loc(dst.loc()), srcVal, dstVal);
      return dstVal;
    }

    emitError(loc(expr.loc()), "error: unknown variable '")
        << dst.getName() << "'";
    return nullptr;
  }

  /// Emit a add expression. It emits specific operations for two builtins:
  /// add(x, y, n) and print(x).
  mlir::Value mlirGen(AddExprAST &call) {
    SmallVector<mlir::Value> argVals;
    for (auto& arg : call.getArgs()) {
      auto val = mlirGen(*arg);
      if (!val)
        return nullptr;

      argVals.push_back(val);
    }

    auto v = builder.create<AddOp>(loc(call.loc()), mlir::TypeRange{argVals[0].getType()}, argVals);
    return v;
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(cast<NumberExprAST>(expr).getValue());
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case ExprAST::Expr_VarAssign:
      return mlirGen(cast<AssignExprAST>(expr));
    case ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case ExprAST::Expr_Add:
      return mlirGen(cast<AddExprAST>(expr));
    case ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value)
      return nullptr;

    // Register the value in the symbol table.
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }

      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);

      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty()) {
      SmallVector<int64_t> shape_vec = {mlir::ShapedType::kDynamic};
      return mlir::RankedTensorType::get(shape_vec, builder.getF32Type());
    }

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF32Type());
  }

  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  mlir::Type getType(const VarType &type) { 
    switch (type.type) {
      case Type::tensor:
        return getType(type.shape);
      case Type::integer:
        return builder.getI64Type();
      default:
        llvm::errs() << "Unknown type\n";
        return nullptr;
    }
  }
};

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace

namespace toy {
namespace compiler {
namespace frontend {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> getModule(mlir::MLIRContext& context, const std::string& inputFilename) {
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  // Handle '.toy' input to the compiler.
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return nullptr;

  mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
  if (!module)
    return nullptr;

  return module;
}

} // namespace frontend
} // namespace compiler
} // namespace toy