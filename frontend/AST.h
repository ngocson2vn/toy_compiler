//===- AST.h - Node definition for the Toy AST ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Lexer.h"

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>
#include <vector>

#include "common/common.h"

namespace toy {
namespace compiler {

using namespace common;
namespace frontend {

class ShapeValue {
 public:
  ShapeValue(uint64_t val)
    : val_(val), static_(true) {}

  ShapeValue(const std::string& id)
    : id_(id), static_(false) {}

  bool isStatic() const {
    return static_;
  }

  bool isDynamic() const {
    return (not static_);
  }

  uint64_t getValue() const {
    return val_;
  }

  llvm::StringRef getId() const {
    return id_;
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ShapeValue& v) {
    if (v.isStatic()) {
      os << v.getValue();
    } else {
      os << v.getId();
    }

    return os;
  }

 private:
  uint64_t val_ = -1;
  std::string id_;
  bool static_;
};

/// A variable type with shape information.
struct VarType {
  Type type;
  Type element_type;
  std::vector<ShapeValue> shape;

  llvm::StringRef getTypeStr(const Type& type) const {
    switch (type) {
      case Type::F32:
        return type::name::F32;
      case Type::F16:
        return type::name::F16;
      case Type::I32:
        return type::name::I32;
      case Type::TENSOR:
        return type::name::TENSOR;

      default:
        return type::name::INVALID;
    }
  }

  llvm::StringRef getTypeName() const {
    return getTypeStr(type);
  }

  llvm::StringRef getElemTypeName() const {
    return getTypeStr(element_type);
  }

  bool isTensor() const {
    return type == Type::TENSOR;
  }

  bool hasStaticShape() const {
    if (type == Type::TENSOR) {
      for (const auto& d : shape) {
        if (!d.isStatic()) {
          return false;
        }
      }
    }

    return true;
  }

  bool hasDynamicDim() const {
    if (type == Type::TENSOR) {
      for (const auto& d : shape) {
        if (!d.isStatic()) {
          return true;
        }
      }
    }

    return false;
  }
};

/// Base class for all expression nodes.

enum class ExprKind {
  Expr_VarDecl,
  Expr_Return,
  Expr_Num,
  Expr_Literal,
  Expr_Var,
  Expr_AssignOp,
  Expr_BinOp,

  // Host ops
  Expr_Random,
  Expr_Call,
  Expr_Print,

  // Device ops
  Expr_Add,
  Expr_Mul,
  Expr_Max,
  Expr_Dot,

  Count
};

template <ExprKind V>
constexpr std::string_view kindName() {
  std::string_view funcName = __PRETTY_FUNCTION__;
  auto eqPos = funcName.find("=");
  auto rpart = funcName.substr(eqPos + 2);
  return rpart.substr(0, rpart.size() - 1);
}

template <std::size_t... I>
constexpr auto makeKindNames(std::index_sequence<I...>) {
  return std::array<std::string_view, sizeof...(I)>{
      kindName<static_cast<ExprKind>(I)>()...};
}

inline constexpr auto kKindNames = makeKindNames(
    std::make_index_sequence<static_cast<std::size_t>(ExprKind::Count)>{});


mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const ExprKind kind);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ExprKind kind);

class ExprAST {
 public:
  ExprAST() = default;
  ExprAST(ExprKind kind, Location location)
    : kind(kind), location(std::move(location)) {};
  virtual ~ExprAST() = default;

  ExprKind getKind() const { return kind; }

  const Location &loc() { return location; }

 private:
  ExprKind kind;
  Location location;
};

class UniformExprAST : public ExprAST {
 public:
  UniformExprAST(ExprKind kind,
                 Location location,
                 std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(kind, std::move(location)), args(std::move(args)) {}

  // Disable all copy operators
  UniformExprAST(const UniformExprAST& other) = delete;
  UniformExprAST& operator==(const UniformExprAST& other) = delete;

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return args; }

  operator bool() {
    return this != nullptr;
  }

private:
  std::vector<std::unique_ptr<ExprAST>> args;
};

/// A block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public UniformExprAST {
  double val;
  Type type;

public:
  NumberExprAST(Location loc, Type type, double val)
      : UniformExprAST(ExprKind::Expr_Num, std::move(loc), {}), type(type), val(val) {}

  double getValue() { return val; }

  Type getType() { return type; }
  void setType(Type ty) { type = ty; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Num; }
};

/// Expression class for a literal value.
class LiteralExprAST : public UniformExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;

public:
  LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : UniformExprAST(ExprKind::Expr_Literal, std::move(loc), {}),
        values(std::move(values)),
        dims(std::move(dims)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }
  llvm::ArrayRef<int64_t> getDims() { return dims; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Literal; }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public UniformExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : UniformExprAST(ExprKind::Expr_Var, std::move(loc), {}), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Var; }
};

/// Expression class for defining a variable.
class VarDeclExprAST : public UniformExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : UniformExprAST(ExprKind::Expr_VarDecl, std::move(loc), {}), name(name),
        type(std::move(type)), initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  ExprAST *getInitVal() { return initVal.get(); }
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_VarDecl; }
};

/// Expression class for defining an assignment
class AssignExprAST : public UniformExprAST {
  char op;
  std::unique_ptr<ExprAST> src;
  std::unique_ptr<ExprAST> dst;

public:
  char getOp() { return op; }
  AssignExprAST(Location loc, char op, 
                std::unique_ptr<ExprAST> dst, std::unique_ptr<ExprAST> src)
      : UniformExprAST(ExprKind::Expr_AssignOp, std::move(loc), {}), op(op),
        dst(std::move(dst)), src(std::move(src)) {}

  ExprAST *getDst() { return dst.get(); }
  ExprAST *getSrc() { return src.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_AssignOp; }
};

/// Expression class for a return operator.
class ReturnExprAST : public UniformExprAST {
  std::optional<std::unique_ptr<ExprAST>> expr;

public:
  ReturnExprAST(Location loc, std::optional<std::unique_ptr<ExprAST>> expr)
      : UniformExprAST(ExprKind::Expr_Return, std::move(loc), {}), expr(std::move(expr)) {}

  std::optional<ExprAST *> getExpr() {
    if (expr.has_value())
      return expr->get();
    return std::nullopt;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Return; }
};

/// Expression class for a binary operator.
class BinaryExprAST : public UniformExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  char getOp() { return op; }
  ExprAST *getLHS() { return lhs.get(); }
  ExprAST *getRHS() { return rhs.get(); }

  BinaryExprAST(Location loc, char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : UniformExprAST(ExprKind::Expr_BinOp, std::move(loc), {}), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_BinOp; }
};

/// Expression class for function call.
class CallExprAST : public UniformExprAST {
  std::string callee;

public:
  CallExprAST(Location loc,
              const std::string& callee,
              std::vector<std::unique_ptr<ExprAST>> args)
      : UniformExprAST(ExprKind::Expr_Call, std::move(loc),
        std::move(args)),
        callee(callee) {}

  llvm::StringRef getCallee() { return callee; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Call; }
};

/// Expression class for builtin print call.
class PrintExprAST : public UniformExprAST {
 public:
  ExprAST* getArg() { return getArgs().front().get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Print; }
};

/// Expression class for builtin add call.
class AddExprAST : public UniformExprAST {
 public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Add; }
};

class MulExprAST : public UniformExprAST {
 public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Mul; }
};

/// Expression class for builtin max call.
class MaxExprAST : public UniformExprAST {
 public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Max; }
};

/// Expression class for builtin dot call.
class DotExprAST : public UniformExprAST {
 public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Dot; }
};

/// Expression class for builtin random call.
class RandomExprAST : public UniformExprAST {
 public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == ExprKind::Expr_Random; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VarDeclExprAST>> args;

public:
  PrototypeAST(Location location, const std::string &name,
               std::vector<std::unique_ptr<VarDeclExprAST>> args)
      : location(std::move(location)), name(name), args(std::move(args)) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VarDeclExprAST>> getArgs() { return args; }
};

/// This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprASTList> body)
      : proto(std::move(proto)), body(std::move(body)) {}
  PrototypeAST *getProto() { return proto.get(); }
  ExprASTList *getBody() { return body.get(); }
};

/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<FunctionAST> functions;

public:
  ModuleAST(std::vector<FunctionAST> functions)
      : functions(std::move(functions)) {}

  auto begin() { return functions.begin(); }
  auto end() { return functions.end(); }
};

void dumpAST(ModuleAST &);

} // namespace frontend
} // namespace compiler
} // namespace toy
