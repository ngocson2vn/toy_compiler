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

#ifndef TOY_AST_H
#define TOY_AST_H

#include "Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>
#include <vector>

namespace toy {
namespace compiler {
namespace frontend {

enum class Type {
  TENSOR = 1,
  I32 = 2,
  F32 = 3,
  F16 = 4,
  INVALID = 99
};

namespace type::name {
  static const std::string F32 = "f32";
  static const std::string F16 = "f16";
  static const std::string I32 = "i32";
  static const std::string TENSOR = "tensor";
  static const std::string INVALID = "invalid";
}

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
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_AssignOp,
    Expr_BinOp,
    Expr_Add,
    Expr_Max,
    Expr_Random,
    Expr_Call,
    Expr_Print
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(std::move(location)) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

/// A block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double val;
  Type type;

public:
  NumberExprAST(Location loc, Type type, double val)
      : ExprAST(Expr_Num, std::move(loc)), type(type), val(val) {}

  double getValue() { return val; }

  Type getType() { return type; }
  void setType(Type ty) { type = ty; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;

public:
  LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : ExprAST(Expr_Literal, std::move(loc)), values(std::move(values)),
        dims(std::move(dims)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }
  llvm::ArrayRef<int64_t> getDims() { return dims; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, std::move(loc)), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : ExprAST(Expr_VarDecl, std::move(loc)), name(name),
        type(std::move(type)), initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  ExprAST *getInitVal() { return initVal.get(); }
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};

/// Expression class for defining an assignment
class AssignExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> src;
  std::unique_ptr<ExprAST> dst;

public:
  char getOp() { return op; }
  AssignExprAST(Location loc, char op, 
                std::unique_ptr<ExprAST> dst, std::unique_ptr<ExprAST> src)
      : ExprAST(Expr_AssignOp, std::move(loc)), op(op),
        dst(std::move(dst)), src(std::move(src)) {}

  ExprAST *getDst() { return dst.get(); }
  ExprAST *getSrc() { return src.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_AssignOp; }
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
  std::optional<std::unique_ptr<ExprAST>> expr;

public:
  ReturnExprAST(Location loc, std::optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(Expr_Return, std::move(loc)), expr(std::move(expr)) {}

  std::optional<ExprAST *> getExpr() {
    if (expr.has_value())
      return expr->get();
    return std::nullopt;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  char getOp() { return op; }
  ExprAST *getLHS() { return lhs.get(); }
  ExprAST *getRHS() { return rhs.get(); }

  BinaryExprAST(Location loc, char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : ExprAST(Expr_BinOp, std::move(loc)), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string callee;
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  CallExprAST(Location loc, const std::string &callee,
              std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Call, std::move(loc)), callee(callee),
        args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
};

/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> arg;

public:
  PrintExprAST(Location loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(Expr_Print, std::move(loc)), arg(std::move(arg)) {}

  ExprAST *getArg() { return arg.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Print; }
};

/// Expression class for builtin add calls.
class AddExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  AddExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Add, std::move(loc)), args(std::move(args)) {}

  std::vector<std::unique_ptr<ExprAST>>& getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Add; }
};

/// Expression class for builtin max calls.
class MaxExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  MaxExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Max, std::move(loc)), args(std::move(args)) {}

  std::vector<std::unique_ptr<ExprAST>>& getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Max; }
};

/// Expression class for builtin random calls.
class RandomExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  RandomExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Random, std::move(loc)), args(std::move(args)) {}

  std::vector<std::unique_ptr<ExprAST>>& getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Random; }
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

#endif // TOY_AST_H
