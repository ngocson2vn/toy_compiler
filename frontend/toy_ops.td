include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "mlir::toy";
}

def Toy_FuncOp : Op<Toy_Dialect, "func", [
    FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "toy.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.

    Example:

    ```mlir
    toy.func @main() {
      %0 = toy.constant dense<5.500000e+00> : tensor<f32>
      %1 = toy.reshape(%0 : tensor<f32>) to tensor<2x2xf32>
      toy.print %1 : tensor<2x2xf32>
      toy.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Toy_CallOp : Op<Toy_Dialect, "call"> {
  let summary = "Call operation for the toy dialect";
  let description = [{
    This operation call a user-defined function.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$callee`(` $args `)` `:` type($args) attr-dict `->` type($result)";
}

def Toy_PrintOp : Op<Toy_Dialect, "print"> {
  let summary = "Print operation for the toy dialect";
  let description = [{
    This operation print a number or a tensor.
  }];
  let arguments = (ins AnyTypeOf<[F32Tensor, F32, I32]>:$arg);
  let results = (outs I64:$result);

  let assemblyFormat = "$arg `:` type($arg) attr-dict `->` type($result)";
}

// def Toy_ConstantOp : Op<Toy_Dialect, "constant", [Pure, AllTypesMatch<["value", "result"]>]> {
def Toy_ConstantOp : Op<Toy_Dialect, "constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant 42 : i64
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins TypedAttrInterface:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict `->` type($result)";

  let builders = [
    OpBuilder<(ins "int32_t":$value), [{
  auto attr = $_builder.getI32IntegerAttr(value);
  build($_builder, $_state, attr.getType(), attr);}]>,

    OpBuilder<(ins "float":$value), [{
  auto attr = $_builder.getF32FloatAttr(value);
  build($_builder, $_state, attr.getType(), attr);}]>
  ];
}

def Toy_TieDimsOp : Op<Toy_Dialect, "tie_dims"> {
  let summary = "Tie dims operation for the toy dialect";
  let description = [{
    This operation tie dims with a tensor.
  }];
  let arguments = (ins AnyTypeOf<[F32Tensor, AnyMemRef]>:$target, Variadic<Index>:$dims);
  let results = (outs AnyTypeOf<[F32Tensor, AnyMemRef]>:$result);

  let assemblyFormat = "$target `,` $dims `:` type($target) `,` type($dims) attr-dict `->` type($result)";

  // let builders = [
  //   // OpBuilder<(ins "Value":$lhs, "Value":$rhs, "int64_t":$size)>,
  //   OpBuilder<(ins "Value":$lhs, "ValueRange":$dynamic_dims), [{
  //     build($_builder, $_state, lhs, dynamic_dims);
  //   }]>,
  // ];

  let hasCanonicalizer = 1;
}

def Toy_AddOp : Op<Toy_Dialect, "add"> {
  let summary = "Addition operation for the toy dialect";
  let description = [{
    This operation computes the sum of two tensors element-wise.
  }];
  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict `->` type($result)";

  // let builders = [
  //   // OpBuilder<(ins "Value":$lhs, "Value":$rhs, "int64_t":$size)>,
  //   OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
  //     build($_builder, $_state, lhs, rhs);
  //   }]>,
  // ];
}

def Toy_MaxOp : Op<Toy_Dialect, "max"> {
  let summary = "Maximum operation for the toy dialect";
  let description = [{
    This operation computes the maximum of two tensors element-wise.
  }];
  let arguments = (ins F32Tensor:$lhs, AnyTypeOf<[F32Tensor, F32]>:$rhs);
  let results = (outs F32Tensor:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict `->` type($result)";

  // let builders = [
  //   // OpBuilder<(ins "Value":$lhs, "Value":$rhs, "int64_t":$size)>,
  //   OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
  //     build($_builder, $_state, lhs, rhs);
  //   }]>,
  // ];
}

def Toy_RandomOp : Op<Toy_Dialect, "random"> {
  let summary = "Random operation for the toy dialect";
  let description = [{
    This operation generates random elements for a tensor.
  }];
  let arguments = (ins F32Tensor:$target);
  let results = (outs F32Tensor:$result);

  let assemblyFormat = "$target `:` type($target) attr-dict `->` type($result)";
}

def Toy_StoreOp : Op<Toy_Dialect, "store"> {
  let summary = "Store operation for the toy dialect";
  let description = [{
    This operation store source tensor into destination tensor.
  }];
  let arguments = (ins F32Tensor:$src, F32Tensor:$dst);
  let results = (outs F32Tensor:$result);

  let assemblyFormat = "$src `,` $dst `:` type($src) `,` type($dst) attr-dict `->` type($result)";
}

def Toy_ReturnOp : Op<Toy_Dialect, "return", [HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func.func @foo() -> tensor<2xf32> {
        ...
        toy.return %0 : tensor<2xf32>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F32Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];
}
