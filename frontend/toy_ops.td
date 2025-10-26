include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
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

// def Toy_ConstantOp : Op<Toy_Dialect, "constant"> {
//   // Provide a summary and description for this operation. This can be used to
//   // auto-generate documentation of the operations within our dialect.
//   let summary = "constant";
//   let description = [{
//     Constant operation turns a literal into an SSA value. The data is attached
//     to the operation as an attribute. For example:

//     ```mlir
//       %0 = toy.constant 42 : i64
//     ```
//   }];

//   // The constant operation takes an attribute as the only input.
//   let arguments = (ins AnyTypeOf<[I64, F32]>:$value);

//   // The constant operation returns a single value of TensorType.
//   let results = (outs AnyTypeOf<[I64, F32]>:$result);

//   let assemblyFormat = "$value `:` type($value) attr-dict `->` type($result)";

//   let builders = [
//     OpBuilder<(ins "int64_t":$value)>,

//     OpBuilder<(ins "double":$value)>
//   ];
// }

def Toy_AddOp : Op<Toy_Dialect, "add"> {
  let summary = "Addition operation for the toy dialect";
  let description = [{
    This operation computes the sum of two tensors element-wise.
  }];
  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs, I64:$size);
  let results = (outs F32Tensor:$result);

  let assemblyFormat = "$lhs `,` $rhs `,` $size `:` type($lhs) `,` type($rhs) `,` type($size) attr-dict `->` type($result)";

  let builders = [
    // OpBuilder<(ins "Value":$lhs, "Value":$rhs, "int64_t":$size)>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "int64_t":$size), [{
      build($_builder, $_state, lhs, rhs, size);
    }]>,
  ];
}

def Toy_StoreOp : Op<Toy_Dialect, "store"> {
  let summary = "Store operation for the toy dialect";
  let description = [{
    This operation store source tensor into destination tensor.
  }];
  let arguments = (ins F32Tensor:$src, F32Tensor:$dst);

  let assemblyFormat = "$src `,` $dst `:` type($src) `,` type($dst) attr-dict";
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
