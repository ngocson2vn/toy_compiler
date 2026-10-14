// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  toy.func @add_two_vectors(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: i32) -> tensor<?xf32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0 : tensor<?xf32>, index -> tensor<?xf32>
    %2 = toy.tie_dims %arg1, %0 : tensor<?xf32>, index -> tensor<?xf32>
    %3 = toy.tie_dims %arg2, %0 : tensor<?xf32>, index -> tensor<?xf32>
    %4 = toy.add %1, %2 : tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
    %5 = toy.store %4, %3 : tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
    toy.return %5 : tensor<?xf32>
  }
  toy.func @main() {
    %0 = toy.constant 256 : i32 -> i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = toy.alloc %1 : -> tensor<?xf32>
    %3 = toy.alloc %1 : -> tensor<?xf32>
    %4 = toy.alloc %1 : -> tensor<?xf32>
    %5 = toy.random %2 : tensor<?xf32> -> tensor<?xf32>
    %6 = toy.random %3 : tensor<?xf32> -> tensor<?xf32>
    %7 = toy.call @add_two_vectors(%2, %3, %4, %0) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, i32) -> tensor<?xf32>
    %8 = toy.print %2 : tensor<?xf32> -> i64
    %9 = toy.print %3 : tensor<?xf32> -> i64
    %10 = toy.print %4 : tensor<?xf32> -> i64
    toy.return
  }
}


