// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  toy.func @add_two_vectors(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %arg3: i32) -> tensor<?x8xf32> {
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %4 = toy.add %1, %2 : tensor<?x8xf32>, tensor<?x8xf32> -> tensor<?x8xf32>
    %5 = toy.store %4, %3 : tensor<?x8xf32>, tensor<?x8xf32> -> tensor<?x8xf32>
    toy.return %5 : tensor<?x8xf32>
  }
  toy.func @main() {
    %c8 = arith.constant 8 : index
    %0 = toy.constant 4 : i32 -> i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = toy.alloc %1, %c8 : -> tensor<?x8xf32>
    %3 = toy.alloc %1, %c8 : -> tensor<?x8xf32>
    %4 = toy.alloc %1, %c8 : -> tensor<?x8xf32>
    %5 = toy.random %2 : tensor<?x8xf32> -> tensor<?x8xf32>
    %6 = toy.random %3 : tensor<?x8xf32> -> tensor<?x8xf32>
    %7 = toy.call @add_two_vectors(%2, %3, %4, %0) : (tensor<?x8xf32>, tensor<?x8xf32>, tensor<?x8xf32>, i32) -> tensor<?x8xf32>
    %8 = toy.print %2 : tensor<?x8xf32> {argName = "a"} -> i64
    %9 = toy.print %3 : tensor<?x8xf32> {argName = "b"} -> i64
    %10 = toy.print %4 : tensor<?x8xf32> {argName = "out"} -> i64
    toy.return
  }
}


