// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  toy.func @add_two_vectors(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>, %arg3: i32, %arg4: i32) -> tensor<?x?xi32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %5 = toy.add %2, %3 : tensor<?x?xi32>, tensor<?x?xi32> -> tensor<?x?xi32>
    %6 = toy.store %5, %4 : tensor<?x?xi32>, tensor<?x?xi32> -> tensor<?x?xi32>
    toy.return %6 : tensor<?x?xi32>
  }
}


