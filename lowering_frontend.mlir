// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  toy.func @add_two_vectors(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: i64) {
    %0 = toy.add %arg0, %arg1, %arg3 : tensor<?xf32>, tensor<?xf32>, i64 -> tensor<?xf32>
    toy.store %0, %arg2 : tensor<?xf32>, tensor<?xf32>
    toy.return
  }
}


