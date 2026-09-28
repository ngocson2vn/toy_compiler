// -----// IR Dump After ConvertToyToStdPass (convert-toy-to-std) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: i64) {
    %0 = toy.add %arg0, %arg1, %arg3 : tensor<?xf32>, tensor<?xf32>, i64 -> tensor<?xf32>
    toy.store %0, %arg2 : tensor<?xf32>, tensor<?xf32>
    return
  }
}


// -----// IR Dump After ConvertTensorToMemRefPass (convert-tensor-to-memref) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i64) {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<?xf32> to tensor<?xf32>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<?xf32> to tensor<?xf32>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<?xf32> to tensor<?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %3 = arith.index_cast %arg3 : i64 to index
    scf.parallel (%arg4) = (%c0) to (%3) step (%c1) {
      %4 = memref.load %arg0[%arg4] : memref<?xf32>
      %5 = memref.load %arg1[%arg4] : memref<?xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %arg2[%arg4] : memref<?xf32>
      scf.reduce 
    }
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i64) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i64 to index
    scf.parallel (%arg4) = (%c0) to (%0) step (%c1) {
      %1 = memref.load %arg0[%arg4] : memref<?xf32>
      %2 = memref.load %arg1[%arg4] : memref<?xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %arg2[%arg4] : memref<?xf32>
      scf.reduce 
    }
    return
  }
}


