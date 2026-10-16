// -----// IR Dump After ConvertToyToStdPass (convert-toy-to-std) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>, %arg3: i32, %arg4: i32) -> tensor<?x?xi32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %5 = toy.add %2, %3 : tensor<?x?xi32>, tensor<?x?xi32> -> tensor<?x?xi32>
    %6 = toy.store %5, %4 : tensor<?x?xi32>, tensor<?x?xi32> -> tensor<?x?xi32>
    return %6 : tensor<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %5 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %6 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %7 = toy.random %4 : tensor<?x?xi32> -> tensor<?x?xi32>
    %8 = toy.random %5 : tensor<?x?xi32> -> tensor<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, i32, i32) -> tensor<?x?xi32>
    %10 = toy.print %4 : tensor<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : tensor<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : tensor<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After ConvertToyToLinalgPass (convert-toy-to-linalg) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>, %arg3: i32, %arg4: i32) -> tensor<?x?xi32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %2, %c0 : tensor<?x?xi32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %2, %c1 : tensor<?x?xi32>
    %5 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%5 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %9 = arith.addi %in, %in_1 : i32
      linalg.yield %9 : i32
    } -> tensor<?x?xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%4 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %9 = arith.addi %in, %in_1 : i32
      linalg.yield %9 : i32
    } -> tensor<?x?xi32>
    %8 = bufferization.materialize_in_destination %7 in %4 : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    return %8 : tensor<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %5 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %6 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %7 = toy.random %4 : tensor<?x?xi32> -> tensor<?x?xi32>
    %8 = toy.random %5 : tensor<?x?xi32> -> tensor<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, i32, i32) -> tensor<?x?xi32>
    %10 = toy.print %4 : tensor<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : tensor<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : tensor<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>, %arg3: i32, %arg4: i32) -> tensor<?x?xi32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%4 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %7 = arith.addi %in, %in_0 : i32
      linalg.yield %7 : i32
    } -> tensor<?x?xi32>
    %6 = bufferization.materialize_in_destination %5 in %4 : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    return %6 : tensor<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %5 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %6 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %7 = toy.random %4 : tensor<?x?xi32> -> tensor<?x?xi32>
    %8 = toy.random %5 : tensor<?x?xi32> -> tensor<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, i32, i32) -> tensor<?x?xi32>
    %10 = toy.print %4 : tensor<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : tensor<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : tensor<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After LinalgElementwiseOpFusionPass (linalg-fuse-elementwise-ops) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>, %arg3: i32, %arg4: i32) -> tensor<?x?xi32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %dim = tensor.dim %4, %c0 : tensor<?x?xi32>
    %dim_0 = tensor.dim %4, %c1 : tensor<?x?xi32>
    %5 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%5 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %8 = arith.addi %in, %in_1 : i32
      linalg.yield %8 : i32
    } -> tensor<?x?xi32>
    %7 = bufferization.materialize_in_destination %6 in %4 : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    return %7 : tensor<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %5 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %6 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %7 = toy.random %4 : tensor<?x?xi32> -> tensor<?x?xi32>
    %8 = toy.random %5 : tensor<?x?xi32> -> tensor<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, i32, i32) -> tensor<?x?xi32>
    %10 = toy.print %4 : tensor<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : tensor<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : tensor<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>, %arg3: i32, %arg4: i32) -> tensor<?x?xi32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %5 = tensor.empty(%0, %1) : tensor<?x?xi32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%5 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %8 = arith.addi %in, %in_0 : i32
      linalg.yield %8 : i32
    } -> tensor<?x?xi32>
    %7 = bufferization.materialize_in_destination %6 in %4 : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    return %7 : tensor<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %5 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %6 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %7 = toy.random %4 : tensor<?x?xi32> -> tensor<?x?xi32>
    %8 = toy.random %5 : tensor<?x?xi32> -> tensor<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, i32, i32) -> tensor<?x?xi32>
    %10 = toy.print %4 : tensor<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : tensor<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : tensor<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After EmptyTensorEliminationPass (eliminate-empty-tensors) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>, %arg3: i32, %arg4: i32) -> tensor<?x?xi32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %5 = tensor.empty(%0, %1) : tensor<?x?xi32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%4 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %8 = arith.addi %in, %in_0 : i32
      linalg.yield %8 : i32
    } -> tensor<?x?xi32>
    %7 = bufferization.materialize_in_destination %6 in %4 : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    return %7 : tensor<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %5 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %6 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %7 = toy.random %4 : tensor<?x?xi32> -> tensor<?x?xi32>
    %8 = toy.random %5 : tensor<?x?xi32> -> tensor<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, i32, i32) -> tensor<?x?xi32>
    %10 = toy.print %4 : tensor<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : tensor<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : tensor<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After FoldMaterializeInDestinationOpPass (fold-materialize-in-destination-op) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>, %arg3: i32, %arg4: i32) -> tensor<?x?xi32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %5 = tensor.empty(%0, %1) : tensor<?x?xi32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%4 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %7 = arith.addi %in, %in_0 : i32
      linalg.yield %7 : i32
    } -> tensor<?x?xi32>
    return %6 : tensor<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %5 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %6 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %7 = toy.random %4 : tensor<?x?xi32> -> tensor<?x?xi32>
    %8 = toy.random %5 : tensor<?x?xi32> -> tensor<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, i32, i32) -> tensor<?x?xi32>
    %10 = toy.print %4 : tensor<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : tensor<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : tensor<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>, %arg3: i32, %arg4: i32) -> tensor<?x?xi32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : tensor<?x?xi32>, index, index -> tensor<?x?xi32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%4 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %6 = arith.addi %in, %in_0 : i32
      linalg.yield %6 : i32
    } -> tensor<?x?xi32>
    return %5 : tensor<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %5 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %6 = toy.alloc %2, %3 : -> tensor<?x?xi32>
    %7 = toy.random %4 : tensor<?x?xi32> -> tensor<?x?xi32>
    %8 = toy.random %5 : tensor<?x?xi32> -> tensor<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, i32, i32) -> tensor<?x?xi32>
    %10 = toy.print %4 : tensor<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : tensor<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : tensor<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After ConvertTensorToMemRefPass (convert-tensor-to-memref) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32) -> memref<?x?xi32> {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<?x?xi32> to tensor<?x?xi32>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<?x?xi32> to tensor<?x?xi32>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<?x?xi32> to tensor<?x?xi32>
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg4 : i32 to index
    %5 = toy.tie_dims %arg0, %3, %4 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %6 = toy.tie_dims %arg1, %3, %4 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %7 = toy.tie_dims %arg2, %3, %4 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %8 = builtin.unrealized_conversion_cast %7 : memref<?x?xi32> to tensor<?x?xi32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %6 : memref<?x?xi32>, memref<?x?xi32>) outs(%7 : memref<?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %10 = arith.addi %in, %in_0 : i32
      linalg.yield %10 : i32
    }
    %9 = builtin.unrealized_conversion_cast %8 : tensor<?x?xi32> to memref<?x?xi32>
    return %9 : memref<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %7 = toy.random %4 : memref<?x?xi32> -> memref<?x?xi32>
    %8 = toy.random %5 : memref<?x?xi32> -> memref<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, i32, i32) -> memref<?x?xi32>
    %10 = toy.print %4 : memref<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : memref<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : memref<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32) -> memref<?x?xi32> {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : memref<?x?xi32>, memref<?x?xi32>) outs(%4 : memref<?x?xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %5 = arith.addi %in, %in_0 : i32
      linalg.yield %5 : i32
    }
    return %4 : memref<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %7 = toy.random %4 : memref<?x?xi32> -> memref<?x?xi32>
    %8 = toy.random %5 : memref<?x?xi32> -> memref<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, i32, i32) -> memref<?x?xi32>
    %10 = toy.print %4 : memref<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : memref<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : memref<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After ConvertLinalgToParallelLoopsPass (convert-linalg-to-parallel-loops) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32) -> memref<?x?xi32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %dim = memref.dim %2, %c0 : memref<?x?xi32>
    %dim_0 = memref.dim %2, %c1 : memref<?x?xi32>
    scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%dim, %dim_0) step (%c1, %c1) {
      %5 = memref.load %2[%arg5, %arg6] : memref<?x?xi32>
      %6 = memref.load %3[%arg5, %arg6] : memref<?x?xi32>
      %7 = arith.addi %5, %6 : i32
      memref.store %7, %4[%arg5, %arg6] : memref<?x?xi32>
      scf.reduce 
    }
    return %4 : memref<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %7 = toy.random %4 : memref<?x?xi32> -> memref<?x?xi32>
    %8 = toy.random %5 : memref<?x?xi32> -> memref<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, i32, i32) -> memref<?x?xi32>
    %10 = toy.print %4 : memref<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : memref<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : memref<?x?xi32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32) -> memref<?x?xi32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%0, %1) step (%c1, %c1) {
      %5 = memref.load %2[%arg5, %arg6] : memref<?x?xi32>
      %6 = memref.load %3[%arg5, %arg6] : memref<?x?xi32>
      %7 = arith.addi %5, %6 : i32
      memref.store %7, %4[%arg5, %arg6] : memref<?x?xi32>
      scf.reduce 
    }
    return %4 : memref<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 4 : i32 -> i32
    %1 = toy.constant 8 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %7 = toy.random %4 : memref<?x?xi32> -> memref<?x?xi32>
    %8 = toy.random %5 : memref<?x?xi32> -> memref<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, i32, i32) -> memref<?x?xi32>
    %10 = toy.print %4 : memref<?x?xi32> {argName = "a"} -> i64
    %11 = toy.print %5 : memref<?x?xi32> {argName = "b"} -> i64
    %12 = toy.print %6 : memref<?x?xi32> {argName = "out"} -> i64
    return
  }
}


