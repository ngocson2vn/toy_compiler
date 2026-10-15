// -----// IR Dump After ConvertToyToStdPass (convert-toy-to-std) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %arg3: i32) -> tensor<?x8xf32> {
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %4 = toy.add %1, %2 : tensor<?x8xf32>, tensor<?x8xf32> -> tensor<?x8xf32>
    %5 = toy.store %4, %3 : tensor<?x8xf32>, tensor<?x8xf32> -> tensor<?x8xf32>
    return %5 : tensor<?x8xf32>
  }
  func.func @main() {
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
    return
  }
}


// -----// IR Dump After ConvertToyToLinalgPass (convert-toy-to-linalg) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %arg3: i32) -> tensor<?x8xf32> {
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %1, %c0 : tensor<?x8xf32>
    %4 = tensor.empty(%dim) : tensor<?x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%4 : tensor<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.addf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<?x8xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%3 : tensor<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.addf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<?x8xf32>
    %7 = bufferization.materialize_in_destination %6 in %3 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
    return %7 : tensor<?x8xf32>
  }
  func.func @main() {
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
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %arg3: i32) -> tensor<?x8xf32> {
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%3 : tensor<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.addf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<?x8xf32>
    %5 = bufferization.materialize_in_destination %4 in %3 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
    return %5 : tensor<?x8xf32>
  }
  func.func @main() {
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
    return
  }
}


// -----// IR Dump After LinalgElementwiseOpFusionPass (linalg-fuse-elementwise-ops) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %arg3: i32) -> tensor<?x8xf32> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %dim = tensor.dim %3, %c0 : tensor<?x8xf32>
    %4 = tensor.empty(%dim) : tensor<?x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%4 : tensor<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<?x8xf32>
    %6 = bufferization.materialize_in_destination %5 in %3 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
    return %6 : tensor<?x8xf32>
  }
  func.func @main() {
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
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %arg3: i32) -> tensor<?x8xf32> {
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %4 = tensor.empty(%0) : tensor<?x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%4 : tensor<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<?x8xf32>
    %6 = bufferization.materialize_in_destination %5 in %3 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
    return %6 : tensor<?x8xf32>
  }
  func.func @main() {
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
    return
  }
}


// -----// IR Dump After EmptyTensorEliminationPass (eliminate-empty-tensors) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %arg3: i32) -> tensor<?x8xf32> {
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %4 = tensor.empty(%0) : tensor<?x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%3 : tensor<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<?x8xf32>
    %6 = bufferization.materialize_in_destination %5 in %3 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
    return %6 : tensor<?x8xf32>
  }
  func.func @main() {
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
    return
  }
}


// -----// IR Dump After FoldMaterializeInDestinationOpPass (fold-materialize-in-destination-op) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %arg3: i32) -> tensor<?x8xf32> {
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %4 = tensor.empty(%0) : tensor<?x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%3 : tensor<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.addf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<?x8xf32>
    return %5 : tensor<?x8xf32>
  }
  func.func @main() {
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
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>, %arg2: tensor<?x8xf32>, %arg3: i32) -> tensor<?x8xf32> {
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : tensor<?x8xf32>, index, index -> tensor<?x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<?x8xf32>, tensor<?x8xf32>) outs(%3 : tensor<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
    } -> tensor<?x8xf32>
    return %4 : tensor<?x8xf32>
  }
  func.func @main() {
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
    return
  }
}


// -----// IR Dump After ConvertTensorToMemRefPass (convert-tensor-to-memref) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xf32>, %arg3: i32) -> memref<?x8xf32> {
    %0 = builtin.unrealized_conversion_cast %arg2 : memref<?x8xf32> to tensor<?x8xf32>
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<?x8xf32> to tensor<?x8xf32>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<?x8xf32> to tensor<?x8xf32>
    %c8 = arith.constant 8 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = toy.tie_dims %arg0, %3, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %5 = toy.tie_dims %arg1, %3, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %6 = toy.tie_dims %arg2, %3, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %7 = builtin.unrealized_conversion_cast %6 : memref<?x8xf32> to tensor<?x8xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : memref<?x8xf32>, memref<?x8xf32>) outs(%6 : memref<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %9 = arith.addf %in, %in_0 : f32
      linalg.yield %9 : f32
    }
    %8 = builtin.unrealized_conversion_cast %7 : tensor<?x8xf32> to memref<?x8xf32>
    return %8 : memref<?x8xf32>
  }
  func.func @main() {
    %c8 = arith.constant 8 : index
    %0 = toy.constant 4 : i32 -> i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %3 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %4 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %5 = toy.random %2 : memref<?x8xf32> -> memref<?x8xf32>
    %6 = toy.random %3 : memref<?x8xf32> -> memref<?x8xf32>
    %7 = toy.call @add_two_vectors(%2, %3, %4, %0) : (memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>, i32) -> memref<?x8xf32>
    %8 = toy.print %2 : memref<?x8xf32> {argName = "a"} -> i64
    %9 = toy.print %3 : memref<?x8xf32> {argName = "b"} -> i64
    %10 = toy.print %4 : memref<?x8xf32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_two_vectors(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xf32>, %arg3: i32) -> memref<?x8xf32> {
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : memref<?x8xf32>, memref<?x8xf32>) outs(%3 : memref<?x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    }
    return %3 : memref<?x8xf32>
  }
  func.func @main() {
    %c8 = arith.constant 8 : index
    %0 = toy.constant 4 : i32 -> i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %3 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %4 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %5 = toy.random %2 : memref<?x8xf32> -> memref<?x8xf32>
    %6 = toy.random %3 : memref<?x8xf32> -> memref<?x8xf32>
    %7 = toy.call @add_two_vectors(%2, %3, %4, %0) : (memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>, i32) -> memref<?x8xf32>
    %8 = toy.print %2 : memref<?x8xf32> {argName = "a"} -> i64
    %9 = toy.print %3 : memref<?x8xf32> {argName = "b"} -> i64
    %10 = toy.print %4 : memref<?x8xf32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After ConvertLinalgToParallelLoopsPass (convert-linalg-to-parallel-loops) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xf32>, %arg3: i32) -> memref<?x8xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %dim = memref.dim %1, %c0 : memref<?x8xf32>
    scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%dim, %c8) step (%c1, %c1) {
      %4 = memref.load %1[%arg4, %arg5] : memref<?x8xf32>
      %5 = memref.load %2[%arg4, %arg5] : memref<?x8xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %3[%arg4, %arg5] : memref<?x8xf32>
      scf.reduce 
    }
    return %3 : memref<?x8xf32>
  }
  func.func @main() {
    %c8 = arith.constant 8 : index
    %0 = toy.constant 4 : i32 -> i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %3 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %4 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %5 = toy.random %2 : memref<?x8xf32> -> memref<?x8xf32>
    %6 = toy.random %3 : memref<?x8xf32> -> memref<?x8xf32>
    %7 = toy.call @add_two_vectors(%2, %3, %4, %0) : (memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>, i32) -> memref<?x8xf32>
    %8 = toy.print %2 : memref<?x8xf32> {argName = "a"} -> i64
    %9 = toy.print %3 : memref<?x8xf32> {argName = "b"} -> i64
    %10 = toy.print %4 : memref<?x8xf32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xf32>, %arg3: i32) -> memref<?x8xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = toy.tie_dims %arg0, %0, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %2 = toy.tie_dims %arg1, %0, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    %3 = toy.tie_dims %arg2, %0, %c8 : memref<?x8xf32>, index, index -> memref<?x8xf32>
    scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%0, %c8) step (%c1, %c1) {
      %4 = memref.load %1[%arg4, %arg5] : memref<?x8xf32>
      %5 = memref.load %2[%arg4, %arg5] : memref<?x8xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %3[%arg4, %arg5] : memref<?x8xf32>
      scf.reduce 
    }
    return %3 : memref<?x8xf32>
  }
  func.func @main() {
    %c8 = arith.constant 8 : index
    %0 = toy.constant 4 : i32 -> i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %3 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %4 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %5 = toy.random %2 : memref<?x8xf32> -> memref<?x8xf32>
    %6 = toy.random %3 : memref<?x8xf32> -> memref<?x8xf32>
    %7 = toy.call @add_two_vectors(%2, %3, %4, %0) : (memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>, i32) -> memref<?x8xf32>
    %8 = toy.print %2 : memref<?x8xf32> {argName = "a"} -> i64
    %9 = toy.print %3 : memref<?x8xf32> {argName = "b"} -> i64
    %10 = toy.print %4 : memref<?x8xf32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After FoldTieDimsOpPass (fold-ties-dims-op) ('builtin.module' operation) //----- //
module {
  func.func @add_two_vectors(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xf32>, %arg3: i32) -> memref<?x8xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%0, %c8) step (%c1, %c1) {
      %1 = memref.load %arg0[%arg4, %arg5] : memref<?x8xf32>
      %2 = memref.load %arg1[%arg4, %arg5] : memref<?x8xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %arg2[%arg4, %arg5] : memref<?x8xf32>
      scf.reduce 
    }
    return %arg2 : memref<?x8xf32>
  }
  func.func @main() {
    %c8 = arith.constant 8 : index
    %0 = toy.constant 4 : i32 -> i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %3 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %4 = toy.alloc %1, %c8 : -> memref<?x8xf32>
    %5 = toy.random %2 : memref<?x8xf32> -> memref<?x8xf32>
    %6 = toy.random %3 : memref<?x8xf32> -> memref<?x8xf32>
    %7 = toy.call @add_two_vectors(%2, %3, %4, %0) : (memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>, i32) -> memref<?x8xf32>
    %8 = toy.print %2 : memref<?x8xf32> {argName = "a"} -> i64
    %9 = toy.print %3 : memref<?x8xf32> {argName = "b"} -> i64
    %10 = toy.print %4 : memref<?x8xf32> {argName = "out"} -> i64
    return
  }
}


