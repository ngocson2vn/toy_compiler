// -----// IR Dump After CollapseParallelLoopsTo1DPass (collapse-parallel-loops-to-1d) ('func.func' operation: @add_max_vectors) //----- //
module {
  func.func @add_max_vectors(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: i32, %arg4: i32) -> memref<?x?xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %5 = toy.constant 1.500000e-01 : f32 -> f32
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    %6 = arith.muli %c1_2, %0 : index
    %7 = arith.muli %6, %1 : index
    scf.parallel (%arg5) = (%c0_0) to (%7) step (%c1_1) {
      %8 = arith.remsi %arg5, %1 : index
      %9 = arith.divsi %arg5, %1 : index
      %10 = memref.load %2[%9, %8] : memref<?x?xf32>
      %11 = memref.load %3[%9, %8] : memref<?x?xf32>
      %12 = arith.addf %10, %11 : f32
      %13 = arith.maximumf %12, %5 : f32
      memref.store %13, %4[%9, %8] : memref<?x?xf32>
      scf.reduce 
    }
    return %4 : memref<?x?xf32>
  }
  func.func @main() {
    %0 = toy.constant 8 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %7 = toy.random %4 : memref<?x?xf32> -> memref<?x?xf32>
    %8 = toy.random %5 : memref<?x?xf32> -> memref<?x?xf32>
    %9 = toy.call @add_max_vectors(%4, %5, %6, %0, %1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, i32, i32) -> memref<?x?xf32>
    %10 = toy.print %4 : memref<?x?xf32> {argName = "a"} -> i64
    %11 = toy.print %5 : memref<?x?xf32> {argName = "b"} -> i64
    %12 = toy.print %6 : memref<?x?xf32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  func.func @add_max_vectors(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: i32, %arg4: i32) -> memref<?x?xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %5 = toy.constant 1.500000e-01 : f32 -> f32
    %6 = arith.muli %0, %1 : index
    scf.parallel (%arg5) = (%c0) to (%6) step (%c1) {
      %7 = arith.remsi %arg5, %1 : index
      %8 = arith.divsi %arg5, %1 : index
      %9 = memref.load %2[%8, %7] : memref<?x?xf32>
      %10 = memref.load %3[%8, %7] : memref<?x?xf32>
      %11 = arith.addf %9, %10 : f32
      %12 = arith.maximumf %11, %5 : f32
      memref.store %12, %4[%8, %7] : memref<?x?xf32>
      scf.reduce 
    }
    return %4 : memref<?x?xf32>
  }
  func.func @main() {
    %0 = toy.constant 8 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %7 = toy.random %4 : memref<?x?xf32> -> memref<?x?xf32>
    %8 = toy.random %5 : memref<?x?xf32> -> memref<?x?xf32>
    %9 = toy.call @add_max_vectors(%4, %5, %6, %0, %1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, i32, i32) -> memref<?x?xf32>
    %10 = toy.print %4 : memref<?x?xf32> {argName = "a"} -> i64
    %11 = toy.print %5 : memref<?x?xf32> {argName = "b"} -> i64
    %12 = toy.print %6 : memref<?x?xf32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After TileLoopsPass (tile-loops) ('func.func' operation: @add_max_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_max_vectors(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: i32, %arg4: i32) -> memref<?x?xf32> {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %5 = toy.constant 1.500000e-01 : f32 -> f32
    %6 = arith.muli %0, %1 : index
    scf.parallel (%arg5) = (%c0) to (%6) step (%c128) {
      %7 = affine.min #map(%c128, %6, %arg5)
      scf.parallel (%arg6) = (%c0) to (%7) step (%c1) {
        %8 = arith.addi %arg6, %arg5 : index
        %9 = arith.remsi %8, %1 : index
        %10 = arith.divsi %8, %1 : index
        %11 = memref.load %2[%10, %9] : memref<?x?xf32>
        %12 = memref.load %3[%10, %9] : memref<?x?xf32>
        %13 = arith.addf %11, %12 : f32
        %14 = arith.maximumf %13, %5 : f32
        memref.store %14, %4[%10, %9] : memref<?x?xf32>
        scf.reduce 
      }
      scf.reduce 
    }
    return %4 : memref<?x?xf32>
  }
  func.func @main() {
    %0 = toy.constant 8 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %7 = toy.random %4 : memref<?x?xf32> -> memref<?x?xf32>
    %8 = toy.random %5 : memref<?x?xf32> -> memref<?x?xf32>
    %9 = toy.call @add_max_vectors(%4, %5, %6, %0, %1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, i32, i32) -> memref<?x?xf32>
    %10 = toy.print %4 : memref<?x?xf32> {argName = "a"} -> i64
    %11 = toy.print %5 : memref<?x?xf32> {argName = "b"} -> i64
    %12 = toy.print %6 : memref<?x?xf32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After GpuMapParallelLoopsPass (gpu-map-parallel-loops) ('func.func' operation: @add_max_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_max_vectors(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: i32, %arg4: i32) -> memref<?x?xf32> {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %5 = toy.constant 1.500000e-01 : f32 -> f32
    %6 = arith.muli %0, %1 : index
    scf.parallel (%arg5) = (%c0) to (%6) step (%c128) {
      %7 = affine.min #map(%c128, %6, %arg5)
      scf.parallel (%arg6) = (%c0) to (%7) step (%c1) {
        %8 = arith.addi %arg6, %arg5 : index
        %9 = arith.remsi %8, %1 : index
        %10 = arith.divsi %8, %1 : index
        %11 = memref.load %2[%10, %9] : memref<?x?xf32>
        %12 = memref.load %3[%10, %9] : memref<?x?xf32>
        %13 = arith.addf %11, %12 : f32
        %14 = arith.maximumf %13, %5 : f32
        memref.store %14, %4[%10, %9] : memref<?x?xf32>
        scf.reduce 
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.reduce 
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %4 : memref<?x?xf32>
  }
  func.func @main() {
    %0 = toy.constant 8 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %7 = toy.random %4 : memref<?x?xf32> -> memref<?x?xf32>
    %8 = toy.random %5 : memref<?x?xf32> -> memref<?x?xf32>
    %9 = toy.call @add_max_vectors(%4, %5, %6, %0, %1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, i32, i32) -> memref<?x?xf32>
    %10 = toy.print %4 : memref<?x?xf32> {argName = "a"} -> i64
    %11 = toy.print %5 : memref<?x?xf32> {argName = "b"} -> i64
    %12 = toy.print %6 : memref<?x?xf32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After ConvertParallelLoopToGpuPass (convert-parallel-loops-to-gpu) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_max_vectors(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: i32, %arg4: i32) -> memref<?x?xf32> {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xf32>, index, index -> memref<?x?xf32>
    %5 = toy.constant 1.500000e-01 : f32 -> f32
    %6 = arith.muli %0, %1 : index
    %c1_0 = arith.constant 1 : index
    %7 = affine.apply #map(%6)[%c0, %c128]
    %c128_1 = arith.constant 128 : index
    %8 = affine.apply #map(%c128_1)[%c0, %c1]
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %7, %arg12 = %c1_0, %arg13 = %c1_0) threads(%arg8, %arg9, %arg10) in (%arg14 = %8, %arg15 = %c1_0, %arg16 = %c1_0) {
      %9 = affine.apply #map1(%arg5)[%c128, %c0]
      %10 = affine.min #map2(%c128, %6, %9)
      %11 = affine.apply #map1(%arg8)[%c1, %c0]
      %12 = arith.cmpi slt, %11, %10 : index
      scf.if %12 {
        %13 = arith.addi %11, %9 : index
        %14 = arith.remsi %13, %1 : index
        %15 = arith.divsi %13, %1 : index
        %16 = memref.load %2[%15, %14] : memref<?x?xf32>
        %17 = memref.load %3[%15, %14] : memref<?x?xf32>
        %18 = arith.addf %16, %17 : f32
        %19 = arith.maximumf %18, %5 : f32
        memref.store %19, %4[%15, %14] : memref<?x?xf32>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    return %4 : memref<?x?xf32>
  }
  func.func @main() {
    %0 = toy.constant 8 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xf32>
    %7 = toy.random %4 : memref<?x?xf32> -> memref<?x?xf32>
    %8 = toy.random %5 : memref<?x?xf32> -> memref<?x?xf32>
    %9 = toy.call @add_max_vectors(%4, %5, %6, %0, %1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, i32, i32) -> memref<?x?xf32>
    %10 = toy.print %4 : memref<?x?xf32> {argName = "a"} -> i64
    %11 = toy.print %5 : memref<?x?xf32> {argName = "b"} -> i64
    %12 = toy.print %6 : memref<?x?xf32> {argName = "out"} -> i64
    return
  }
}


// -----// IR Dump After LowerMemRefToLLVMPass (lower-memref-to-llvm) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
    %3 = arith.muli %0, %1 : index
    %c1_0 = arith.constant 1 : index
    %4 = affine.apply #map(%3)[%c0, %c128]
    %c128_1 = arith.constant 128 : index
    %c128_2 = arith.constant 128 : index
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %4, %arg12 = %c1_0, %arg13 = %c1_0) threads(%arg8, %arg9, %arg10) in (%arg14 = %c128_2, %arg15 = %c1_0, %arg16 = %c1_0) {
      %5 = affine.apply #map1(%arg5)[%c128, %c0]
      %6 = affine.min #map2(%c128, %3, %5)
      %7 = affine.apply #map1(%arg8)[%c1, %c0]
      %8 = arith.cmpi slt, %7, %6 : index
      scf.if %8 {
        %9 = arith.addi %7, %5 : index
        %10 = arith.remsi %9, %1 : index
        %11 = builtin.unrealized_conversion_cast %10 : index to i64
        %12 = arith.divsi %9, %1 : index
        %13 = builtin.unrealized_conversion_cast %12 : index to i64
        %14 = llvm.mlir.constant(1 : i64) : i64
        %15 = arith.index_cast %1 : index to i64
        %16 = llvm.mul %14, %15 : i64
        %17 = llvm.mlir.constant(0 : i64) : i64
        %18 = llvm.mul %13, %16 : i64
        %19 = llvm.add %18, %17 : i64
        %20 = llvm.mul %11, %14 : i64
        %21 = llvm.add %20, %19 : i64
        %22 = llvm.getelementptr %arg0[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %23 = llvm.load %22 : !llvm.ptr -> f32
        %24 = llvm.mlir.constant(1 : i64) : i64
        %25 = arith.index_cast %1 : index to i64
        %26 = llvm.mul %24, %25 : i64
        %27 = llvm.mlir.constant(0 : i64) : i64
        %28 = llvm.mul %13, %26 : i64
        %29 = llvm.add %28, %27 : i64
        %30 = llvm.mul %11, %24 : i64
        %31 = llvm.add %30, %29 : i64
        %32 = llvm.getelementptr %arg1[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %33 = llvm.load %32 : !llvm.ptr -> f32
        %34 = arith.addf %23, %33 : f32
        %35 = arith.maximumf %34, %2 : f32
        %36 = llvm.mlir.constant(1 : i64) : i64
        %37 = arith.index_cast %1 : index to i64
        %38 = llvm.mul %36, %37 : i64
        %39 = llvm.mlir.constant(0 : i64) : i64
        %40 = llvm.mul %13, %38 : i64
        %41 = llvm.add %40, %39 : i64
        %42 = llvm.mul %11, %36 : i64
        %43 = llvm.add %42, %41 : i64
        %44 = llvm.getelementptr %arg2[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %35, %44 : f32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return %arg2 : !llvm.ptr
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(8 : i32) : i32
    %1 = llvm.mlir.constant(32 : i32) : i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.index_cast %2 : index to i32
    %5 = arith.index_cast %3 : index to i32
    %6 = llvm.mul %4, %5 : i32
    %7 = llvm.mlir.constant(3 : i32) : i32
    %8 = llvm.call @toy_alloc(%6, %7) : (i32, i32) -> !llvm.ptr
    %9 = arith.index_cast %2 : index to i32
    %10 = arith.index_cast %3 : index to i32
    %11 = llvm.mul %9, %10 : i32
    %12 = llvm.mlir.constant(3 : i32) : i32
    %13 = llvm.call @toy_alloc(%11, %12) : (i32, i32) -> !llvm.ptr
    %14 = arith.index_cast %2 : index to i32
    %15 = arith.index_cast %3 : index to i32
    %16 = llvm.mul %14, %15 : i32
    %17 = llvm.mlir.constant(3 : i32) : i32
    %18 = llvm.call @toy_alloc(%16, %17) : (i32, i32) -> !llvm.ptr
    %19 = arith.index_cast %2 : index to i32
    %20 = arith.index_cast %3 : index to i32
    %21 = llvm.mul %19, %20 : i32
    %22 = llvm.mlir.constant(3 : i32) : i32
    %23 = llvm.call @toy_random(%8, %21, %22) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %24 = arith.index_cast %2 : index to i32
    %25 = arith.index_cast %3 : index to i32
    %26 = llvm.mul %24, %25 : i32
    %27 = llvm.mlir.constant(3 : i32) : i32
    %28 = llvm.call @toy_random(%13, %26, %27) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %29 = llvm.call @add_max_vectors(%8, %13, %18, %0, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %30 = arith.index_cast %2 : index to i32
    %31 = arith.index_cast %3 : index to i32
    %32 = llvm.mul %30, %31 : i32
    %33 = llvm.mlir.constant(3 : i32) : i32
    %34 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %35 = llvm.mlir.constant(1 : i32) : i32
    %36 = llvm.alloca %35 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %34, %36 : !llvm.array<2 x i8>, !llvm.ptr
    %37 = llvm.call @toy_print(%8, %32, %33, %36) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %38 = arith.index_cast %2 : index to i32
    %39 = arith.index_cast %3 : index to i32
    %40 = llvm.mul %38, %39 : i32
    %41 = llvm.mlir.constant(3 : i32) : i32
    %42 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %43 = llvm.mlir.constant(1 : i32) : i32
    %44 = llvm.alloca %43 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %42, %44 : !llvm.array<2 x i8>, !llvm.ptr
    %45 = llvm.call @toy_print(%13, %40, %41, %44) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %46 = arith.index_cast %2 : index to i32
    %47 = arith.index_cast %3 : index to i32
    %48 = llvm.mul %46, %47 : i32
    %49 = llvm.mlir.constant(3 : i32) : i32
    %50 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %51 = llvm.mlir.constant(1 : i32) : i32
    %52 = llvm.alloca %51 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %50, %52 : !llvm.array<4 x i8>, !llvm.ptr
    %53 = llvm.call @toy_print(%18, %48, %49, %52) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * -128 + s1, 128)>
module {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg4 : i32 to index
    %5 = arith.muli %3, %4 : index
    %6 = affine.apply #map()[%5]
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %6, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c128, %arg15 = %c1, %arg16 = %c1) {
      %7 = affine.apply #map1()[%arg5]
      %8 = affine.min #map2()[%arg5, %5]
      %9 = arith.cmpi slt, %arg8, %8 : index
      scf.if %9 {
        %10 = arith.addi %arg8, %7 : index
        %11 = arith.remsi %10, %4 : index
        %12 = builtin.unrealized_conversion_cast %11 : index to i64
        %13 = arith.divsi %10, %4 : index
        %14 = builtin.unrealized_conversion_cast %13 : index to i64
        %15 = arith.index_cast %4 : index to i64
        %16 = llvm.mul %15, %1 : i64
        %17 = llvm.mul %14, %16 : i64
        %18 = llvm.add %17, %0 : i64
        %19 = llvm.mul %12, %1 : i64
        %20 = llvm.add %19, %18 : i64
        %21 = llvm.getelementptr %arg0[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %22 = llvm.load %21 : !llvm.ptr -> f32
        %23 = arith.index_cast %4 : index to i64
        %24 = llvm.mul %23, %1 : i64
        %25 = llvm.mul %14, %24 : i64
        %26 = llvm.add %25, %0 : i64
        %27 = llvm.mul %12, %1 : i64
        %28 = llvm.add %27, %26 : i64
        %29 = llvm.getelementptr %arg1[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %30 = llvm.load %29 : !llvm.ptr -> f32
        %31 = arith.addf %22, %30 : f32
        %32 = arith.maximumf %31, %2 : f32
        %33 = arith.index_cast %4 : index to i64
        %34 = llvm.mul %33, %1 : i64
        %35 = llvm.mul %14, %34 : i64
        %36 = llvm.add %35, %0 : i64
        %37 = llvm.mul %12, %1 : i64
        %38 = llvm.add %37, %36 : i64
        %39 = llvm.getelementptr %arg2[%38] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %32, %39 : f32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return %arg2 : !llvm.ptr
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.mul %c8_i32, %c32_i32 : i32
    %10 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.mul %c8_i32, %c32_i32 : i32
    %12 = llvm.call @toy_alloc(%11, %4) : (i32, i32) -> !llvm.ptr
    %13 = llvm.mul %c8_i32, %c32_i32 : i32
    %14 = llvm.call @toy_random(%8, %13, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %15 = llvm.mul %c8_i32, %c32_i32 : i32
    %16 = llvm.call @toy_random(%10, %15, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %17 = llvm.call @add_max_vectors(%8, %10, %12, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %18 = llvm.mul %c8_i32, %c32_i32 : i32
    %19 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %19 : !llvm.array<2 x i8>, !llvm.ptr
    %20 = llvm.call @toy_print(%8, %18, %4, %19) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %21 = llvm.mul %c8_i32, %c32_i32 : i32
    %22 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %22 : !llvm.array<2 x i8>, !llvm.ptr
    %23 = llvm.call @toy_print(%10, %21, %4, %22) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %24 = llvm.mul %c8_i32, %c32_i32 : i32
    %25 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %25 : !llvm.array<4 x i8>, !llvm.ptr
    %26 = llvm.call @toy_print(%12, %24, %4, %25) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After CSE (cse) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * -128 + s1, 128)>
module {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg4 : i32 to index
    %5 = arith.muli %3, %4 : index
    %6 = affine.apply #map()[%5]
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %6, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c128, %arg15 = %c1, %arg16 = %c1) {
      %7 = affine.apply #map1()[%arg5]
      %8 = affine.min #map2()[%arg5, %5]
      %9 = arith.cmpi slt, %arg8, %8 : index
      scf.if %9 {
        %10 = arith.addi %arg8, %7 : index
        %11 = arith.remsi %10, %4 : index
        %12 = builtin.unrealized_conversion_cast %11 : index to i64
        %13 = arith.divsi %10, %4 : index
        %14 = builtin.unrealized_conversion_cast %13 : index to i64
        %15 = arith.index_cast %4 : index to i64
        %16 = llvm.mul %15, %1 : i64
        %17 = llvm.mul %14, %16 : i64
        %18 = llvm.add %17, %0 : i64
        %19 = llvm.mul %12, %1 : i64
        %20 = llvm.add %19, %18 : i64
        %21 = llvm.getelementptr %arg0[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %22 = llvm.load %21 : !llvm.ptr -> f32
        %23 = llvm.getelementptr %arg1[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %24 = llvm.load %23 : !llvm.ptr -> f32
        %25 = arith.addf %22, %24 : f32
        %26 = arith.maximumf %25, %2 : f32
        %27 = llvm.getelementptr %arg2[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %26, %27 : f32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return %arg2 : !llvm.ptr
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuLaunchSinkIndexComputationsPass (gpu-launch-sink-index-computations) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * -128 + s1, 128)>
module {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg4 : i32 to index
    %5 = arith.muli %3, %4 : index
    %6 = affine.apply #map()[%5]
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %6, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c128, %arg15 = %c1, %arg16 = %c1) {
      %7 = llvm.mlir.constant(1 : i64) : i64
      %8 = llvm.mlir.constant(0 : i64) : i64
      %9 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %10 = affine.apply #map1()[%arg5]
      %11 = affine.min #map2()[%arg5, %5]
      %12 = arith.cmpi slt, %arg8, %11 : index
      scf.if %12 {
        %13 = arith.addi %arg8, %10 : index
        %14 = arith.remsi %13, %4 : index
        %15 = builtin.unrealized_conversion_cast %14 : index to i64
        %16 = arith.divsi %13, %4 : index
        %17 = builtin.unrealized_conversion_cast %16 : index to i64
        %18 = arith.index_cast %4 : index to i64
        %19 = llvm.mul %18, %7 : i64
        %20 = llvm.mul %17, %19 : i64
        %21 = llvm.add %20, %8 : i64
        %22 = llvm.mul %15, %7 : i64
        %23 = llvm.add %22, %21 : i64
        %24 = llvm.getelementptr %arg0[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %25 = llvm.load %24 : !llvm.ptr -> f32
        %26 = llvm.getelementptr %arg1[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %27 = llvm.load %26 : !llvm.ptr -> f32
        %28 = arith.addf %25, %27 : f32
        %29 = arith.maximumf %28, %9 : f32
        %30 = llvm.getelementptr %arg2[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %29, %30 : f32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return %arg2 : !llvm.ptr
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuKernelOutliningPass (gpu-kernel-outlining) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * -128 + s1, 128)>
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg4 : i32 to index
    %5 = arith.muli %3, %4 : index
    %6 = affine.apply #map()[%5]
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%6, %c1, %c1) threads in (%c128, %c1, %c1)  args(%5 : index, %4 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_max_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_max_vectors_kernel(%arg0: index, %arg1: index, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = llvm.mlir.constant(1 : i64) : i64
      %1 = llvm.mlir.constant(0 : i64) : i64
      %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %3 = affine.apply #map1()[%block_id_x]
      %4 = affine.min #map2()[%block_id_x, %arg0]
      %5 = arith.cmpi slt, %thread_id_x, %4 : index
      scf.if %5 {
        %6 = arith.addi %thread_id_x, %3 : index
        %7 = arith.remsi %6, %arg1 : index
        %8 = builtin.unrealized_conversion_cast %7 : index to i64
        %9 = arith.divsi %6, %arg1 : index
        %10 = builtin.unrealized_conversion_cast %9 : index to i64
        %11 = arith.index_cast %arg1 : index to i64
        %12 = llvm.mul %11, %0 : i64
        %13 = llvm.mul %10, %12 : i64
        %14 = llvm.add %13, %1 : i64
        %15 = llvm.mul %8, %0 : i64
        %16 = llvm.add %15, %14 : i64
        %17 = llvm.getelementptr %arg2[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %18 = llvm.load %17 : !llvm.ptr -> f32
        %19 = llvm.getelementptr %arg3[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %20 = llvm.load %19 : !llvm.ptr -> f32
        %21 = arith.addf %18, %20 : f32
        %22 = arith.maximumf %21, %2 : f32
        %23 = llvm.getelementptr %arg4[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %22, %23 : f32, !llvm.ptr
      }
      gpu.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After LowerAffinePass (lower-affine) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg4 : i32 to index
    %5 = arith.muli %3, %4 : index
    %c128_0 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %6 = arith.cmpi sle, %5, %c0 : index
    %7 = arith.subi %c0, %5 : index
    %8 = arith.subi %5, %c1_1 : index
    %9 = arith.select %6, %7, %8 : index
    %10 = arith.divsi %9, %c128_0 : index
    %11 = arith.subi %c0, %10 : index
    %12 = arith.addi %10, %c1_1 : index
    %13 = arith.select %6, %11, %12 : index
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%13, %c1, %c1) threads in (%c128, %c1, %c1)  args(%5 : index, %4 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_max_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_max_vectors_kernel(%arg0: index, %arg1: index, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = llvm.mlir.constant(1 : i64) : i64
      %1 = llvm.mlir.constant(0 : i64) : i64
      %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %c128 = arith.constant 128 : index
      %3 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %c-128 = arith.constant -128 : index
      %4 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %5 = arith.addi %4, %arg0 : index
      %c128_0 = arith.constant 128 : index
      %6 = arith.minsi %5, %c128_0 : index
      %7 = arith.cmpi slt, %thread_id_x, %6 : index
      scf.if %7 {
        %8 = arith.addi %thread_id_x, %3 : index
        %9 = arith.remsi %8, %arg1 : index
        %10 = builtin.unrealized_conversion_cast %9 : index to i64
        %11 = arith.divsi %8, %arg1 : index
        %12 = builtin.unrealized_conversion_cast %11 : index to i64
        %13 = arith.index_cast %arg1 : index to i64
        %14 = llvm.mul %13, %0 : i64
        %15 = llvm.mul %12, %14 : i64
        %16 = llvm.add %15, %1 : i64
        %17 = llvm.mul %10, %0 : i64
        %18 = llvm.add %17, %16 : i64
        %19 = llvm.getelementptr %arg2[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %20 = llvm.load %19 : !llvm.ptr -> f32
        %21 = llvm.getelementptr %arg3[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %22 = llvm.load %21 : !llvm.ptr -> f32
        %23 = arith.addf %20, %22 : f32
        %24 = arith.maximumf %23, %2 : f32
        %25 = llvm.getelementptr %arg4[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %24, %25 : f32, !llvm.ptr
      }
      gpu.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %0, %1 : index
    %3 = arith.cmpi sle, %2, %c0 : index
    %4 = arith.subi %c0, %2 : index
    %5 = arith.subi %2, %c1 : index
    %6 = arith.select %3, %4, %5 : index
    %7 = arith.divsi %6, %c128 : index
    %8 = arith.subi %c0, %7 : index
    %9 = arith.addi %7, %c1 : index
    %10 = arith.select %3, %8, %9 : index
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_max_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_max_vectors_kernel(%arg0: index, %arg1: index, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c-128 = arith.constant -128 : index
      %c128 = arith.constant 128 : index
      %0 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %1 = llvm.mlir.constant(0 : i64) : i64
      %2 = llvm.mlir.constant(1 : i64) : i64
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %3 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %4 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %5 = arith.addi %4, %arg0 : index
      %6 = arith.minsi %5, %c128 : index
      %7 = arith.cmpi slt, %thread_id_x, %6 : index
      scf.if %7 {
        %8 = arith.addi %thread_id_x, %3 : index
        %9 = arith.remsi %8, %arg1 : index
        %10 = builtin.unrealized_conversion_cast %9 : index to i64
        %11 = arith.divsi %8, %arg1 : index
        %12 = builtin.unrealized_conversion_cast %11 : index to i64
        %13 = arith.index_cast %arg1 : index to i64
        %14 = llvm.mul %13, %2 : i64
        %15 = llvm.mul %12, %14 : i64
        %16 = llvm.add %15, %1 : i64
        %17 = llvm.mul %10, %2 : i64
        %18 = llvm.add %17, %16 : i64
        %19 = llvm.getelementptr %arg2[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %20 = llvm.load %19 : !llvm.ptr -> f32
        %21 = llvm.getelementptr %arg3[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %22 = llvm.load %21 : !llvm.ptr -> f32
        %23 = arith.addf %20, %22 : f32
        %24 = arith.maximumf %23, %0 : f32
        %25 = llvm.getelementptr %arg4[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %24, %25 : f32, !llvm.ptr
      }
      gpu.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After SCFToControlFlowPass (convert-scf-to-cf) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %0, %1 : index
    %3 = arith.cmpi sle, %2, %c0 : index
    %4 = arith.subi %c0, %2 : index
    %5 = arith.subi %2, %c1 : index
    %6 = arith.select %3, %4, %5 : index
    %7 = arith.divsi %6, %c128 : index
    %8 = arith.subi %c0, %7 : index
    %9 = arith.addi %7, %c1 : index
    %10 = arith.select %3, %8, %9 : index
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_max_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_max_vectors_kernel(%arg0: index, %arg1: index, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c-128 = arith.constant -128 : index
      %c128 = arith.constant 128 : index
      %0 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %1 = llvm.mlir.constant(0 : i64) : i64
      %2 = llvm.mlir.constant(1 : i64) : i64
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %3 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %4 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %5 = arith.addi %4, %arg0 : index
      %6 = arith.minsi %5, %c128 : index
      %7 = arith.cmpi slt, %thread_id_x, %6 : index
      cf.cond_br %7, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %8 = arith.addi %thread_id_x, %3 : index
      %9 = arith.remsi %8, %arg1 : index
      %10 = builtin.unrealized_conversion_cast %9 : index to i64
      %11 = arith.divsi %8, %arg1 : index
      %12 = builtin.unrealized_conversion_cast %11 : index to i64
      %13 = arith.index_cast %arg1 : index to i64
      %14 = llvm.mul %13, %2 : i64
      %15 = llvm.mul %12, %14 : i64
      %16 = llvm.add %15, %1 : i64
      %17 = llvm.mul %10, %2 : i64
      %18 = llvm.add %17, %16 : i64
      %19 = llvm.getelementptr %arg2[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %20 = llvm.load %19 : !llvm.ptr -> f32
      %21 = llvm.getelementptr %arg3[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %22 = llvm.load %21 : !llvm.ptr -> f32
      %23 = arith.addf %20, %22 : f32
      %24 = arith.maximumf %23, %0 : f32
      %25 = llvm.getelementptr %arg4[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %24, %25 : f32, !llvm.ptr
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      gpu.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After ConvertGpuOpsToNVVMOps (convert-gpu-to-nvvm) ('gpu.module' operation: @add_max_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %0, %1 : index
    %3 = arith.cmpi sle, %2, %c0 : index
    %4 = arith.subi %c0, %2 : index
    %5 = arith.subi %2, %c1 : index
    %6 = arith.select %3, %4, %5 : index
    %7 = arith.divsi %6, %c128 : index
    %8 = arith.subi %c0, %7 : index
    %9 = arith.addi %7, %c1 : index
    %10 = arith.select %3, %8, %9 : index
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_max_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_max_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %3 = llvm.mlir.constant(0 : i64) : i64
      %4 = llvm.mlir.constant(1 : i64) : i64
      %5 = nvvm.read.ptx.sreg.ctaid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %8 = llvm.sext %7 : i32 to i64
      %9 = llvm.mul %6, %1 overflow<nsw> : i64
      %10 = llvm.mul %6, %0 overflow<nsw> : i64
      %11 = llvm.add %10, %arg0 : i64
      %12 = llvm.intr.smin(%11, %1) : (i64, i64) -> i64
      %13 = llvm.icmp "slt" %8, %12 : i64
      llvm.cond_br %13, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %14 = llvm.add %8, %9 : i64
      %15 = llvm.srem %14, %arg1 : i64
      %16 = builtin.unrealized_conversion_cast %15 : i64 to index
      %17 = builtin.unrealized_conversion_cast %16 : index to i64
      %18 = llvm.sdiv %14, %arg1 : i64
      %19 = builtin.unrealized_conversion_cast %18 : i64 to index
      %20 = builtin.unrealized_conversion_cast %19 : index to i64
      %21 = llvm.mul %arg1, %4 : i64
      %22 = llvm.mul %20, %21 : i64
      %23 = llvm.add %22, %3 : i64
      %24 = llvm.mul %17, %4 : i64
      %25 = llvm.add %24, %23 : i64
      %26 = llvm.getelementptr %arg2[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %27 = llvm.load %26 : !llvm.ptr -> f32
      %28 = llvm.getelementptr %arg3[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %29 = llvm.load %28 : !llvm.ptr -> f32
      %30 = llvm.fadd %27, %29 : f32
      %31 = llvm.intr.maximum(%30, %2) : (f32, f32) -> f32
      %32 = llvm.getelementptr %arg4[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %31, %32 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After ConvertNVVMToLLVMPass (convert-nvvm-to-llvm) ('gpu.module' operation: @add_max_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %0, %1 : index
    %3 = arith.cmpi sle, %2, %c0 : index
    %4 = arith.subi %c0, %2 : index
    %5 = arith.subi %2, %c1 : index
    %6 = arith.select %3, %4, %5 : index
    %7 = arith.divsi %6, %c128 : index
    %8 = arith.subi %c0, %7 : index
    %9 = arith.addi %7, %c1 : index
    %10 = arith.select %3, %8, %9 : index
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_max_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_max_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %3 = llvm.mlir.constant(0 : i64) : i64
      %4 = llvm.mlir.constant(1 : i64) : i64
      %5 = nvvm.read.ptx.sreg.ctaid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %8 = llvm.sext %7 : i32 to i64
      %9 = llvm.mul %6, %1 overflow<nsw> : i64
      %10 = llvm.mul %6, %0 overflow<nsw> : i64
      %11 = llvm.add %10, %arg0 : i64
      %12 = llvm.intr.smin(%11, %1) : (i64, i64) -> i64
      %13 = llvm.icmp "slt" %8, %12 : i64
      llvm.cond_br %13, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %14 = llvm.add %8, %9 : i64
      %15 = llvm.srem %14, %arg1 : i64
      %16 = builtin.unrealized_conversion_cast %15 : i64 to index
      %17 = llvm.sdiv %14, %arg1 : i64
      %18 = builtin.unrealized_conversion_cast %17 : i64 to index
      %19 = llvm.mul %arg1, %4 : i64
      %20 = llvm.mul %17, %19 : i64
      %21 = llvm.add %20, %3 : i64
      %22 = llvm.mul %15, %4 : i64
      %23 = llvm.add %22, %21 : i64
      %24 = llvm.getelementptr %arg2[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %25 = llvm.load %24 : !llvm.ptr -> f32
      %26 = llvm.getelementptr %arg3[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %27 = llvm.load %26 : !llvm.ptr -> f32
      %28 = llvm.fadd %25, %27 : f32
      %29 = llvm.intr.maximum(%28, %2) : (f32, f32) -> f32
      %30 = llvm.getelementptr %arg4[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %29, %30 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After NVVMOptimizeForTargetPass (llvm-optimize-for-nvvm-target) ('gpu.module' operation: @add_max_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %0, %1 : index
    %3 = arith.cmpi sle, %2, %c0 : index
    %4 = arith.subi %c0, %2 : index
    %5 = arith.subi %2, %c1 : index
    %6 = arith.select %3, %4, %5 : index
    %7 = arith.divsi %6, %c128 : index
    %8 = arith.subi %c0, %7 : index
    %9 = arith.addi %7, %c1 : index
    %10 = arith.select %3, %8, %9 : index
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_max_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_max_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %3 = llvm.mlir.constant(0 : i64) : i64
      %4 = llvm.mlir.constant(1 : i64) : i64
      %5 = nvvm.read.ptx.sreg.ctaid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %8 = llvm.sext %7 : i32 to i64
      %9 = llvm.mul %6, %1 overflow<nsw> : i64
      %10 = llvm.mul %6, %0 overflow<nsw> : i64
      %11 = llvm.add %10, %arg0 : i64
      %12 = llvm.intr.smin(%11, %1) : (i64, i64) -> i64
      %13 = llvm.icmp "slt" %8, %12 : i64
      llvm.cond_br %13, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %14 = llvm.add %8, %9 : i64
      %15 = llvm.srem %14, %arg1 : i64
      %16 = llvm.sdiv %14, %arg1 : i64
      %17 = llvm.mul %arg1, %4 : i64
      %18 = llvm.mul %16, %17 : i64
      %19 = llvm.add %18, %3 : i64
      %20 = llvm.mul %15, %4 : i64
      %21 = llvm.add %20, %19 : i64
      %22 = llvm.getelementptr %arg2[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %23 = llvm.load %22 : !llvm.ptr -> f32
      %24 = llvm.getelementptr %arg3[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %25 = llvm.load %24 : !llvm.ptr -> f32
      %26 = llvm.fadd %23, %25 : f32
      %27 = llvm.intr.maximum(%26, %2) : (f32, f32) -> f32
      %28 = llvm.getelementptr %arg4[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %27, %28 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c8_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuToLLVMConversionPass (gpu-to-llvm) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.sext %arg3 : i32 to i64
    %4 = llvm.sext %arg4 : i32 to i64
    %5 = llvm.mul %3, %4 : i64
    %6 = llvm.icmp "sle" %5, %0 : i64
    %7 = llvm.sub %0, %5 : i64
    %8 = llvm.sub %5, %2 : i64
    %9 = llvm.select %6, %7, %8 : i1, i64
    %10 = llvm.sdiv %9, %1 : i64
    %11 = llvm.sub %0, %10 : i64
    %12 = llvm.add %10, %2 : i64
    %13 = llvm.select %6, %11, %12 : i1, i64
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%13, %2, %2) threads in (%1, %2, %2) : i64 args(%5 : i64, %4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_max_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_max_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %3 = llvm.mlir.constant(0 : i64) : i64
      %4 = llvm.mlir.constant(1 : i64) : i64
      %5 = nvvm.read.ptx.sreg.ctaid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %8 = llvm.sext %7 : i32 to i64
      %9 = llvm.mul %6, %1 overflow<nsw> : i64
      %10 = llvm.mul %6, %0 overflow<nsw> : i64
      %11 = llvm.add %10, %arg0 : i64
      %12 = llvm.intr.smin(%11, %1) : (i64, i64) -> i64
      %13 = llvm.icmp "slt" %8, %12 : i64
      llvm.cond_br %13, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %14 = llvm.add %8, %9 : i64
      %15 = llvm.srem %14, %arg1 : i64
      %16 = llvm.sdiv %14, %arg1 : i64
      %17 = llvm.mul %arg1, %4 : i64
      %18 = llvm.mul %16, %17 : i64
      %19 = llvm.add %18, %3 : i64
      %20 = llvm.mul %15, %4 : i64
      %21 = llvm.add %20, %19 : i64
      %22 = llvm.getelementptr %arg2[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %23 = llvm.load %22 : !llvm.ptr -> f32
      %24 = llvm.getelementptr %arg3[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %25 = llvm.load %24 : !llvm.ptr -> f32
      %26 = llvm.fadd %23, %25 : f32
      %27 = llvm.intr.maximum(%26, %2) : (f32, f32) -> f32
      %28 = llvm.getelementptr %arg4[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %27, %28 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mul %7, %8 : i32
    %10 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %13 = llvm.call @toy_random(%10, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.call @toy_random(%11, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %15 = llvm.call @add_max_vectors(%10, %11, %12, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%10, %9, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %18 : !llvm.array<2 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%11, %9, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %20 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %20 : !llvm.array<4 x i8>, !llvm.ptr
    %21 = llvm.call @toy_print(%12, %9, %4, %20) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuNVVMAttachTarget (nvvm-attach-target) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.sext %arg3 : i32 to i64
    %4 = llvm.sext %arg4 : i32 to i64
    %5 = llvm.mul %3, %4 : i64
    %6 = llvm.icmp "sle" %5, %0 : i64
    %7 = llvm.sub %0, %5 : i64
    %8 = llvm.sub %5, %2 : i64
    %9 = llvm.select %6, %7, %8 : i1, i64
    %10 = llvm.sdiv %9, %1 : i64
    %11 = llvm.sub %0, %10 : i64
    %12 = llvm.add %10, %2 : i64
    %13 = llvm.select %6, %11, %12 : i1, i64
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%13, %2, %2) threads in (%1, %2, %2) : i64 args(%5 : i64, %4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_max_vectors_kernel [#nvvm.target<chip = "sm_86", features = "+ptx84">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_max_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(1.500000e-01 : f32) : f32
      %3 = llvm.mlir.constant(0 : i64) : i64
      %4 = llvm.mlir.constant(1 : i64) : i64
      %5 = nvvm.read.ptx.sreg.ctaid.x : i32
      %6 = llvm.sext %5 : i32 to i64
      %7 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %8 = llvm.sext %7 : i32 to i64
      %9 = llvm.mul %6, %1 overflow<nsw> : i64
      %10 = llvm.mul %6, %0 overflow<nsw> : i64
      %11 = llvm.add %10, %arg0 : i64
      %12 = llvm.intr.smin(%11, %1) : (i64, i64) -> i64
      %13 = llvm.icmp "slt" %8, %12 : i64
      llvm.cond_br %13, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %14 = llvm.add %8, %9 : i64
      %15 = llvm.srem %14, %arg1 : i64
      %16 = llvm.sdiv %14, %arg1 : i64
      %17 = llvm.mul %arg1, %4 : i64
      %18 = llvm.mul %16, %17 : i64
      %19 = llvm.add %18, %3 : i64
      %20 = llvm.mul %15, %4 : i64
      %21 = llvm.add %20, %19 : i64
      %22 = llvm.getelementptr %arg2[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %23 = llvm.load %22 : !llvm.ptr -> f32
      %24 = llvm.getelementptr %arg3[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %25 = llvm.load %24 : !llvm.ptr -> f32
      %26 = llvm.fadd %23, %25 : f32
      %27 = llvm.intr.maximum(%26, %2) : (f32, f32) -> f32
      %28 = llvm.getelementptr %arg4[%21] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %27, %28 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mul %7, %8 : i32
    %10 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %13 = llvm.call @toy_random(%10, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.call @toy_random(%11, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %15 = llvm.call @add_max_vectors(%10, %11, %12, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%10, %9, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %18 : !llvm.array<2 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%11, %9, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %20 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %20 : !llvm.array<4 x i8>, !llvm.ptr
    %21 = llvm.call @toy_print(%12, %9, %4, %20) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuModuleToBinaryPass (gpu-module-to-binary) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.sext %arg3 : i32 to i64
    %4 = llvm.sext %arg4 : i32 to i64
    %5 = llvm.mul %3, %4 : i64
    %6 = llvm.icmp "sle" %5, %0 : i64
    %7 = llvm.sub %0, %5 : i64
    %8 = llvm.sub %5, %2 : i64
    %9 = llvm.select %6, %7, %8 : i1, i64
    %10 = llvm.sdiv %9, %1 : i64
    %11 = llvm.sub %0, %10 : i64
    %12 = llvm.add %10, %2 : i64
    %13 = llvm.select %6, %11, %12 : i1, i64
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%13, %2, %2) threads in (%1, %2, %2) : i64 args(%5 : i64, %4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.binary @add_max_vectors_kernel  [#gpu.object<#nvvm.target<chip = "sm_86", features = "+ptx84">, properties = {ISAToBinaryTimeInMs = 10 : i64, LLVMIRToISATimeInMs = 12 : i64}, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\00\00\00\00\09\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_max_vectors_kernel\00.nv.info.add_max_vectors_kernel\00.nv.shared.add_max_vectors_kernel\00.nv.constant0.add_max_vectors_kernel\00.rel.nv.constant0.add_max_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_max_vectors_kernel\00.nv.info.add_max_vectors_kernel\00.nv.shared.add_max_vectors_kernel\00.rel.nv.constant0.add_max_vectors_kernel\00.nv.constant0.add_max_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_max_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\04<\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01(\00\03\19(\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F5!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03\1B\FF\00\04\1C\08\00\80\00\00\00\80\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\80\00\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00X\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00Y\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\07\05\02\00\00\00\03\02\01\00\00\E2\0F\04$x\06\05\04\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\06\00^\00\00\FF\E0\F3\07\00\E4\0F\04\10z\02\06\00\\\00\00\FF\E0\F1\07\00\E4\0F\00\10z\05\07\00_\00\00\FF\E4\FF\00\00\E4\0F\04\10z\03\07\00]\00\00\FF\E4\7F\00\00\C8\0F\00\81y\05\04\04\00\00\00\00\19\1E\0C\00\A8\0E\00\81y\02\02\04\00\00\00\00\19\1E\0C\00\A2\0E\00\10z\06\06\00`\00\00\FF\E0\F1\07\00\C8\0F\00\10z\07\07\00a\00\00\FF\E4\7F\00\00\E2\0F\00!r\00\02\05\00\00\00\00\00\00\00\00\CAO\00\09x\09\00\9A\99\19>\00\00\82\07\00\CA\0F\00\86y\00\06\09\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00\8C\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A4\04\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\04\00\00\00\00\00\00\88\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\E8\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00\18\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mul %7, %8 : i32
    %10 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %13 = llvm.call @toy_random(%10, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.call @toy_random(%11, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %15 = llvm.call @add_max_vectors(%10, %11, %12, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%10, %9, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %18 : !llvm.array<2 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%11, %9, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %20 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %20 : !llvm.array<4 x i8>, !llvm.ptr
    %21 = llvm.call @toy_print(%12, %9, %4, %20) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_max_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.sext %arg3 : i32 to i64
    %4 = llvm.sext %arg4 : i32 to i64
    %5 = llvm.mul %3, %4 : i64
    %6 = llvm.icmp "sle" %5, %0 : i64
    %7 = llvm.sub %0, %5 : i64
    %8 = llvm.sub %5, %2 : i64
    %9 = llvm.select %6, %7, %8 : i1, i64
    %10 = llvm.sdiv %9, %1 : i64
    %11 = llvm.sub %0, %10 : i64
    %12 = llvm.add %10, %2 : i64
    %13 = llvm.select %6, %11, %12 : i1, i64
    gpu.launch_func  @add_max_vectors_kernel::@add_max_vectors_kernel blocks in (%13, %2, %2) threads in (%1, %2, %2) : i64 args(%5 : i64, %4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.binary @add_max_vectors_kernel  [#gpu.object<#nvvm.target<chip = "sm_86", features = "+ptx84">, properties = {ISAToBinaryTimeInMs = 10 : i64, LLVMIRToISATimeInMs = 12 : i64}, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\00\00\00\00\09\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_max_vectors_kernel\00.nv.info.add_max_vectors_kernel\00.nv.shared.add_max_vectors_kernel\00.nv.constant0.add_max_vectors_kernel\00.rel.nv.constant0.add_max_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_max_vectors_kernel\00.nv.info.add_max_vectors_kernel\00.nv.shared.add_max_vectors_kernel\00.rel.nv.constant0.add_max_vectors_kernel\00.nv.constant0.add_max_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_max_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\04<\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01(\00\03\19(\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F5!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03\1B\FF\00\04\1C\08\00\80\00\00\00\80\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\80\00\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00X\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00Y\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\07\05\02\00\00\00\03\02\01\00\00\E2\0F\04$x\06\05\04\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\06\00^\00\00\FF\E0\F3\07\00\E4\0F\04\10z\02\06\00\\\00\00\FF\E0\F1\07\00\E4\0F\00\10z\05\07\00_\00\00\FF\E4\FF\00\00\E4\0F\04\10z\03\07\00]\00\00\FF\E4\7F\00\00\C8\0F\00\81y\05\04\04\00\00\00\00\19\1E\0C\00\A8\0E\00\81y\02\02\04\00\00\00\00\19\1E\0C\00\A2\0E\00\10z\06\06\00`\00\00\FF\E0\F1\07\00\C8\0F\00\10z\07\07\00a\00\00\FF\E4\7F\00\00\E2\0F\00!r\00\02\05\00\00\00\00\00\00\00\00\CAO\00\09x\09\00\9A\99\19>\00\00\82\07\00\CA\0F\00\86y\00\06\09\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00\8C\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A4\04\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\04\00\00\00\00\00\00\88\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\E8\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00\18\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mul %5, %6 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_max_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<2 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<2 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<4 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


