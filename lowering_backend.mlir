// -----// IR Dump After CollapseParallelLoopsTo1DPass (collapse-parallel-loops-to-1d) ('func.func' operation: @add_two_vectors) //----- //
module {
  func.func @add_two_vectors(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32) -> memref<?x?xi32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    %5 = arith.muli %c1_2, %0 : index
    %6 = arith.muli %5, %1 : index
    scf.parallel (%arg5) = (%c0_0) to (%6) step (%c1_1) {
      %7 = arith.remsi %arg5, %1 : index
      %8 = arith.divsi %arg5, %1 : index
      %9 = memref.load %2[%8, %7] : memref<?x?xi32>
      %10 = memref.load %3[%8, %7] : memref<?x?xi32>
      %11 = arith.addi %9, %10 : i32
      memref.store %11, %4[%8, %7] : memref<?x?xi32>
      scf.reduce 
    }
    return %4 : memref<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 16 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %7 = toy.random %4 : memref<?x?xi32> -> memref<?x?xi32>
    %8 = toy.random %5 : memref<?x?xi32> -> memref<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, i32, i32) -> memref<?x?xi32>
    %10 = toy.print %4 : memref<?x?xi32> {argName = "t1"} -> i64
    %11 = toy.print %5 : memref<?x?xi32> {argName = "t2"} -> i64
    %12 = toy.print %6 : memref<?x?xi32> {argName = "t3"} -> i64
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
    %5 = arith.muli %0, %1 : index
    scf.parallel (%arg5) = (%c0) to (%5) step (%c1) {
      %6 = arith.remsi %arg5, %1 : index
      %7 = arith.divsi %arg5, %1 : index
      %8 = memref.load %2[%7, %6] : memref<?x?xi32>
      %9 = memref.load %3[%7, %6] : memref<?x?xi32>
      %10 = arith.addi %8, %9 : i32
      memref.store %10, %4[%7, %6] : memref<?x?xi32>
      scf.reduce 
    }
    return %4 : memref<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 16 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %7 = toy.random %4 : memref<?x?xi32> -> memref<?x?xi32>
    %8 = toy.random %5 : memref<?x?xi32> -> memref<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, i32, i32) -> memref<?x?xi32>
    %10 = toy.print %4 : memref<?x?xi32> {argName = "t1"} -> i64
    %11 = toy.print %5 : memref<?x?xi32> {argName = "t2"} -> i64
    %12 = toy.print %6 : memref<?x?xi32> {argName = "t3"} -> i64
    return
  }
}


// -----// IR Dump After TileLoopsPass (tile-loops) ('func.func' operation: @add_two_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32) -> memref<?x?xi32> {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %5 = arith.muli %0, %1 : index
    scf.parallel (%arg5) = (%c0) to (%5) step (%c128) {
      %6 = affine.min #map(%c128, %5, %arg5)
      scf.parallel (%arg6) = (%c0) to (%6) step (%c1) {
        %7 = arith.addi %arg6, %arg5 : index
        %8 = arith.remsi %7, %1 : index
        %9 = arith.divsi %7, %1 : index
        %10 = memref.load %2[%9, %8] : memref<?x?xi32>
        %11 = memref.load %3[%9, %8] : memref<?x?xi32>
        %12 = arith.addi %10, %11 : i32
        memref.store %12, %4[%9, %8] : memref<?x?xi32>
        scf.reduce 
      }
      scf.reduce 
    }
    return %4 : memref<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 16 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %7 = toy.random %4 : memref<?x?xi32> -> memref<?x?xi32>
    %8 = toy.random %5 : memref<?x?xi32> -> memref<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, i32, i32) -> memref<?x?xi32>
    %10 = toy.print %4 : memref<?x?xi32> {argName = "t1"} -> i64
    %11 = toy.print %5 : memref<?x?xi32> {argName = "t2"} -> i64
    %12 = toy.print %6 : memref<?x?xi32> {argName = "t3"} -> i64
    return
  }
}


// -----// IR Dump After GpuMapParallelLoopsPass (gpu-map-parallel-loops) ('func.func' operation: @add_two_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32) -> memref<?x?xi32> {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %5 = arith.muli %0, %1 : index
    scf.parallel (%arg5) = (%c0) to (%5) step (%c128) {
      %6 = affine.min #map(%c128, %5, %arg5)
      scf.parallel (%arg6) = (%c0) to (%6) step (%c1) {
        %7 = arith.addi %arg6, %arg5 : index
        %8 = arith.remsi %7, %1 : index
        %9 = arith.divsi %7, %1 : index
        %10 = memref.load %2[%9, %8] : memref<?x?xi32>
        %11 = memref.load %3[%9, %8] : memref<?x?xi32>
        %12 = arith.addi %10, %11 : i32
        memref.store %12, %4[%9, %8] : memref<?x?xi32>
        scf.reduce 
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.reduce 
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %4 : memref<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 16 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %7 = toy.random %4 : memref<?x?xi32> -> memref<?x?xi32>
    %8 = toy.random %5 : memref<?x?xi32> -> memref<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, i32, i32) -> memref<?x?xi32>
    %10 = toy.print %4 : memref<?x?xi32> {argName = "t1"} -> i64
    %11 = toy.print %5 : memref<?x?xi32> {argName = "t2"} -> i64
    %12 = toy.print %6 : memref<?x?xi32> {argName = "t3"} -> i64
    return
  }
}


// -----// IR Dump After ConvertParallelLoopToGpuPass (convert-parallel-loops-to-gpu) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?x?xi32>, %arg1: memref<?x?xi32>, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32) -> memref<?x?xi32> {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = toy.tie_dims %arg0, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %3 = toy.tie_dims %arg1, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %4 = toy.tie_dims %arg2, %0, %1 : memref<?x?xi32>, index, index -> memref<?x?xi32>
    %5 = arith.muli %0, %1 : index
    %c1_0 = arith.constant 1 : index
    %6 = affine.apply #map(%5)[%c0, %c128]
    %c128_1 = arith.constant 128 : index
    %7 = affine.apply #map(%c128_1)[%c0, %c1]
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %6, %arg12 = %c1_0, %arg13 = %c1_0) threads(%arg8, %arg9, %arg10) in (%arg14 = %7, %arg15 = %c1_0, %arg16 = %c1_0) {
      %8 = affine.apply #map1(%arg5)[%c128, %c0]
      %9 = affine.min #map2(%c128, %5, %8)
      %10 = affine.apply #map1(%arg8)[%c1, %c0]
      %11 = arith.cmpi slt, %10, %9 : index
      scf.if %11 {
        %12 = arith.addi %10, %8 : index
        %13 = arith.remsi %12, %1 : index
        %14 = arith.divsi %12, %1 : index
        %15 = memref.load %2[%14, %13] : memref<?x?xi32>
        %16 = memref.load %3[%14, %13] : memref<?x?xi32>
        %17 = arith.addi %15, %16 : i32
        memref.store %17, %4[%14, %13] : memref<?x?xi32>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    return %4 : memref<?x?xi32>
  }
  func.func @main() {
    %0 = toy.constant 16 : i32 -> i32
    %1 = toy.constant 32 : i32 -> i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %5 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %6 = toy.alloc %2, %3 : -> memref<?x?xi32>
    %7 = toy.random %4 : memref<?x?xi32> -> memref<?x?xi32>
    %8 = toy.random %5 : memref<?x?xi32> -> memref<?x?xi32>
    %9 = toy.call @add_two_vectors(%4, %5, %6, %0, %1) : (memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>, i32, i32) -> memref<?x?xi32>
    %10 = toy.print %4 : memref<?x?xi32> {argName = "t1"} -> i64
    %11 = toy.print %5 : memref<?x?xi32> {argName = "t2"} -> i64
    %12 = toy.print %6 : memref<?x?xi32> {argName = "t3"} -> i64
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
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.muli %0, %1 : index
    %c1_0 = arith.constant 1 : index
    %3 = affine.apply #map(%2)[%c0, %c128]
    %c128_1 = arith.constant 128 : index
    %c128_2 = arith.constant 128 : index
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %3, %arg12 = %c1_0, %arg13 = %c1_0) threads(%arg8, %arg9, %arg10) in (%arg14 = %c128_2, %arg15 = %c1_0, %arg16 = %c1_0) {
      %4 = affine.apply #map1(%arg5)[%c128, %c0]
      %5 = affine.min #map2(%c128, %2, %4)
      %6 = affine.apply #map1(%arg8)[%c1, %c0]
      %7 = arith.cmpi slt, %6, %5 : index
      scf.if %7 {
        %8 = arith.addi %6, %4 : index
        %9 = arith.remsi %8, %1 : index
        %10 = builtin.unrealized_conversion_cast %9 : index to i64
        %11 = arith.divsi %8, %1 : index
        %12 = builtin.unrealized_conversion_cast %11 : index to i64
        %13 = llvm.mlir.constant(1 : i64) : i64
        %14 = arith.index_cast %1 : index to i64
        %15 = llvm.mul %13, %14 : i64
        %16 = llvm.mlir.constant(0 : i64) : i64
        %17 = llvm.mul %12, %15 : i64
        %18 = llvm.add %17, %16 : i64
        %19 = llvm.mul %10, %13 : i64
        %20 = llvm.add %19, %18 : i64
        %21 = llvm.getelementptr %arg0[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %22 = llvm.load %21 : !llvm.ptr -> i32
        %23 = llvm.mlir.constant(1 : i64) : i64
        %24 = arith.index_cast %1 : index to i64
        %25 = llvm.mul %23, %24 : i64
        %26 = llvm.mlir.constant(0 : i64) : i64
        %27 = llvm.mul %12, %25 : i64
        %28 = llvm.add %27, %26 : i64
        %29 = llvm.mul %10, %23 : i64
        %30 = llvm.add %29, %28 : i64
        %31 = llvm.getelementptr %arg1[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %32 = llvm.load %31 : !llvm.ptr -> i32
        %33 = arith.addi %22, %32 : i32
        %34 = llvm.mlir.constant(1 : i64) : i64
        %35 = arith.index_cast %1 : index to i64
        %36 = llvm.mul %34, %35 : i64
        %37 = llvm.mlir.constant(0 : i64) : i64
        %38 = llvm.mul %12, %36 : i64
        %39 = llvm.add %38, %37 : i64
        %40 = llvm.mul %10, %34 : i64
        %41 = llvm.add %40, %39 : i64
        %42 = llvm.getelementptr %arg2[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %33, %42 : i32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return %arg2 : !llvm.ptr
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(16 : i32) : i32
    %1 = llvm.mlir.constant(32 : i32) : i32
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.index_cast %2 : index to i32
    %5 = arith.index_cast %3 : index to i32
    %6 = llvm.mul %4, %5 : i32
    %7 = llvm.mlir.constant(2 : i32) : i32
    %8 = llvm.call @toy_alloc(%6, %7) : (i32, i32) -> !llvm.ptr
    %9 = arith.index_cast %2 : index to i32
    %10 = arith.index_cast %3 : index to i32
    %11 = llvm.mul %9, %10 : i32
    %12 = llvm.mlir.constant(2 : i32) : i32
    %13 = llvm.call @toy_alloc(%11, %12) : (i32, i32) -> !llvm.ptr
    %14 = arith.index_cast %2 : index to i32
    %15 = arith.index_cast %3 : index to i32
    %16 = llvm.mul %14, %15 : i32
    %17 = llvm.mlir.constant(2 : i32) : i32
    %18 = llvm.call @toy_alloc(%16, %17) : (i32, i32) -> !llvm.ptr
    %19 = arith.index_cast %2 : index to i32
    %20 = arith.index_cast %3 : index to i32
    %21 = llvm.mul %19, %20 : i32
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.call @toy_random(%8, %21, %22) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %24 = arith.index_cast %2 : index to i32
    %25 = arith.index_cast %3 : index to i32
    %26 = llvm.mul %24, %25 : i32
    %27 = llvm.mlir.constant(2 : i32) : i32
    %28 = llvm.call @toy_random(%13, %26, %27) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %29 = llvm.call @add_two_vectors(%8, %13, %18, %0, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %30 = arith.index_cast %2 : index to i32
    %31 = arith.index_cast %3 : index to i32
    %32 = llvm.mul %30, %31 : i32
    %33 = llvm.mlir.constant(2 : i32) : i32
    %34 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %35 = llvm.mlir.constant(1 : i32) : i32
    %36 = llvm.alloca %35 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %34, %36 : !llvm.array<3 x i8>, !llvm.ptr
    %37 = llvm.call @toy_print(%8, %32, %33, %36) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %38 = arith.index_cast %2 : index to i32
    %39 = arith.index_cast %3 : index to i32
    %40 = llvm.mul %38, %39 : i32
    %41 = llvm.mlir.constant(2 : i32) : i32
    %42 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %43 = llvm.mlir.constant(1 : i32) : i32
    %44 = llvm.alloca %43 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %42, %44 : !llvm.array<3 x i8>, !llvm.ptr
    %45 = llvm.call @toy_print(%13, %40, %41, %44) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %46 = arith.index_cast %2 : index to i32
    %47 = arith.index_cast %3 : index to i32
    %48 = llvm.mul %46, %47 : i32
    %49 = llvm.mlir.constant(2 : i32) : i32
    %50 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %51 = llvm.mlir.constant(1 : i32) : i32
    %52 = llvm.alloca %51 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %50, %52 : !llvm.array<3 x i8>, !llvm.ptr
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
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = arith.muli %2, %3 : index
    %5 = affine.apply #map()[%4]
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %5, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c128, %arg15 = %c1, %arg16 = %c1) {
      %6 = affine.apply #map1()[%arg5]
      %7 = affine.min #map2()[%arg5, %4]
      %8 = arith.cmpi slt, %arg8, %7 : index
      scf.if %8 {
        %9 = arith.addi %arg8, %6 : index
        %10 = arith.remsi %9, %3 : index
        %11 = builtin.unrealized_conversion_cast %10 : index to i64
        %12 = arith.divsi %9, %3 : index
        %13 = builtin.unrealized_conversion_cast %12 : index to i64
        %14 = arith.index_cast %3 : index to i64
        %15 = llvm.mul %14, %1 : i64
        %16 = llvm.mul %13, %15 : i64
        %17 = llvm.add %16, %0 : i64
        %18 = llvm.mul %11, %1 : i64
        %19 = llvm.add %18, %17 : i64
        %20 = llvm.getelementptr %arg0[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %21 = llvm.load %20 : !llvm.ptr -> i32
        %22 = arith.index_cast %3 : index to i64
        %23 = llvm.mul %22, %1 : i64
        %24 = llvm.mul %13, %23 : i64
        %25 = llvm.add %24, %0 : i64
        %26 = llvm.mul %11, %1 : i64
        %27 = llvm.add %26, %25 : i64
        %28 = llvm.getelementptr %arg1[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %29 = llvm.load %28 : !llvm.ptr -> i32
        %30 = arith.addi %21, %29 : i32
        %31 = arith.index_cast %3 : index to i64
        %32 = llvm.mul %31, %1 : i64
        %33 = llvm.mul %13, %32 : i64
        %34 = llvm.add %33, %0 : i64
        %35 = llvm.mul %11, %1 : i64
        %36 = llvm.add %35, %34 : i64
        %37 = llvm.getelementptr %arg2[%36] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %30, %37 : i32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return %arg2 : !llvm.ptr
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.mul %c16_i32, %c32_i32 : i32
    %10 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.mul %c16_i32, %c32_i32 : i32
    %12 = llvm.call @toy_alloc(%11, %4) : (i32, i32) -> !llvm.ptr
    %13 = llvm.mul %c16_i32, %c32_i32 : i32
    %14 = llvm.call @toy_random(%8, %13, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %15 = llvm.mul %c16_i32, %c32_i32 : i32
    %16 = llvm.call @toy_random(%10, %15, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %17 = llvm.call @add_two_vectors(%8, %10, %12, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %18 = llvm.mul %c16_i32, %c32_i32 : i32
    %19 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %19 : !llvm.array<3 x i8>, !llvm.ptr
    %20 = llvm.call @toy_print(%8, %18, %4, %19) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %21 = llvm.mul %c16_i32, %c32_i32 : i32
    %22 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %22 : !llvm.array<3 x i8>, !llvm.ptr
    %23 = llvm.call @toy_print(%10, %21, %4, %22) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %24 = llvm.mul %c16_i32, %c32_i32 : i32
    %25 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %25 : !llvm.array<3 x i8>, !llvm.ptr
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
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = arith.muli %2, %3 : index
    %5 = affine.apply #map()[%4]
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %5, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c128, %arg15 = %c1, %arg16 = %c1) {
      %6 = affine.apply #map1()[%arg5]
      %7 = affine.min #map2()[%arg5, %4]
      %8 = arith.cmpi slt, %arg8, %7 : index
      scf.if %8 {
        %9 = arith.addi %arg8, %6 : index
        %10 = arith.remsi %9, %3 : index
        %11 = builtin.unrealized_conversion_cast %10 : index to i64
        %12 = arith.divsi %9, %3 : index
        %13 = builtin.unrealized_conversion_cast %12 : index to i64
        %14 = arith.index_cast %3 : index to i64
        %15 = llvm.mul %14, %1 : i64
        %16 = llvm.mul %13, %15 : i64
        %17 = llvm.add %16, %0 : i64
        %18 = llvm.mul %11, %1 : i64
        %19 = llvm.add %18, %17 : i64
        %20 = llvm.getelementptr %arg0[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %21 = llvm.load %20 : !llvm.ptr -> i32
        %22 = llvm.getelementptr %arg1[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %23 = llvm.load %22 : !llvm.ptr -> i32
        %24 = arith.addi %21, %23 : i32
        %25 = llvm.getelementptr %arg2[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %24, %25 : i32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return %arg2 : !llvm.ptr
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
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
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = arith.muli %2, %3 : index
    %5 = affine.apply #map()[%4]
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %5, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c128, %arg15 = %c1, %arg16 = %c1) {
      %6 = llvm.mlir.constant(1 : i64) : i64
      %7 = llvm.mlir.constant(0 : i64) : i64
      %8 = affine.apply #map1()[%arg5]
      %9 = affine.min #map2()[%arg5, %4]
      %10 = arith.cmpi slt, %arg8, %9 : index
      scf.if %10 {
        %11 = arith.addi %arg8, %8 : index
        %12 = arith.remsi %11, %3 : index
        %13 = builtin.unrealized_conversion_cast %12 : index to i64
        %14 = arith.divsi %11, %3 : index
        %15 = builtin.unrealized_conversion_cast %14 : index to i64
        %16 = arith.index_cast %3 : index to i64
        %17 = llvm.mul %16, %6 : i64
        %18 = llvm.mul %15, %17 : i64
        %19 = llvm.add %18, %7 : i64
        %20 = llvm.mul %13, %6 : i64
        %21 = llvm.add %20, %19 : i64
        %22 = llvm.getelementptr %arg0[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %23 = llvm.load %22 : !llvm.ptr -> i32
        %24 = llvm.getelementptr %arg1[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %25 = llvm.load %24 : !llvm.ptr -> i32
        %26 = arith.addi %23, %25 : i32
        %27 = llvm.getelementptr %arg2[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %26, %27 : i32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return %arg2 : !llvm.ptr
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
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
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = arith.muli %2, %3 : index
    %5 = affine.apply #map()[%4]
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%5, %c1, %c1) threads in (%c128, %c1, %c1)  args(%4 : index, %3 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: index, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
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
      %2 = affine.apply #map1()[%block_id_x]
      %3 = affine.min #map2()[%block_id_x, %arg0]
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      scf.if %4 {
        %5 = arith.addi %thread_id_x, %2 : index
        %6 = arith.remsi %5, %arg1 : index
        %7 = builtin.unrealized_conversion_cast %6 : index to i64
        %8 = arith.divsi %5, %arg1 : index
        %9 = builtin.unrealized_conversion_cast %8 : index to i64
        %10 = arith.index_cast %arg1 : index to i64
        %11 = llvm.mul %10, %0 : i64
        %12 = llvm.mul %9, %11 : i64
        %13 = llvm.add %12, %1 : i64
        %14 = llvm.mul %7, %0 : i64
        %15 = llvm.add %14, %13 : i64
        %16 = llvm.getelementptr %arg2[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %17 = llvm.load %16 : !llvm.ptr -> i32
        %18 = llvm.getelementptr %arg3[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %19 = llvm.load %18 : !llvm.ptr -> i32
        %20 = arith.addi %17, %19 : i32
        %21 = llvm.getelementptr %arg4[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %20, %21 : i32, !llvm.ptr
      }
      gpu.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After LowerAffinePass (lower-affine) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = arith.muli %2, %3 : index
    %c128_0 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %5 = arith.cmpi sle, %4, %c0 : index
    %6 = arith.subi %c0, %4 : index
    %7 = arith.subi %4, %c1_1 : index
    %8 = arith.select %5, %6, %7 : index
    %9 = arith.divsi %8, %c128_0 : index
    %10 = arith.subi %c0, %9 : index
    %11 = arith.addi %9, %c1_1 : index
    %12 = arith.select %5, %10, %11 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%12, %c1, %c1) threads in (%c128, %c1, %c1)  args(%4 : index, %3 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: index, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
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
      %c128 = arith.constant 128 : index
      %2 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %c-128 = arith.constant -128 : index
      %3 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %4 = arith.addi %3, %arg0 : index
      %c128_0 = arith.constant 128 : index
      %5 = arith.minsi %4, %c128_0 : index
      %6 = arith.cmpi slt, %thread_id_x, %5 : index
      scf.if %6 {
        %7 = arith.addi %thread_id_x, %2 : index
        %8 = arith.remsi %7, %arg1 : index
        %9 = builtin.unrealized_conversion_cast %8 : index to i64
        %10 = arith.divsi %7, %arg1 : index
        %11 = builtin.unrealized_conversion_cast %10 : index to i64
        %12 = arith.index_cast %arg1 : index to i64
        %13 = llvm.mul %12, %0 : i64
        %14 = llvm.mul %11, %13 : i64
        %15 = llvm.add %14, %1 : i64
        %16 = llvm.mul %9, %0 : i64
        %17 = llvm.add %16, %15 : i64
        %18 = llvm.getelementptr %arg2[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %19 = llvm.load %18 : !llvm.ptr -> i32
        %20 = llvm.getelementptr %arg3[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %21 = llvm.load %20 : !llvm.ptr -> i32
        %22 = arith.addi %19, %21 : i32
        %23 = llvm.getelementptr %arg4[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %22, %23 : i32, !llvm.ptr
      }
      gpu.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
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
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: index, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c-128 = arith.constant -128 : index
      %c128 = arith.constant 128 : index
      %0 = llvm.mlir.constant(0 : i64) : i64
      %1 = llvm.mlir.constant(1 : i64) : i64
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %2 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %3 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %4 = arith.addi %3, %arg0 : index
      %5 = arith.minsi %4, %c128 : index
      %6 = arith.cmpi slt, %thread_id_x, %5 : index
      scf.if %6 {
        %7 = arith.addi %thread_id_x, %2 : index
        %8 = arith.remsi %7, %arg1 : index
        %9 = builtin.unrealized_conversion_cast %8 : index to i64
        %10 = arith.divsi %7, %arg1 : index
        %11 = builtin.unrealized_conversion_cast %10 : index to i64
        %12 = arith.index_cast %arg1 : index to i64
        %13 = llvm.mul %12, %1 : i64
        %14 = llvm.mul %11, %13 : i64
        %15 = llvm.add %14, %0 : i64
        %16 = llvm.mul %9, %1 : i64
        %17 = llvm.add %16, %15 : i64
        %18 = llvm.getelementptr %arg2[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %19 = llvm.load %18 : !llvm.ptr -> i32
        %20 = llvm.getelementptr %arg3[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %21 = llvm.load %20 : !llvm.ptr -> i32
        %22 = arith.addi %19, %21 : i32
        %23 = llvm.getelementptr %arg4[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %22, %23 : i32, !llvm.ptr
      }
      gpu.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After SCFToControlFlowPass (convert-scf-to-cf) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
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
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: index, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c-128 = arith.constant -128 : index
      %c128 = arith.constant 128 : index
      %0 = llvm.mlir.constant(0 : i64) : i64
      %1 = llvm.mlir.constant(1 : i64) : i64
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %2 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %3 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %4 = arith.addi %3, %arg0 : index
      %5 = arith.minsi %4, %c128 : index
      %6 = arith.cmpi slt, %thread_id_x, %5 : index
      cf.cond_br %6, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %7 = arith.addi %thread_id_x, %2 : index
      %8 = arith.remsi %7, %arg1 : index
      %9 = builtin.unrealized_conversion_cast %8 : index to i64
      %10 = arith.divsi %7, %arg1 : index
      %11 = builtin.unrealized_conversion_cast %10 : index to i64
      %12 = arith.index_cast %arg1 : index to i64
      %13 = llvm.mul %12, %1 : i64
      %14 = llvm.mul %11, %13 : i64
      %15 = llvm.add %14, %0 : i64
      %16 = llvm.mul %9, %1 : i64
      %17 = llvm.add %16, %15 : i64
      %18 = llvm.getelementptr %arg2[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %19 = llvm.load %18 : !llvm.ptr -> i32
      %20 = llvm.getelementptr %arg3[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %21 = llvm.load %20 : !llvm.ptr -> i32
      %22 = arith.addi %19, %21 : i32
      %23 = llvm.getelementptr %arg4[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %22, %23 : i32, !llvm.ptr
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      gpu.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After ConvertGpuOpsToNVVMOps (convert-gpu-to-nvvm) ('gpu.module' operation: @add_two_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
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
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(0 : i64) : i64
      %3 = llvm.mlir.constant(1 : i64) : i64
      %4 = nvvm.read.ptx.sreg.ctaid.x : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %7 = llvm.sext %6 : i32 to i64
      %8 = llvm.mul %5, %1 overflow<nsw> : i64
      %9 = llvm.mul %5, %0 overflow<nsw> : i64
      %10 = llvm.add %9, %arg0 : i64
      %11 = llvm.intr.smin(%10, %1) : (i64, i64) -> i64
      %12 = llvm.icmp "slt" %7, %11 : i64
      llvm.cond_br %12, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %13 = llvm.add %7, %8 : i64
      %14 = llvm.srem %13, %arg1 : i64
      %15 = builtin.unrealized_conversion_cast %14 : i64 to index
      %16 = builtin.unrealized_conversion_cast %15 : index to i64
      %17 = llvm.sdiv %13, %arg1 : i64
      %18 = builtin.unrealized_conversion_cast %17 : i64 to index
      %19 = builtin.unrealized_conversion_cast %18 : index to i64
      %20 = llvm.mul %arg1, %3 : i64
      %21 = llvm.mul %19, %20 : i64
      %22 = llvm.add %21, %2 : i64
      %23 = llvm.mul %16, %3 : i64
      %24 = llvm.add %23, %22 : i64
      %25 = llvm.getelementptr %arg2[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %26 = llvm.load %25 : !llvm.ptr -> i32
      %27 = llvm.getelementptr %arg3[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %28 = llvm.load %27 : !llvm.ptr -> i32
      %29 = llvm.add %26, %28 : i32
      %30 = llvm.getelementptr %arg4[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %29, %30 : i32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After ConvertNVVMToLLVMPass (convert-nvvm-to-llvm) ('gpu.module' operation: @add_two_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
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
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(0 : i64) : i64
      %3 = llvm.mlir.constant(1 : i64) : i64
      %4 = nvvm.read.ptx.sreg.ctaid.x : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %7 = llvm.sext %6 : i32 to i64
      %8 = llvm.mul %5, %1 overflow<nsw> : i64
      %9 = llvm.mul %5, %0 overflow<nsw> : i64
      %10 = llvm.add %9, %arg0 : i64
      %11 = llvm.intr.smin(%10, %1) : (i64, i64) -> i64
      %12 = llvm.icmp "slt" %7, %11 : i64
      llvm.cond_br %12, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %13 = llvm.add %7, %8 : i64
      %14 = llvm.srem %13, %arg1 : i64
      %15 = builtin.unrealized_conversion_cast %14 : i64 to index
      %16 = llvm.sdiv %13, %arg1 : i64
      %17 = builtin.unrealized_conversion_cast %16 : i64 to index
      %18 = llvm.mul %arg1, %3 : i64
      %19 = llvm.mul %16, %18 : i64
      %20 = llvm.add %19, %2 : i64
      %21 = llvm.mul %14, %3 : i64
      %22 = llvm.add %21, %20 : i64
      %23 = llvm.getelementptr %arg2[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %24 = llvm.load %23 : !llvm.ptr -> i32
      %25 = llvm.getelementptr %arg3[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %26 = llvm.load %25 : !llvm.ptr -> i32
      %27 = llvm.add %24, %26 : i32
      %28 = llvm.getelementptr %arg4[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %27, %28 : i32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After NVVMOptimizeForTargetPass (llvm-optimize-for-nvvm-target) ('gpu.module' operation: @add_two_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
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
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%10, %c1, %c1) threads in (%c128, %c1, %c1)  args(%2 : index, %1 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(0 : i64) : i64
      %3 = llvm.mlir.constant(1 : i64) : i64
      %4 = nvvm.read.ptx.sreg.ctaid.x : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %7 = llvm.sext %6 : i32 to i64
      %8 = llvm.mul %5, %1 overflow<nsw> : i64
      %9 = llvm.mul %5, %0 overflow<nsw> : i64
      %10 = llvm.add %9, %arg0 : i64
      %11 = llvm.intr.smin(%10, %1) : (i64, i64) -> i64
      %12 = llvm.icmp "slt" %7, %11 : i64
      llvm.cond_br %12, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %13 = llvm.add %7, %8 : i64
      %14 = llvm.srem %13, %arg1 : i64
      %15 = llvm.sdiv %13, %arg1 : i64
      %16 = llvm.mul %arg1, %3 : i64
      %17 = llvm.mul %15, %16 : i64
      %18 = llvm.add %17, %2 : i64
      %19 = llvm.mul %14, %3 : i64
      %20 = llvm.add %19, %18 : i64
      %21 = llvm.getelementptr %arg2[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %22 = llvm.load %21 : !llvm.ptr -> i32
      %23 = llvm.getelementptr %arg3[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %24 = llvm.load %23 : !llvm.ptr -> i32
      %25 = llvm.add %22, %24 : i32
      %26 = llvm.getelementptr %arg4[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %25, %26 : i32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %7 = llvm.mul %c16_i32, %c32_i32 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuToLLVMConversionPass (gpu-to-llvm) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
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
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%13, %2, %2) threads in (%1, %2, %2) : i64 args(%5 : i64, %4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(0 : i64) : i64
      %3 = llvm.mlir.constant(1 : i64) : i64
      %4 = nvvm.read.ptx.sreg.ctaid.x : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %7 = llvm.sext %6 : i32 to i64
      %8 = llvm.mul %5, %1 overflow<nsw> : i64
      %9 = llvm.mul %5, %0 overflow<nsw> : i64
      %10 = llvm.add %9, %arg0 : i64
      %11 = llvm.intr.smin(%10, %1) : (i64, i64) -> i64
      %12 = llvm.icmp "slt" %7, %11 : i64
      llvm.cond_br %12, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %13 = llvm.add %7, %8 : i64
      %14 = llvm.srem %13, %arg1 : i64
      %15 = llvm.sdiv %13, %arg1 : i64
      %16 = llvm.mul %arg1, %3 : i64
      %17 = llvm.mul %15, %16 : i64
      %18 = llvm.add %17, %2 : i64
      %19 = llvm.mul %14, %3 : i64
      %20 = llvm.add %19, %18 : i64
      %21 = llvm.getelementptr %arg2[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %22 = llvm.load %21 : !llvm.ptr -> i32
      %23 = llvm.getelementptr %arg3[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %24 = llvm.load %23 : !llvm.ptr -> i32
      %25 = llvm.add %22, %24 : i32
      %26 = llvm.getelementptr %arg4[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %25, %26 : i32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mlir.constant(16 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mul %7, %8 : i32
    %10 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %13 = llvm.call @toy_random(%10, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.call @toy_random(%11, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %15 = llvm.call @add_two_vectors(%10, %11, %12, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%10, %9, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%11, %9, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %20 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %20 : !llvm.array<3 x i8>, !llvm.ptr
    %21 = llvm.call @toy_print(%12, %9, %4, %20) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuNVVMAttachTarget (nvvm-attach-target) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
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
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%13, %2, %2) threads in (%1, %2, %2) : i64 args(%5 : i64, %4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel [#nvvm.target<chip = "sm_120", features = "+ptx91">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = llvm.mlir.constant(0 : i64) : i64
      %3 = llvm.mlir.constant(1 : i64) : i64
      %4 = nvvm.read.ptx.sreg.ctaid.x : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %7 = llvm.sext %6 : i32 to i64
      %8 = llvm.mul %5, %1 overflow<nsw> : i64
      %9 = llvm.mul %5, %0 overflow<nsw> : i64
      %10 = llvm.add %9, %arg0 : i64
      %11 = llvm.intr.smin(%10, %1) : (i64, i64) -> i64
      %12 = llvm.icmp "slt" %7, %11 : i64
      llvm.cond_br %12, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %13 = llvm.add %7, %8 : i64
      %14 = llvm.srem %13, %arg1 : i64
      %15 = llvm.sdiv %13, %arg1 : i64
      %16 = llvm.mul %arg1, %3 : i64
      %17 = llvm.mul %15, %16 : i64
      %18 = llvm.add %17, %2 : i64
      %19 = llvm.mul %14, %3 : i64
      %20 = llvm.add %19, %18 : i64
      %21 = llvm.getelementptr %arg2[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %22 = llvm.load %21 : !llvm.ptr -> i32
      %23 = llvm.getelementptr %arg3[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %24 = llvm.load %23 : !llvm.ptr -> i32
      %25 = llvm.add %22, %24 : i32
      %26 = llvm.getelementptr %arg4[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %25, %26 : i32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mlir.constant(16 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mul %7, %8 : i32
    %10 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %13 = llvm.call @toy_random(%10, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.call @toy_random(%11, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %15 = llvm.call @add_two_vectors(%10, %11, %12, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%10, %9, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%11, %9, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %20 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %20 : !llvm.array<3 x i8>, !llvm.ptr
    %21 = llvm.call @toy_print(%12, %9, %4, %20) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuModuleToBinaryPass (gpu-module-to-binary) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
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
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%13, %2, %2) threads in (%1, %2, %2) : i64 args(%5 : i64, %4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.binary @add_two_vectors_kernel  [#gpu.object<#nvvm.target<chip = "sm_120", features = "+ptx91">, properties = {ISAToBinaryTimeInMs = 8 : i64, LLVMIRToISATimeInMs = 9 : i64}, bin = "\7FELF\02\01\01A\08\00\00\00\00\00\00\00\02\00\BE\00\01\00\00\00\00\00\00\00\00\00\00\00\C8\15\00\00\00\00\00\00H\10\00\00\00\00\00\00\02x\00\06@\008\00\05\00@\00\16\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.note.nv.tkinfo\00.note.nv.cuinfo\00.nv.info\00.nv.compat\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.shared.reserved.0\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.constant0.add_two_vectors_kernel\00.nv.capmerc.text.add_two_vectors_kernel\00.nv.merc.debug_frame\00.nv.merc.nv.info\00.nv.merc.nv.info.add_two_vectors_kernel\00.nv.merc.rela.debug_frame\00.nv.merc.nv.shared.reserved.0\00.nv.merc.symtab\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.note.nv.tkinfo\00.note.nv.cuinfo\00.nv.info\00.nv.compat\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.reservedSmem.offset0\00.nv.shared.reserved.0\00__nv_reservedSMEM_offset_0_alias\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\03\00\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\009\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00]\00\00\00\03\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BC\00\00\00!\00\00\00@\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\EB\00\00\00 \A0\0D\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0C\01\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00<\01\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00X\01\00\00\12\10\0C\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00o\01\00\00\03\00\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\04$\00\00\00\0C\81\80\80(\00\044\00\00\00\00\00\00\00\0C\00\00\00\8C\00\00\00\D0\07\00\00NVIDIA Corp\00\02\00\00\00\00\00\00\00\01\00\00\00\07\00\00\006\00\00\00`\00\00\00\00ptxas\00Cuda compilation tools, release 13.1, V13.1.80\00Build cuda_13.1.r13.1/compiler.36836380_0\00-O 2 -arch sm_120 \00\00\0C\00\00\00\08\00\00\00\E8\03\00\00NVIDIA Corp\00\02\00x\00\83\00\00\00\04/\08\00\08\00\00\00\0C\00\00\00\04\11\08\00\08\00\00\00\00\00\00\00\04\12\08\00\08\00\00\00\00\00\00\00\02\09\00\00\02\02\01\00\03\07\01\01\02\03\00\00\04\0B\08\00P\00\00\00\00\00\00\00\047\04\00\83\00\00\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F5!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03P\00\00\03\1B\FF\00\03_\01\01\02J\00\00\04\1C\08\00\80\00\00\00`\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\03\19(\00\04\0A\08\00\09\00\00\00\80\03(\00\046\04\00\00\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FFD\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\82{\01\FF\00\DF\00\00\00\08\00\00\00\E2\0F\00\19y\04\00\00\00\00\00\00%\00\00\00\22\0E\00\ACw\04\FF\00p\00\00\00\0A\00\08\00b\0E\001t\03\FF\00\00\00\00\FF\01\00\00\00\E2\0F\00\19y\02\00\00\00\00\00\00!\00\00\00\A2\0E\00%x\04\04\80\00\00\00\FF\00\8E\07\00\CA\1F\005|\06\04\04\00\00\00\00\03\8E\0F\00\CC/\00\0Cr\00\06\02\00\00\00pB\F1\03\00\DCO\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\ACw\08\FF\00r\00\00\00\0C\00\08\00\22\0E\00\12r\03\04\02\00\00\00\FF\FC\8E\07\00\E2\0F\00\ACw\04\FF\00k\00\00\00\0A\00\08\00f\0E\00\19x\05\03\02\00\00\00\05\02\01\00\00\E2\0F\04$x\04\03\04\00\00\00\FF\00\8E\07\00\E2\0F\00\ACw\06\FF\00t\00\00\00\0A\00\08\00\A8\0E\005|\02\04\08\00\00\00\00\02\8E\0F\00\C4\1F\005|\06\04\0A\00\00\00\00\02\8E\0F\00\CA\0F\00\81y\02\02\04\00\00\00\00\19\1E\0C\00\E8.\00\81y\07\06\04\00\00\00\00\19\1E\0C\00\E2\0E\005|\04\04\06\00\00\00\00\02\8E\0F\00\E4O\005r\09\02\07\00\00\00\00\00\8E\07\00\CA\8F\00\86y\00\04\09\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\FC\00\FC\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\01\00\00\C0\18\00\00\00(5F\00\01\0B\04\0A\F8\00\04\00\00\00A\00\00\04\00\00\01\0B\04\0A\F8\00\04\00\00\00\01\01\04\02\00\00\02\22\08\06\FA\00R\00\00\00\03\01@\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\0B\04\0A\F8\00\04\00\00\00\81\00\01\02\00\00A\0CT\04A\10r\0A\02\22\08\06\FA\00b\00\00\00\07\02@\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\10\00\00\00\01\0B\0E\0A\FA\00\05\00\00\00\03\019\04\00\00\02\22\08\06\FA\00R\00\00\00\83\01@\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00 \00\00\00A\0CT\04A\0CT\04A\0CT\04A\0C@\04\028\0E2\F8\00@\11\00\00\00\00\02\01\0A\00\00\02\01@\02\00\00\00\00\00\00\00\00\00\00\00\D0\04\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\01|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\02\00\00\00\00\00\00\04\10\00\00\00\04\B0\00\00\00\0C\81\80\80(\00\04@\01\00\00\00\00\00\00\00\00\00\00\00\04/\08\00\08\00\00\00\0C\00\00\00\04\11\08\00\08\00\00\00\00\00\00\00\04\12\08\00\08\00\00\00\00\00\00\00\047\04\00\83\00\00\00\04Z \00\8A\9D\22\A4\B1\9D\14m\00\B4*\F3\F7\F5\AE\08\8A\81\8C\9Dd\B3j\22\A2\A1\E4\C1\A7\82\00\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F5!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03P\00\00\03\1B\FF\00\03_\01\01\02J\00\00\04\1C\08\00\B0\00\00\00\00\02\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00D\00\00\00\00\00\00\00=\00\01\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\03\00\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\009\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00]\00\00\00\03\00\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BC\00\00\00!\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\EB\00\00\00 \A0\14\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0C\01\00\00\03\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00<\01\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00X\01\00\00\12\10\0F\00\00\00\00\00\00\00\00\00\10\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\01\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00e\02\00\00\00\00\00\00\94\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\F0\00\00\00\00\00\00\00\02\00\00\00\0A\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\D2\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0\04\00\00\00\00\00\00h\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\07\00\00\00\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00X\05\00\00\00\00\00\00\A4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\009\00\00\00\07\00\00\00@\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\FC\05\00\00\00\00\00\00 \00\00\00\00\00\00\00\05\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00I\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1C\06\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00R\00\00\00\86\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\06\00\00\00\00\00\00\1C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\\\06\00\00\00\00\00\00\9C\00\00\00\00\00\00\00\03\00\00\00\0C\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F8\06\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\F0\00\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\07\00\00\00\00\00\00\18\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00]\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\07\00\00\00\00\00\00\00\02\00\00\00\00\00\00\03\00\00\00\08\00\00\00\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BC\00\00\00\08\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\09\00\00\00\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1E\01\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\09\00\00\00\00\00\00\A8\03\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00C\01\00\00\16\00\00p\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\000\0D\00\00\00\00\00\00\EA\00\00\00\00\00\00\00\15\00\00\00\08\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00k\01\00\00\01\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\1A\0E\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\01\00\00\83\00\00p\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\8C\0E\00\00\00\00\00\00$\00\00\00\00\00\00\00\15\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\91\01\00\00\83\00\00p@\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\B0\0E\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\15\00\00\00\0F\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B9\01\00\00\82\00\00p@\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00X\0F\00\00\00\00\00\00\18\00\00\00\00\00\00\00\15\00\00\00\10\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\D3\01\00\00\15\00\00p\03\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00p\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F1\01\00\00\85\00\00p\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00p\0F\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\06\00\00\00\04\00\00\00\C8\15\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\01\00\00\00\00\00\00\18\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\C8\15\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\01\00\00\00\00\00\00\18\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\80\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\80\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\80\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\03\00\00\00\00\00\00\A8\03\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mlir.constant(16 : i32) : i32
    %8 = llvm.mlir.constant(32 : i32) : i32
    %9 = llvm.mul %7, %8 : i32
    %10 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_alloc(%9, %4) : (i32, i32) -> !llvm.ptr
    %13 = llvm.call @toy_random(%10, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.call @toy_random(%11, %9, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %15 = llvm.call @add_two_vectors(%10, %11, %12, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%10, %9, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%11, %9, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %20 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %20 : !llvm.array<3 x i8>, !llvm.ptr
    %21 = llvm.call @toy_print(%12, %9, %4, %20) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: i32) -> !llvm.ptr {
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
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%13, %2, %2) threads in (%1, %2, %2) : i64 args(%5 : i64, %4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.binary @add_two_vectors_kernel  [#gpu.object<#nvvm.target<chip = "sm_120", features = "+ptx91">, properties = {ISAToBinaryTimeInMs = 8 : i64, LLVMIRToISATimeInMs = 9 : i64}, bin = "\7FELF\02\01\01A\08\00\00\00\00\00\00\00\02\00\BE\00\01\00\00\00\00\00\00\00\00\00\00\00\C8\15\00\00\00\00\00\00H\10\00\00\00\00\00\00\02x\00\06@\008\00\05\00@\00\16\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.note.nv.tkinfo\00.note.nv.cuinfo\00.nv.info\00.nv.compat\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.shared.reserved.0\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.constant0.add_two_vectors_kernel\00.nv.capmerc.text.add_two_vectors_kernel\00.nv.merc.debug_frame\00.nv.merc.nv.info\00.nv.merc.nv.info.add_two_vectors_kernel\00.nv.merc.rela.debug_frame\00.nv.merc.nv.shared.reserved.0\00.nv.merc.symtab\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.note.nv.tkinfo\00.note.nv.cuinfo\00.nv.info\00.nv.compat\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.reservedSmem.offset0\00.nv.shared.reserved.0\00__nv_reservedSMEM_offset_0_alias\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\03\00\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\009\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00]\00\00\00\03\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BC\00\00\00!\00\00\00@\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\EB\00\00\00 \A0\0D\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0C\01\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00<\01\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00X\01\00\00\12\10\0C\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00o\01\00\00\03\00\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\04$\00\00\00\0C\81\80\80(\00\044\00\00\00\00\00\00\00\0C\00\00\00\8C\00\00\00\D0\07\00\00NVIDIA Corp\00\02\00\00\00\00\00\00\00\01\00\00\00\07\00\00\006\00\00\00`\00\00\00\00ptxas\00Cuda compilation tools, release 13.1, V13.1.80\00Build cuda_13.1.r13.1/compiler.36836380_0\00-O 2 -arch sm_120 \00\00\0C\00\00\00\08\00\00\00\E8\03\00\00NVIDIA Corp\00\02\00x\00\83\00\00\00\04/\08\00\08\00\00\00\0C\00\00\00\04\11\08\00\08\00\00\00\00\00\00\00\04\12\08\00\08\00\00\00\00\00\00\00\02\09\00\00\02\02\01\00\03\07\01\01\02\03\00\00\04\0B\08\00P\00\00\00\00\00\00\00\047\04\00\83\00\00\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F5!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03P\00\00\03\1B\FF\00\03_\01\01\02J\00\00\04\1C\08\00\80\00\00\00`\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\03\19(\00\04\0A\08\00\09\00\00\00\80\03(\00\046\04\00\00\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FFD\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\82{\01\FF\00\DF\00\00\00\08\00\00\00\E2\0F\00\19y\04\00\00\00\00\00\00%\00\00\00\22\0E\00\ACw\04\FF\00p\00\00\00\0A\00\08\00b\0E\001t\03\FF\00\00\00\00\FF\01\00\00\00\E2\0F\00\19y\02\00\00\00\00\00\00!\00\00\00\A2\0E\00%x\04\04\80\00\00\00\FF\00\8E\07\00\CA\1F\005|\06\04\04\00\00\00\00\03\8E\0F\00\CC/\00\0Cr\00\06\02\00\00\00pB\F1\03\00\DCO\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\ACw\08\FF\00r\00\00\00\0C\00\08\00\22\0E\00\12r\03\04\02\00\00\00\FF\FC\8E\07\00\E2\0F\00\ACw\04\FF\00k\00\00\00\0A\00\08\00f\0E\00\19x\05\03\02\00\00\00\05\02\01\00\00\E2\0F\04$x\04\03\04\00\00\00\FF\00\8E\07\00\E2\0F\00\ACw\06\FF\00t\00\00\00\0A\00\08\00\A8\0E\005|\02\04\08\00\00\00\00\02\8E\0F\00\C4\1F\005|\06\04\0A\00\00\00\00\02\8E\0F\00\CA\0F\00\81y\02\02\04\00\00\00\00\19\1E\0C\00\E8.\00\81y\07\06\04\00\00\00\00\19\1E\0C\00\E2\0E\005|\04\04\06\00\00\00\00\02\8E\0F\00\E4O\005r\09\02\07\00\00\00\00\00\8E\07\00\CA\8F\00\86y\00\04\09\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\FC\00\FC\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\01\00\00\C0\18\00\00\00(5F\00\01\0B\04\0A\F8\00\04\00\00\00A\00\00\04\00\00\01\0B\04\0A\F8\00\04\00\00\00\01\01\04\02\00\00\02\22\08\06\FA\00R\00\00\00\03\01@\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\0B\04\0A\F8\00\04\00\00\00\81\00\01\02\00\00A\0CT\04A\10r\0A\02\22\08\06\FA\00b\00\00\00\07\02@\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\10\00\00\00\01\0B\0E\0A\FA\00\05\00\00\00\03\019\04\00\00\02\22\08\06\FA\00R\00\00\00\83\01@\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00 \00\00\00A\0CT\04A\0CT\04A\0CT\04A\0C@\04\028\0E2\F8\00@\11\00\00\00\00\02\01\0A\00\00\02\01@\02\00\00\00\00\00\00\00\00\00\00\00\D0\04\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\01|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\02\00\00\00\00\00\00\04\10\00\00\00\04\B0\00\00\00\0C\81\80\80(\00\04@\01\00\00\00\00\00\00\00\00\00\00\00\04/\08\00\08\00\00\00\0C\00\00\00\04\11\08\00\08\00\00\00\00\00\00\00\04\12\08\00\08\00\00\00\00\00\00\00\047\04\00\83\00\00\00\04Z \00\8A\9D\22\A4\B1\9D\14m\00\B4*\F3\F7\F5\AE\08\8A\81\8C\9Dd\B3j\22\A2\A1\E4\C1\A7\82\00\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F5!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03P\00\00\03\1B\FF\00\03_\01\01\02J\00\00\04\1C\08\00\B0\00\00\00\00\02\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00D\00\00\00\00\00\00\00=\00\01\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\03\00\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\009\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00]\00\00\00\03\00\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BC\00\00\00!\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\EB\00\00\00 \A0\14\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0C\01\00\00\03\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00<\01\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00X\01\00\00\12\10\0F\00\00\00\00\00\00\00\00\00\10\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\01\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00e\02\00\00\00\00\00\00\94\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\F0\00\00\00\00\00\00\00\02\00\00\00\0A\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\D2\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0\04\00\00\00\00\00\00h\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\07\00\00\00\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00X\05\00\00\00\00\00\00\A4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\009\00\00\00\07\00\00\00@\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\FC\05\00\00\00\00\00\00 \00\00\00\00\00\00\00\05\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00I\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1C\06\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00R\00\00\00\86\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\06\00\00\00\00\00\00\1C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\\\06\00\00\00\00\00\00\9C\00\00\00\00\00\00\00\03\00\00\00\0C\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F8\06\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\F0\00\00\00\04\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\07\00\00\00\00\00\00\18\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00]\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\07\00\00\00\00\00\00\00\02\00\00\00\00\00\00\03\00\00\00\08\00\00\00\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BC\00\00\00\08\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\09\00\00\00\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1E\01\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\09\00\00\00\00\00\00\A8\03\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00C\01\00\00\16\00\00p\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\000\0D\00\00\00\00\00\00\EA\00\00\00\00\00\00\00\15\00\00\00\08\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00k\01\00\00\01\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\1A\0E\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\01\00\00\83\00\00p\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\8C\0E\00\00\00\00\00\00$\00\00\00\00\00\00\00\15\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\91\01\00\00\83\00\00p@\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\B0\0E\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\15\00\00\00\0F\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B9\01\00\00\82\00\00p@\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00X\0F\00\00\00\00\00\00\18\00\00\00\00\00\00\00\15\00\00\00\10\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\D3\01\00\00\15\00\00p\03\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00p\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F1\01\00\00\85\00\00p\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00p\0F\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\06\00\00\00\04\00\00\00\C8\15\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\01\00\00\00\00\00\00\18\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\C8\15\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\01\00\00\00\00\00\00\18\01\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\80\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\80\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\04\00\00\00\80\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\03\00\00\00\00\00\00\A8\03\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  llvm.func @main() {
    %0 = llvm.mlir.constant("t3\00") : !llvm.array<3 x i8>
    %1 = llvm.mlir.constant("t2\00") : !llvm.array<3 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("t1\00") : !llvm.array<3 x i8>
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(16 : i32) : i32
    %6 = llvm.mlir.constant(32 : i32) : i32
    %7 = llvm.mul %5, %6 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %14 : !llvm.array<3 x i8>, !llvm.ptr
    %15 = llvm.call @toy_print(%8, %7, %4, %14) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %16 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %16 : !llvm.array<3 x i8>, !llvm.ptr
    %17 = llvm.call @toy_print(%9, %7, %4, %16) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %18 = llvm.alloca %2 x !llvm.array<3 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %18 : !llvm.array<3 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%10, %7, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


