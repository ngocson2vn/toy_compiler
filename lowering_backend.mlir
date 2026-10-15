// -----// IR Dump After TileLoopsPass (tile-loops) ('func.func' operation: @add_two_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (512, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xf32>, %arg3: i32) -> memref<?x8xf32> {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%0, %c8) step (%c512, %c1) {
      %1 = affine.min #map(%c512, %0, %arg4)
      scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%1, %c1) step (%c1, %c1) {
        %2 = arith.addi %arg6, %arg4 : index
        %3 = arith.addi %arg7, %arg5 : index
        %4 = memref.load %arg0[%2, %3] : memref<?x8xf32>
        %5 = memref.load %arg1[%2, %3] : memref<?x8xf32>
        %6 = arith.addf %4, %5 : f32
        memref.store %6, %arg2[%2, %3] : memref<?x8xf32>
        scf.reduce 
      }
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


// -----// IR Dump After GpuMapParallelLoopsPass (gpu-map-parallel-loops) ('func.func' operation: @add_two_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (512, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xf32>, %arg3: i32) -> memref<?x8xf32> {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%0, %c8) step (%c512, %c1) {
      %1 = affine.min #map(%c512, %0, %arg4)
      scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%1, %c1) step (%c1, %c1) {
        %2 = arith.addi %arg6, %arg4 : index
        %3 = arith.addi %arg7, %arg5 : index
        %4 = memref.load %arg0[%2, %3] : memref<?x8xf32>
        %5 = memref.load %arg1[%2, %3] : memref<?x8xf32>
        %6 = arith.addf %4, %5 : f32
        memref.store %6, %arg2[%2, %3] : memref<?x8xf32>
        scf.reduce 
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.reduce 
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
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


// -----// IR Dump After ConvertParallelLoopToGpuPass (convert-parallel-loops-to-gpu) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (512, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xf32>, %arg3: i32) -> memref<?x8xf32> {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %c1_0 = arith.constant 1 : index
    %1 = affine.apply #map(%0)[%c0, %c512]
    %2 = affine.apply #map(%c8)[%c0, %c1]
    %c512_1 = arith.constant 512 : index
    %3 = affine.apply #map(%c512_1)[%c0, %c1]
    %4 = affine.apply #map(%c1)[%c0, %c1]
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %1, %arg11 = %2, %arg12 = %c1_0) threads(%arg7, %arg8, %arg9) in (%arg13 = %3, %arg14 = %4, %arg15 = %c1_0) {
      %5 = affine.apply #map1(%arg4)[%c512, %c0]
      %6 = affine.apply #map1(%arg5)[%c1, %c0]
      %7 = affine.min #map2(%c512, %0, %5)
      %8 = affine.apply #map1(%arg7)[%c1, %c0]
      %9 = arith.cmpi slt, %8, %7 : index
      scf.if %9 {
        %10 = affine.apply #map1(%arg8)[%c1, %c0]
        %11 = arith.addi %8, %5 : index
        %12 = arith.addi %10, %6 : index
        %13 = memref.load %arg0[%11, %12] : memref<?x8xf32>
        %14 = memref.load %arg1[%11, %12] : memref<?x8xf32>
        %15 = arith.addf %13, %14 : f32
        memref.store %15, %arg2[%11, %12] : memref<?x8xf32>
      }
      gpu.terminator
    } {SCFToGPU_visited}
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


// -----// IR Dump After LowerMemRefToLLVMPass (lower-memref-to-llvm) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (512, d1 - d2)>
module {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %c1_0 = arith.constant 1 : index
    %1 = affine.apply #map(%0)[%c0, %c512]
    %c8_1 = arith.constant 8 : index
    %c512_2 = arith.constant 512 : index
    %c512_3 = arith.constant 512 : index
    %c1_4 = arith.constant 1 : index
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %1, %arg11 = %c8_1, %arg12 = %c1_0) threads(%arg7, %arg8, %arg9) in (%arg13 = %c512_3, %arg14 = %c1_4, %arg15 = %c1_0) {
      %2 = affine.apply #map1(%arg4)[%c512, %c0]
      %3 = affine.apply #map1(%arg5)[%c1, %c0]
      %4 = affine.min #map2(%c512, %0, %2)
      %5 = affine.apply #map1(%arg7)[%c1, %c0]
      %6 = arith.cmpi slt, %5, %4 : index
      scf.if %6 {
        %7 = affine.apply #map1(%arg8)[%c1, %c0]
        %8 = arith.addi %5, %2 : index
        %9 = builtin.unrealized_conversion_cast %8 : index to i64
        %10 = arith.addi %7, %3 : index
        %11 = llvm.getelementptr %arg0[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %12 = llvm.load %11 : !llvm.ptr -> f32
        %13 = llvm.getelementptr %arg1[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %14 = llvm.load %13 : !llvm.ptr -> f32
        %15 = arith.addf %12, %14 : f32
        %16 = llvm.getelementptr %arg2[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %15, %16 : f32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return %arg2 : !llvm.ptr
  }
  llvm.func @main() {
    %c8 = arith.constant 8 : index
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %1 : index to i32
    %3 = arith.index_cast %c8 : index to i32
    %4 = llvm.mul %2, %3 : i32
    %5 = llvm.mlir.constant(3 : i32) : i32
    %6 = llvm.call @toy_alloc(%4, %5) : (i32, i32) -> !llvm.ptr
    %7 = arith.index_cast %1 : index to i32
    %8 = arith.index_cast %c8 : index to i32
    %9 = llvm.mul %7, %8 : i32
    %10 = llvm.mlir.constant(3 : i32) : i32
    %11 = llvm.call @toy_alloc(%9, %10) : (i32, i32) -> !llvm.ptr
    %12 = arith.index_cast %1 : index to i32
    %13 = arith.index_cast %c8 : index to i32
    %14 = llvm.mul %12, %13 : i32
    %15 = llvm.mlir.constant(3 : i32) : i32
    %16 = llvm.call @toy_alloc(%14, %15) : (i32, i32) -> !llvm.ptr
    %17 = arith.index_cast %1 : index to i32
    %18 = arith.index_cast %c8 : index to i32
    %19 = llvm.mul %17, %18 : i32
    %20 = llvm.mlir.constant(3 : i32) : i32
    %21 = llvm.call @toy_random(%6, %19, %20) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %22 = arith.index_cast %1 : index to i32
    %23 = arith.index_cast %c8 : index to i32
    %24 = llvm.mul %22, %23 : i32
    %25 = llvm.mlir.constant(3 : i32) : i32
    %26 = llvm.call @toy_random(%11, %24, %25) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %27 = llvm.call @add_two_vectors(%6, %11, %16, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %28 = arith.index_cast %1 : index to i32
    %29 = arith.index_cast %c8 : index to i32
    %30 = llvm.mul %28, %29 : i32
    %31 = llvm.mlir.constant(3 : i32) : i32
    %32 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %33 = llvm.mlir.constant(1 : i32) : i32
    %34 = llvm.alloca %33 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %32, %34 : !llvm.array<2 x i8>, !llvm.ptr
    %35 = llvm.call @toy_print(%6, %30, %31, %34) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %36 = arith.index_cast %1 : index to i32
    %37 = arith.index_cast %c8 : index to i32
    %38 = llvm.mul %36, %37 : i32
    %39 = llvm.mlir.constant(3 : i32) : i32
    %40 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %41 = llvm.mlir.constant(1 : i32) : i32
    %42 = llvm.alloca %41 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %40, %42 : !llvm.array<2 x i8>, !llvm.ptr
    %43 = llvm.call @toy_print(%11, %38, %39, %42) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %44 = arith.index_cast %1 : index to i32
    %45 = arith.index_cast %c8 : index to i32
    %46 = llvm.mul %44, %45 : i32
    %47 = llvm.mlir.constant(3 : i32) : i32
    %48 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %49 = llvm.mlir.constant(1 : i32) : i32
    %50 = llvm.alloca %49 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %48, %50 : !llvm.array<4 x i8>, !llvm.ptr
    %51 = llvm.call @toy_print(%16, %46, %47, %50) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<()[s0] -> (s0 * 512)>
#map2 = affine_map<()[s0, s1] -> (s0 * -512 + s1, 512)>
module {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = affine.apply #map()[%0]
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %1, %arg11 = %c8, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c512, %arg14 = %c1, %arg15 = %c1) {
      %2 = affine.apply #map1()[%arg4]
      %3 = affine.min #map2()[%arg4, %0]
      %4 = arith.cmpi slt, %arg7, %3 : index
      scf.if %4 {
        %5 = arith.addi %arg7, %2 : index
        %6 = builtin.unrealized_conversion_cast %5 : index to i64
        %7 = llvm.getelementptr %arg0[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %8 = llvm.load %7 : !llvm.ptr -> f32
        %9 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %10 = llvm.load %9 : !llvm.ptr -> f32
        %11 = arith.addf %8, %10 : f32
        %12 = llvm.getelementptr %arg2[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %11, %12 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = llvm.mul %c4_i32, %c8_i32 : i32
    %7 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %8 = llvm.mul %c4_i32, %c8_i32 : i32
    %9 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.mul %c4_i32, %c8_i32 : i32
    %11 = llvm.call @toy_alloc(%10, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.mul %c4_i32, %c8_i32 : i32
    %13 = llvm.call @toy_random(%7, %12, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.mul %c4_i32, %c8_i32 : i32
    %15 = llvm.call @toy_random(%9, %14, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %16 = llvm.call @add_two_vectors(%7, %9, %11, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %17 = llvm.mul %c4_i32, %c8_i32 : i32
    %18 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %18 : !llvm.array<2 x i8>, !llvm.ptr
    %19 = llvm.call @toy_print(%7, %17, %4, %18) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %20 = llvm.mul %c4_i32, %c8_i32 : i32
    %21 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %21 : !llvm.array<2 x i8>, !llvm.ptr
    %22 = llvm.call @toy_print(%9, %20, %4, %21) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %23 = llvm.mul %c4_i32, %c8_i32 : i32
    %24 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %24 : !llvm.array<4 x i8>, !llvm.ptr
    %25 = llvm.call @toy_print(%11, %23, %4, %24) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After CSE (cse) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<()[s0] -> (s0 * 512)>
#map2 = affine_map<()[s0, s1] -> (s0 * -512 + s1, 512)>
module {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = affine.apply #map()[%0]
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %1, %arg11 = %c8, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c512, %arg14 = %c1, %arg15 = %c1) {
      %2 = affine.apply #map1()[%arg4]
      %3 = affine.min #map2()[%arg4, %0]
      %4 = arith.cmpi slt, %arg7, %3 : index
      scf.if %4 {
        %5 = arith.addi %arg7, %2 : index
        %6 = builtin.unrealized_conversion_cast %5 : index to i64
        %7 = llvm.getelementptr %arg0[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %8 = llvm.load %7 : !llvm.ptr -> f32
        %9 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %10 = llvm.load %9 : !llvm.ptr -> f32
        %11 = arith.addf %8, %10 : f32
        %12 = llvm.getelementptr %arg2[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %11, %12 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = llvm.mul %c4_i32, %c8_i32 : i32
    %7 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %8 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_random(%7, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @add_two_vectors(%7, %8, %9, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %13 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %13 : !llvm.array<2 x i8>, !llvm.ptr
    %14 = llvm.call @toy_print(%7, %6, %4, %13) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%8, %6, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %17 : !llvm.array<4 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%9, %6, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuKernelOutliningPass (gpu-kernel-outlining) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<()[s0] -> (s0 * 512)>
#map2 = affine_map<()[s0, s1] -> (s0 * -512 + s1, 512)>
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = affine.apply #map()[%0]
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%1, %c8, %c1) threads in (%c512, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) kernel attributes {known_block_size = array<i32: 512, 1, 1>} {
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
      %0 = affine.apply #map1()[%block_id_x]
      %1 = affine.min #map2()[%block_id_x, %arg0]
      %2 = arith.cmpi slt, %thread_id_x, %1 : index
      scf.if %2 {
        %3 = arith.addi %thread_id_x, %0 : index
        %4 = builtin.unrealized_conversion_cast %3 : index to i64
        %5 = llvm.getelementptr %arg1[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %6 = llvm.load %5 : !llvm.ptr -> f32
        %7 = llvm.getelementptr %arg2[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %8 = llvm.load %7 : !llvm.ptr -> f32
        %9 = arith.addf %6, %8 : f32
        %10 = llvm.getelementptr %arg3[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %9, %10 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = llvm.mul %c4_i32, %c8_i32 : i32
    %7 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %8 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_random(%7, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @add_two_vectors(%7, %8, %9, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %13 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %13 : !llvm.array<2 x i8>, !llvm.ptr
    %14 = llvm.call @toy_print(%7, %6, %4, %13) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%8, %6, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %17 : !llvm.array<4 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%9, %6, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After LowerAffinePass (lower-affine) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %c512_0 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1_1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c512_0 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1_1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c8, %c1) threads in (%c512, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) kernel attributes {known_block_size = array<i32: 512, 1, 1>} {
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
      %c512 = arith.constant 512 : index
      %0 = arith.muli %block_id_x, %c512 overflow<nsw> : index
      %c-512 = arith.constant -512 : index
      %1 = arith.muli %block_id_x, %c-512 overflow<nsw> : index
      %2 = arith.addi %1, %arg0 : index
      %c512_0 = arith.constant 512 : index
      %3 = arith.minsi %2, %c512_0 : index
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      scf.if %4 {
        %5 = arith.addi %thread_id_x, %0 : index
        %6 = builtin.unrealized_conversion_cast %5 : index to i64
        %7 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %8 = llvm.load %7 : !llvm.ptr -> f32
        %9 = llvm.getelementptr %arg2[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %10 = llvm.load %9 : !llvm.ptr -> f32
        %11 = arith.addf %8, %10 : f32
        %12 = llvm.getelementptr %arg3[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %11, %12 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = llvm.mul %c4_i32, %c8_i32 : i32
    %7 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %8 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_random(%7, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @add_two_vectors(%7, %8, %9, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %13 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %13 : !llvm.array<2 x i8>, !llvm.ptr
    %14 = llvm.call @toy_print(%7, %6, %4, %13) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%8, %6, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %17 : !llvm.array<4 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%9, %6, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c512 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c8, %c1) threads in (%c512, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) kernel attributes {known_block_size = array<i32: 512, 1, 1>} {
      %c-512 = arith.constant -512 : index
      %c512 = arith.constant 512 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c512 overflow<nsw> : index
      %1 = arith.muli %block_id_x, %c-512 overflow<nsw> : index
      %2 = arith.addi %1, %arg0 : index
      %3 = arith.minsi %2, %c512 : index
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      scf.if %4 {
        %5 = arith.addi %thread_id_x, %0 : index
        %6 = builtin.unrealized_conversion_cast %5 : index to i64
        %7 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %8 = llvm.load %7 : !llvm.ptr -> f32
        %9 = llvm.getelementptr %arg2[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %10 = llvm.load %9 : !llvm.ptr -> f32
        %11 = arith.addf %8, %10 : f32
        %12 = llvm.getelementptr %arg3[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %11, %12 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = llvm.mul %c4_i32, %c8_i32 : i32
    %7 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %8 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_random(%7, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @add_two_vectors(%7, %8, %9, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %13 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %13 : !llvm.array<2 x i8>, !llvm.ptr
    %14 = llvm.call @toy_print(%7, %6, %4, %13) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%8, %6, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %17 : !llvm.array<4 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%9, %6, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After SCFToControlFlowPass (convert-scf-to-cf) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c512 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c8, %c1) threads in (%c512, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) kernel attributes {known_block_size = array<i32: 512, 1, 1>} {
      %c-512 = arith.constant -512 : index
      %c512 = arith.constant 512 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c512 overflow<nsw> : index
      %1 = arith.muli %block_id_x, %c-512 overflow<nsw> : index
      %2 = arith.addi %1, %arg0 : index
      %3 = arith.minsi %2, %c512 : index
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      cf.cond_br %4, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %5 = arith.addi %thread_id_x, %0 : index
      %6 = builtin.unrealized_conversion_cast %5 : index to i64
      %7 = llvm.getelementptr %arg1[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %8 = llvm.load %7 : !llvm.ptr -> f32
      %9 = llvm.getelementptr %arg2[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %10 = llvm.load %9 : !llvm.ptr -> f32
      %11 = arith.addf %8, %10 : f32
      %12 = llvm.getelementptr %arg3[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %11, %12 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = llvm.mul %c4_i32, %c8_i32 : i32
    %7 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %8 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_random(%7, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @add_two_vectors(%7, %8, %9, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %13 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %13 : !llvm.array<2 x i8>, !llvm.ptr
    %14 = llvm.call @toy_print(%7, %6, %4, %13) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%8, %6, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %17 : !llvm.array<4 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%9, %6, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After ConvertGpuOpsToNVVMOps (convert-gpu-to-nvvm) ('gpu.module' operation: @add_two_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c512 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c8, %c1) threads in (%c512, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 512, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 512, 1, 1>} {
      %0 = llvm.mlir.constant(-512 : index) : i64
      %1 = llvm.mlir.constant(512 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 512> : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = llvm.mul %3, %1 overflow<nsw> : i64
      %7 = llvm.mul %3, %0 overflow<nsw> : i64
      %8 = llvm.add %7, %arg0 : i64
      %9 = llvm.intr.smin(%8, %1) : (i64, i64) -> i64
      %10 = llvm.icmp "slt" %5, %9 : i64
      llvm.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %11 = llvm.add %5, %6 : i64
      %12 = builtin.unrealized_conversion_cast %11 : i64 to index
      %13 = builtin.unrealized_conversion_cast %12 : index to i64
      %14 = llvm.getelementptr %arg1[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %15 = llvm.load %14 : !llvm.ptr -> f32
      %16 = llvm.getelementptr %arg2[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %17 = llvm.load %16 : !llvm.ptr -> f32
      %18 = llvm.fadd %15, %17 : f32
      %19 = llvm.getelementptr %arg3[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %18, %19 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = llvm.mul %c4_i32, %c8_i32 : i32
    %7 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %8 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_random(%7, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @add_two_vectors(%7, %8, %9, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %13 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %13 : !llvm.array<2 x i8>, !llvm.ptr
    %14 = llvm.call @toy_print(%7, %6, %4, %13) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%8, %6, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %17 : !llvm.array<4 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%9, %6, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After ConvertNVVMToLLVMPass (convert-nvvm-to-llvm) ('gpu.module' operation: @add_two_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c512 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c8, %c1) threads in (%c512, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 512, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 512, 1, 1>} {
      %0 = llvm.mlir.constant(-512 : index) : i64
      %1 = llvm.mlir.constant(512 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 512> : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = llvm.mul %3, %1 overflow<nsw> : i64
      %7 = llvm.mul %3, %0 overflow<nsw> : i64
      %8 = llvm.add %7, %arg0 : i64
      %9 = llvm.intr.smin(%8, %1) : (i64, i64) -> i64
      %10 = llvm.icmp "slt" %5, %9 : i64
      llvm.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %11 = llvm.add %5, %6 : i64
      %12 = builtin.unrealized_conversion_cast %11 : i64 to index
      %13 = llvm.getelementptr %arg1[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %14 = llvm.load %13 : !llvm.ptr -> f32
      %15 = llvm.getelementptr %arg2[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %16 = llvm.load %15 : !llvm.ptr -> f32
      %17 = llvm.fadd %14, %16 : f32
      %18 = llvm.getelementptr %arg3[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %17, %18 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = llvm.mul %c4_i32, %c8_i32 : i32
    %7 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %8 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_random(%7, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @add_two_vectors(%7, %8, %9, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %13 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %13 : !llvm.array<2 x i8>, !llvm.ptr
    %14 = llvm.call @toy_print(%7, %6, %4, %13) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%8, %6, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %17 : !llvm.array<4 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%9, %6, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After NVVMOptimizeForTargetPass (llvm-optimize-for-nvvm-target) ('gpu.module' operation: @add_two_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c512 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c8, %c1) threads in (%c512, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 512, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 512, 1, 1>} {
      %0 = llvm.mlir.constant(-512 : index) : i64
      %1 = llvm.mlir.constant(512 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 512> : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = llvm.mul %3, %1 overflow<nsw> : i64
      %7 = llvm.mul %3, %0 overflow<nsw> : i64
      %8 = llvm.add %7, %arg0 : i64
      %9 = llvm.intr.smin(%8, %1) : (i64, i64) -> i64
      %10 = llvm.icmp "slt" %5, %9 : i64
      llvm.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %11 = llvm.add %5, %6 : i64
      %12 = llvm.getelementptr %arg1[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %13 = llvm.load %12 : !llvm.ptr -> f32
      %14 = llvm.getelementptr %arg2[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %15 = llvm.load %14 : !llvm.ptr -> f32
      %16 = llvm.fadd %13, %15 : f32
      %17 = llvm.getelementptr %arg3[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %16, %17 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = llvm.mul %c4_i32, %c8_i32 : i32
    %7 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %8 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%6, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_random(%7, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %6, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @add_two_vectors(%7, %8, %9, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %13 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %13 : !llvm.array<2 x i8>, !llvm.ptr
    %14 = llvm.call @toy_print(%7, %6, %4, %13) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%8, %6, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %17 : !llvm.array<4 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%9, %6, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuToLLVMConversionPass (gpu-to-llvm) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(512 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(8 : index) : i64
    %4 = llvm.sext %arg3 : i32 to i64
    %5 = llvm.icmp "sle" %4, %0 : i64
    %6 = llvm.sub %0, %4 : i64
    %7 = llvm.sub %4, %2 : i64
    %8 = llvm.select %5, %6, %7 : i1, i64
    %9 = llvm.sdiv %8, %1 : i64
    %10 = llvm.sub %0, %9 : i64
    %11 = llvm.add %9, %2 : i64
    %12 = llvm.select %5, %10, %11 : i1, i64
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%12, %3, %2) threads in (%1, %2, %2) : i64 args(%4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 512, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 512, 1, 1>} {
      %0 = llvm.mlir.constant(-512 : index) : i64
      %1 = llvm.mlir.constant(512 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 512> : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = llvm.mul %3, %1 overflow<nsw> : i64
      %7 = llvm.mul %3, %0 overflow<nsw> : i64
      %8 = llvm.add %7, %arg0 : i64
      %9 = llvm.intr.smin(%8, %1) : (i64, i64) -> i64
      %10 = llvm.icmp "slt" %5, %9 : i64
      llvm.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %11 = llvm.add %5, %6 : i64
      %12 = llvm.getelementptr %arg1[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %13 = llvm.load %12 : !llvm.ptr -> f32
      %14 = llvm.getelementptr %arg2[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %15 = llvm.load %14 : !llvm.ptr -> f32
      %16 = llvm.fadd %13, %15 : f32
      %17 = llvm.getelementptr %arg3[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %16, %17 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mul %6, %7 : i32
    %9 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %8, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @toy_random(%10, %8, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.call @add_two_vectors(%9, %10, %11, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%9, %8, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %17 : !llvm.array<2 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%10, %8, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %19 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %19 : !llvm.array<4 x i8>, !llvm.ptr
    %20 = llvm.call @toy_print(%11, %8, %4, %19) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuNVVMAttachTarget (nvvm-attach-target) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(512 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(8 : index) : i64
    %4 = llvm.sext %arg3 : i32 to i64
    %5 = llvm.icmp "sle" %4, %0 : i64
    %6 = llvm.sub %0, %4 : i64
    %7 = llvm.sub %4, %2 : i64
    %8 = llvm.select %5, %6, %7 : i1, i64
    %9 = llvm.sdiv %8, %1 : i64
    %10 = llvm.sub %0, %9 : i64
    %11 = llvm.add %9, %2 : i64
    %12 = llvm.select %5, %10, %11 : i1, i64
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%12, %3, %2) threads in (%1, %2, %2) : i64 args(%4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.module @add_two_vectors_kernel [#nvvm.target<chip = "sm_86", features = "+ptx84">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 512, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 512, 1, 1>} {
      %0 = llvm.mlir.constant(-512 : index) : i64
      %1 = llvm.mlir.constant(512 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 512> : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = llvm.mul %3, %1 overflow<nsw> : i64
      %7 = llvm.mul %3, %0 overflow<nsw> : i64
      %8 = llvm.add %7, %arg0 : i64
      %9 = llvm.intr.smin(%8, %1) : (i64, i64) -> i64
      %10 = llvm.icmp "slt" %5, %9 : i64
      llvm.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %11 = llvm.add %5, %6 : i64
      %12 = llvm.getelementptr %arg1[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %13 = llvm.load %12 : !llvm.ptr -> f32
      %14 = llvm.getelementptr %arg2[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %15 = llvm.load %14 : !llvm.ptr -> f32
      %16 = llvm.fadd %13, %15 : f32
      %17 = llvm.getelementptr %arg3[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %16, %17 : f32, !llvm.ptr
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
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mul %6, %7 : i32
    %9 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %8, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @toy_random(%10, %8, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.call @add_two_vectors(%9, %10, %11, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%9, %8, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %17 : !llvm.array<2 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%10, %8, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %19 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %19 : !llvm.array<4 x i8>, !llvm.ptr
    %20 = llvm.call @toy_print(%11, %8, %4, %19) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After GpuModuleToBinaryPass (gpu-module-to-binary) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(512 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(8 : index) : i64
    %4 = llvm.sext %arg3 : i32 to i64
    %5 = llvm.icmp "sle" %4, %0 : i64
    %6 = llvm.sub %0, %4 : i64
    %7 = llvm.sub %4, %2 : i64
    %8 = llvm.select %5, %6, %7 : i1, i64
    %9 = llvm.sdiv %8, %1 : i64
    %10 = llvm.sub %0, %9 : i64
    %11 = llvm.add %9, %2 : i64
    %12 = llvm.select %5, %10, %11 : i1, i64
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%12, %3, %2) threads in (%1, %2, %2) : i64 args(%4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.binary @add_two_vectors_kernel  [#gpu.object<#nvvm.target<chip = "sm_86", features = "+ptx84">, properties = {ISAToBinaryTimeInMs = 9 : i64, LLVMIRToISATimeInMs = 12 : i64}, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\00\00\00\00\09\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\048\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01 \00\03\19 \00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F5!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03\1B\80\00\04\1C\08\00\80\00\00\00p\01\00\00\04\05\0C\00\00\02\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\00\02\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00X\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00Y\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\00\05\02\00\00\00\03\02\01\00\00\E2\0F\04$x\06\05\04\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\06\00\\\00\00\FF\E0\F1\07\00\E4\0F\04\10z\02\06\00Z\00\00\FF\E0\F3\07\00\E4\0F\00\10z\05\00\00]\00\00\FF\E4\7F\00\00\E4\0F\04\10z\03\00\00[\00\00\FF\E4\FF\00\00\C8\0F\00\81y\05\04\04\00\00\00\00\19\1E\0C\00\A8\0E\00\81y\02\02\04\00\00\00\00\19\1E\0C\00\A2\0E\00\10z\06\06\00^\00\00\FF\E0\F1\07\00\C8\0F\00\10z\07\00\00_\00\00\FF\E4\7F\00\00\E2\0F\00!r\09\02\05\00\00\00\00\00\00\00\00\CAO\00\86y\00\06\09\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00|\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\94\04\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D8\04\00\00\00\00\00\00\80\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\D8\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\04\00\00\00\00\00\00(\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(8 : i32) : i32
    %8 = llvm.mul %6, %7 : i32
    %9 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_alloc(%8, %4) : (i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %8, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @toy_random(%10, %8, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %14 = llvm.call @add_two_vectors(%9, %10, %11, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    %15 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %3, %15 : !llvm.array<2 x i8>, !llvm.ptr
    %16 = llvm.call @toy_print(%9, %8, %4, %15) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %17 = llvm.alloca %2 x !llvm.array<2 x i8> : (i32) -> !llvm.ptr
    llvm.store %1, %17 : !llvm.array<2 x i8>, !llvm.ptr
    %18 = llvm.call @toy_print(%10, %8, %4, %17) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    %19 = llvm.alloca %2 x !llvm.array<4 x i8> : (i32) -> !llvm.ptr
    llvm.store %0, %19 : !llvm.array<4 x i8>, !llvm.ptr
    %20 = llvm.call @toy_print(%11, %8, %4, %19) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i64
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @toy_print(!llvm.ptr, i32, i32, !llvm.ptr) -> i64
  llvm.func @toy_random(!llvm.ptr, i32, i32) -> !llvm.ptr
  llvm.func @toy_alloc(i32, i32) -> !llvm.ptr
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> !llvm.ptr {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(512 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(8 : index) : i64
    %4 = llvm.sext %arg3 : i32 to i64
    %5 = llvm.icmp "sle" %4, %0 : i64
    %6 = llvm.sub %0, %4 : i64
    %7 = llvm.sub %4, %2 : i64
    %8 = llvm.select %5, %6, %7 : i1, i64
    %9 = llvm.sdiv %8, %1 : i64
    %10 = llvm.sub %0, %9 : i64
    %11 = llvm.add %9, %2 : i64
    %12 = llvm.select %5, %10, %11 : i1, i64
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%12, %3, %2) threads in (%1, %2, %2) : i64 args(%4 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return %arg2 : !llvm.ptr
  }
  gpu.binary @add_two_vectors_kernel  [#gpu.object<#nvvm.target<chip = "sm_86", features = "+ptx84">, properties = {ISAToBinaryTimeInMs = 9 : i64, LLVMIRToISATimeInMs = 12 : i64}, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\00\00\00\00\09\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\048\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01 \00\03\19 \00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F5!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03\1B\80\00\04\1C\08\00\80\00\00\00p\01\00\00\04\05\0C\00\00\02\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\00\02\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00X\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00Y\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\00\05\02\00\00\00\03\02\01\00\00\E2\0F\04$x\06\05\04\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\06\00\\\00\00\FF\E0\F1\07\00\E4\0F\04\10z\02\06\00Z\00\00\FF\E0\F3\07\00\E4\0F\00\10z\05\00\00]\00\00\FF\E4\7F\00\00\E4\0F\04\10z\03\00\00[\00\00\FF\E4\FF\00\00\C8\0F\00\81y\05\04\04\00\00\00\00\19\1E\0C\00\A8\0E\00\81y\02\02\04\00\00\00\00\19\1E\0C\00\A2\0E\00\10z\06\06\00^\00\00\FF\E0\F1\07\00\C8\0F\00\10z\07\00\00_\00\00\FF\E4\7F\00\00\E2\0F\00!r\09\02\05\00\00\00\00\00\00\00\00\CAO\00\86y\00\06\09\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00|\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\94\04\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D8\04\00\00\00\00\00\00\80\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\D8\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\04\00\00\00\00\00\00(\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  llvm.func @main() {
    %0 = llvm.mlir.constant("out\00") : !llvm.array<4 x i8>
    %1 = llvm.mlir.constant("b\00") : !llvm.array<2 x i8>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant("a\00") : !llvm.array<2 x i8>
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.mlir.constant(8 : i32) : i32
    %7 = llvm.mul %5, %6 : i32
    %8 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %9 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %10 = llvm.call @toy_alloc(%7, %4) : (i32, i32) -> !llvm.ptr
    %11 = llvm.call @toy_random(%8, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %12 = llvm.call @toy_random(%9, %7, %4) : (!llvm.ptr, i32, i32) -> !llvm.ptr
    %13 = llvm.call @add_two_vectors(%8, %9, %10, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
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


