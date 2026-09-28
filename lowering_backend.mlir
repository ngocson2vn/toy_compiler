// -----// IR Dump After TileLoopsPass (tile-loops) ('func.func' operation: @add_two_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i64) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i64 to index
    scf.parallel (%arg4) = (%c0) to (%0) step (%c128) {
      %1 = affine.min #map(%c128, %0, %arg4)
      scf.parallel (%arg5) = (%c0) to (%1) step (%c1) {
        %2 = arith.addi %arg5, %arg4 : index
        %3 = memref.load %arg0[%2] : memref<?xf32>
        %4 = memref.load %arg1[%2] : memref<?xf32>
        %5 = arith.addf %3, %4 : f32
        memref.store %5, %arg2[%2] : memref<?xf32>
        scf.reduce 
      }
      scf.reduce 
    }
    return
  }
}


// -----// IR Dump After GpuMapParallelLoopsPass (gpu-map-parallel-loops) ('func.func' operation: @add_two_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i64) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i64 to index
    scf.parallel (%arg4) = (%c0) to (%0) step (%c128) {
      %1 = affine.min #map(%c128, %0, %arg4)
      scf.parallel (%arg5) = (%c0) to (%1) step (%c1) {
        %2 = arith.addi %arg5, %arg4 : index
        %3 = memref.load %arg0[%2] : memref<?xf32>
        %4 = memref.load %arg1[%2] : memref<?xf32>
        %5 = arith.addf %3, %4 : f32
        memref.store %5, %arg2[%2] : memref<?xf32>
        scf.reduce 
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.reduce 
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return
  }
}


// -----// IR Dump After ConvertParallelLoopToGpuPass (convert-parallel-loops-to-gpu) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i64) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i64 to index
    %c1_0 = arith.constant 1 : index
    %1 = affine.apply #map(%0)[%c0, %c128]
    %c128_1 = arith.constant 128 : index
    %2 = affine.apply #map(%c128_1)[%c0, %c1]
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %1, %arg11 = %c1_0, %arg12 = %c1_0) threads(%arg7, %arg8, %arg9) in (%arg13 = %2, %arg14 = %c1_0, %arg15 = %c1_0) {
      %3 = affine.apply #map1(%arg4)[%c128, %c0]
      %4 = affine.min #map2(%c128, %0, %3)
      %5 = affine.apply #map1(%arg7)[%c1, %c0]
      %6 = arith.cmpi slt, %5, %4 : index
      scf.if %6 {
        %7 = arith.addi %5, %3 : index
        %8 = memref.load %arg0[%7] : memref<?xf32>
        %9 = memref.load %arg1[%7] : memref<?xf32>
        %10 = arith.addf %8, %9 : f32
        memref.store %10, %arg2[%7] : memref<?xf32>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    return
  }
}


// -----// IR Dump After LowerMemRefToLLVMPass (lower-memref-to-llvm) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg3 : i64 to index
    %c1_0 = arith.constant 1 : index
    %1 = affine.apply #map(%0)[%c0, %c128]
    %c128_1 = arith.constant 128 : index
    %2 = affine.apply #map(%c128_1)[%c0, %c1]
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %1, %arg11 = %c1_0, %arg12 = %c1_0) threads(%arg7, %arg8, %arg9) in (%arg13 = %2, %arg14 = %c1_0, %arg15 = %c1_0) {
      %3 = affine.apply #map1(%arg4)[%c128, %c0]
      %4 = affine.min #map2(%c128, %0, %3)
      %5 = affine.apply #map1(%arg7)[%c1, %c0]
      %6 = arith.cmpi slt, %5, %4 : index
      scf.if %6 {
        %7 = arith.addi %5, %3 : index
        %8 = arith.index_cast %7 : index to i64
        %9 = llvm.getelementptr %arg0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %10 = llvm.load %9 : !llvm.ptr -> f32
        %11 = llvm.getelementptr %arg1[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %12 = llvm.load %11 : !llvm.ptr -> f32
        %13 = arith.addf %10, %12 : f32
        %14 = llvm.getelementptr %arg2[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %13, %14 : f32, !llvm.ptr
      }
      gpu.terminator
    } {SCFToGPU_visited}
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * -128 + s1, 128)>
module {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i64 to index
    %1 = affine.apply #map()[%0]
    gpu.launch blocks(%arg4, %arg5, %arg6) in (%arg10 = %1, %arg11 = %c1, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c128, %arg14 = %c1, %arg15 = %c1) {
      %2 = affine.apply #map1()[%arg4]
      %3 = affine.min #map2()[%arg4, %0]
      %4 = arith.cmpi slt, %arg7, %3 : index
      scf.if %4 {
        %5 = arith.addi %arg7, %2 : index
        %6 = arith.index_cast %5 : index to i64
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
    llvm.return
  }
}


// -----// IR Dump After GpuKernelOutliningPass (gpu-kernel-outlining) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * -128 + s1, 128)>
module attributes {gpu.container_module} {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i64 to index
    %1 = affine.apply #map()[%0]
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%1, %c1, %c1) threads in (%c128, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
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
        %4 = arith.index_cast %3 : index to i64
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
}


// -----// IR Dump After LowerAffinePass (lower-affine) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i64 to index
    %c128_0 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1_1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c128_0 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1_1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c1, %c1) threads in (%c128, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
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
      %c128 = arith.constant 128 : index
      %0 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %c-128 = arith.constant -128 : index
      %1 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %2 = arith.addi %1, %arg0 : index
      %c128_0 = arith.constant 128 : index
      %3 = arith.minsi %2, %c128_0 : index
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      scf.if %4 {
        %5 = arith.addi %thread_id_x, %0 : index
        %6 = arith.index_cast %5 : index to i64
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
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i64 to index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c128 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c1, %c1) threads in (%c128, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c-128 = arith.constant -128 : index
      %c128 = arith.constant 128 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %1 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %2 = arith.addi %1, %arg0 : index
      %3 = arith.minsi %2, %c128 : index
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      scf.if %4 {
        %5 = arith.addi %thread_id_x, %0 : index
        %6 = arith.index_cast %5 : index to i64
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
}


// -----// IR Dump After SCFToControlFlowPass (convert-scf-to-cf) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i64 to index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c128 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c1, %c1) threads in (%c128, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: index, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c-128 = arith.constant -128 : index
      %c128 = arith.constant 128 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %1 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %2 = arith.addi %1, %arg0 : index
      %3 = arith.minsi %2, %c128 : index
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      cf.cond_br %4, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %5 = arith.addi %thread_id_x, %0 : index
      %6 = arith.index_cast %5 : index to i64
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
}


// -----// IR Dump After ConvertGpuOpsToNVVMOps (convert-gpu-to-nvvm) ('gpu.module' operation: @add_two_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i64 to index
    %1 = arith.cmpi sle, %0, %c0 : index
    %2 = arith.subi %c0, %0 : index
    %3 = arith.subi %0, %c1 : index
    %4 = arith.select %1, %2, %3 : index
    %5 = arith.divsi %4, %c128 : index
    %6 = arith.subi %c0, %5 : index
    %7 = arith.addi %5, %c1 : index
    %8 = arith.select %1, %6, %7 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%8, %c1, %c1) threads in (%c128, %c1, %c1)  args(%0 : index, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
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
}


// -----// IR Dump After GpuToLLVMConversionPass (gpu-to-llvm) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.icmp "sle" %arg3, %0 : i64
    %4 = llvm.sub %0, %arg3 : i64
    %5 = llvm.sub %arg3, %2 : i64
    %6 = llvm.select %3, %4, %5 : i1, i64
    %7 = llvm.sdiv %6, %1 : i64
    %8 = llvm.sub %0, %7 : i64
    %9 = llvm.add %7, %2 : i64
    %10 = llvm.select %3, %8, %9 : i1, i64
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%10, %2, %2) threads in (%1, %2, %2) : i64 args(%arg3 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
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
}


// -----// IR Dump After GpuNVVMAttachTarget (nvvm-attach-target) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.icmp "sle" %arg3, %0 : i64
    %4 = llvm.sub %0, %arg3 : i64
    %5 = llvm.sub %arg3, %2 : i64
    %6 = llvm.select %3, %4, %5 : i1, i64
    %7 = llvm.sdiv %6, %1 : i64
    %8 = llvm.sub %0, %7 : i64
    %9 = llvm.add %7, %2 : i64
    %10 = llvm.select %3, %8, %9 : i1, i64
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%10, %2, %2) threads in (%1, %2, %2) : i64 args(%arg3 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return
  }
  gpu.module @add_two_vectors_kernel [#nvvm.target<chip = "sm_86", features = "+ptx84">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(-128 : index) : i64
      %1 = llvm.mlir.constant(128 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
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
}


// -----// IR Dump After GpuModuleToBinaryPass (gpu-module-to-binary) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.icmp "sle" %arg3, %0 : i64
    %4 = llvm.sub %0, %arg3 : i64
    %5 = llvm.sub %arg3, %2 : i64
    %6 = llvm.select %3, %4, %5 : i1, i64
    %7 = llvm.sdiv %6, %1 : i64
    %8 = llvm.sub %0, %7 : i64
    %9 = llvm.add %7, %2 : i64
    %10 = llvm.select %3, %8, %9 : i1, i64
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%10, %2, %2) threads in (%1, %2, %2) : i64 args(%arg3 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr)
    llvm.return
  }
  gpu.binary @add_two_vectors_kernel  [#gpu.object<#nvvm.target<chip = "sm_86", features = "+ptx84">, properties = {ISAToBinaryTimeInMs = 9 : i64, LLVMIRToISATimeInMs = 13 : i64}, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\00\00\00\00\09\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\048\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01 \00\03\19 \00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F5!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03\1B\FF\00\04\1C\08\00\80\00\00\00p\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\80\00\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00X\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00Y\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\00\05\02\00\00\00\03\02\01\00\00\E2\0F\04$x\06\05\04\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\06\00\\\00\00\FF\E0\F1\07\00\E4\0F\04\10z\02\06\00Z\00\00\FF\E0\F3\07\00\E4\0F\00\10z\05\00\00]\00\00\FF\E4\7F\00\00\E4\0F\04\10z\03\00\00[\00\00\FF\E4\FF\00\00\C8\0F\00\81y\05\04\04\00\00\00\00\19\1E\0C\00\A8\0E\00\81y\02\02\04\00\00\00\00\19\1E\0C\00\A2\0E\00\10z\06\06\00^\00\00\FF\E0\F1\07\00\C8\0F\00\10z\07\00\00_\00\00\FF\E4\7F\00\00\E2\0F\00!r\09\02\05\00\00\00\00\00\00\00\00\CAO\00\86y\00\06\09\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00|\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\94\04\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D8\04\00\00\00\00\00\00\80\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\D8\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\04\00\00\00\00\00\00(\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
}


// -----// IR Dump After InjectRuntimeCtxPass (inject-runtime-ctx) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64) attributes {RuntimeCtx = true} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.getelementptr %arg0[%0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.ptr
    %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(128 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.icmp "sle" %arg4, %3 : i64
    %7 = llvm.sub %3, %arg4 : i64
    %8 = llvm.sub %arg4, %5 : i64
    %9 = llvm.select %6, %7, %8 : i1, i64
    %10 = llvm.sdiv %9, %4 : i64
    %11 = llvm.sub %3, %10 : i64
    %12 = llvm.add %10, %5 : i64
    %13 = llvm.select %6, %11, %12 : i1, i64
    gpu.launch_func <%2 : !llvm.ptr> @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%13, %5, %5) threads in (%4, %5, %5) : i64 args(%arg4 : i64, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr, %arg3 : !llvm.ptr) {RuntimeCtx = true}
    llvm.return
  }
  gpu.binary @add_two_vectors_kernel  [#gpu.object<#nvvm.target<chip = "sm_86", features = "+ptx84">, properties = {ISAToBinaryTimeInMs = 9 : i64, LLVMIRToISATimeInMs = 13 : i64}, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0C\00\00\00\00\00\00\00\09\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\048\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01 \00\03\19 \00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F5!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F5!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F5!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\03\1B\FF\00\04\1C\08\00\80\00\00\00p\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\80\00\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00X\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00Y\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\00\05\02\00\00\00\03\02\01\00\00\E2\0F\04$x\06\05\04\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\06\00\\\00\00\FF\E0\F1\07\00\E4\0F\04\10z\02\06\00Z\00\00\FF\E0\F3\07\00\E4\0F\00\10z\05\00\00]\00\00\FF\E4\7F\00\00\E4\0F\04\10z\03\00\00[\00\00\FF\E4\FF\00\00\C8\0F\00\81y\05\04\04\00\00\00\00\19\1E\0C\00\A8\0E\00\81y\02\02\04\00\00\00\00\19\1E\0C\00\A2\0E\00\10z\06\06\00^\00\00\FF\E0\F1\07\00\C8\0F\00\10z\07\00\00_\00\00\FF\E4\7F\00\00\E2\0F\00!r\09\02\05\00\00\00\00\00\00\00\00\CAO\00\86y\00\06\09\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00|\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\94\04\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\04\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D8\04\00\00\00\00\00\00\80\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\D8\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\04\00\00\00\00\00\00(\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
}


