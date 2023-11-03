// RUN: buddy-opt %s \
// RUN:     -polyhedral-tiling="tile-sizes=256,256"
// RUN: | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (256, 4096-d0)>
func.func @tiling(%arg0: memref<4096x4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) {
  linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins (%arg0, %arg1: memref<4096x4096xf32>, memref<4096xf32>) outs(%arg2: memref<4096xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %2 = arith.addf %in, %in_1 : f32
    linalg.yield %2 : f32 
  }
  return
}
// CHECK-LABEL: func.func @tiling(
// CHECK-DAG:   %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C4096:.*]] = arith.constant 4096 : index
// CHECK:       scf.for {{.*}} step %[[C256]]
// CHECK:         scf.for {{.*}} step %[[C256]]
// CHECK:           %[[subview:.*]] = memref.subview
// CHECK:           %[[subview:.*]] = memref.subview
// CHECK:           linalg.generic


// #map = affine_map<(d0, d1) -> (d0, d1)>
// #map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @manul(%arg0: memref<4096x4096xf32>, %arg1: memref<4096xf32>) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    scf.for %arg2 = %c0 to %c4096 step %c32 {
      scf.for %arg3 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %arg0[%arg2, %arg3] [32, 32] [1, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
        %subview_0 = memref.subview %arg1[%arg2] [32] [1] : memref<4096xf32> to memref<32xf32, strided<[1], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<32x32xf32, strided<[4096, 1], offset: ?>>) outs(%subview_0 : memref<32xf32, strided<[1], offset: ?>>) {
        ^bb0(%in: f32, %out: f32):
          %0 = arith.addf %in, %out : f32
          linalg.yield %0 : f32
        }
      }
    }
    return
  }
}