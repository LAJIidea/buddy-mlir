// RUN: buddy-opt %s \
// RUN:     -generic-vectorization
// RUN: | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @vectorization(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>) {
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins (%arg0, %arg1: memref<16x16xf32>, memref<16x16xf32>) outs(%arg2: memref<16x16xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.mulf %in, %in_1 : f32
      linalg.yield %0 : f32
  }
  return
}
// CHECK-LABEL: func @vectorization(
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %cst = arith.constant 0.000000e+00 : f32
// CHECK:       {{.*}} = vector.transfer_read
// CHECK:       {{.*}} = vector.transfer_read
// CHECK:       {{.*}} = arith.mulf {{.*}}, {{.*}} : vector<16x16xf32>
// CHECK:       vector.transfer_write