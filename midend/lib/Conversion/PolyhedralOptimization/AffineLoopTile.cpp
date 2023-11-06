#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "affine-loop-tiling"

using namespace mlir;

namespace {
struct GenericTilePattern : public ConversionPattern {
  explicit GenericTilePattern(MLIRContext *context)
      : ConversionPattern(linalg::Conv2DOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::GenericOp>(op);
    auto loc = op->getLoc();
    auto ctx = op->getContext();

    auto maps = linalgOp.getIndexingMapsArray();
    
    return failure();
  }
};

struct AffineLoopTilePass
    : public PassWrapper<AffineLoopTilePass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

void AffineLoopTilePass::runOnOperation() {}
} // namespace

namespace mlir {
namespace buddy {
void registerAffineLoopTilingPass() { PassRegistration<AffineLoopTilePass>(); }
} // namespace buddy
} // namespace mlir
