#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include <iostream>

#define DEBUG_TYPE "affine-loop-tiling"

using namespace mlir;

namespace {

SmallVector<Range, 4> getLoopRanges(OpBuilder &b, Location loc, linalg::GenericOp *op) {
  AffineMap map = op->getLoopsToShapesMap();
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  // auto viewSizes = 
  SmallVector<Range, 4> res(numDims);
  return res;
}

struct GenericTilePattern : public ConversionPattern {
  explicit GenericTilePattern(MLIRContext *context)
      : ConversionPattern(linalg::GenericOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    auto loc = op->getLoc();
    auto ctx = op->getContext();
    // linalgOp.getShapesToLoopsMap().dump();

    
    SmallVector<int, 4> cacheReuses;
    SmallVector<unsigned, 3> permutation{1, 2, 0};
    auto maps = linalgOp.getIndexingMapsArray();
    for (auto const map : maps) {
      if (map.getNumResults() != map.getNumInputs()) {
        // map.getPermutationMap(permutation, ctx).dump();
        for (size_t i = 0; i < map.getNumResults(); ++i) {
          // map.getNumDims
          if (map.getDimPosition(i) != i)
            cacheReuses.push_back(map.getDimPosition(i));
        }
      }
      // rewriter.replaceAllUsesWith
    }
    return failure();
  }
};

struct AffineLoopTilePass
    : public PassWrapper<AffineLoopTilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineLoopTilePass)
  StringRef getArgument() const final { return "polyhedral-loop-tile"; }
  StringRef getDescription() const final {
    return "Polyhedral loop tile optimization";
  }
  AffineLoopTilePass() = default;
  AffineLoopTilePass(const AffineLoopTilePass &) {}

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    arith::ArithDialect, vector::VectorDialect,
                    affine::AffineDialect, func::FuncDialect>();
  }
};

void AffineLoopTilePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         scf::SCFDialect, func::FuncDialect,
                         memref::MemRefDialect, vector::VectorDialect>();
  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
  
  RewritePatternSet patterns(context);
  patterns.add<GenericTilePattern>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
} // namespace

namespace mlir {
namespace buddy {
void registerAffineLoopTilingPass() { PassRegistration<AffineLoopTilePass>(); }
} // namespace buddy
} // namespace mlir
