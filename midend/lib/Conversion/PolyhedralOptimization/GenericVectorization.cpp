#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "generic-vectorization"

using namespace mlir;

namespace {

using SizesAndScalableFlags = std::pair<SmallVector<int64_t>, SmallVector<bool>>;

FailureOr<SmallVector<int64_t>>
inferVectorSizesFromIR(linalg::GenericOp linalgOp) {
  SmallVector<int64_t> vectorSizes;
  unsigned numDims = linalgOp.getNumLoops();

  for (int dim = 0; dim < numDims; ++dim) {
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    linalgOp.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty()) {
      return failure();
    }

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                          .getShape()[firstOperandDim];
    if (!ShapedType::isDynamic(dimSize)) {
      vectorSizes.push_back(dimSize);
      continue;
    }

    // Use ValueBounds analysis to infer `dim` size upper bound.
    FailureOr<int64_t> maybeDimBound;
    for (auto operandDimPair : operandDimPairs) {
      Value operand = operandDimPair.first;
      unsigned operandDim = operandDimPair.second;
      maybeDimBound = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, operand, operandDim, nullptr, true);
      if (succeeded(maybeDimBound))
        break;
    }

    if (failed(maybeDimBound)) 
      return failure();

    dimSize = maybeDimBound.value();
    vectorSizes.push_back(dimSize);
  }

  return vectorSizes;
}


class GenericVectorizationPass
    : public PassWrapper<GenericVectorizationPass, OperationPass<func::FuncOp>> {
public:
  GenericVectorizationPass() = default;
  GenericVectorizationPass(const GenericVectorizationPass &) {}
  GenericVectorizationPass(ArrayRef<int64_t> tileParam) { 
    tile = tileParam; 
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenericVectorizationPass)
  StringRef getArgument() const final { return "generic-vectorization"; }
  StringRef getDescription() const final {
    return "Vectorization for linalg.generic ops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect, 
                    vector::VectorDialect>();
  }

  void runOnOperation() override;

  FailureOr<SizesAndScalableFlags>
  getVectorSizes(linalg::GenericOp linalgOp) {
    if (!tile.empty()) {
      SmallVector<int64_t> vectorSizes(tile.begin(), tile.end());
      SmallVector<bool> scalableFlags(tile.size(), false);
      // Replace zeros in canonical vector shape to turn it into a valid shape.
      std::replace(vectorSizes.begin(), vectorSizes.end(), 0, 1);
      return std::make_pair(vectorSizes, scalableFlags);
    }

    // Try to infer the vector sizes from the IR.
    auto vectorSizes = inferVectorSizesFromIR(linalgOp);
    if (succeeded(vectorSizes))
      return std::make_pair(*vectorSizes, SmallVector<bool>(vectorSizes->size(), false));
    return failure();
  }

  ListOption<int64_t> tile{*this, "tile-sizes", llvm::cl::desc("Tile sizes."),
                           llvm::cl::ZeroOrMore};
};

void GenericVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  IRRewriter rewriter(context);
  SmallVector<Operation *> candiates;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::GenericOp>(op)) {
      candiates.push_back(op);
    }
  });
  for (auto op : candiates) {
    SmallVector<int64_t> vectorSizes;
    SmallVector<bool> scalableVecDims;
    if (auto linalgOp = dyn_cast<linalg::GenericOp>(op)) {
      auto vectorSizesAndScalableDims = getVectorSizes(linalgOp);
      if (succeeded(vectorSizesAndScalableDims)) {
        auto [sizes, scalableVecDims] = *vectorSizesAndScalableDims;
        vectorSizes.append(sizes.begin(), sizes.end());
        scalableVecDims.append(scalableVecDims.begin(), scalableVecDims.end());
      }
      // Pad scalable dims with `false` to match the vector sizes.
      scalableVecDims.resize(vectorSizes.size());
      (void)linalg::vectorize(rewriter, linalgOp, vectorSizes, false);
    }
  }

  // Canonicalize mask replated ops before lower them.
  RewritePatternSet maskCanonPatterns(funcOp.getContext());
  vector::CreateMaskOp::getCanonicalizationPatterns(maskCanonPatterns, context);
  vector::ConstantMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                           context);
  vector::MaskOp::getCanonicalizationPatterns(maskCanonPatterns, context);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(maskCanonPatterns))))
    return signalPassFailure();

  RewritePatternSet vectorizationPatterns(funcOp.getContext());
  vector::populateVectorTransferPermutationMapLoweringPatterns(
      vectorizationPatterns);
  vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
      vectorizationPatterns);
  vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                      context);
  vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                       context);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));
}

}

namespace mlir {
namespace buddy {
void registerGenericVectorizationPass() {
  PassRegistration<GenericVectorizationPass>();
}
} // namespace buddy
} // namespace mlir