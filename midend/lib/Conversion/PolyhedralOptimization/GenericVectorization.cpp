#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


#define DEBUG_TYPE "polyhedral-generic-vectorization"

using namespace mlir;

namespace {
  struct GenericVectorizationPass : public PassWrapper<GenericVectorizationPass, OperationPass<ModuleOp>> {
    
    GenericVectorizationPass() = default;
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenericVectorizationPass)
    StringRef getArgument() const final { return "polyhedral-generic-vectorization"; }
    StringRef getDescription() const final {
      return "Vectorization for polyhedral optimization";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                      vector::VectorDialect>();
    }

    void runOnOperation() override;
  };
}