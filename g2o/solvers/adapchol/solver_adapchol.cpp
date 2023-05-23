#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_dogleg.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/stuff/macros.h"
#include "linear_solver_adapchol.h"

using namespace std;

namespace g2o {

namespace {
template <int p, int l, bool blockorder>
std::unique_ptr<BlockSolverBase> AllocateSolver() {
  std::cerr << "# Using AdapCholSparseCholesky poseDim " << p << " landMarkDim "
            << l << " blockordering " << blockorder << std::endl;
  auto linearSolver = std::make_unique<
      LinearSolverAdapChol<typename BlockSolverPL<p, l>::PoseMatrixType>>();
  linearSolver->setBlockOrdering(blockorder);
  return std::make_unique<BlockSolverPL<p, l>>(std::move(linearSolver));
}
}  // namespace

/**
 * helper function for allocating
 */
static OptimizationAlgorithm* createSolver(const std::string& fullSolverName) {
  static const std::map<std::string,
                        std::function<std::unique_ptr<BlockSolverBase>()>>
      solver_factories{
          {"var", &AllocateSolver<-1, -1, true>},
      };

  string solverName = fullSolverName.substr(3);
  auto solverf = solver_factories.find(solverName);
  if (solverf == solver_factories.end()) return nullptr;

  string methodName = fullSolverName.substr(0, 2);

  if (methodName == "gn") {
    return new OptimizationAlgorithmGaussNewton(solverf->second());
  } else if (methodName == "lm") {
    return new OptimizationAlgorithmLevenberg(solverf->second());
  } else if (methodName == "dl") {
    return new OptimizationAlgorithmDogleg(solverf->second());
  }

  return nullptr;
}

class AdapCholSolverCreator : public AbstractOptimizationAlgorithmCreator {
 public:
  explicit AdapCholSolverCreator(const OptimizationAlgorithmProperty& p)
      : AbstractOptimizationAlgorithmCreator(p) {}
  virtual OptimizationAlgorithm* construct() {
    return createSolver(property().name);
  }
};

// clang-format off
  G2O_REGISTER_OPTIMIZATION_LIBRARY(AdapChol);

//  G2O_REGISTER_OPTIMIZATION_ALGORITHM(gn_var, new AdapCholSolverCreator(OptimizationAlgorithmProperty("gn_var", "Gauss-Newton: Cholesky solver using AdapChol's Sparse Cholesky methods (variable blocksize)", "AdapChol", false, AdapChol::Dynamic, AdapChol::Dynamic)));
//  G2O_REGISTER_OPTIMIZATION_ALGORITHM(gn_fix3_2, new AdapCholSolverCreator(OptimizationAlgorithmProperty("gn_fix3_2", "Gauss-Newton: Cholesky solver using  AdapChol's Sparse Cholesky (fixed blocksize)", "AdapChol", true, 3, 2)));
//  G2O_REGISTER_OPTIMIZATION_ALGORITHM(gn_fix6_3, new AdapCholSolverCreator(OptimizationAlgorithmProperty("gn_fix6_3", "Gauss-Newton: Cholesky solver using  AdapChol's Sparse Cholesky (fixed blocksize)", "AdapChol", true, 6, 3)));
//  G2O_REGISTER_OPTIMIZATION_ALGORITHM(gn_fix7_3, new AdapCholSolverCreator(OptimizationAlgorithmProperty("gn_fix7_3", "Gauss-Newton: Cholesky solver using  AdapChol's Sparse Cholesky (fixed blocksize)", "AdapChol", true, 7, 3)));
//
//  G2O_REGISTER_OPTIMIZATION_ALGORITHM(lm_var, new AdapCholSolverCreator(OptimizationAlgorithmProperty("lm_var", "Levenberg: Cholesky solver using AdapChol's Sparse Cholesky methods (variable blocksize)", "AdapChol", false, AdapChol::Dynamic, AdapChol::Dynamic)));
//  G2O_REGISTER_OPTIMIZATION_ALGORITHM(lm_fix3_2, new AdapCholSolverCreator(OptimizationAlgorithmProperty("lm_fix3_2", "Levenberg: Cholesky solver using  AdapChol's Sparse Cholesky (fixed blocksize)", "AdapChol", true, 3, 2)));
//  G2O_REGISTER_OPTIMIZATION_ALGORITHM(lm_fix6_3, new AdapCholSolverCreator(OptimizationAlgorithmProperty("lm_fix6_3", "Levenberg: Cholesky solver using  AdapChol's Sparse Cholesky (fixed blocksize)", "AdapChol", true, 6, 3)));
//  G2O_REGISTER_OPTIMIZATION_ALGORITHM(lm_fix7_3, new AdapCholSolverCreator(OptimizationAlgorithmProperty("lm_fix7_3", "Levenberg: Cholesky solver using  AdapChol's Sparse Cholesky (fixed blocksize)", "AdapChol", true, 7, 3)));
//
//  G2O_REGISTER_OPTIMIZATION_ALGORITHM(dl_var, new AdapCholSolverCreator(OptimizationAlgorithmProperty("dl_var", "Dogleg: Cholesky solver using AdapChol's Sparse Cholesky methods (variable blocksize)", "AdapChol", false, AdapChol::Dynamic, AdapChol::Dynamic)));
// clang-format on
}  // namespace g2o
