#ifndef G2O_LINEAR_SOLVER_ADAPCHOL_H
#define G2O_LINEAR_SOLVER_ADAPCHOL_H

#include <cassert>
#include <iostream>
#include <vector>

#include "adapchol/host/include/adapchol.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/linear_solver.h"
#include "g2o/core/marginal_covariance_cholesky.h"
#include "g2o/stuff/timeutil.h"

namespace g2o {

template <typename MatrixType>
class LinearSolverAdapChol : public LinearSolverCCS<MatrixType> {
 public:
  AdapChol::AdapCholContext* context = nullptr;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  LinearSolverAdapChol() : LinearSolverCCS<MatrixType>(), _init(false) {}

  virtual bool init() {
    if (_init) return true;
    context = AdapChol::allocateContext();
    auto* CPUBackend = AdapChol::allocateCPUBackend();
    auto* FPGABackend = AdapChol::allocateFPGABackend(
        std::string("/lib/firmware/xilinx/adapchol/binary_container_1.bin"), 4);
    AdapChol::setBackend(context, CPUBackend, FPGABackend);
    std::cerr << "AdapCHOL Init Completed!" << std::endl;
    return true;
  }

  bool solve(const SparseBlockMatrix<MatrixType>& A, double* x, double* b) {
    assert(A.cols() == A.rows());
    int rowcol = A.cols();
    int nz = A.nonZeros();

    auto* cs_A = AdapChol::allocateSparse(rowcol, nz);
    auto *Ap = AdapChol::getSparseP(cs_A), *Ai = AdapChol::getSparseI(cs_A);
    auto* Ax = AdapChol::getSparseX(cs_A);
    A.fillCCS(Ap, Ai,
              Ax, false);

//    for (int i = 0; i < rowcol; i++) {
//      std::cerr << Ap[i] << " ";
//    }
//    std::cerr << "\n";
//    for (int i = 0; i < rowcol; i++) {
//      std::cerr << Ai[i] << " ";
//    }
//    std::cerr << "\n";

//    std::cerr << "Setting A for problem " << rowcol << " " << nz << '\n';
        AdapChol::setA(context, cs_A);
        AdapChol::run(context);
        AdapChol::postSolve(context, b);
        memcpy(x, b, sizeof(double) * rowcol);
    return true;
  }
  bool solveBlocks_impl(
      const SparseBlockMatrix<MatrixType>& A,
      std::function<void(MarginalCovarianceCholesky&)> compute) {
    std::cerr << "no implemented: solveBlocks_impl called!" << std::endl;
    return true;
  }

 protected:
  bool _init;
};

}  // namespace g2o

#endif
