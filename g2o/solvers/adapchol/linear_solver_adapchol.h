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
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  LinearSolverAdapChol() : LinearSolverCCS<MatrixType>(), _init(false) {}

  virtual bool init() {
    if (_init) return true;
    auto* AdapCHOLContext = AdapChol::allocateContext();
    auto* CPUBackend = AdapChol::allocateCPUBackend();
    auto* FPGABackend = AdapChol::allocateFPGABackend(
        std::string("/lib/firmware/xilinx/adapchol/binary_container_1.bin"), 4);
    AdapChol::setBackend(AdapCHOLContext, CPUBackend, FPGABackend);
    std::cerr << "AdapCHOL Init Completed!" << std::endl;
    return true;
  }

  bool solve(const SparseBlockMatrix<MatrixType>& A, double* x, double* b) {
    std::cerr << "solve called!" << std::endl;
    assert(A.cols() == A.rows());
    int rowcol = A.cols();
    auto Cp = new int[rowcol + 1];
    auto Ci = new int[rowcol * rowcol];
    auto Cx = new double[rowcol * rowcol];
    auto matrix = new double*[rowcol];
    for (int i = 0; i < rowcol; i++) {
      matrix[i] = new double[rowcol];
    }

    A.fillCCS(Cp, Ci, Cx, false);
    for (int i = 0; i < Cp[rowcol]; i++) {
      int row = Ci[i],
          col = (int)(std::upper_bound(Cp, Cp + rowcol, i) - Cp) - 1;
      matrix[row][col] = Cx[i];
    }

    for (int i = 0; i < rowcol; i++) {
      for (int j = 0; j < rowcol; j++) {
        std::cerr << matrix[i][j] << "\t";
      }
      std::cerr << "\n";
    }

    std::cerr << "A.cols: " << A.cols() << ", A.rows: " << A.rows()
              << std::endl;
    std::cerr << "Col Ptr: ";
    for (int i = 0; i < rowcol + 1; i++) {
      std::cerr << Cp[i] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "Row Index: ";
    for (int i = 0; i < Cp[rowcol]; i++) {
      std::cerr << Ci[i] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "Values: ";
    for (int i = 0; i < Cp[rowcol]; i++) {
      std::cerr << Cx[i] << " ";
    }
    std::cerr << std::endl;

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
