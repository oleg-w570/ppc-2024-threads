// Copyright 2024 Zorin Oleg

#ifndef TASKS_SEQ_ZORIN_O_CRS_MATMULT_INCLUDE_CRS_MATRIX_HPP_
#define TASKS_SEQ_ZORIN_O_CRS_MATMULT_INCLUDE_CRS_MATRIX_HPP_

#include <random>
#include <vector>

struct CRSMatrix {
  std::vector<size_t> row_ptr;
  std::vector<size_t> col_index;
  std::vector<double> values;
  size_t n_rows;
  size_t n_cols;

  CRSMatrix(size_t n_rows, size_t n_cols);
  CRSMatrix(const double* matrix, size_t n_rows, size_t n_cols);
};

std::vector<double> getRandomMatrix(const size_t& n_rows, const size_t& n_cols, const double& density = 0.3,
                                    const double& a = 1.0, const double& b = 100.0);

std::vector<double> getIdentityMatrix(const size_t& n);

#endif  // TASKS_SEQ_ZORIN_O_CRS_MATMULT_INCLUDE_CRS_MATRIX_HPP_
