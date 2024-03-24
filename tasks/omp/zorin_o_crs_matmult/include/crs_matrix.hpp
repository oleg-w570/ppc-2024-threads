// Copyright 2024 Zorin Oleg

#ifndef TASKS_SEQ_ZORIN_O_CRS_MATMULT_INCLUDE_CRS_MATRIX_HPP_
#define TASKS_SEQ_ZORIN_O_CRS_MATMULT_INCLUDE_CRS_MATRIX_HPP_

#include <random>
#include <vector>

#define EPS 1e-8

struct CRSMatrix {
  std::vector<int> row_ptr;
  std::vector<int> col_index;
  std::vector<double> values;
  int n_rows;
  int n_cols;

  CRSMatrix(int n_rows, int n_cols);
  CRSMatrix(const double* matrix, int n_rows, int n_cols);
};

std::vector<double> getRandomMatrix(const int& n_rows, const int& n_cols, const double& density = 0.3,
                                    const double& a = 1.0, const double& b = 100.0);

std::vector<double> getIdentityMatrix(const int& n);

#endif  // TASKS_SEQ_ZORIN_O_CRS_MATMULT_INCLUDE_CRS_MATRIX_HPP_
