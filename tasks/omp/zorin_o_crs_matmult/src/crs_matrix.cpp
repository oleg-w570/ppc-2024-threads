// Copyright 2024 Zorin Oleg

#include "omp/zorin_o_crs_matmult/include/crs_matrix.hpp"

#include <cmath>
#include <iostream>

CRSMatrix::CRSMatrix(int n_rows, int n_cols) : n_rows(n_rows), n_cols(n_cols) { row_ptr.reserve(n_rows + 1); }

CRSMatrix::CRSMatrix(const double* matrix, int n_rows, int n_cols) : CRSMatrix(n_rows, n_cols) {
  for (int i = 0; i < n_rows; ++i) {
    row_ptr.emplace_back(static_cast<int>(values.size()));
    for (int j = 0; j < n_cols; ++j) {
      const double& val = matrix[i * n_cols + j];
      if (std::abs(val) > EPS) {
        values.emplace_back(val);
        col_index.emplace_back(j);
      }
    }
  }
  row_ptr.emplace_back(static_cast<int>(values.size()));
}

std::vector<double> getRandomMatrix(const int& n_rows, const int& n_cols, const double& density, const double& a,
                                    const double& b) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> value_dist(a, b);
  std::uniform_int_distribution<int> index_dist(0, n_rows * n_cols - 1);
  std::vector<double> matrix(n_rows * n_cols);
  auto nnz = static_cast<int>(static_cast<double>(n_rows * n_cols) * density);
  for (int i = 0; i < nnz; ++i) {
    matrix[index_dist(gen)] = value_dist(gen);
  }
  return matrix;
}

std::vector<double> getIdentityMatrix(const int& n) {
  std::vector<double> matrix(n * n);
  for (int i = 0; i < n; ++i) {
    matrix[i * n + i] = 1.0;
  }
  return matrix;
}