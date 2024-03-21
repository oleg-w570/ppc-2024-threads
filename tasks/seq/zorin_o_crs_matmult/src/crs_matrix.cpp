// Copyright 2024 Zorin Oleg

#include "seq/zorin_o_crs_matmult/include/crs_matrix.hpp"

#include <cmath>

CRSMatrix::CRSMatrix(size_t n_rows, size_t n_cols) : n_rows(n_rows), n_cols(n_cols) {}

CRSMatrix::CRSMatrix(const double* matrix, size_t n_rows, size_t n_cols) : n_rows(n_rows), n_cols(n_cols) {
  for (size_t i = 0; i < n_rows; ++i) {
    row_ptr.push_back(values.size());
    for (size_t j = 0; j < n_cols; ++j) {
      const double& val = matrix[i * n_cols + j];
      if (std::abs(val) > 1e-8) {
        values.push_back(val);
        col_index.push_back(j);
      }
    }
  }
  row_ptr.push_back(values.size());
}

std::vector<double> getRandomMatrix(const size_t& n_rows, const size_t& n_cols, const double& density, const double& a,
                                    const double& b) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> value_dist(a, b);
  std::uniform_int_distribution<size_t> index_dist(0, n_rows * n_cols - 1);
  std::vector<double> matrix(n_rows * n_cols);
  auto nnz = static_cast<size_t>(static_cast<double>(n_rows * n_cols) * density);
  for (size_t i = 0; i < nnz; ++i) {
    matrix[index_dist(gen)] = value_dist(gen);
  }
  return matrix;
}

std::vector<double> getIdentityMatrix(const size_t& n) {
  std::vector<double> matrix(n * n);
  for (size_t i = 0; i < n; ++i) {
    matrix[i * n + i] = 1.0;
  }
  return matrix;
}