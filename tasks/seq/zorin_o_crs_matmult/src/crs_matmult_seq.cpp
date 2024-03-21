// Copyright 2024 Zorin Oleg
#include "seq/zorin_o_crs_matmult/include/crs_matmult_seq.hpp"

bool CRSMatMult::validation() {
  internal_order_test();

  return taskData->inputs_count[1] == taskData->inputs_count[2] &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[3];
}

bool CRSMatMult::pre_processing() {
  internal_order_test();

  A = std::make_unique<CRSMatrix>(reinterpret_cast<double*>(taskData->inputs[0]), taskData->inputs_count[0],
                                  taskData->inputs_count[1]);
  B = std::make_unique<CRSMatrix>(reinterpret_cast<double*>(taskData->inputs[1]), taskData->inputs_count[2],
                                  taskData->inputs_count[3]);
  C = std::make_unique<CRSMatrix>(taskData->outputs_count[0], taskData->outputs_count[1]);

  return true;
}

bool CRSMatMult::run() {
  internal_order_test();

  std::vector<double> temp_row(C->n_cols);

  for (size_t row_i = 0; row_i < A->n_rows; ++row_i) {
    for (size_t i = A->row_ptr[row_i]; i < A->row_ptr[row_i + 1]; ++i) {
      const size_t& col_i = A->col_index[i];
      const double& val = A->values[i];
      for (size_t j = B->row_ptr[col_i]; j < B->row_ptr[col_i + 1]; ++j) {
        temp_row[B->col_index[j]] += val * B->values[j];
      }
    }

    C->row_ptr.push_back(C->values.size());
    for (size_t i = 0; i < temp_row.size(); ++i) {
      double& val = temp_row[i];
      if (std::abs(val) > 1e-8) {
        C->values.push_back(val);
        C->col_index.push_back(i);
        val = 0.0;
      }
    }
  }
  C->row_ptr.push_back(C->values.size());

  return true;
}

bool CRSMatMult::post_processing() {
  internal_order_test();

  auto* out_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  for (size_t i = 0; i < C->n_rows; ++i) {
    for (size_t j = C->row_ptr[i]; j < C->row_ptr[i + 1]; ++j) {
      out_ptr[i * C->n_cols + C->col_index[j]] = C->values[j];
    }
  }

  return true;
}
