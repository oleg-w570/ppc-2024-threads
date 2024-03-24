// Copyright 2024 Zorin Oleg
#pragma once

#include <vector>

#include "core/task/include/task.hpp"
#include "crs_matrix.hpp"

class CRSMatMult : public ppc::core::Task {
  std::unique_ptr<CRSMatrix> A;
  std::unique_ptr<CRSMatrix> B;
  std::unique_ptr<CRSMatrix> C;

 public:
  explicit CRSMatMult(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};
