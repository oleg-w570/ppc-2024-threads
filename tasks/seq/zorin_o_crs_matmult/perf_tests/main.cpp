// Copyright 2024 Zorin Oleg
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/zorin_o_crs_matmult/include/crs_matmult_seq.hpp"

TEST(Zorin_O_CRS_MatMult_Seq, test_pipeline_run) {
  // Create data
  size_t p = 901;
  size_t q = 900;
  size_t r = 901;
  std::vector<double> lhs_in(p * q);
  for (size_t i = 0; i < p; ++i) {
    if (i % 2 == 0)
      for (size_t j = 0; j < q; ++j) {
        lhs_in[i * q + j] = 1.0;
      }
  }
  std::vector<double> rhs_in(q * r);
  for (size_t i = 0; i < q; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (j % 2 == 0) rhs_in[i * r + j] = 1.0;
    }
  }
  std::vector<double> out(p * r);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs_in.data()));
  taskDataSeq->inputs_count.emplace_back(p);
  taskDataSeq->inputs_count.emplace_back(q);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs_in.data()));
  taskDataSeq->inputs_count.emplace_back(q);
  taskDataSeq->inputs_count.emplace_back(r);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(p);
  taskDataSeq->outputs_count.emplace_back(r);

  // Create Task
  auto testTaskSeq = std::make_shared<CRSMatMult>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (i % 2 == 0 && j % 2 == 0)
        EXPECT_DOUBLE_EQ(out[i * r + j], q);
      else
        EXPECT_DOUBLE_EQ(out[i * r + j], 0.0);
    }
  }
}

TEST(Zorin_O_CRS_MatMult_Seq, test_task_run) {
  // Create data
  size_t p = 901;
  size_t q = 900;
  size_t r = 901;
  std::vector<double> lhs_in(p * q);
  for (size_t i = 0; i < p; ++i) {
    if (i % 2 == 0)
      for (size_t j = 0; j < q; ++j) {
        lhs_in[i * q + j] = 1.0;
      }
  }
  std::vector<double> rhs_in(q * r);
  for (size_t i = 0; i < q; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (j % 2 == 0) rhs_in[i * r + j] = 1.0;
    }
  }
  std::vector<double> out(p * r);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs_in.data()));
  taskDataSeq->inputs_count.emplace_back(p);
  taskDataSeq->inputs_count.emplace_back(q);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs_in.data()));
  taskDataSeq->inputs_count.emplace_back(q);
  taskDataSeq->inputs_count.emplace_back(r);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(p);
  taskDataSeq->outputs_count.emplace_back(r);

  // Create Task
  auto testTaskSeq = std::make_shared<CRSMatMult>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (i % 2 == 0 && j % 2 == 0)
        EXPECT_DOUBLE_EQ(out[i * r + j], q);
      else
        EXPECT_DOUBLE_EQ(out[i * r + j], 0.0);
    }
  }
}
