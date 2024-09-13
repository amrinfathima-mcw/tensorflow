// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// NOLINTNEXTLINE
#include <filesystem>
#include <fstream>
#include <string_view>

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/api/graph_tools.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/api/lite_rt_model_api.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/model/lite_rt_model_init.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/test_data/test_data_util.h"

namespace {

class TestWithPath : public ::testing::TestWithParam<std::string_view> {};

TEST(LrtModelTest, TestLoadTestDataBadFilepath) {
  LrtModel model = nullptr;
  ASSERT_STATUS_HAS_CODE(LoadModelFromFile("bad_path", &model),
                         kLrtStatusBadFileOp);
}

TEST(LrtModelTest, TestLoadTestDataBadFileData) {
  // NOLINTBEGIN
#ifndef NDEBUG
  // In debug mode, flatbuffers will `assert` while verifying. This will
  // cause this test to crash (as expected).
  GTEST_SKIP();
#endif
  std::filesystem::path test_file_path(::testing::TempDir());
  test_file_path.append("bad_file.txt");

  std::ofstream bad_file;
  bad_file.open(test_file_path.c_str());
  bad_file << "not_tflite";
  bad_file.close();

  LrtModel model = nullptr;
  ASSERT_STATUS_HAS_CODE(LoadModelFromFile(test_file_path.c_str(), &model),
                         kLrtStatusFlatbufferFailedVerify);
  // NOLINTEND
}

TEST_P(TestWithPath, TestConstructDestroy) {
  UniqueLrtModel model = LoadTestFileModel(GetParam());
}

INSTANTIATE_TEST_SUITE_P(InstTestWithPath, TestWithPath,
                         ::testing::Values("add_simple.tflite",
                                           "add_cst.tflite",
                                           "simple_multi_op.tflite"));

TEST(LrtModelTest, TestBuildModelAddSimple) {
  auto model = LoadTestFileModel("add_simple.tflite");

  // func(arg0)
  //  output = tfl.add(arg0, arg0)
  //  return(output)
  //

  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_inputs,
                          graph_tools::GetSubgraphInputs(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_outputs,
                          graph_tools::GetSubgraphOutputs(subgraph));

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  ASSERT_TRUE(graph_tools::ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 1);
  auto op = ops[0];

  graph_tools::RankedTypeInfo float_2by2_type(kLrtElementTypeFloat32, {2, 2});
  ASSERT_TRUE(graph_tools::MatchOpType(op, {float_2by2_type, float_2by2_type},
                                       {float_2by2_type}, kLrtOpCodeTflAdd));

  ASSERT_RESULT_OK_ASSIGN(auto op_inputs, graph_tools::GetOpIns(op));
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs[0], subgraph_inputs[0]);
  ASSERT_EQ(op_inputs[0], op_inputs[1]);

  ASSERT_RESULT_OK_ASSIGN(auto op_out, graph_tools::GetOnlyOpOut(op));
  ASSERT_EQ(op_out, subgraph_outputs[0]);

  ASSERT_TRUE(graph_tools::MatchNoBuffer(subgraph_outputs[0]));
  ASSERT_TRUE(graph_tools::MatchNoBuffer(subgraph_inputs[0]));
}

TEST(LrtModelTest, TestBuildModelAddCst) {
  auto model = LoadTestFileModel("add_cst.tflite");

  // func(arg0)
  //  cst = ConstantTensor([1, 2, 3, 4])
  //  output = tfl.add(arg0, cst)
  //  return(output)
  //

  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_inputs,
                          graph_tools::GetSubgraphInputs(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_outputs,
                          graph_tools::GetSubgraphOutputs(subgraph));

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  ASSERT_TRUE(graph_tools::ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 1);
  auto op = ops[0];

  graph_tools::RankedTypeInfo float_2by2_type(kLrtElementTypeFloat32, {4});
  ASSERT_TRUE(graph_tools::MatchOpType(op, {float_2by2_type, float_2by2_type},
                                       {float_2by2_type}, kLrtOpCodeTflAdd));

  ASSERT_RESULT_OK_ASSIGN(auto op_inputs, graph_tools::GetOpIns(op));
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs[0], subgraph_inputs[0]);
  ASSERT_TRUE(graph_tools::MatchBuffer(
      op_inputs[1], llvm::ArrayRef<float>{1.0, 2.0, 3.0, 4.0}));

  ASSERT_RESULT_OK_ASSIGN(auto op_out, graph_tools::GetOnlyOpOut(op));
  ASSERT_EQ(op_out, subgraph_outputs[0]);

  ASSERT_TRUE(graph_tools::MatchNoBuffer(subgraph_outputs[0]));
  ASSERT_TRUE(graph_tools::MatchNoBuffer(subgraph_inputs[0]));
}

TEST(LrtModelTest, TestSimpleMultiAdd) {
  auto model = LoadTestFileModel("simple_multi_op.tflite");

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  ASSERT_RESULT_OK_ASSIGN(LrtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_inputs,
                          graph_tools::GetSubgraphInputs(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_outputs,
                          graph_tools::GetSubgraphOutputs(subgraph));

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  ASSERT_TRUE(graph_tools::ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 4);
  for (auto op : ops) {
    ASSERT_RESULT_OK_ASSIGN(auto inputs, graph_tools::GetOpIns(op));
    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(inputs[0], inputs[1]);
  }

  graph_tools::RankedTypeInfo float_2by2_type(kLrtElementTypeFloat32, {2, 2});

  ASSERT_TRUE(graph_tools::MatchOpType(ops[2],
                                       {float_2by2_type, float_2by2_type},
                                       {float_2by2_type}, kLrtOpCodeTflMul));
}

}  // namespace
