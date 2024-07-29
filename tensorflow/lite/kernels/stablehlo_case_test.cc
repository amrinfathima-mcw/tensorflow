/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <vector>

//#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"

namespace tflite{
namespace {

using ::testing::ElementsAreArray;

class StablehloCaseOpModel : public SingleOpModel {
 public:
  StablehloCaseOpModel(const TensorData& input, const TensorData& output_shape, const TfLiteStablehloCaseParams& params) {
    input_ = AddInput(input);
    output_shape_ = AddInput(output_shape);
    output_ = AddOutput(TensorData(input.type, {}));
    SetBuiltinOp(
        BuiltinOperator_STABLEHLO_CASE,
        BuiltinOptions2_StablehloCaseOptions,
        CreateStablehloCaseOptions(
            builder_,
            builder_.CreateVector(
                std::vector(params.subgraph_indices,
                            params.subgraph_indices + params.num_branches))
            ).Union());
    BuildInterpreter({GetShape(input_), GetShape(output_shape_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  void SetOutputShape(std::initializer_list<T> data) {
    PopulateTensor<T>(output_shape_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  int output_shape_;
  int output_;
};

TEST(StablehloCaseTest, CaseWorks1) {
  TfLiteStablehloCaseParams params = {
      2,
      {0, 2},       
  };
  
  StablehloCaseOpModel model(
      {TensorType_INT32, {}}, 
      {TensorType_INT32, {4}}, 
      params);

  model.SetInput<int>({-1});
  model.SetOutputShape<int>({2, 3, 4, 2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

} //namespace
}  // namespace tflite
