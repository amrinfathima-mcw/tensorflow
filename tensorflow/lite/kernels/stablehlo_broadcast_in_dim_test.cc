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


namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_broadcast_in_dim {
namespace {

using ::testing::ElementsAreArray;

class StablehloBroadcastInDimOpModel : public SingleOpModel {
 public:
  StablehloBroadcastInDimOpModel(const TensorData& input, const TensorData& output_shape, const TfLiteStablehloBroadcastInDimParams& params) {
    input_ = AddInput(input);
    output_shape_ = AddInput(output_shape);
    output_ = AddOutput(TensorData(input.type, {}));
    SetBuiltinOp(
        BuiltinOperator_STABLEHLO_BROADCAST_IN_DIM,
        BuiltinOptions2_StablehloBroadcastInDimOptions,
        CreateStablehloBroadcastInDimOptions(
            builder_,
            builder_.CreateVector(
                std::vector(params.broadcast_dimensions,
                            params.broadcast_dimensions + params.num_broadcast_dimensions))
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

TEST(StablehloBroadcastInDimOpTest, BroadcastsCorrectly) {
  TfLiteStablehloBroadcastInDimParams params = {
      {0, 2}, // broadcast_dimensions
      2       // num_broadcast_dimensions
  };
  
  StablehloBroadcastInDimOpModel model(
      {TensorType_FLOAT32, {2, 3}}, 
      {TensorType_INT32, {4}}, 
      params);

  model.SetInput<float>({1, 2, 3, 4, 5, 6});
  model.SetOutputShape<int>({2, 3, 4, 2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  
  std::vector<float> expected_values = {
      1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
      5, 6, 5, 6, 5, 6, 5, 6, 1, 2, 1, 2, 1, 2, 1, 2,
      3, 4, 3, 4, 3, 4, 3, 4, 5, 6, 5, 6, 5, 6, 5, 6
  };
  
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

TEST(StablehloBroadcastInDimOpTest, HandlesDynamicShapes) {
  TfLiteStablehloBroadcastInDimParams params = {
      {0, 2}, // broadcast_dimensions
      2       // num_broadcast_dimensions
  };

  TensorData output_shape_tensor = {
      TensorType_INT32,
      /*shape*/ {4},
      /*min*/ 0.0f,
      /*max*/ 0.0f,
      /*scale*/ 0.0f,
      /*zero_point*/ 0,
      /*per_channel_quantization*/ false,
      /*per_channel_quantization_scales*/ {},
      /*per_channel_quantization_offsets*/ {},
      /*channel_index*/ 0,
      /*traversal_order*/ {},
      /*format*/ {},
      /*block_size*/ {},
      /*block_map*/ {},
      /*shape_signature*/ {{-1}}
  };

  StablehloBroadcastInDimOpModel model(
      {TensorType_FLOAT32, {2, 3}}, 
      output_shape_tensor, 
      params);

  model.SetInput<float>({1, 2, 3, 4, 5, 6});
  model.SetOutputShape<int>({2, 3, 4, 2});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  
  std::vector<float> expected_values = {
      1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4,
      5, 6, 5, 6, 5, 6, 5, 6, 1, 2, 1, 2, 1, 2, 1, 2,
      3, 4, 3, 4, 3, 4, 3, 4, 5, 6, 5, 6, 5, 6, 5, 6
  };
  
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

} //namespace
}  // namespace stablehlo_broadcast_in_dim
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
