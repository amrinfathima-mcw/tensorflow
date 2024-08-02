
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "flatbuffers/flatbuffers.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_reduce_precision {
namespace {

using ::testing::ElementsAreArray;

class StablehloReshapeOpModel : public SingleOpModel {
 public:
  StablehloReshapeOpModel(const TensorData& input,const TensorData& output_shape) {
    input_ = AddInput(input);
    std::vector<int> outputShape = output_shape_; 
    output_ = AddOutput(TensorData(input.type,outputShape));
    SetBuiltinOp(BuiltinOperator_STABLEHLO_RESHAPE, BuiltinOptions_NONE,0);
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  std::vector<int>output_shape_;
  int output_;
};

TEST(StablehloReshapeOpTest, ReshapesTensor) {
   //resultType{TensorType_FLOAT32, {2, 3, 2, 2}};
  StablehloReshapeOpModel model({TensorType_FLOAT32, {3, 4, 2}}, {TensorType_FLOAT32, {2, 3, 2, 2}});

  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,
                         9,  10, 11, 12, 13, 14, 15, 16,
                         17, 18, 19, 20, 21, 22, 23, 24});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  std::vector<float> expected_values = {1,  2,  3,  4,  5,  6,  7,  8,
                                        9,  10, 11, 12, 13, 14, 15, 16,
                                        17, 18, 19, 20, 21, 22, 23, 24};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}

}  // namespace
}  // namespace stablehlo_reduce_precision
}  // namespace builtin
}  // namespace ops
}  // namespace tflite

/*
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "flatbuffers/flatbuffers.h"
#include <iostream>
namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class StablehloReshapeOpModel : public SingleOpModel {
 public:
  StablehloReshapeOpModel(const TensorData& input) {

    input_ = AddInput(input);

    std :: cout << "input done" << std::endl;

    output_ = AddOutput(TensorData(input.type, {2, 3, 2, 2}));
    
    // :: flatbuffers::Offset<StablehloReshapeOptions> reshape_options;

    :: flatbuffers::Offset<void> reshape_options = :: flatbuffers::Offset<void>();


    SetBuiltinOp(
        BuiltinOperator_STABLEHLO_RESHAPE,
        BuiltinOptions2_NONE,
        reshape_options.Union()
        );
    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  int output_;
};

TEST(StablehloScatterOpTest, SOMETHING_CHECK) {
 
  StablehloReshapeOpModel model({TensorType_FLOAT32, {3, 4, 2}});
  model.SetInput<float>({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});


  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  std::vector<float> expected_values = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
}
}
}
*/