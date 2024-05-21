/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/status/status.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/i4.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/ops/transpose.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using shlo_ref::testing::StatusIs;
using testing::ElementsAreArray;
using testing::Eq;
using testing::FloatEq;
using testing::NanSensitiveFloatEq;
using testing::Pointwise;

namespace shlo_ref {
namespace {

template <class T>
struct NonQuantizedBoolTransposeTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedBoolTransposeTest, BoolTestType, TestParamNames);

//transpose_dtypes_shape_bool_2_3__permutation__1_0.mlir
TYPED_TEST(NonQuantizedBoolTransposeTest, BoolTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_r({3, 2});
  Vector<StorageT> operand_data{true, true, true, true, true, true};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<StorageT> expected_data{true, true, true, true, true, true};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, expected_data);
}

template <class T>
struct NonQuantizedIntTransposeTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedIntTransposeTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntTransposeTest, IntTestTypesTensorrsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3, 2});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<StorageT> expected_data = Vector<StorageT>{
      (1), (2), (7), (8), (3), (4), (9), (10), (5), (6), (11), (12)};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, expected_data);
}

TYPED_TEST(NonQuantizedIntTransposeTest, IntTestTypesRaiseAnError1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3, 2});
  // const Shape shapePermutation({3});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data = Vector<StorageT>{
      (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12)};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 3, 2};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              "stablehlo.transpose: The permutation should be in the range of "
              "operand rank.");
}

TYPED_TEST(NonQuantizedIntTransposeTest, IntTestTypesRaiseAnError2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3, 2});
  // const Shape shapePermutation({3});
  const Shape shape_r({2, 2, 3});
  Vector<StorageT> operand_data = Vector<StorageT>{
      (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12)};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              "stablehlo.transpose: The output shape should be equal to the "
              "permutation of operand shape.");
}

template <class T>
struct NonQuantizedFloatTransposeTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedFloatTransposeTest, FloatTestTypes,
                 TestParamNames);

TYPED_TEST(NonQuantizedFloatTransposeTest, FloatTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3, 1, 2, 3, 2});
  const Shape shape_r({2, 3, 2, 1, 3, 2});
  Vector<float> operand_data_float{
      0.8892613,  0.96106523, 0.3297237,  0.01481166, 0.40585053, 0.6295124,
      0.65750533, 0.9323971,  0.8402479,  0.60150546, 0.55197155, 0.45298976,
      0.06617546, 0.7375449,  0.10545965, 0.36622655, 0.3403618,  0.05371938,
      0.20413005, 0.6899744,  0.06575581, 0.9133887,  0.16854943, 0.95533204,
      0.4127409,  0.71123004, 0.04607956, 0.53329164, 0.63794744, 0.72146165,
      0.5062086,  0.52149826, 0.5982147,  0.04738012, 0.02777152, 0.68853503,
      0.90734386, 0.4707448,  0.16416517, 0.5578099,  0.5012608,  0.7054341,
      0.93245703, 0.3422893,  0.3091052,  0.73739254, 0.5609417,  0.1827155,
      0.46748862, 0.5770362,  0.09115962, 0.19891712, 0.17695905, 0.44523406,
      0.16706759, 0.33241633, 0.7358104,  0.53436184, 0.9387657,  0.47540718,
      0.1061028,  0.00765424, 0.74906033, 0.4662233,  0.17728485, 0.58914083,
      0.9620191,  0.8419449,  0.3718606,  0.3346534,  0.11427356, 0.1969535};
  Vector<StorageT> operand_data(operand_data_float.begin(),
                                operand_data_float.end());
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {5, 4, 3, 2, 1, 0};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<float> expected_data_float{
      0.8892613,  0.90734386, 0.06617546, 0.46748862, 0.4127409,  0.1061028,
      0.65750533, 0.93245703, 0.20413005, 0.16706759, 0.5062086,  0.9620191,
      0.3297237,  0.16416517, 0.10545965, 0.09115962, 0.04607956, 0.74906033,
      0.8402479,  0.3091052,  0.06575581, 0.7358104,  0.5982147,  0.3718606,
      0.40585053, 0.5012608,  0.3403618,  0.17695905, 0.63794744, 0.17728485,
      0.55197155, 0.5609417,  0.16854943, 0.9387657,  0.02777152, 0.11427356,
      0.96106523, 0.4707448,  0.7375449,  0.5770362,  0.71123004, 0.00765424,
      0.9323971,  0.3422893,  0.6899744,  0.33241633, 0.52149826, 0.8419449,
      0.01481166, 0.5578099,  0.36622655, 0.19891712, 0.53329164, 0.4662233,
      0.60150546, 0.73739254, 0.9133887,  0.53436184, 0.04738012, 0.3346534,
      0.6295124,  0.7054341,  0.05371938, 0.44523406, 0.72146165, 0.58914083,
      0.45298976, 0.1827155,  0.95533204, 0.47540718, 0.68853503, 0.1969535};
  Vector<StorageT> expected_data(expected_data_float.begin(),
                                 expected_data_float.end());

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

// stablehlo/testdata/transpose_permutations_shape_float32_2_3_4__permutation__1_2_0.mlir
TYPED_TEST(NonQuantizedFloatTransposeTest, FloatTestTypesTensorrsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3, 4});
  const Shape shape_r({3, 4, 2});
  Vector<float> operand_data_float = Vector<float>{
      1.58909702,  -3.29110479,  -4.5229125,  -2.02355504, -5.2291913,
      -2.32745957, 0.410715669,  -6.9613409,  -6.73274517, 5.25102949,
      1.60699403,  6.31244135,   0.815644443, 6.24662971,  1.3186307,
      -2.22678375, -0.796898603, -4.74064922, 2.40567923,  3.60277486,
      -1.24683356, 0.389498383,  -2.51559758, 4.69905949};
  Vector<StorageT> operand_data(operand_data_float.begin(),
                                operand_data_float.end());
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 2, 0};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<float> expected_data_float = Vector<float>{
      1.58909702,  0.815644443, -3.29110479, 6.24662971, -4.5229125,
      1.3186307,   -2.02355504, -2.22678375, -5.2291913, -0.796898603,
      -2.32745957, -4.74064922, 0.410715669, 2.40567923, -6.9613409,
      3.60277486,  -6.73274517, -1.24683356, 5.25102949, 0.389498383,
      1.60699403,  -2.51559758, 6.31244135,  4.69905949};
  Vector<StorageT> expected_data(expected_data_float.begin(),
                                 expected_data_float.end());

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, expected_data);
}

using kBF16TestTypes = ::testing::Types<TestParam<DataType::kBF16>>;
template <class T>
struct NonQuantizedkBF16TransposeTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkBF16TransposeTest, kBF16TestTypes,
                 TestParamNames);

// stablehlo/testdata/transpose_dtypes_shape_bfloat16_2_3__permutation__1_0.mlir
TYPED_TEST(NonQuantizedkBF16TransposeTest, kBF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_r({3, 2});
  Vector<float> operand_data_float{9.921870e-01, 8.828120e-01,  -1.179690e+00,
                                   1.726560e+00, -1.156250e+00, 7.656250e-01};
  Vector<StorageT> operand_data(operand_data_float.begin(),
                                operand_data_float.end());
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<float> expected_data_float{9.921870e-01,  1.726560e+00,  8.828120e-01,
                                    -1.156250e+00, -1.179690e+00, 7.656250e-01};
  Vector<StorageT> expected_data(expected_data_float.begin(),
                                 expected_data_float.end());

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkBF16TransposeTest, kBF16TestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape_operand({2, 3});
  const Shape shape_r({3, 2});
  Vector<float> operand_data_float{9.921870e-01, 8.828120e-01,  -1.179690e+00,
                                   1.726560e+00, -1.156250e+00, 7.656250e-01};
  Vector<StorageT> operand_data(operand_data_float.begin(),
                                operand_data_float.end());
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0};
  Vector<StorageT> output_data(shape_r.NumElements());

  Tensor operand{.type = TensorType{.shape = shape_operand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape_r, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<float> expected_data_float{9.921870e-01,  1.726560e+00,  8.828120e-01,
                                    -1.156250e+00, -1.179690e+00, 7.656250e-01};
  Vector<StorageT> expected_data(expected_data_float.begin(),
                                 expected_data_float.end());

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct QuantizedIntTransposeTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedIntTransposeTest, QuantizedTestTypes, TestParamNames);
TYPED_TEST(QuantizedIntTransposeTest, QuantizedTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;
  const Shape shape_operand({2, 3, 2});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(0);
  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);

  Tensor operand{
      .type = QuantizedPerTensorTensorType{.shape = shape_operand,
                                           .element_type = tensor_type},
      .data = operand_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerTensorTensorType{.shape = shape_r,
                                           .element_type = tensor_type},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<float> expected_data{1.5,  3,  10.5, 12, 4.5,  6,
                              13.5, 15, 7.5,  9,  16.5, 18};
  Vector<StorageT> expected_quantized(shape_r.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_quantized.begin(), [&](float val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zero_point,
                       static_cast<ExpressedT>(1.0) / scale);
                 });

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, Pointwise(Eq(), expected_quantized));
}

TYPED_TEST(QuantizedIntTransposeTest, QuantizedTestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;
  const Shape shape_operand({1, 3, 2});
  const Shape shape_r({3, 1, 2});
  Vector<StorageT> operand_data = Vector<StorageT>{1, 2, 3, 4, 5, 6};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  std::vector<StorageT> zeroes = {0};
  std::vector<float> scales = {1.2f};

  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zeroes, TypeParam::kExpressed, scales, 0);
  QuantizedElementTypePerAxis tensor_type_axis_output(
      TypeParam::kStorage, zeroes, TypeParam::kExpressed, scales, 1);

  Tensor operand{
      .type = QuantizedPerAxisTensorType{.shape = shape_operand,
                                         .element_type = tensor_type_axis},
      .data = operand_data.data()};
  Tensor output_tensor{
      .type =
          QuantizedPerAxisTensorType{.shape = shape_r,
                                     .element_type = tensor_type_axis_output},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  Vector<float> expected_data{1.2002f,  2.40039f, 3.60156f,
                              4.80078f, 6,        7.20312f};
  Vector<float> expected_quantized(shape_r.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_quantized.begin(), [&](float val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zeroes[0],
                       static_cast<ExpressedT>(
                           (1.0) / static_cast<ExpressedT>(scales[0])));
                 });

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_quantized));
}

TYPED_TEST(QuantizedIntTransposeTest, QuantizedTestTypesTensorsRaiseAnError1) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;
  const Shape shape_operand({2, 3, 2});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  std::initializer_list<float> zero_points = {0, 0};
  std::initializer_list<float> scales = {1.2, 1.1};
  std::vector<int> zeroes = {0, 0};
  std::vector<float> scalesv = {1.2, 1.1};
  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 0);

  Tensor operand{
      .type = QuantizedPerAxisTensorType{.shape = shape_operand,
                                         .element_type = tensor_type_axis},
      .data = operand_data.data()};
  Tensor output_tensor{
      .type = QuantizedPerAxisTensorType{.shape = shape_r,
                                         .element_type = tensor_type_axis},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      status.message(),
      "stablehlo.transpose: The quantization dimension of operand should be "
      "equal to the permutation of quantization dimension of output.");
}

TYPED_TEST(QuantizedIntTransposeTest, QuantizedTestTypesTensorsRaiseAnError2) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;
  const Shape shape_operand({2, 3, 2});
  const Shape shape_r({3, 2, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  absl::InlinedVector<Axis, kMaxNumDimensions> permutation = {1, 0, 2};
  Vector<StorageT> output_data(shape_r.NumElements());
  std::initializer_list<float> zero_points = {0, 0};
  std::initializer_list<float> scales = {1.2, 1.1};
  std::initializer_list<float> zero_points_output = {1, 0, 0};
  std::initializer_list<float> scales_output = {2, 1.2, 1.1};
  std::vector<int> zeroes = {0, 0};
  std::vector<float> scalesv = {1.2, 1.1};
  QuantizedElementTypePerAxis tensor_type_axis(
      TypeParam::kStorage, zero_points, TypeParam::kExpressed, scales, 0);

  QuantizedElementTypePerAxis tensor_type_axis_output(
      TypeParam::kStorage, zero_points_output, TypeParam::kExpressed,
      scales_output, 0);
  Tensor operand{
      .type = QuantizedPerAxisTensorType{.shape = shape_operand,
                                         .element_type = tensor_type_axis},
      .data = operand_data.data()};
  Tensor output_tensor{
      .type =
          QuantizedPerAxisTensorType{.shape = shape_r,
                                     .element_type = tensor_type_axis_output},
      .data = output_data.data()};

  auto op = Create(TransposeOp::Attributes{
      .permutation = permutation,
  });

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      status.message(),
      ::testing::ContainsRegex(
          "stablehlo.transpose: baseline type constraint is not satisfied"));
}

}  // namespace
}  // namespace shlo_ref