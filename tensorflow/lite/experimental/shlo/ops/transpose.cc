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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/transpose.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

absl::Status CheckParameters(
    const Tensor& operand,
    absl::InlinedVector<Axis, kMaxNumDimensions>& permutation, Tensor& output) {
  // C1
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("transpose"), operand, output));
  // C2
  for (int64_t perm : permutation) {
    if (perm < 0 || perm >= operand.Rank()) {
      return absl::FailedPreconditionError(
          "stablehlo.transpose: The permutation should be in the range of "
          "operand rank.");
    }
  }
  // C3
  for (int i = 0; i < operand.Rank(); ++i) {
    if (output.shape().Dim(i) != operand.shape().Dim(permutation[i])) {
      return absl::FailedPreconditionError(
          "stablehlo.transpose: The output shape should be equal to the "
          "permutation of operand shape.");
    }
  }
  // C4
  if (output.IsPerAxisQuantized()) {
    if (operand.quantized_per_axis_element_type().QuantizedDimension() !=
        permutation[output.quantized_per_axis_element_type()
                        .QuantizedDimension()]) {
      return absl::FailedPreconditionError(
          "stablehlo.transpose: The quantization dimension of operand should "
          "be equal to the permutation of quantization dimension of output.");
    }
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status PrepareTensors(TransposeOp& op, const Tensor& operand,
                            Tensor& output) {
  using StorageT = StorageType<storage_type>;

  const DimensionSize operand_size = operand.NumElements();
  const DimensionSize output_size = output.NumElements();

  op.operand_dequantized_data =
      std::vector<std::byte>(operand_size * sizeof(StorageT));
  const Shape operand_dequantized_shape = operand.shape();
  Tensor operand_dequantized{
      .type = TensorType{.shape = operand_dequantized_shape,
                         .element_type = storage_type},
      .data = op.operand_dequantized_data.data()};

  op.output_dequantized_data =
      std::vector<std::byte>(output_size * sizeof(StorageT));
  const Shape output_dequantized_shape = output.shape();
  Tensor output_dequantized{
      .type = TensorType{.shape = output_dequantized_shape,
                         .element_type = storage_type},
      .data = op.output_dequantized_data.data()};

  op.operand_dequantized = std::move(operand_dequantized);
  op.output_dequantized = std::move(output_dequantized);
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status EvaluateImpl(
    const Tensor& operand,
    absl::InlinedVector<Axis, kMaxNumDimensions>& permutation, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const DimensionSize operand_size = operand.NumElements();
  const Axis operand_rank = operand.Rank();

  absl::InlinedVector<Axis, kMaxNumDimensions> operand_index(operand_rank);
  absl::InlinedVector<Axis, kMaxNumDimensions> output_index(operand_rank);

  for (DimensionSize k = 0; k < operand_size; ++k) {
    operand.GetNdIndex(k, operand_index);
    for (DimensionSize d = 0; d < operand_rank; ++d) {
      output_index[d] = operand_index[permutation[d]];
    }
    output_buffer[output.FlattenIndex(output_index)] =
        operand.Get<storage_type>(operand_index);
  }
  return absl::OkStatus();
}

template <DataType storage_type, DataType expressed_type>
void DequantizeOpQuantizePerTensor(TransposeOp& op, const Tensor& operand,
                                   Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  const StorageT* operand_data = operand.GetDataAs<storage_type>();
  ExpressedT* operand_dequantized_data =
      op.operand_dequantized.GetDataAs<expressed_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  ExpressedT* output_dequantized_data =
      op.output_dequantized.GetDataAs<expressed_type>();
  const DimensionSize operand_num_elements = operand.NumElements();
  const StorageT operand_zero_point =
      operand.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT operand_scale =
      operand.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

  for (DimensionSize i = 0; i < operand_num_elements;
       ++i, ++operand_data, ++operand_dequantized_data) {
    *operand_dequantized_data =
        Dequantize(*operand_data, operand_zero_point, operand_scale);
  }

  absl::Status status =
      Evaluate(op, op.operand_dequantized, op.output_dequantized);

  const DimensionSize output_num_elements = output.NumElements();
  const StorageT output_zero_point =
      output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT output_scale =
      output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const ExpressedT inv_scale = static_cast<ExpressedT>(1 / output_scale);

  for (DimensionSize i = 0; i < output_num_elements;
       ++i, ++output_dequantized_data, ++output_data) {
    *output_data = Quantize<storage_type, expressed_type>(
        *output_dequantized_data, output_zero_point, inv_scale);
  }
}

template <typename StorageT, typename ExpressedT>
void DequantizeOpQuantizePerAxisImpl(
    const Shape& shape, const Axis quantization_dimension,
    const StorageT quantization_min, const StorageT quantization_max,
    const absl::Span<const StorageT> input_zero_points,
    const absl::Span<const ExpressedT> input_scales, const Strides& strides,
    const StorageT* input_data, ExpressedT* inputDeQuantized_data,
    const size_t depth, size_t quantization_index) {
  const DimensionSize dim = shape.Dim(depth);
  if (depth + 1 >= shape.Rank()) {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      *inputDeQuantized_data =
          Dequantize(*input_data, input_zero_points[quantization_index],
                     input_scales[quantization_index]);
      input_data += strides[depth];
      inputDeQuantized_data += strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      DequantizeOpQuantizePerAxisImpl(
          shape, quantization_dimension, quantization_min, quantization_max,
          input_zero_points, input_scales, strides, input_data,
          inputDeQuantized_data, depth + 1, quantization_index);
      input_data += strides[depth];
      inputDeQuantized_data += strides[depth];
    }
  }
}

template <typename StorageT, typename ExpressedT>
void QuantizeOpQuantizePerAxisImpl(
    const Shape& shape, const Axis quantization_dimension,
    const StorageT quantization_min, const StorageT quantization_max,
    const absl::Span<const StorageT> input_zero_points,
    const absl::Span<const ExpressedT> input_scales, const Strides& strides,
    StorageT* input_data, const ExpressedT* inputDequantized_data,
    const size_t depth, size_t quantization_index) {
  const DimensionSize dim = shape.Dim(depth);
  if (depth + 1 >= shape.Rank()) {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      *input_data = Quantize<StorageT, ExpressedT>(
          *inputDequantized_data, input_zero_points[quantization_index],
          static_cast<ExpressedT>(1 / input_scales[quantization_index]),
          quantization_min, quantization_max);
      input_data += strides[depth];
      inputDequantized_data += strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      QuantizeOpQuantizePerAxisImpl(
          shape, quantization_dimension, quantization_min, quantization_max,
          input_zero_points, input_scales, strides, input_data,
          inputDequantized_data, depth + 1, quantization_index);
      input_data += strides[depth];
      inputDequantized_data += strides[depth];
    }
  }
}

template <DataType storage_type, DataType expressed_type>
void DequantizeOpQuantizePerAxis(TransposeOp& op, const Tensor& operand,
                                 Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  const StorageT* operand_data = operand.GetDataAs<storage_type>();
  ExpressedT* operand_dequantized_data =
      op.operand_dequantized.GetDataAs<expressed_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  ExpressedT* output_dequantized_data =
      op.output_dequantized.GetDataAs<expressed_type>();

  const Shape& shape = operand.shape();
  const Axis operand_quantization_dimension =
      operand.quantized_per_axis_element_type().QuantizedDimension();
  const absl::Span<const StorageT> operand_zero_points =
      operand.quantized_per_axis_element_type().ZeroPointsAs<storage_type>();
  const absl::Span<const ExpressedT> operand_scales =
      operand.quantized_per_axis_element_type().ScalesAs<expressed_type>();
  const Strides& strides = ComputeStrides(shape);
  DequantizeOpQuantizePerAxisImpl(
      shape, operand_quantization_dimension, Storage<storage_type>::kMinValue,
      Storage<storage_type>::kMaxValue, operand_zero_points, operand_scales,
      strides, operand_data, operand_dequantized_data, /*depth=*/0,
      /*quantization_index=*/0);

  absl::Status status =
      Evaluate(op, op.operand_dequantized, op.output_dequantized);
  const Shape& shape_output = output.shape();
  const Axis output_quantization_dimension =
      output.quantized_per_axis_element_type().QuantizedDimension();
  const absl::Span<const StorageT> output_zero_points =
      output.quantized_per_axis_element_type().ZeroPointsAs<storage_type>();
  const absl::Span<const ExpressedT> output_scales =
      output.quantized_per_axis_element_type().ScalesAs<expressed_type>();
  const Strides& strides_output = ComputeStrides(shape_output);
  QuantizeOpQuantizePerAxisImpl(
      shape_output, output_quantization_dimension,
      Storage<storage_type>::kMinValue, Storage<storage_type>::kMaxValue,
      output_zero_points, output_scales, strides_output, output_data,
      output_dequantized_data, /*depth=*/0,
      /*quantization_index=*/0);
}

TransposeOp Create(TransposeOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(TransposeOp& op, const Tensor& operand, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(
      CheckParameters(operand, op.attributes.permutation, output));
  if (operand.IsQuantized()) {
    DISPATCH_BOOL_INT_FLOAT(PrepareTensors, operand.ExpressedType(), op,
                            operand, output);
  }
  return absl::OkStatus();
}

absl::Status Evaluate(TransposeOp& op, const Tensor& operand, Tensor& output) {
  if (operand.IsQuantized()) {
    if (operand.IsPerTensorQuantized()) {
      DISPATCH_QUANTIZED(
          DequantizeOpQuantizePerTensor,
          operand.quantized_per_tensor_element_type().StorageType(),
          operand.quantized_per_tensor_element_type().ExpressedType(), op,
          operand, output);

    } else if (operand.IsPerAxisQuantized()) {
      DISPATCH_QUANTIZED(DequantizeOpQuantizePerAxis, operand.StorageType(),
                         operand.ExpressedType(), op, operand, output);
    }
  } else {
    DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, output.tensor_element_type(), operand,
                            op.attributes.permutation, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.transpose: Unsupported tensor type.");
}
}  // namespace shlo_ref
