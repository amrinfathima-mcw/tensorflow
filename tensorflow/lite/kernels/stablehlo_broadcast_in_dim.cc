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
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
// File: tensorflow/lite/kernels/stablehlo_broadcast_in_dim.cc


namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_broadcast_in_dim {
namespace{

constexpr int kOperandTensor = 0;
constexpr int kOutputTensor = 0;

using TfLiteIntArrayUniquePtr =
    std::unique_ptr<TfLiteIntArray, decltype(&TfLiteIntArrayFree)>;

template <typename DataType>
TfLiteStatus EvalWithTypes(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  int operand_rank = operand->dims->size;
  RuntimeShape operand_shape = GetTensorShape(operand);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  int output_rank = output->dims->size;
  RuntimeShape output_shape = GetTensorShape(output);

  const TfLiteStablehloBroadcastInDimParams* data =
      reinterpret_cast<TfLiteStablehloBroadcastInDimParams*>(node->builtin_data);

  const DataType* operand_data = GetTensorData<DataType>(operand);
  DataType* output_data = GetTensorData<DataType>(output);

  std::vector<int> result_index = std::vector<int>(output_rank, 0);
  do {
    std::vector<int> operand_index(operand_rank, 0);
    for (int d = 0; d < operand_rank; ++d) {
      if (operand_shape.Dims(d) == 1) continue;
      operand_index[d] = result_index[data->broadcast_dimensions[d]];
    }

    // int flat_operand_index =
    //     TensorIndexToFlat(operand_index.data(), operand_index.size(), operand_shape);
    // int flat_result_index =
    //     TensorIndexToFlat(result_index.data(), result_index.size(), output_shape);
    
    output_data[TensorIndexToFlat(result_index.data(), result_index.size(), output_shape)] = operand_data[TensorIndexToFlat(operand_index.data(), operand_index.size(), operand_shape)];
  } while (NextIndex(output_rank, output_shape.DimsData(), result_index.data()));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  TfLiteType data_type = operand->type;

  switch (data_type) {
    case kTfLiteFloat16:
      return EvalWithTypes<Eigen::half>(context, node);
    case kTfLiteFloat32:
      return EvalWithTypes<float>(context, node);
    case kTfLiteFloat64:
      return EvalWithTypes<double>(context, node);
    case kTfLiteInt8:
      return EvalWithTypes<int8_t>(context, node);
    case kTfLiteInt16:
      return EvalWithTypes<int16_t>(context, node);
    case kTfLiteInt32:
      return EvalWithTypes<int32_t>(context, node);
    case kTfLiteInt64:
      return EvalWithTypes<int64_t>(context, node);
    case kTfLiteUInt8:
      return EvalWithTypes<uint8_t>(context, node);
    case kTfLiteUInt16:
      return EvalWithTypes<uint16_t>(context, node);
    case kTfLiteUInt32:
      return EvalWithTypes<uint32_t>(context, node);
    case kTfLiteUInt64:
      return EvalWithTypes<uint64_t>(context, node);
    default:
      TF_LITE_KERNEL_LOG(context, "(Data Type: %s) currently not supported.\n",
                         TfLiteTypeGetName(data_type));
      return kTfLiteError;
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOperandTensor, &operand));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  const TfLiteStablehloBroadcastInDimParams* data =
      reinterpret_cast<TfLiteStablehloBroadcastInDimParams*>(node->builtin_data);

  RuntimeShape operand_shape = GetTensorShape(operand);

  std::vector<int> result_shape(output->dims->size);
  for (int i = 0; i < result_shape.size(); ++i) {
    if (i < data->num_broadcast_dimensions) {
      result_shape[i] = operand_shape.Dims(data->broadcast_dimensions[i]);
    } else {
      result_shape[i] = 1;
    }
  }

  TfLiteIntArrayUniquePtr result_shape_array =
      TfLiteIntArrayUniquePtr(TfLiteIntArrayCreate(result_shape.size()), &TfLiteIntArrayFree);
  std::copy(result_shape.begin(), result_shape.end(), result_shape_array->data);

  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, output, result_shape_array.release()));

  return kTfLiteOk;
}

 
} // namespace
}  // namespace stablehlo_broadcast_in_dim
TfLiteRegistration* Register_STABLEHLO_BROADCAST_IN_DIM() {
  static TfLiteRegistration r = {nullptr, nullptr, stablehlo_broadcast_in_dim::Prepare,
                                 stablehlo_broadcast_in_dim::Eval};
  return &r;
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite
