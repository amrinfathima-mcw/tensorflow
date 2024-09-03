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
#include <iostream>

#include "kernel_util.h"
#include "tensorflow/lite/kernels/dequantize.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/stablehlo_batch_norm_training.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_batch_norm_grad {
namespace {

constexpr int kMaxTemporaryTensors = 18;
constexpr int32_t kMaxReduceRank = 6;
struct OpData {
 public:
  enum {
    kOperandTensor,
    kScaleTensor,
    kMeanTensor,
    kVarianceTensor,
    kGradOutputTensor
  };
  enum {
    kOutputGradOperandTensor,
    kOutputGradScaleTensor,
    kOutputGradOffsetTensor
  };
  int scratch_tensor_index;
  int32_t mul_multiplier;
  int mul_shift;
  int32_t div_multiplier;
  int div_shift;
  int32_t output_multiplier;
  int output_shift;
  int left_shift;
  int32_t input_multiplier;
  int input_shift;
};

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  OpData* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus PrepareTemporaries(TfLiteContext* context, TfLiteNode* node,
                                const TfLiteBatchNormGradParams* params,
                                const TfLiteTensor* operand,
                                const TfLiteTensor* grad_output,
                                const TfLiteTensor* scale) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  context->AddTensors(context, kMaxTemporaryTensors,
                      &data->scratch_tensor_index);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(kMaxTemporaryTensors);

  node->temporaries->data[0] = data->scratch_tensor_index;
  TfLiteTensor* epsilon_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 0, &epsilon_tensor));
  TfLiteIntArray* epsilon_tensor_shape = TfLiteIntArrayCreate(1);
  epsilon_tensor_shape->data[0] = 1;
  epsilon_tensor->type = operand->type;
  epsilon_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, epsilon_tensor,
                                                   epsilon_tensor_shape));

  node->temporaries->data[1] = data->scratch_tensor_index + 1;
  TfLiteTensor* centered_operand;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 1, &centered_operand));
  TfLiteIntArray* centered_operand_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    centered_operand_bcast_shape->data[i] = operand->dims->data[i];
  }
  centered_operand->type = operand->type;
  centered_operand->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, centered_operand,
                                          centered_operand_bcast_shape));

  node->temporaries->data[2] = data->scratch_tensor_index + 2;
  TfLiteTensor* stddev;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 2, &stddev));
  TfLiteIntArray* stddev_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    stddev_bcast_shape->data[i] = operand->dims->data[i];
  }
  stddev->type = operand->type;
  stddev->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, stddev, stddev_bcast_shape));

  node->temporaries->data[3] = data->scratch_tensor_index + 3;
  TfLiteTensor* normalized_operand;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 3, &normalized_operand));
  TfLiteIntArray* normalized_operand_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    normalized_operand_bcast_shape->data[i] = operand->dims->data[i];
  }
  normalized_operand->type = operand->type;
  normalized_operand->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, normalized_operand,
                                          normalized_operand_bcast_shape));

  node->temporaries->data[4] = data->scratch_tensor_index + 4;
  TfLiteTensor* elements_per_feature_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 4,
                                              &elements_per_feature_tensor));
  TfLiteIntArray* elements_per_feature_tensor_shape = TfLiteIntArrayCreate(0);

  elements_per_feature_tensor->type = operand->type;
  elements_per_feature_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, elements_per_feature_tensor,
                                          elements_per_feature_tensor_shape));

  node->temporaries->data[5] = data->scratch_tensor_index + 5;
  TfLiteTensor* i6;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 5, &i6));
  TfLiteIntArray* i6_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    i6_bcast_shape->data[i] = operand->dims->data[i];
  }
  i6->type = operand->type;
  i6->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, i6, i6_bcast_shape));

  node->temporaries->data[6] = data->scratch_tensor_index + 6;
  TfLiteTensor* grad_output_centered_operand_mul;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, 6, &grad_output_centered_operand_mul));
  TfLiteIntArray* grad_output_centered_operand_mul_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    grad_output_centered_operand_mul_bcast_shape->data[i] =
        operand->dims->data[i];
  }
  grad_output_centered_operand_mul->type = operand->type;
  grad_output_centered_operand_mul->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                 context, grad_output_centered_operand_mul,
                                 grad_output_centered_operand_mul_bcast_shape));

  node->temporaries->data[7] = data->scratch_tensor_index + 7;
  TfLiteTensor* grad_output_reduced;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 7, &grad_output_reduced));
  TfLiteIntArray* grad_output_reduced_shape = TfLiteIntArrayCreate(1);
  grad_output_reduced_shape->data[0] =
      grad_output->dims->data[params->feature_index];

  grad_output_reduced->type = operand->type;
  grad_output_reduced->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_output_reduced,
                                                   grad_output_reduced_shape));

  node->temporaries->data[8] = data->scratch_tensor_index + 8;
  TfLiteTensor* i3_intermediate;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 8, &i3_intermediate));
  TfLiteIntArray* i3_intermediate_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    i3_intermediate_shape->data[i] =
        operand->dims->data[i];
  }
  i3_intermediate->type = operand->type;
  i3_intermediate->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, i3_intermediate,
                                                   i3_intermediate_shape));

  node->temporaries->data[9] = data->scratch_tensor_index + 9;
  TfLiteTensor* grad_scale_intermediate;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, 9, &grad_scale_intermediate));
  TfLiteIntArray* grad_scale_intermediate_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    grad_scale_intermediate_shape->data[i] = operand->dims->data[i];
  }

  grad_scale_intermediate->type = operand->type;
  grad_scale_intermediate->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, grad_scale_intermediate,
                                          grad_scale_intermediate_shape));

  if (operand->type == kTfLiteInt8 || operand->type == kTfLiteInt16) {
    TfLiteIntArray* i3_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      i3_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[10] = data->scratch_tensor_index + 10;
    TfLiteTensor* i3;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, 10, &i3));
    i3->type = kTfLiteInt8;
    i3->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, i3,
                                            i3_shape));

    TfLiteIntArray* i4_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      i4_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[11] = data->scratch_tensor_index + 11;
    TfLiteTensor* i4;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, 11, &i4));
    i4->type = kTfLiteInt8;
    i4->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, i4,
                                                     i4_shape));

    TfLiteIntArray* var_eps_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      var_eps_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[12] = data->scratch_tensor_index + 12;
    TfLiteTensor* var_eps;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, 12, &var_eps));
    var_eps->type = kTfLiteInt8;
    var_eps->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, var_eps,
                                                     var_eps_shape));

    TfLiteIntArray* i5_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      i5_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[13] = data->scratch_tensor_index + 13;
    TfLiteTensor* i5;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 13, &i5));
    i5->type = kTfLiteInt8;
    i5->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, i5,
                                            i5_shape));

    TfLiteIntArray* i1_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      i1_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[14] = data->scratch_tensor_index + 14;
    TfLiteTensor* i1;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 14, &i1));
    i1->type = kTfLiteInt8;
    i1->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, i1,
                                            i1_shape));

    TfLiteIntArray* grad_operand_dequantize_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      grad_operand_dequantize_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[15] = data->scratch_tensor_index + 15;
    TfLiteTensor* grad_operand_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 15, &grad_operand_dequantize));
    grad_operand_dequantize->type = kTfLiteFloat32;
    grad_operand_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, grad_operand_dequantize,
                                            grad_operand_dequantize_shape));

    TfLiteIntArray* grad_scale_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      grad_scale_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[16] = data->scratch_tensor_index + 16;
    TfLiteTensor* grad_scale_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 16, &grad_scale_dequantize));
    grad_scale_dequantize->type = kTfLiteFloat32;
    grad_scale_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, grad_scale_dequantize,
                                            grad_scale_dequantize_shape));

    TfLiteIntArray* grad_offset_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      grad_offset_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[17] = data->scratch_tensor_index + 17;
    TfLiteTensor* grad_offset_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 17, &grad_offset_dequantize));
    grad_offset_dequantize->type = kTfLiteFloat32;
    grad_offset_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, grad_offset_dequantize,
                                            grad_offset_dequantize_shape));
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 5);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kOperandTensor, &operand));

  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kScaleTensor, &scale));

  const TfLiteTensor* mean;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kMeanTensor, &mean));

  const TfLiteTensor* variance;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kVarianceTensor, &variance));

  const TfLiteTensor* grad_output;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, OpData::kGradOutputTensor, &grad_output));

  TfLiteTensor* grad_operand;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOperandTensor,
                             &grad_operand));

  TfLiteTensor* grad_scale;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, OpData::kOutputGradScaleTensor,
                                  &grad_scale));

  TfLiteTensor* grad_offset;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOffsetTensor,
                             &grad_offset));

  const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const double real_div_multiplier =
        operand->params.scale / (operand->params.scale * operand->params.scale);
  QuantizeMultiplier(real_div_multiplier, &data->div_multiplier, &data->div_shift);

  data->left_shift = (operand->type == kTfLiteInt16) ? 15 : 20;
  const double twice_max_input_scale =
        2 * std::max(operand->params.scale, operand->params.scale);
    const double real_input_multiplier =
        operand->params.scale / twice_max_input_scale;
    const double real_output_multiplier =
        twice_max_input_scale / ((1 << data->left_shift) * operand->params.scale);

    tflite::QuantizeMultiplierSmallerThanOneExp(
        real_input_multiplier, &data->input_multiplier, &data->input_shift);
    if (real_output_multiplier > 1) {
      tflite::QuantizeMultiplierGreaterThanOne(
          real_output_multiplier, &data->output_multiplier, &data->output_shift);
    } else {
      tflite::QuantizeMultiplierSmallerThanOneExp(
          real_output_multiplier, &data->output_multiplier, &data->output_shift);
    }

  int operand_rank = NumDimensions(operand);
  TF_LITE_ENSURE(context, params->feature_index >= 0 &&
                              params->feature_index < operand_rank);

  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, scale->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, mean->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, variance->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, grad_output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, grad_operand->type);
  TF_LITE_ENSURE_TYPES_EQ(context, scale->type, grad_scale->type);
  TF_LITE_ENSURE_TYPES_EQ(context, scale->type, grad_offset->type);

  TF_LITE_ENSURE_EQ(
      context, TfLiteIntArrayEqual(operand->dims, grad_output->dims), true);

  TF_LITE_ENSURE_EQ(context, scale->dims->data[0],
                    operand->dims->data[params->feature_index]);

  TfLiteIntArray* grad_operand_size = TfLiteIntArrayCopy(operand->dims);
  TfLiteIntArray* grad_scale_size = TfLiteIntArrayCreate(1);
  grad_scale_size->data[0] = operand->dims->data[params->feature_index];
  TfLiteIntArray* grad_offset_size = TfLiteIntArrayCreate(1);
  grad_offset_size->data[0] = operand->dims->data[params->feature_index];

  TF_LITE_ENSURE_OK(context, PrepareTemporaries(context, node, params, operand,
                                                grad_output, scale));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_operand, grad_operand_size));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_scale, grad_scale_size));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_offset, grad_offset_size));
  TF_LITE_ENSURE_EQ(
      context, TfLiteIntArrayEqual(operand->dims, grad_operand->dims), true);

  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(scale->dims, mean->dims),
                    true);
  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(scale->dims, variance->dims),
                    true);
  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(scale->dims, grad_scale->dims),
                    true);
  TF_LITE_ENSURE_EQ(context,
                    TfLiteIntArrayEqual(scale->dims, grad_offset->dims), true);

  return kTfLiteOk;
}

}  // namespace

template <typename T>
T quantize_value(const float value, const double scale, int zero_point) {
  int min_val = std::numeric_limits<T>::min();
  int max_val = std::numeric_limits<T>::max();
  int unclamped =
      static_cast<int>(TfLiteRound(value / static_cast<float>(scale))) +
      zero_point;
  int clamped = std::min(std::max(unclamped, min_val), max_val);
  return static_cast<T>(clamped);
}

template <typename DataType>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                      const TfLiteTensor* operand, const TfLiteTensor* scale,
                      const TfLiteTensor* mean, const TfLiteTensor* variance,
                      const TfLiteTensor* grad_output,
                      TfLiteTensor* grad_operand, TfLiteTensor* grad_scale,
                      TfLiteTensor* grad_offset) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* epsilon_tensor = GetTemporary(context, node, 0);
  TfLiteTensor* centered_operand = GetTemporary(context, node, 1);
  TfLiteTensor* stddev = GetTemporary(context, node, 2);
  TfLiteTensor* normalized_operand = GetTemporary(context, node, 3);
  TfLiteTensor* elements_per_feature_tensor = GetTemporary(context, node, 4);
  TfLiteTensor* i6 = GetTemporary(context, node, 5);
  TfLiteTensor* grad_output_centered_operand_mul =
      GetTemporary(context, node, 6);
  TfLiteTensor* grad_output_reduced = GetTemporary(context, node, 7);
  TfLiteTensor* i3_intermediate = GetTemporary(context, node, 8);
  TfLiteTensor* grad_scale_intermediate = GetTemporary(context, node, 9);


  const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(node->builtin_data);

  const int feature_index = params->feature_index;
  const float epsilon = params->epsilon;

  TfLiteIntArray* feature_dims = TfLiteIntArrayCreate(1);
  feature_dims->data[0] = feature_index;

  const DataType* scale_data = GetTensorData<DataType>(scale);
  int scale_size = NumElements(scale);

  epsilon_tensor->data.f[0] = epsilon;

  ArithmeticParams op_params;
  op_params.broadcast_category = BroadcastableOpCategory::kGenericBroadcast;

  DataType* centered_operand_buffer = GetTensorData<DataType>(centered_operand);
  const float* operand_buffer = GetTensorData<float>(operand);
  const DataType* mean_data = GetTensorData<DataType>(mean);
  for (int i = 0; i < NumElements(centered_operand); ++i) {
    centered_operand_buffer[i] = static_cast<DataType>(
        operand_buffer[i] - mean_data[i % NumElements(mean)]);
  }

  int num_elements = NumElements(stddev);

  const DataType* variance_data = GetTensorData<DataType>(variance);
  int variance_size = NumElements(variance);
  DataType* stddev_buffer = GetTensorData<DataType>(stddev);
  for (int i = 0; i < NumElements(stddev); ++i) {
    stddev_buffer[i] = static_cast<DataType>(
        std::sqrt(variance_data[i % (NumElements(variance))] +
                  static_cast<DataType>(epsilon)));
  }

  float* normalized_buffer = GetTensorData<float>(normalized_operand);

  int operand_size = NumElements(operand);
  int feature_size = GetTensorShape(operand).Dims(feature_index);
  float elements_per_feature = static_cast<float>(operand_size) / feature_size;
  elements_per_feature_tensor->data.f[0] = elements_per_feature;

  TfLiteIntArray* a = TfLiteIntArrayCreate(0);

  DataType* element_per_feature_tensor_buffer =
      GetTensorData<DataType>(elements_per_feature_tensor);
  const DataType* grad_output_buffer = GetTensorData<DataType>(grad_output);

  tflite::stablehlo_batch_norm_training::reference::ComputeSum<DataType>(
      context, node, grad_output, feature_index, grad_output_reduced);

  DataType* grad_output_centered_operand_mul_buffer =
      GetTensorData<DataType>(grad_output_centered_operand_mul);
  for (int i = 0; i < NumElements(grad_output_centered_operand_mul); ++i) {
    grad_output_centered_operand_mul_buffer[i] =
        grad_output_buffer[i] * centered_operand_buffer[i];
  }

  tflite::stablehlo_batch_norm_training::reference::ComputeSum<DataType>(
      context, node, grad_output_centered_operand_mul, feature_index,
      i3_intermediate);

  DataType* i3_intermediate_buffer = GetTensorData<DataType>(i3_intermediate);

  DataType* i6_buffer = GetTensorData<DataType>(i6);
  DataType* grad_output_reduced_buffer =
      GetTensorData<DataType>(grad_output_reduced);

  for (int i = 0; i < NumElements(i6); ++i) {
    i6_buffer[i] =
        ((grad_output_buffer[i] *
              element_per_feature_tensor_buffer
                  [i % (NumElements(elements_per_feature_tensor))] -
          grad_output_reduced_buffer[i % NumElements(grad_output_reduced)]) -
         (i3_intermediate_buffer[i % (NumElements(i3_intermediate))] *
          centered_operand_buffer[i]) /
             (variance_data[i % (NumElements(variance))] +
              static_cast<DataType>(epsilon)));
  }

  DataType* grad_operand_buffer = GetTensorData<DataType>(grad_operand);

  for (int i = 0; i < NumElements(grad_operand); ++i) {
    grad_operand_buffer[i] =
        ((scale_data[i % scale_size] / stddev_buffer[i]) /
         element_per_feature_tensor_buffer
             [i % (NumElements(elements_per_feature_tensor))]) *
        i6_buffer[i];
  }

  DataType* grad_scale_intermediate_buffer =
      GetTensorData<DataType>(grad_scale_intermediate);
  for (int i = 0; i < NumElements(grad_scale_intermediate); ++i) {
    grad_scale_intermediate_buffer[i] =
        static_cast<DataType>(grad_output_buffer[i] * normalized_buffer[i]);
  }

  tflite::stablehlo_batch_norm_training::reference::ComputeSum<DataType>(
      context, node, grad_scale_intermediate, feature_index, grad_scale);

  tflite::stablehlo_batch_norm_training::reference::ComputeSum<DataType>(
      context, node, grad_output, feature_index, grad_offset);

  DataType* grad_offset_buffer = GetTensorData<DataType>(grad_offset);
  for (int i = 0; i < NumElements(grad_offset); ++i) {
  }
  TfLiteIntArrayFree(feature_dims);
  TfLiteIntArrayFree(a);
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus EvalQuantImp(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteTensor* operand,
                          const TfLiteTensor* scale, const TfLiteTensor* mean,
                          const TfLiteTensor* variance,
                          const TfLiteTensor* grad_output,
                          TfLiteTensor* grad_operand, TfLiteTensor* grad_scale,
                          TfLiteTensor* grad_offset) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* epsilon_tensor = GetTemporary(context, node, 0);
  TfLiteTensor* centered_operand = GetTemporary(context, node, 1);
  TfLiteTensor* stddev = GetTemporary(context, node, 2);
  TfLiteTensor* normalized_operand = GetTemporary(context, node, 3);
  TfLiteTensor* elements_per_feature_tensor = GetTemporary(context, node, 4);
  TfLiteTensor* i6 = GetTemporary(context, node, 5);
  TfLiteTensor* grad_output_centered_operand_mul =
      GetTemporary(context, node, 6);
  TfLiteTensor* grad_output_reduced = GetTemporary(context, node, 7);
  TfLiteTensor* i3_intermediate = GetTemporary(context, node, 8);
  TfLiteTensor* grad_scale_intermediate = GetTemporary(context, node, 9);
  TfLiteTensor* i3 = GetTemporary(context, node, 10);
  TfLiteTensor* i4 = GetTemporary(context, node, 11);
  TfLiteTensor* var_eps = GetTemporary(context, node, 12);
  TfLiteTensor* i5 = GetTemporary(context, node, 13);
  TfLiteTensor* i1 = GetTemporary(context, node, 14);

  const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(node->builtin_data);

  const int feature_index = params->feature_index;
  const float epsilon = params->epsilon;

  const int operand_rank = operand->dims->size;

  const DataType* scale_data = GetTensorData<DataType>(scale);
  const DataType* mean_data = GetTensorData<DataType>(mean);
  const DataType* variance_data = GetTensorData<DataType>(variance);
  const DataType* operand_data = GetTensorData<DataType>(operand);
  const DataType* grad_output_buffer = GetTensorData<DataType>(grad_output);

  DataType* grad_operand_buffer = GetTensorData<DataType>(grad_operand);
  DataType* grad_scale_buffer = GetTensorData<DataType>(grad_scale);
  DataType* grad_offset_buffer = GetTensorData<DataType>(grad_offset);
  DataType* i3_buffer = GetTensorData<DataType>(i3);
  DataType* i3_intermediate_buffer = GetTensorData<DataType>(i3_intermediate);
  DataType* grad_scale_intermediate_buffer = GetTensorData<DataType>(grad_scale_intermediate);
  DataType* i4_buffer = GetTensorData<DataType>(i4);
  DataType* centered_operand_buffer = GetTensorData<DataType>(centered_operand);
  DataType* var_eps_buffer = GetTensorData<DataType>(var_eps);
  DataType* i5_buffer = GetTensorData<DataType>(i5);
  DataType* i1_buffer = GetTensorData<DataType>(i1);  
  DataType* i6_buffer = GetTensorData<DataType>(i6); 
  DataType* stddev_buffer = GetTensorData<DataType>(stddev);
  DataType* elements_per_feature_buffer = GetTensorData<DataType>(elements_per_feature_tensor);

   // grad offset calculation
  TF_LITE_ENSURE_OK(
      context,
      tflite::stablehlo_batch_norm_training::reference::ComputeQuantizedSum<
          DataType>(context, node, grad_output, feature_index, grad_offset));

  const int kMin = std::numeric_limits<DataType>::min();
  const int kMax = std::numeric_limits<DataType>::max();

  std::cout<<"scale and zp "<<operand->params.scale<<"  "<<operand->params.zero_point<<std::endl;
  // const int left_shift = (operand->type == kTfLiteInt16) ? 15 : 20;
  for (int64_t i = 0; i < NumElements(operand); ++i) {
    int64_t feature_index_value = i % operand->dims->data[feature_index];
    const int operand_val = -operand->params.zero_point + operand_data[i];
    const int mean_val =
        -operand->params.zero_point + mean_data[i % NumElements(mean)];
    const int shifted_operand_val = operand_val * (1 << data->left_shift);
    const int shifted_mean_val = mean_val * (1 << data->left_shift);
    const int scaled_operand_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_operand_val, data->input_multiplier, data->input_shift);
    const int scaled_mean_val = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        shifted_mean_val, data->input_multiplier, data->input_shift);
    const int raw_centered_val = scaled_operand_val - scaled_mean_val;
    const int raw_centered_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_centered_val, data->output_multiplier, data->output_shift) +
        operand->params.zero_point;
    centered_operand_buffer[i]=raw_centered_output;
    std::cout<<"raw centred val "<<raw_centered_output<<std::endl;

    const int variance_val =
        -operand->params.zero_point + variance_data[i % NumElements(variance)];
    const int epsilon_quantized =
        (epsilon * operand->params.scale) - operand->params.zero_point;
    const int epsilon_val = -operand->params.zero_point + epsilon_quantized;
    const int shifted_variance_val = variance_val * (1 << data->left_shift);
    const int shifted_epsilon_val = epsilon_val * (1 << data->left_shift);
    const int scaled_variance_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_variance_val, data->input_multiplier, data->input_shift);
    const int scaled_epsilon_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_epsilon_val, data->input_multiplier, data->input_shift);
    const int32_t raw_add_output = scaled_variance_val + scaled_epsilon_val;
    const int raw_addition_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_add_output, data->output_multiplier, data->output_shift) +
        operand->params.zero_point;
    var_eps_buffer[i%NumElements(var_eps)] = raw_addition_output;
    float input_sqrt = operand->params.scale *
                       (raw_addition_output - operand->params.zero_point);
    float stddev_deq = std::sqrt(input_sqrt);
    int stddev = static_cast<int>(quantize_value<DataType>(
        stddev_deq, operand->params.scale, operand->params.zero_point));
    stddev_buffer[i] =(quantize_value<DataType>(
        stddev_deq, operand->params.scale, operand->params.zero_point));

  
    TFLITE_DCHECK_NE(stddev - operand->params.zero_point, 0);
    int input2_val = stddev - operand->params.zero_point;
    int input1_val = raw_centered_output - operand->params.zero_point;
    if (input2_val < 0) {
      // Invert signs to avoid a negative input2_val as input2_inv needs to be
      // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
      input1_val = -input1_val;
      input2_val = -input2_val;
    }
    int recip_shift;

    const int32_t input2_inv = GetReciprocal(input2_val, 31, &recip_shift);
    const int headroom = CountLeadingSignBits(input1_val);
    const int32_t unscaled_quotient =
        MultiplyByQuantizedMultiplierGreaterThanOne(input1_val, input2_inv,
                                                    headroom);
    const int total_shift = data->div_shift - recip_shift - headroom;
    int32_t unclamped_result;
    if (std::abs(total_shift) > 31) {
      unclamped_result = operand->params.zero_point +
                         MultiplyByQuantizedMultiplierGreaterThanOne(
                             unscaled_quotient, data->div_multiplier, total_shift);
    } else {
      unclamped_result = operand->params.zero_point +
                         MultiplyByQuantizedMultiplierSmallerThanOneExp(
                             unscaled_quotient, data->div_multiplier, total_shift);
    }
    const int32_t clamped_div_output = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 unclamped_result));

    int operand_size = NumElements(operand);
    int feature_size = GetTensorShape(operand).Dims(feature_index);
    DataType elements_per_feature =(quantize_value<DataType>(
        float(operand_size/feature_size), operand->params.scale, operand->params.zero_point));
    elements_per_feature_buffer[i%NumElements(elements_per_feature_tensor)]=elements_per_feature;
    // need to check if multipilier shift

    // i1
    //  int32_t mul_multiplier;
    //      int mul_shift;
    const double real_mul_multiplier = operand->params.scale;
    QuantizeMultiplier(real_mul_multiplier, &data->mul_multiplier,
                       &data->mul_shift);
    int32_t raw_output = (-operand->params.zero_point + grad_output_buffer[i]) *
                         (elements_per_feature - operand->params.zero_point);
    int mul_final_output =
        MultiplyByQuantizedMultiplier(raw_output, data->mul_multiplier, data->mul_shift) +
        operand->params.scale;
    const int clamped_mul_output = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 mul_final_output));
    i1_buffer[i]= clamped_mul_output;
    std::cout<<"i1 "<<int(i1_buffer[i])<<std::endl;
    // std::cout << "mul final out" << mul_final_output << std::endl;
    // std::cout << "clamp mul out" << clamped_mul_output << std::endl;
    // std::cout << "operand input " << operand->params.scale << std::endl;
    // std::cout << "zero point " << operand->params.zero_point << std::endl;

    //grad_scale intermediate 
    int32_t raw_grad_scale_intermediate = (clamped_div_output -operand->params.zero_point)*(grad_output_buffer[i]-operand->params.zero_point);
    int grad_scale_intermediate_output =
        MultiplyByQuantizedMultiplier(raw_grad_scale_intermediate, data->mul_multiplier, data->mul_shift) +
        operand->params.scale;
    grad_scale_intermediate_buffer[i] = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 grad_scale_intermediate_output));
    std::cout<<"grad scale intermediate: "<<int(grad_scale_intermediate_buffer[i])<<std::endl;



    // i2  -> is the same as grad_offset so we will use that value for i2
    //  TF_LITE_ENSURE_OK(context,tflite::stablehlo_batch_norm_training::reference::ComputeQuantizedSum<DataType>(context,
    //  node,
    //                                   grad_output,
    //                                   feature_index,
    //                                   i2));
    // i3
    int raw_i3_intermediate =
        (grad_output_buffer[i] - operand->params.zero_point) *
        (raw_centered_output - operand->params.zero_point);
    int i3_intermediate_value =
        MultiplyByQuantizedMultiplier(raw_i3_intermediate, data->mul_multiplier,
                                      data->mul_shift) +
        operand->params.scale;
    i3_intermediate_buffer[i] = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 i3_intermediate_value));
      std::cout<<"i3 intermediate: "<<i3_intermediate_value<<" clamped_i3_intermediate "<<int(i3_intermediate_buffer[i])<<std::endl;
  }
  TF_LITE_ENSURE_OK(
      context,
      tflite::stablehlo_batch_norm_training::reference::ComputeQuantizedSum<
          DataType>(context, node, i3_intermediate, feature_index, i3));
  for(int i=0;i<NumElements(i3);++i){
    std::cout<<"i3 val: "<<int(i3_buffer[i])<<std::endl;
  }
  //i4,i5 and i6 calc 
  for(int i=0;i<NumElements(i4);++i){
    int raw_i4 = (i3_buffer[i%NumElements(i3)]-operand->params.zero_point)*(centered_operand_buffer[i] -operand->params.zero_point);
    int i4_value =
        MultiplyByQuantizedMultiplier(raw_i4, data->mul_multiplier,
                                      data->mul_shift) +
        operand->params.scale;
    i4_buffer[i] = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 i4_value));
    std::cout<<"i4 val "<<i4_value<<"final "<<int(i4_buffer[i])<<std::endl;
    //i5 calc
    TFLITE_DCHECK_NE(var_eps_buffer[i%NumElements(var_eps)] - operand->params.zero_point, 0);
    int input2_val = var_eps_buffer[i%NumElements(var_eps)] - operand->params.zero_point;
    int input1_val = i4_buffer[i] - operand->params.zero_point;
    if (input2_val < 0) {
      // Invert signs to avoid a negative input2_val as input2_inv needs to be
      // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
      input1_val = -input1_val;
      input2_val = -input2_val;
    }
    int recip_shift;

    const int32_t input2_inv = GetReciprocal(input2_val, 31, &recip_shift);
    const int headroom = CountLeadingSignBits(input1_val);
    const int32_t unscaled_quotient =
        MultiplyByQuantizedMultiplierGreaterThanOne(input1_val, input2_inv,
                                                    headroom);
    const int total_shift = data->div_shift - recip_shift - headroom;
    int32_t unclamped_result;
    if (std::abs(total_shift) > 31) {
      unclamped_result = operand->params.zero_point +
                         MultiplyByQuantizedMultiplierGreaterThanOne(
                             unscaled_quotient, data->div_multiplier, total_shift);
    } else {
      unclamped_result = operand->params.zero_point +
                         MultiplyByQuantizedMultiplierSmallerThanOneExp(
                             unscaled_quotient, data->div_multiplier, total_shift);
    }
    const int32_t clamped_div_output = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 unclamped_result));

    i5_buffer[i] = clamped_div_output;

    //i6 calc
    const int i2_val = -operand->params.zero_point + grad_offset_buffer[i];
    const int i1_val = -operand->params.zero_point + i1_buffer[i];
    const int shifted_i1_val = i1_val * (1 << data->left_shift);
    const int shifted_i2_val = i2_val * (1 << data->left_shift);
    const int scaled_i1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_i1_val, data->input_multiplier, data->input_shift);
    const int scaled_i2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_i2_val, data->input_multiplier, data->input_shift);
    const int32_t raw_sub_output = scaled_i1_val - scaled_i2_val;
    const int raw_subtraction_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sub_output, data->output_multiplier, data->output_shift) +
        operand->params.zero_point;
   
   const int shifted_i6_intermediate = raw_sub_output * (1<<data->left_shift);
   const int shifted_i5_val = (clamped_div_output -operand->params.zero_point) * (1<<data->left_shift);
  const int scaled_i6_intermediate =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_i6_intermediate, data->input_multiplier, data->input_shift);
  const int scaled_i5_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_i5_val, data->input_multiplier, data->input_shift);
  const int32_t raw_i6 = scaled_i6_intermediate - scaled_i5_val;
    const int i6_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_i6, data->output_multiplier, data->output_shift) +
        operand->params.zero_point;
  i6_buffer[i] =static_cast<DataType>(i6_output);

  }

  //grad operand calculation
  for(int i=0;i<NumElements(grad_operand);++i){
    // scale/stddev
    TFLITE_DCHECK_NE(stddev_buffer[i] - operand->params.zero_point, 0);
    int input2_val = stddev_buffer[i] - operand->params.zero_point;
    int input1_val = scale_data[i%NumElements(scale)] - operand->params.zero_point;
    if (input2_val < 0) {
      // Invert signs to avoid a negative input2_val as input2_inv needs to be
      // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
      input1_val = -input1_val;
      input2_val = -input2_val;
    }
    int recip_shift;

    const int32_t input2_inv = GetReciprocal(input2_val, 31, &recip_shift);
    const int headroom = CountLeadingSignBits(input1_val);
    const int32_t unscaled_quotient =
        MultiplyByQuantizedMultiplierGreaterThanOne(input1_val, input2_inv,
                                                    headroom);
    const int total_shift = data->div_shift - recip_shift - headroom;
    int32_t unclamped_result;
    if (std::abs(total_shift) > 31) {
      unclamped_result = operand->params.zero_point +
                         MultiplyByQuantizedMultiplierGreaterThanOne(
                             unscaled_quotient, data->div_multiplier, total_shift);
    } else {
      unclamped_result = operand->params.zero_point +
                         MultiplyByQuantizedMultiplierSmallerThanOneExp(
                             unscaled_quotient, data->div_multiplier, total_shift);
    }
    const int32_t clamped_div_output = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 unclamped_result));
    
    // (scale/stddev)/elements per feature
    TFLITE_DCHECK_NE(elements_per_feature_buffer[0] - operand->params.zero_point, 0);
    int input4_val = elements_per_feature_buffer[0] - operand->params.zero_point;
    int input3_val = clamped_div_output - operand->params.zero_point;
    if (input4_val < 0) {
      // Invert signs to avoid a negative input2_val as input2_inv needs to be
      // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
      input3_val = -input3_val;
      input4_val = -input4_val;
    }
    int recip_shift_1;

    const int32_t input4_inv = GetReciprocal(input4_val, 31, &recip_shift_1);
    const int headroom1 = CountLeadingSignBits(input3_val);
    const int32_t unscaled_quotient1 =
        MultiplyByQuantizedMultiplierGreaterThanOne(input3_val, input4_inv,
                                                    headroom1);
    const int total_shift1 = data->div_shift - recip_shift_1 - headroom1;
    std::cout<<"total shift 1 "<<total_shift1<<std::endl;
    int32_t unclamped_result1;
    if (std::abs(total_shift1) > 31) {
      unclamped_result1 = operand->params.zero_point +
                         MultiplyByQuantizedMultiplierGreaterThanOne(
                             unscaled_quotient1, data->div_multiplier, total_shift1);
    } else {
      unclamped_result1 = operand->params.zero_point +
                         MultiplyByQuantizedMultiplierSmallerThanOneExp(
                             unscaled_quotient1, data->div_multiplier, total_shift1);
    }
    const int32_t clamped_div_output1 = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 unclamped_result1));
    
    // i6 * ((scale/stddev)/elements per feature)
    int32_t raw_output = (-operand->params.zero_point + i6_buffer[i]) *
                         (clamped_div_output1- operand->params.zero_point);
    int mul_final_output =
        MultiplyByQuantizedMultiplier(raw_output, data->mul_multiplier, data->mul_shift) +
        operand->params.scale;
    const int clamped_mul_output = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 mul_final_output));
    grad_operand_buffer[i]= clamped_mul_output;


  }

  // grad scale calculation
  TF_LITE_ENSURE_OK(
      context,
      tflite::stablehlo_batch_norm_training::reference::ComputeQuantizedSum<
          DataType>(context, node,
                    grad_scale_intermediate /* normalised operand * grad_output */,
                    feature_index, grad_scale));


  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kOperandTensor, &operand));

  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kScaleTensor, &scale));

  const TfLiteTensor* mean;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kMeanTensor, &mean));

  const TfLiteTensor* variance;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kVarianceTensor, &variance));

  const TfLiteTensor* grad_output;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, OpData::kGradOutputTensor, &grad_output));

  TfLiteTensor* grad_operand;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOperandTensor,
                             &grad_operand));

  TfLiteTensor* grad_scale;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, OpData::kOutputGradScaleTensor,
                                  &grad_scale));
  TfLiteTensor* grad_offset;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOffsetTensor,
                             &grad_offset));

  switch (operand->type) {
    case kTfLiteFloat32: {
      return EvalImpl<float>(context, node, operand, scale, mean, variance,
                             grad_output, grad_operand, grad_scale,
                             grad_offset);
    }
    case kTfLiteFloat16: {
      return EvalImpl<Eigen::half>(context, node, operand, scale, mean,
                                   variance, grad_output, grad_operand,
                                   grad_scale, grad_offset);
    }
    case kTfLiteBFloat16: {
      return EvalImpl<Eigen::bfloat16>(context, node, operand, scale, mean,
                                       variance, grad_output, grad_operand,
                                       grad_scale, grad_offset);
    }
    case kTfLiteInt8: {
      return EvalQuantImp<int8_t>(context, node, operand, scale, mean, variance,
                                  grad_output, grad_operand, grad_scale,
                                  grad_offset);
    }
    case kTfLiteInt16: {
      return EvalQuantImp<int16_t>(context, node, operand, scale, mean,
                                   variance, grad_output, grad_operand,
                                   grad_scale, grad_offset);
    }
    default: {
      TF_LITE_KERNEL_LOG(
          context, "Type '%s' is not supported by stablehlo.batch_norm_grad.",
          TfLiteTypeGetName(operand->type));
      return kTfLiteError;
    }
  }
}
}  // namespace stablehlo_batch_norm_grad

TfLiteRegistration* Register_STABLEHLO_BATCH_NORM_GRAD() {
  static TfLiteRegistration r = {
      stablehlo_batch_norm_grad::Init, stablehlo_batch_norm_grad::Free,
      stablehlo_batch_norm_grad::Prepare, stablehlo_batch_norm_grad::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
