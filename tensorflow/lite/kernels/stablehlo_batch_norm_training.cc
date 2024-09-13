/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,5fg
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <limits>
#include <vector>

#include "Eigen/Core"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_batch_norm_training {
namespace {

constexpr int kInputTensor = 0;
constexpr int kScaleTensor = 1;
constexpr int kOffsetTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kBatchMeanTensor = 1;
constexpr int kBatchVarTensor = 2;
constexpr int kMaxReduceRank = 8;

struct OpData {
  int scratch_tensor_index;
};

TfLiteStatus GetOutputShape(TfLiteContext* context, TfLiteIntArray* input_dims,
                            const int input_num_dims, std::vector<int64_t> axis,
                            int64_t num_axis, TfLiteIntArray** output_shape) {
  if (input_num_dims == 0) {
    *output_shape = TfLiteIntArrayCreate(0);
    return kTfLiteOk;
  }
  // Calculates size of reducing axis.
  int64_t num_reduce_axis = num_axis;
  for (int64_t i = 0; i < num_axis; ++i) {
    int current = axis[i];
    if (current < 0) {
      current += input_num_dims;
    }
    TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims);
    for (int64_t j = 0; j < i; ++j) {
      int previous = axis[j];
      if (previous < 0) {
        previous += input_num_dims;
      }
      if (current == previous) {
        --num_reduce_axis;
        break;
      }
    }
    // Determines output dimensions.
    TfLiteIntArray* output_dims =
        TfLiteIntArrayCreate(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int64_t axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis) {
        output_dims->data[idx - num_skip_axis] = input_dims->data[idx];
      }
    }
    *output_shape = output_dims;
    return kTfLiteOk;
  }
}

template <typename DataType>
TfLiteStatus ComputeQuantizedMean(TfLiteContext* context, TfLiteNode* node,
                                  const TfLiteTensor* operand,
                                  int64_t feature_index,
                                  TfLiteTensor* batch_mean) {
  const int operand_rank = operand->dims->size;
  std::vector<int> dimarray;
  for (int i = 0; i < operand_rank; ++i) {
    if (i != feature_index) {
      dimarray.push_back(i);
    }
  }
  int resolved_axis[kMaxReduceRank] = {0};
  int temp_index[kMaxReduceRank] = {0};
  std::vector<int> temp_sum(dimarray.size(), 0);
  int32_t multiplier;
  int shift;
  QuantizeMultiplier(1.0, &multiplier, &shift);
  TF_LITE_ENSURE(
      context, reference_ops::QuantizedMeanOrSum(
                   GetTensorData<DataType>(operand), operand->params.zero_point,
                   operand->dims->data, operand->dims->size,
                   GetTensorData<DataType>(batch_mean), multiplier, shift,
                   operand->params.zero_point, batch_mean->dims->data,
                   batch_mean->dims->size, dimarray.data(), dimarray.size(),
                   false, temp_index, resolved_axis, temp_sum.data(), false));
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus ComputeMean(TfLiteContext* context, TfLiteNode* node,
                         const TfLiteTensor* operand, int64_t feature_index,
                         TfLiteTensor* batch_mean) {
  const int operand_rank = operand->dims->size;
  std::vector<int> dimarray;
  for (int i = 0; i < operand_rank; ++i) {
    if (i != feature_index) {
      dimarray.push_back(i);
    }
  }
  int resolved_axis[kMaxReduceRank];
  int temp_index[kMaxReduceRank];
  int temp_sum[NumElements(batch_mean)];
  if (batch_mean->type == kTfLiteInt8 || batch_mean->type == kTfLiteInt16) {
    ComputeQuantizedMean<DataType>(context, node, operand, feature_index,
                                   batch_mean);
  } else {
    TF_LITE_ENSURE(
        context, reference_ops::ReduceGeneric<DataType>(
                     GetTensorData<DataType>(operand), operand->dims->data,
                     operand->dims->size, GetTensorData<DataType>(batch_mean),
                     batch_mean->dims->data, batch_mean->dims->size,
                     dimarray.data(), dimarray.size(), false, temp_index,
                     resolved_axis, static_cast<DataType>(0),
                     [](const DataType current, const DataType in) -> DataType {
                       return in + current;
                     }));
    int64_t operand_size = 1;
    for (int i = 0; i < operand->dims->size; ++i) {
      operand_size *= operand->dims->data[i];
    }
    int64_t feature_dim = operand->dims->data[feature_index];
    int64_t divisor = operand_size / feature_dim;

    DataType* mean_data = GetTensorData<DataType>(batch_mean);
    for (int64_t i = 0; i < NumElements(batch_mean); ++i) {
      mean_data[i] = mean_data[i] / divisor;
    }
  }
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus ComputeVariance(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTensor* operand, int64_t feature_index,
                             TfLiteTensor* batch_mean, TfLiteTensor* batch_var,
                             TfLiteTensor* centered_operand) {
  TF_LITE_ENSURE_STATUS(
      ComputeMean<DataType>(context, node, operand, feature_index, batch_mean));

  DataType* mean_data = GetTensorData<DataType>(batch_mean);
  const int operand_rank = operand->dims->size;
  std::vector<int> broadcast_shape(operand_rank, 1);
  broadcast_shape[feature_index] = operand->dims->data[feature_index];

  const DataType* operand_data = GetTensorData<DataType>(operand);
  DataType* centered_operand_data = GetTensorData<DataType>(centered_operand);
  if (operand->type == kTfLiteInt8 || operand->type == kTfLiteInt16) {
    const int left_shift = (operand->type == kTfLiteInt16) ? 15 : 20;
    for (int64_t i = 0; i < NumElements(operand); ++i) {
      const double twice_max_input_scale =
          2 * std::max(operand->params.scale, operand->params.scale);
      const double real_input_multiplier =
          operand->params.scale / twice_max_input_scale;
      const double real_output_multiplier =
          twice_max_input_scale / ((1 << left_shift) * operand->params.scale);
      int32_t output_multiplier;
      int output_shift;
      int32_t input_multiplier;
      int input_shift;

      tflite::QuantizeMultiplierSmallerThanOneExp(
          real_input_multiplier, &input_multiplier, &input_shift);
      if (real_output_multiplier > 1) {
        tflite::QuantizeMultiplierGreaterThanOne(
            real_output_multiplier, &output_multiplier, &output_shift);
      } else {
        tflite::QuantizeMultiplierSmallerThanOneExp(
            real_output_multiplier, &output_multiplier, &output_shift);
      }
      const int operand_val = -operand->params.zero_point + operand_data[i];
      const int mean_val =
          -operand->params.zero_point + mean_data[i % NumElements(batch_mean)];
      const int shifted_operand_val = operand_val * (1 << left_shift);
      const int shifted_mean_val = mean_val * (1 << left_shift);
      const int scaled_operand_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_operand_val, input_multiplier, input_shift);
      const int scaled_mean_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_mean_val, input_multiplier, input_shift);
      const int raw_centered_val = scaled_operand_val - scaled_mean_val;
      const int raw_centered_output =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              raw_centered_val, output_multiplier, output_shift) +
          operand->params.zero_point;
      double real_multiplier_mul = operand->params.scale;
      int32_t mul_multiplier;
      int mul_shift;
      QuantizeMultiplier(real_multiplier_mul, &mul_multiplier, &mul_shift);

      int centered_operand_square =
          (raw_centered_output - operand->params.zero_point) *
          (raw_centered_output - operand->params.zero_point);
      int mul_output = MultiplyByQuantizedMultiplier(
                           centered_operand_square, mul_multiplier, mul_shift) +
                       operand->params.scale;
      const int clamped_output = std::min(
          static_cast<int>(std::numeric_limits<DataType>::max()),
          std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                   mul_output));
      centered_operand_data[i] = static_cast<DataType>(clamped_output);
    }
  } else {
    for (int64_t i = 0; i < NumElements(operand); ++i) {
      centered_operand_data[i] =
          operand_data[i] - mean_data[i % broadcast_shape[feature_index]];
      centered_operand_data[i] *= centered_operand_data[i];
    }
  }
  return ComputeMean<DataType>(context, node, centered_operand, feature_index,
                               batch_var);
}

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
TfLiteStatus BatchNormInference(
    TfLiteContext* context, TfLiteNode* node, const TfLiteTensor* operand,
    const TfLiteTensor* scale, const TfLiteTensor* offset,
    const TfLiteTensor* mean, const TfLiteTensor* variance, const float epsilon,
    const int64_t feature_index, TfLiteTensor* output) {
  
  const int operand_rank = operand->dims->size;

  const DataType* scale_data = GetTensorData<DataType>(scale);
  const DataType* offset_data = GetTensorData<DataType>(offset);
  const DataType* mean_data = GetTensorData<DataType>(mean);
  const DataType* variance_data = GetTensorData<DataType>(variance);
  const DataType* operand_data = GetTensorData<DataType>(operand);
  DataType* output_data = GetTensorData<DataType>(output);

  const int kMin = std::numeric_limits<DataType>::min();
  const int kMax = std::numeric_limits<DataType>::max();

  if (operand->type == kTfLiteInt8 || operand->type == kTfLiteInt16) {
    const int left_shift = (operand->type == kTfLiteInt16) ? 15 : 20;
    for (int64_t i = 0; i < NumElements(operand); ++i) {
      int64_t feature_index_value = i % operand->dims->data[feature_index];
      const double twice_max_input_scale =
          2 * std::max(operand->params.scale, operand->params.scale);
      const double real_input_multiplier =
          operand->params.scale / twice_max_input_scale;
      const double real_output_multiplier =
          twice_max_input_scale / ((1 << left_shift) * operand->params.scale);
      int32_t output_multiplier;
      int output_shift;
      int32_t input_multiplier;
      int input_shift;

      tflite::QuantizeMultiplierSmallerThanOneExp(
          real_input_multiplier, &input_multiplier, &input_shift);
      if (real_output_multiplier > 1) {
        tflite::QuantizeMultiplierGreaterThanOne(
            real_output_multiplier, &output_multiplier, &output_shift);
      } else {
        tflite::QuantizeMultiplierSmallerThanOneExp(
            real_output_multiplier, &output_multiplier, &output_shift);
      }
      const int operand_val = -operand->params.zero_point + operand_data[i];
      const int mean_val =
          -operand->params.zero_point + mean_data[i % NumElements(mean)];
      const int shifted_operand_val = operand_val * (1 << left_shift);
      const int shifted_mean_val = mean_val * (1 << left_shift);
      const int scaled_operand_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_operand_val, input_multiplier, input_shift);
      const int scaled_mean_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_mean_val, input_multiplier, input_shift);
      const int raw_centered_val = scaled_operand_val - scaled_mean_val;
      const int raw_centered_output =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              raw_centered_val, output_multiplier, output_shift) +
          operand->params.zero_point;

      const int variance_val = -operand->params.zero_point +
                               variance_data[i % NumElements(variance)];
      const int epsilon_quantized =
          (epsilon * operand->params.scale) - operand->params.zero_point;
      const int epsilon_val = -operand->params.zero_point + epsilon_quantized;
      const int shifted_variance_val = variance_val * (1 << left_shift);
      const int shifted_epsilon_val = epsilon_val * (1 << left_shift);
      const int scaled_variance_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_variance_val, input_multiplier, input_shift);
      const int scaled_epsilon_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_epsilon_val, input_multiplier, input_shift);
      const int32_t raw_add_output = scaled_variance_val + scaled_epsilon_val;
      const int raw_addition_output =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              raw_add_output, output_multiplier, output_shift) +
          operand->params.zero_point;
      float input_sqrt = operand->params.scale *
                         (raw_addition_output - operand->params.zero_point);
      float stddev_deq = std::sqrt(input_sqrt);
      int stddev = static_cast<int>(quantize_value<DataType>(
          stddev_deq, operand->params.scale, operand->params.zero_point));

      int32_t div_multiplier;
      int div_shift;
      const double real_div_multiplier =
          operand->params.scale /
          (operand->params.scale * operand->params.scale);
      QuantizeMultiplier(real_div_multiplier, &div_multiplier, &div_shift);
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
      const int total_shift = div_shift - recip_shift - headroom;
      int32_t unclamped_result;
      if (std::abs(total_shift) > 31) {
        unclamped_result = operand->params.zero_point +
                           MultiplyByQuantizedMultiplierGreaterThanOne(
                               unscaled_quotient, div_multiplier, total_shift);
      } else {
        unclamped_result = operand->params.zero_point +
                           MultiplyByQuantizedMultiplierSmallerThanOneExp(
                               unscaled_quotient, div_multiplier, total_shift);
      }
      const int32_t clamped_div_output = std::min(
          static_cast<int>(std::numeric_limits<DataType>::max()),
          std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                   unclamped_result));
      int32_t mul_multiplier;
      int mul_shift;
      const double real_mul_multiplier =operand->params.scale;
      QuantizeMultiplier(real_mul_multiplier, &mul_multiplier, &mul_shift);
      int32_t raw_output =
          (-operand->params.zero_point + scale_data[i % NumElements(scale)]) *
          (clamped_div_output - operand->params.zero_point);
      int mul_final_output =
          MultiplyByQuantizedMultiplier(raw_output, mul_multiplier, mul_shift) +
          operand->params.scale;
      const int clamped_mul_output = std::min(
          static_cast<int>(std::numeric_limits<DataType>::max()),
          std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                   mul_final_output));
      // Add offset to normalized_operand*scale
      const int offset_val =
          -operand->params.zero_point + offset_data[i % NumElements(offset)];
      const int shifted_mul_val = clamped_mul_output * (1 << left_shift);
      const int shifted_offset_val = offset_val * (1 << left_shift);
      const int scaled_mul_val = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_mul_val, input_multiplier, input_shift);
      const int scaled_offset_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_offset_val, input_multiplier, input_shift);
      const int raw_final_add_output = scaled_mul_val + scaled_offset_val;
      const int raw_final_addition_output =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              raw_final_add_output, output_multiplier, output_shift) +
          operand->params.zero_point;
      output_data[i] = static_cast<DataType>(raw_final_addition_output);
    }
  } else {
    for (int64_t i = 0; i < NumElements(operand); ++i) {
      int64_t feature_index_value = i % operand->dims->data[feature_index];
      DataType centered_value =
          operand_data[i] - mean_data[feature_index_value];
      DataType stddev = static_cast<DataType>(std::sqrt(
          static_cast<float>(variance_data[feature_index_value]) + epsilon));
      output_data[i] =
          scale_data[feature_index_value] * (centered_value / stddev) +
          offset_data[feature_index_value];
    }
  }
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                      const TfLiteTensor* operand, const TfLiteTensor* scale,
                      const TfLiteTensor* offset, TfLiteTensor* output,
                      TfLiteTensor* batch_mean, TfLiteTensor* batch_var,
                      const int64_t feature_index, const float epsilon) {
  TF_LITE_ENSURE_OK(
      context, ComputeVariance<DataType>(context, node, operand, feature_index,
                                         batch_mean, batch_var, output));

  TF_LITE_ENSURE_OK(
      context, BatchNormInference<DataType>(context, node, operand, scale,
                                            offset, batch_mean, batch_var,
                                            epsilon, feature_index, output));

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kScaleTensor, &scale));
  const TfLiteTensor* offset;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOffsetTensor, &offset));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteTensor* batch_mean;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kBatchMeanTensor, &batch_mean));
  TfLiteTensor* batch_var;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kBatchVarTensor, &batch_var));

  const TfLiteStablehloBatchNormTrainingParams* data =
      reinterpret_cast<TfLiteStablehloBatchNormTrainingParams*>(
          node->builtin_data);
  const int64_t feature_index = data->feature_index;

  const int input_rank = input->dims->size;
  std::vector<int64_t> axis;
  for (int64_t i = 0; i < input_rank; ++i) {
    if (i != feature_index) {
      axis.push_back(i);
    }
  }

  TfLiteIntArray* batch_mean_var_shape =nullptr;
  TF_LITE_ENSURE_OK(
      context, GetOutputShape(context, input->dims, input->dims->size, axis,
                              input_rank - 1, &batch_mean_var_shape));
  context->ResizeTensor(context, batch_mean, batch_mean_var_shape);
  TF_LITE_ENSURE_OK(
      context, GetOutputShape(context, input->dims, input->dims->size, axis,
                              input_rank - 1, &batch_mean_var_shape));
  context->ResizeTensor(context, batch_var, batch_mean_var_shape);

  TF_LITE_ENSURE(context,
                 feature_index >= 0 && feature_index < input->dims->size);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);
  TF_LITE_ENSURE_EQ(context, scale->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, scale->dims->data[0],
                    input->dims->data[feature_index]);
  TF_LITE_ENSURE_EQ(context, offset->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, offset->dims->data[0],
                    input->dims->data[feature_index]);
  TF_LITE_ENSURE_EQ(context, batch_mean->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, batch_mean->dims->data[0],
                    input->dims->data[feature_index]);
  TF_LITE_ENSURE_EQ(context, batch_var->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, batch_var->dims->data[0],
                    input->dims->data[feature_index]);

  TF_LITE_ENSURE_OK(
      context,
      context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims)));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor, &operand));
  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kScaleTensor, &scale));
  const TfLiteTensor* offset;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOffsetTensor, &offset));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteTensor* batch_mean;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kBatchMeanTensor, &batch_mean));
  TfLiteTensor* batch_var;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kBatchVarTensor, &batch_var));

  const TfLiteStablehloBatchNormTrainingParams* data =
      reinterpret_cast<TfLiteStablehloBatchNormTrainingParams*>(
          node->builtin_data);
  const int64_t feature_index = data->feature_index;
  const float epsilon = data->epsilon;

  if (operand->type == kTfLiteFloat32) {
    return EvalImpl<float>(context, node, operand, scale, offset, output,
                           batch_mean, batch_var, feature_index, epsilon);
  } else if (operand->type == kTfLiteFloat16) {
    return EvalImpl<Eigen::half>(context, node, operand, scale, offset, output,
                                 batch_mean, batch_var, feature_index, epsilon);
  } else if (operand->type == kTfLiteBFloat16) {
    return EvalImpl<Eigen::bfloat16>(context, node, operand, scale, offset,
                                     output, batch_mean, batch_var,
                                     feature_index, epsilon);
  } else if (operand->type == kTfLiteInt8) {
    return EvalImpl<int8_t>(context, node, operand, scale, offset, output,
                            batch_mean, batch_var, feature_index, epsilon);
  } else if (operand->type == kTfLiteInt16) {
    return EvalImpl<int16_t>(context, node, operand, scale, offset, output,
                             batch_mean, batch_var, feature_index, epsilon);
  } else {
    TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                       TfLiteTypeGetName(operand->type));
    return kTfLiteError;
  }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* node_data) {
  delete static_cast<OpData*>(node_data);
}

}  // namespace
}  // namespace stablehlo_batch_norm_training

TfLiteRegistration* Register_STABLEHLO_BATCH_NORM_TRAINING() {
  static TfLiteRegistration r = {stablehlo_batch_norm_training::Init,
                                 stablehlo_batch_norm_training::Free,
                                 stablehlo_batch_norm_training::Prepare,
                                 stablehlo_batch_norm_training::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
