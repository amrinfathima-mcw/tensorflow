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

#ifndef TENSORFLOW_LITE_KERNELS_STABLEHLO_BATCH_NORM_TRAINING_H_
#define TENSORFLOW_LITE_KERNELS_STABLEHLO_BATCH_NORM_TRAINING_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "kernel_util.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/optimized/reduce.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace stablehlo_batch_norm_training {
namespace reference {

constexpr int kMaxReduceRank = 8;

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
  TF_LITE_ENSURE(context,
                 reference_ops::ReduceGeneric<DataType>(
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
  // }
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus ComputeQuantizedVariance(TfLiteContext* context, TfLiteNode* node,
                                      const TfLiteTensor* operand,
                                      int64_t feature_index,
                                      TfLiteTensor* batch_mean,
                                      TfLiteTensor* batch_var,
                                      TfLiteTensor* centered_operand) {
  TF_LITE_ENSURE_STATUS(ComputeQuantizedMean<DataType>(
      context, node, operand, feature_index, batch_mean));

  DataType* mean_data = GetTensorData<DataType>(batch_mean);
  const int operand_rank = operand->dims->size;
  std::vector<int> broadcast_shape(operand_rank, 1);
  broadcast_shape[feature_index] = operand->dims->data[feature_index];

  const DataType* operand_data = GetTensorData<DataType>(operand);
  DataType* centered_operand_data = GetTensorData<DataType>(centered_operand);
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
    const int scaled_mean_val = MultiplyByQuantizedMultiplierSmallerThanOneExp(
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
    int mul_output = MultiplyByQuantizedMultiplier(centered_operand_square,
                                                   mul_multiplier, mul_shift) +
                     operand->params.scale;
    const int clamped_output = std::min(
        static_cast<int>(std::numeric_limits<DataType>::max()),
        std::max(static_cast<int>(std::numeric_limits<DataType>::min()),
                 mul_output));
    centered_operand_data[i] = static_cast<DataType>(clamped_output);
  }
  return ComputeQuantizedMean<DataType>(context, node, centered_operand,
                                        feature_index, batch_var);
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
  for (int64_t i = 0; i < NumElements(operand); ++i) {
    centered_operand_data[i] =
        operand_data[i] - mean_data[i % broadcast_shape[feature_index]];
    centered_operand_data[i] *= centered_operand_data[i];
  }
  return ComputeMean<DataType>(context, node, centered_operand, feature_index,
                               batch_var);
}

template <typename DataType>
TfLiteStatus ComputeQuantizedSum(TfLiteContext* context, TfLiteNode* node,
                                  const TfLiteTensor* operand,
                                  int64_t feature_index,
                                  TfLiteTensor* batch_mean) {
  const int operand_rank = operand->dims->size;
  std::vector<int> dimarray;
  std::cout<<"operand rank "<<operand_rank<<std::endl;
  std::cout<<"feature index "<<feature_index<<std::endl;
  for (int i = 0; i < operand_rank; ++i) {
    if (i != feature_index) {
      std::cout<<"i "<<i<<std::endl;
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
                   false, temp_index, resolved_axis, temp_sum.data(), true));
  DataType* batch_mean_buffer= GetTensorData<DataType>(batch_mean);
  // std::cout<<"operand params and scale "<<operand->params.scale<<"   "<<operand->params.zero_point<<std::endl;
  // TF_LITE_ENSURE(context,
  //       optimized_ops::QuantizedMeanOrSum(
  //           GetTensorData<DataType>(operand),
  //           -1, float(0.996078),
  //           operand->dims->data, operand->dims->size,
  //           GetTensorData<DataType>(batch_mean),
  //           -1,
  //           float(0.996078), batch_mean->dims->data,
  //           batch_mean->dims->size, dimarray.data(),
  //           dimarray.size(), false,
  //           temp_index, resolved_axis,
  //           temp_sum.data(), true));
  for(int i=0;i<NumElements(batch_mean);++i){
    std::cout<<"batch sum  "<<int(batch_mean_buffer[i])<<std::endl;
  }
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus ComputeSum(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* operand, const int64_t feature_index,
                        TfLiteTensor* batch_sum) {
  const int operand_rank = operand->dims->size;
  std::vector<int> dimarray;
  for (int i = 0; i < operand_rank; ++i) {
    if (i != feature_index) {
      dimarray.push_back(i);
    }
  }
  int resolved_axis[kMaxReduceRank];
  int temp_index[kMaxReduceRank];
  TF_LITE_ENSURE(context,
                 reference_ops::ReduceGeneric<DataType>(
                     GetTensorData<DataType>(operand), operand->dims->data,
                     operand->dims->size, GetTensorData<DataType>(batch_sum),
                     batch_sum->dims->data, batch_sum->dims->size,
                     dimarray.data(), dimarray.size(), false, temp_index,
                     resolved_axis, static_cast<DataType>(0),
                     [](const DataType current, const DataType in) -> DataType {
                       return in + current;
                     }));
  DataType* batch_sum_buffer = GetTensorData<DataType>(batch_sum);

  return kTfLiteOk;
}

}  // namespace reference
}  // namespace reduce_window
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_STABLEHLO_BATCH_NORM_TRAINING_H_
