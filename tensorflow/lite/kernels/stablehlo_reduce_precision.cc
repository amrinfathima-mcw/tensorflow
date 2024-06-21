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
#include <cmath>
#include <limits>

#include "Eigen/Core"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tsl/platform/bfloat16.h"
namespace tensorflow {
typedef tsl::bfloat16 bfloat16;
}

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_reduce_precision {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename DataType>
struct FloatingPointTraits {
  static constexpr int mantissa_bits =
      std::numeric_limits<DataType>::digits - 1;
  static constexpr int maxExponent =
      std::numeric_limits<DataType>::max_exponent;
  static constexpr int minExponent =
      std::numeric_limits<DataType>::min_exponent;
  static constexpr int exponent_bits =
      (__builtin_clz(static_cast<unsigned int>(maxExponent - minExponent + 1)) ^
       31) +
      1;
};
template <typename DataType>
DataType reducePrecision(DataType value, int32_t exponentBits,
                         int32_t mantissaBits);

template <typename DataType>
TfLiteStatus ReducePrecisionOp(const TfLiteTensor* operand,
                               int32_t exponentBits, int32_t mantissaBits,
                               TfLiteTensor* result) {
  for (int i = 0; i < NumElements(result); ++i) {
    DataType value = GetTensorData<DataType>(operand)[i];
    DataType reduced_value = reducePrecision(value, exponentBits, mantissaBits);
    GetTensorData<DataType>(result)[i] = reduced_value;
  }
  return TfLiteStatus::kTfLiteOk;
}

template <typename DataType>
DataType reducePrecision(DataType value, int32_t exponentBits,
                         int32_t mantissaBits) {
  uint32_t intVal = *reinterpret_cast<uint32_t*>(&value);

  const int32_t srcMantissaBits = FloatingPointTraits<DataType>::mantissa_bits;
  const int32_t srcExponentBits = FloatingPointTraits<DataType>::exponent_bits;

  if (mantissaBits < srcMantissaBits) {
    uint32_t lastMantissaBitMask = 1ull << (srcMantissaBits - mantissaBits);
    uint32_t baseRoundingBias = (lastMantissaBitMask >> 1) - 1;
    uint32_t xLastMantissaBit =
        (intVal & lastMantissaBitMask) >> (srcMantissaBits - mantissaBits);
    uint32_t xRoundingBias = xLastMantissaBit + baseRoundingBias;

    uint32_t truncationMask = ~(lastMantissaBitMask - 1);
    intVal = intVal + xRoundingBias;
    intVal = intVal & truncationMask;
  }

  if (exponentBits < srcExponentBits) {
    uint32_t signBitMask = 1ull << 31;
    uint32_t expBitsMask = ((1ull << srcExponentBits) - 1) << srcMantissaBits;
    uint32_t exponentBias = (1ull << (srcExponentBits - 1)) - 1;
    uint32_t reducedExponentBias = (1ull << (exponentBits - 1)) - 1;
    uint32_t reducedMaxExponent = exponentBias + reducedExponentBias;
    uint32_t reducedMinExponent = exponentBias - reducedExponentBias;

    uint32_t xExponent = intVal & expBitsMask;
    bool xOverflows = xExponent > (reducedMaxExponent << srcMantissaBits);
    bool xUnderflows = xExponent <= (reducedMinExponent << srcMantissaBits);

    uint32_t xSignedZero = intVal & signBitMask;
    uint32_t xSignedInf = xSignedZero | expBitsMask;

    intVal = xOverflows ? xSignedInf : intVal;
    intVal = xUnderflows ? xSignedZero : intVal;
  }
  DataType reducedResult;
  std::memcpy(&reducedResult, &intVal, sizeof(DataType));
  if (std::isnan(value)) {
    reducedResult =
        mantissaBits > 0 ? value : std::numeric_limits<DataType>::infinity();
  }
  return reducedResult;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  int input_rank = input->dims->size;
  RuntimeShape input_shape = GetTensorShape(input);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  int result_rank = output->dims->size;
  RuntimeShape result_runtime_shape(result_rank, output->dims->data);
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteType data_type = input->type;
  const TfLiteStablehloReducePrecisionParams* data =
      reinterpret_cast<TfLiteStablehloReducePrecisionParams*>(
          node->builtin_data);
  int32_t exponentBits = data->exponent_bits;
  int32_t mantissaBits = data->mantissa_bits;
  if (data_type == kTfLiteFloat32) {
    return ReducePrecisionOp<float>(input, exponentBits, mantissaBits, output);
  } else if (data_type == kTfLiteFloat16) {
    return ReducePrecisionOp<Eigen::half>(input, exponentBits, mantissaBits,
                                          output);
  } else if (data_type == kTfLiteBFloat16) {
    return ReducePrecisionOp<tensorflow::bfloat16>(input, exponentBits,
                                                   mantissaBits, output);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(data_type));
    return TfLiteStatus::kTfLiteError;
  }
}

}  // namespace stablehlo_reduce_precision

TfLiteRegistration* Register_REDUCE_PRECISION() {
  static TfLiteRegistration r = {
      nullptr,
      nullptr,
      stablehlo_reduce_precision::Prepare,
      stablehlo_reduce_precision::Eval,
      /*profiling_string=*/nullptr,
      /*builtin_code=*/0,
      /*custom_name=*/nullptr,
      /*version=*/0,
      /*registration_external=*/nullptr,
      /*async_kernel=*/nullptr,
      kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpDataUnmodified};
  return &r;
}
TfLiteRegistration* Register_STABLEHLO_REDUCE_PRECISION() {
  // std::cout << "Register_REDUCE_PRECISION called" << std::endl;
  static TfLiteRegistration r = {nullptr, nullptr,
                                 stablehlo_reduce_precision::Prepare,
                                 stablehlo_reduce_precision::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
