/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <stddef.h>
#include <stdint.h>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <iostream>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pow {
namespace {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// Op data for pow op.
struct OpData {
  bool requires_broadcast;
  int32_t output_multiplier;
  int output_shift;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);

  const TfLiteType type = input1->type;
  // if (type != kTfLiteInt32 && type != kTfLiteFloat32) {
  //   TF_LITE_KERNEL_LOG(context, "Unsupported data type %s.",
  //                      TfLiteTypeGetName(type));
  //   return kTfLiteError;
  // }
  output->type = type;

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

template <typename T>
void PowImpl(const TfLiteTensor* input1, const TfLiteTensor* input2,
             TfLiteTensor* output, bool requires_broadcast) {
  if (requires_broadcast) {
    optimized_ops::BroadcastPow4D(
        GetTensorShape(input1), GetTensorData<T>(input1),
        GetTensorShape(input2), GetTensorData<T>(input2),
        GetTensorShape(output), GetTensorData<T>(output));
  } else {
    reference_ops::Pow(GetTensorShape(input1), GetTensorData<T>(input1),
                       GetTensorShape(input2), GetTensorData<T>(input2),
                       GetTensorShape(output), GetTensorData<T>(output));
  }
}

// template <typename T>
// void PowQuantizedImpl(const TfLiteTensor* input1, const TfLiteTensor* input2,
//              TfLiteTensor* output, bool requires_broadcast) {
//               double real_multiplier =
    
//   if (requires_broadcast) {
//     optimized_ops::BroadcastPow4D(
//         GetTensorShape(input1), GetTensorData<T>(input1),
//         GetTensorShape(input2), GetTensorData<T>(input2),
//         GetTensorShape(output), GetTensorData<T>(output));
//   } else {
//     reference_ops::Pow(GetTensorShape(input1), GetTensorData<T>(input1),
//                        GetTensorShape(input2), GetTensorData<T>(input2),
//                        GetTensorShape(output), GetTensorData<T>(output));
//   }
//}
inline int64_t ModularExponentiation(int64_t base, int32_t exponent, int64_t modulus) {
    int64_t result = 1;
    base = base % modulus; // Handle base larger than modulus
   std::cout<<"base at start of modulo "<<base<<std::endl;
    while (exponent > 0) {
        // If exponent is odd, multiply the result with the base
           std::cout<<"base in loop "<<base<<std::endl;
        if (exponent % 2 == 1) {
            result = (result * static_cast<int64_t>(base)) % int64_t(modulus);
            std::cout<<"params before intermediate result "<<result<<" "<<base<<" "<<modulus<<std::endl;
             std::cout << "Intermediate result: " << result << std::endl;
        }
        // Square the base
        base = (base * base) % modulus;
        exponent >>= 1; // Divide exponent by 2
                std::cout << "Base after squaring: " << base << ", exponent: " << exponent << std::endl;
    }

    return result;
}

// Quantizes the result from floating-point to int16
template <typename T>
inline T QuantizedPower(T base,T exponent, int32_t output_multiplier, int32_t output_shift) {
    const int64_t modulus = 4294967296; // Example modulus for int16 range (2^16)
    
    std::cout<<" base and exp from quantpow "<<base<<" "<<exponent<<std::endl;
    int64_t result = ModularExponentiation(static_cast<int64_t>(base), static_cast<int32_t>(exponent), modulus);
    std::cout<<"result "<<result<<std::endl;

    // Apply the output multiplier and shift
    int64_t scaled_result = static_cast<int64_t>(result) * output_multiplier;
    scaled_result >>= output_shift;

    // Clamp to the quantized range
    const int32_t min_val = std::numeric_limits<int16_t>::min();
    const int32_t max_val = std::numeric_limits<int16_t>::max();
    return static_cast<int16_t>(std::min(std::max(scaled_result, static_cast<int64_t>(min_val)), static_cast<int64_t>(max_val)));
}

template <typename T>
inline void QuantizedPow(const TfLiteTensor* input1,const TfLiteTensor* input2,TfLiteTensor* output,OpData* data) {
    const int flat_size = MatchingFlatSize(GetTensorShape(input1), GetTensorShape(input2), GetTensorShape(output));
    const T* input1_data=GetTensorData<T>(input1);
    const T* input2_data=GetTensorData<T>(input2);
    for(int i=0;i<NumElements(input1) ;++i){
      std::cout<<"inputs "<<input1_data[i]<<"  "<<input2_data[i]<<std::endl;
    }
    T* output_data=GetTensorData<T>(output);
    for (int i = 0; i < flat_size; ++i) {
        T base = input1_data[i];
        T exponent =input2_data[i];
        double real_exponent = (input2_data[i]-input2->params.zero_point)*input2->params.scale;
        double real_multiplier=std::pow(input1->params.scale,real_exponent) / output->params.scale;
        int32_t output_multiplier;
        int32_t output_shift;
        std::cout<<"scale "<<input1->params.scale<<input1->params.zero_point<<std::endl;
        std::cout<<"real_multiplier "<<real_multiplier<<"\n\n";
        QuantizeMultiplier(input1->params.scale,&output_multiplier,&output_shift);
         std::cout<<"quantizemultiplier "<<output_multiplier<<"   "<<output_shift<<"\n\n ";
        QuantizeMultiplier(real_multiplier,&output_multiplier,
                       &output_shift);
                       std::cout<<"quantizemultiplier "<<output_multiplier<<"   "<<output_shift<<"\n\n ";
        output_data[i]=QuantizedPower(base,exponent, output_multiplier, output_shift);

    }
    return;
}
// inline void QuantizedPow(const TfLiteTensor* input1,const TfLiteTensor* input2,TfLiteTensor* output,OpData* data) {
//     const int flat_size = MatchingFlatSize(GetTensorShape(input1), GetTensorShape(input2), GetTensorShape(output));
//     const T* input1_data=GetTensorData<T>(input1);
//     const T* input2_data=GetTensorData<T>(input2);
//     T* output_data=GetTensorData<T>(output);
//     for (int i = 0; i < flat_size; ++i) {
    //     T base = input1_data[i];
    //     double real_exponent = (input2_data[i]-input2->params.zero_point)*input2->params.scale;
    // double real_multiplier=std::pow(input1->params.scale,real_exponent) / output->params.scale;
    // QuantizeMultiplier(real_multiplier, &data->output_multiplier,
    //                    &data->output_shift);  
//         int64_t result = 1;
//         int exponent = static_cast<int>(std::round(real_exponent));
//         std::cout<<"exponent "<<exponent<<std::endl;
//         for (int exp = 0; exp < exponent; ++exp) {
//             // Multiply the base with result using fixed-point arithmetic
//             //std::cout<<"base "<<base<<std::endl;
//             if(exp==0){
//             std::cout<<"result "<<result<<std::endl;
//            std::cout<<"operands "<<result<<" "<<static_cast<int64_t>(base)<<std::endl;
//             }
//             int64_t temp_result = static_cast<int64_t>(result) * static_cast<int64_t>(base);
//             // Apply the multiplier and shift to scale the result
//             std::cout<<"temp_result "<<temp_result<<std::endl;
//             temp_result = (temp_result * data->output_multiplier) >> data->output_shift;

//             // Clamp to the quantized range
//             result = static_cast<T>(std::min(std::max(temp_result, static_cast<int64_t>(std::numeric_limits<T>::min())),
//                                                static_cast<int64_t>(std::numeric_limits<T>::max())));
//         }

//         // Store the final result
//         output_data[i] = result;
//     }
// }

// template <typename T>
// inline void QuantizedPow(const TfLiteTensor* input1,const TfLiteTensor* input2,TfLiteTensor* output,OpData* data) {
//     const int flat_size = MatchingFlatSize(GetTensorShape(input1), GetTensorShape(input2), GetTensorShape(output));
//     const T* input1_data=GetTensorData<T>(input1);
//     const T* input2_data=GetTensorData<T>(input2);
//     T* output_data=GetTensorData<T>(output);
//     for (int i = 0; i < flat_size; ++i) {
//         T base = input1_data[i];
//         T exponent = input2_data[i];
//         float ln_base = ln_lookup[base];
//         double real_multiplier = exponent* ln_base;
//         int32_t output_multiplier;
//         int32_t output_shift;
//         QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
//         int32_t result = (base * output_multiplier) >> output_shift;
//         // Clamp to valid range
//         result = std::max(static_cast<int32_t>(std::numeric_limits<T>::min(), std::min(std::numeric_limits<T>::max(), result)));
//         // Add the zero point for output quantization
//         output[i] = static_cast<T>(result + output->params.zero_point);
//         output_data[i] = result;
//     }
// }


TfLiteStatus CheckValue(TfLiteContext* context, const TfLiteTensor* input) {
  const int64_t num_elements = NumElements(input);
  const int32_t* data = GetTensorData<int32_t>(input);
  for (int i = 0; i < num_elements; ++i) {
    if (data[i] < 0) {
      TF_LITE_KERNEL_LOG(context,
                         "POW does not support negative value for int32.");
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (output->type) {
    case kTfLiteInt32: {
      // TensorFlow does not support negative for int32.
      TF_LITE_ENSURE_OK(context, CheckValue(context, input2));
      PowImpl<int32_t>(input1, input2, output, data->requires_broadcast);
      break;
    }
    // case kTfLiteInt8:
    //     reference_integer_ops::LookupTable(
    //         GetTensorData<int8_t>(op_context.input),
    //         NumElements(op_context.input), data->lut_int8,
    //         GetTensorData<int8_t>(op_context.output));
    //     break;
    // case kTfLiteInt16:
    //     reference_integer_ops::LookupTable(
    //         GetTensorData<int16_t>(input1),
    //         NumElements(input1), data->lut_int16,
    //         GetTensorData<int16_t>(output));
    //     break;
    case kTfLiteInt16: {
      // TensorFlow does not support negative for int32.
      std::cout<<"entered the case int16"<<std::endl;
    //  TF_LITE_ENSURE_OK(context, CheckValue(context, input2));
      QuantizedPow<int16_t>(input1, input2, output, data);
      break;
    }
      case kTfLiteInt8: {
      // TensorFlow does not support negative for int32.
      std::cout<<"entered the case int16"<<std::endl;
    //  TF_LITE_ENSURE_OK(context, CheckValue(context, input2));
      QuantizedPow<int8_t>(input1, input2, output, data);
      break;
    }
    case kTfLiteFloat32: {
      PowImpl<float>(input1, input2, output, data->requires_broadcast);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Unsupported data type: %d", output->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace pow

TfLiteRegistration* Register_POW() {
  static TfLiteRegistration r = {pow::Init, pow::Free, pow::Prepare, pow::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
