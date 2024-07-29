#include "tensorflow/lite/core/subgraph.h"
#include <numeric>
#include <vector>
#include <iostream>
#include "Eigen/Core"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/tensor_slice_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_case {

struct OpData {
  std::vector<int> subgraph_indices;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const auto* params = reinterpret_cast<const TfLiteStablehloCaseParams*>(buffer);
  op_data->subgraph_indices.assign(params->subgraph_indices,
                                   params->subgraph_indices + params->num_branches);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, !op_data->subgraph_indices.empty());

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();

  for (int subgraph_idx : op_data->subgraph_indices) {
    TF_LITE_ENSURE(context, subgraph_idx < subgraphs->size());
  }

  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();

  // Get index input tensor
  const TfLiteTensor* index_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &index_tensor));
  TF_LITE_ENSURE_EQ(context, index_tensor->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumElements(index_tensor), 1);

  int32_t index_value = index_tensor->data.i32[0];
  if (index_value < 0 || index_value >= op_data->subgraph_indices.size()) {
    index_value = op_data->subgraph_indices.size() - 1; 
  }

  int selected_subgraph_index = op_data->subgraph_indices[index_value];
  TF_LITE_ENSURE(context, selected_subgraph_index < subgraphs->size());
  Subgraph* selected_subgraph = (*subgraphs)[selected_subgraph_index].get();

  TF_LITE_ENSURE_OK(context, selected_subgraph->Invoke());

  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteTensor* output_tensor;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output_tensor));

    TfLiteTensor* subgraph_output = selected_subgraph->tensor(selected_subgraph->outputs()[i]);
    TF_LITE_ENSURE_EQ(context, output_tensor->type, subgraph_output->type);
    std::memcpy(output_tensor->data.raw, subgraph_output->data.raw, subgraph_output->bytes);
  }

  return kTfLiteOk;
}

TfLiteRegistration* Register_STABLEHLO_CASE() {
  static TfLiteRegistration r = {stablehlo_case::Init, stablehlo_case::Free,
                                 stablehlo_case::Prepare, stablehlo_case::Eval};
  return &r;
}

}  // namespace stablehlo_case
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
