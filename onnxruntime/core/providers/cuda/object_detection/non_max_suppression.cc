// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "non_max_suppression.h"
#include "non_max_suppression_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    NonMaxSuppression,
    kOnnxDomain,
    10,
    kCudaExecutionProvider,
    KernelDefBuilder(),
    cuda::NonMaxSuppression);


Status NonMaxSuppression::ComputeInternal(OpKernelContext* ctx) const {

  const auto* boxes = ctx->Input<Tensor>(0);
  ORT_ENFORCE(boxes);
  const auto* scores = ctx->Input<Tensor>(1);
  ORT_ENFORCE(scores);

  PrepareContext pc;
  auto ret = PrepareCompute(ctx, pc);
  ORT_RETURN_IF_NOT(ret.IsOK(), ret.ErrorMessage());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
