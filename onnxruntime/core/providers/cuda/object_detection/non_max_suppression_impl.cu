// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "non_max_suppression_impl.h"

namespace onnxruntime {
namespace cuda {

__global__
void NonMaxSuppressionImplDevice(const float* boxes, const float* scores,
                                 int64_t num_batches, int64_t num_classes,
                                 int64_t num_boxes, float score_threshold,
                                 float iou_threshold, CUDA_LONG N) {

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

}

void NonMaxSuppressionImpl(const float* boxes, const float* scores,
                           int64_t num_batches, int64_t num_classes, int64_t num_boxes,
                           float score_threshold, float iou_threshold) {
  auto N = num_batches * num_classes;

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  // Need to scan all the scores and select the ones that qualify, skip score scanning if
  // we do not have score threshold
}

}  // namespace cuda
}  // namespace onnxruntime
