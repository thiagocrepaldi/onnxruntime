// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "non_max_suppression_impl.h"
#include "core/providers/cpu/object_detection/non_max_suppression_helper.h"

namespace onnxruntime {
namespace cuda {

struct ClassResult {
  SelectedIndex* result_ = nullptr;
  int32_t size_ = 0;
};

__global__ void NonMaxSuppressionImplDevice(const float* boxes, const float* scores,
                                            int64_t* output_data,
                                            int32_t num_batches, int32_t num_classes,
                                            int32_t num_boxes, float score_threshold,
                                            float iou_threshold, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // XXX: Need fast_div here
  auto batch_index = id % num_batches;
  auto class_index = id % num_classes;

  // Compute for each of the classes/batches
  

  // __syncthreads() so they are all done
  // One thread copies results into output
}

void NonMaxSuppressionImpl(const PrepareContext& pc, int64_t max_boxes_per_class, float score_threshold,
                           float iou_threshold, int64_t* output_data) {

  auto N = pc.num_batches_ * pc.num_classes_;

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  // XXX: How do we gather the input into one big buffer? Collect pairs of buffer/size?
  // NonMaxSuppressionImplDevice<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>
}

}  // namespace cuda
}  // namespace onnxruntime
