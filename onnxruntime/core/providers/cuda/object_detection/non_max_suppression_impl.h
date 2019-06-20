// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

void NonMaxSuppressionImpl(const float* boxes, const float* scores, int64_t num_batches,
                           int64_t num_classes, int64_t num_boxes,
                           float score_threshold, float iou_threshold);

}  // namespace cuda
}  // namespace onnxruntime
