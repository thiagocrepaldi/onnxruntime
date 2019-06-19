// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class NonMaxSuppressionBase {
 protected:
  explicit NonMaxSuppressionBase(const OpKernelInfo& info) {
    center_point_box_ = info.GetAttrOrDefault<int64_t>("center_point_box", 0);
    ORT_ENFORCE(0 == center_point_box_ || 1 == center_point_box_, "center_point_box only support 0 or 1");
  }

  static bool SuppressByIOU(const float* boxes_data, int64_t box_index1, int64_t box_index2, int64_t center_point_box,
                            float iou_threshold);

  static void MaxMin(const float& lhs, const float& rhs, float& min, float& max) {
    if (lhs >= rhs) {
      min = rhs;
      max = lhs;
    } else {
      min = lhs;
      max = rhs;
    }
  }

  struct SelectedIndex {
    SelectedIndex(int64_t batch_index, int64_t class_index, int64_t box_index)
        : batch_index_(batch_index), class_index_(class_index), box_index_(box_index) {}
    int64_t batch_index_ = 0;
    int64_t class_index_ = 0;
    int64_t box_index_ = 0;
  };

  struct PrepareContext {
    int64_t num_batches_ = 0;
    int64_t num_classes_ = 0;
    int64_t num_boxes_ = 0;
    int64_t max_output_boxes_per_class_ = 0;
    int64_t max_output_boxes_per_batch_ = 0;
    float iou_threshold_ = .0f;
    bool has_score_threshold_ = false;
    float score_threshold_ = .0f;
  };

  static Status PrepareCompute(OpKernelContext* ctx, const TensorShape& boxes_shape, const TensorShape& scores_shape,
                               PrepareContext& pc);

  int64_t GetCenterPointBox() const {
    return center_point_box_;
  }

 private:
  int64_t center_point_box_;
};

class NonMaxSuppression final : public OpKernel, public NonMaxSuppressionBase {
 public:
  explicit NonMaxSuppression(const OpKernelInfo& info) : OpKernel(info), NonMaxSuppressionBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace onnxruntime
