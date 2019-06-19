/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
/* Modifications Copyright (c) Microsoft. */

#include "non_max_suppression.h"
#include <queue>

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    NonMaxSuppression,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder(),
    NonMaxSuppression);

bool NonMaxSuppressionBase::SuppressByIOU(const float* boxes_data, int64_t box_index1, int64_t box_index2,
                                      int64_t center_point_box, float iou_threshold) {
  float x1_min{};
  float y1_min{};
  float x1_max{};
  float y1_max{};
  float x2_min{};
  float y2_min{};
  float x2_max{};
  float y2_max{};
  // center_point_box_ only support 0 or 1
  if (0 == center_point_box) {
    // boxes data format [y1, x1, y2, x2],
    MaxMin(boxes_data[4 * box_index1 + 1], boxes_data[4 * box_index1 + 3], x1_min, x1_max);
    MaxMin(boxes_data[4 * box_index1 + 0], boxes_data[4 * box_index1 + 2], y1_min, y1_max);
    MaxMin(boxes_data[4 * box_index2 + 1], boxes_data[4 * box_index2 + 3], x2_min, x2_max);
    MaxMin(boxes_data[4 * box_index2 + 0], boxes_data[4 * box_index2 + 2], y2_min, y2_max);
  } else {
    // 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
    float box1_width_half = boxes_data[4 * box_index1 + 2] / 2;
    float box1_height_half = boxes_data[4 * box_index1 + 3] / 2;
    float box2_width_half = boxes_data[4 * box_index2 + 2] / 2;
    float box2_height_half = boxes_data[4 * box_index2 + 3] / 2;

    x1_min = boxes_data[4 * box_index1 + 0] - box1_width_half;
    x1_max = boxes_data[4 * box_index1 + 0] + box1_width_half;
    y1_min = boxes_data[4 * box_index1 + 1] - box1_height_half;
    y1_max = boxes_data[4 * box_index1 + 1] + box1_height_half;

    x2_min = boxes_data[4 * box_index2 + 0] - box2_width_half;
    x2_max = boxes_data[4 * box_index2 + 0] + box2_width_half;
    y2_min = boxes_data[4 * box_index2 + 1] - box2_height_half;
    y2_max = boxes_data[4 * box_index2 + 1] + box2_height_half;
  }

  const float intersection_x_min = std::max(x1_min, x2_min);
  const float intersection_y_min = std::max(y1_min, y2_min);
  const float intersection_x_max = std::min(x1_max, x2_max);
  const float intersection_y_max = std::min(y1_max, y2_max);

  const float intersection_area = std::max(intersection_x_max - intersection_x_min, .0f) *
                                  std::max(intersection_y_max - intersection_y_min, .0f);

  if (intersection_area <= .0f) {
    return false;
  }

  const float area1 = (x1_max - x1_min) * (y1_max - y1_min);
  const float area2 = (x2_max - x2_min) * (y2_max - y2_min);
  const float union_area = area1 + area2 - intersection_area;

  if (area1 <= .0f || area2 <= .0f || union_area <= .0f) {
    return false;
  }

  const float intersection_over_union = intersection_area / union_area;

  return intersection_over_union > iou_threshold;
}

Status NonMaxSuppressionBase::PrepareCompute(OpKernelContext* ctx, const TensorShape& boxes_shape, const TensorShape& scores_shape,
                                         PrepareContext& pc) {
  ORT_RETURN_IF_NOT(boxes_shape.NumDimensions() == 3, "boxes must be a 3D tensor.");
  ORT_RETURN_IF_NOT(scores_shape.NumDimensions() == 3, "scores must be a 3D tensor.");

  const auto num_inputs = ctx->InputCount();
  auto boxes_dims = boxes_shape.GetDims();
  auto scores_dims = scores_shape.GetDims();
  ORT_RETURN_IF_NOT(boxes_dims[0] == scores_dims[0], "boxes and scores should have same num_batches.");
  ORT_RETURN_IF_NOT(boxes_dims[1] == scores_dims[2], "boxes and scores should have same spatial_dimension.");
  ORT_RETURN_IF_NOT(boxes_dims[2] == 4, "The most inner dimension in boxes must have 4 data.");

  pc.num_batches_ = boxes_dims[0];
  pc.num_classes_ = scores_dims[1];
  pc.num_boxes_ = boxes_dims[1];

  const auto* max_output_boxes_per_class_tensor = ctx->Input<Tensor>(2);
  if (max_output_boxes_per_class_tensor != nullptr) {
    pc.max_output_boxes_per_class_ = *(max_output_boxes_per_class_tensor->Data<int64_t>());
    pc.max_output_boxes_per_class_ = std::max(pc.max_output_boxes_per_class_, 0ll);
  }

  if (num_inputs > 2) {
    const auto* iou_threshold_tensor = ctx->Input<Tensor>(3);
    if (iou_threshold_tensor != nullptr) {
      pc.iou_threshold_ = *(iou_threshold_tensor->Data<float>());
      ORT_RETURN_IF_NOT((pc.iou_threshold_ >= 0 && pc.iou_threshold_ <= 1), "iou_threshold must be in range [0, 1].");
    }
  }

  if (num_inputs > 3) {
    const auto* score_threshold_tensor = ctx->Input<Tensor>(4);
    if (score_threshold_tensor != nullptr) {
      pc.has_score_threshold_ = true;
      pc.score_threshold_ = *(score_threshold_tensor->Data<float>());
    }
  }

  return Status::OK();
}

Status NonMaxSuppression::Compute(OpKernelContext* ctx) const {
  const auto* boxes = ctx->Input<Tensor>(0);
  ORT_ENFORCE(boxes);
  const auto* scores = ctx->Input<Tensor>(1);
  ORT_ENFORCE(scores);

  auto& boxes_shape = boxes->Shape();
  auto& scores_shape = scores->Shape();

  PrepareContext pc;

  auto ret = PrepareCompute(ctx, boxes_shape, scores_shape, pc);
  ORT_RETURN_IF_NOT(ret.IsOK(), ret.ErrorMessage());

  if (0 == pc.max_output_boxes_per_class_) {
    ctx->Output(0, {0, 3});
    return Status::OK();
  }

  const auto* boxes_data = boxes->Data<float>();
  const auto* scores_data = scores->Data<float>();

  struct ScoreIndexPair {
    float score_{};
    int64_t index_{};

    ScoreIndexPair() = default;
    explicit ScoreIndexPair(float score, int64_t idx) : score_(score), index_(idx) {}

    bool operator<(const ScoreIndexPair& rhs) const {
      return score_ < rhs.score_;
    }
  };

  const bool has_score_threshold = pc.has_score_threshold_;
  const auto center_point_box = GetCenterPointBox();

  std::vector<SelectedIndex> selected_indices;
  for (int64_t batch_index = 0; batch_index < pc.num_batches_; ++batch_index) {
    for (int64_t class_index = 0; class_index < pc.num_classes_; ++class_index) {
      int64_t box_score_offset = (batch_index * pc.num_classes_ + class_index) * pc.num_boxes_;
      int64_t box_offset = batch_index * pc.num_classes_ * pc.num_boxes_ * 4;
      // Filter by score_threshold_
      std::priority_queue<ScoreIndexPair, std::deque<ScoreIndexPair>> sorted_scores_with_index;
      for (int64_t box_index = 0; box_index < pc.num_boxes_; ++box_index) {
        if (!has_score_threshold || (has_score_threshold && scores_data[box_score_offset + box_index] > pc.score_threshold_)) {
          sorted_scores_with_index.emplace(scores_data[box_score_offset + box_index], box_index);
        }
      }

      ScoreIndexPair next_top_score;
      std::vector<int64_t> selected_indicies_inside_class;
      // Get the next box with top score, filter by iou_threshold_
      while (!sorted_scores_with_index.empty()) {
        next_top_score = sorted_scores_with_index.top();
        sorted_scores_with_index.pop();

        bool selected = true;
        // Check with existing selected boxes for this class, suppress if exceed the IOU (Intersection Over Union) threshold
        for (int64_t selected_index : selected_indicies_inside_class) {
          if (SuppressByIOU(boxes_data + box_offset, selected_index, next_top_score.index_,
                            center_point_box, pc.iou_threshold_)) {
            selected = false;
            break;
          }
        }

        if (selected) {
          if (pc.max_output_boxes_per_class_ > 0 &&
              static_cast<int64_t>(selected_indicies_inside_class.size()) >= pc.max_output_boxes_per_class_) {
            break;
          }
          selected_indicies_inside_class.push_back(next_top_score.index_);
          selected_indices.emplace_back(batch_index, class_index, next_top_score.index_);
        }
      }  //while
    }    //for class_index
  }      //for batch_index

  const auto num_selected = selected_indices.size();
  Tensor* output = ctx->Output(0, {static_cast<int64_t>(num_selected), 3});
  ORT_ENFORCE(output != nullptr);
  memcpy(output->MutableData<int64_t>(), selected_indices.data(), num_selected * sizeof(SelectedIndex));

  return Status::OK();
}

}  // namespace onnxruntime
