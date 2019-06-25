// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "non_max_suppression_impl.h"
#include "core/providers/cpu/object_detection/non_max_suppression_helper.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "thrust/sort.h"

namespace onnxruntime {
namespace cuda {

// XXX: Move to common?
template<class T, int max_size>
class FixesSizeContainer {
  char* buffer_ = nullptr;
  int size_ = 0; // Number of contained elements

public:
  __device__ __host__ explicit FixesSizeContainer(int size) {
    Allocate(size);
  }

  FixesSizeContainer(const FixesSizeContainer&) = delete;
  FixesSizeContainer& operator=(const FixesSizeContainer&) = delete;
  __device__ __host__ ~FixesSizeContainer() noexcept {
    uninitialized_default_destruct();
    delete[] buffer_;
  }

  __device__ __host__ int size() const { return size_; }

  __device__ __host__ T* operator[](int idx) noexcept {
    // no exceptions possible
    return GetTyped()[idx];
  }

  __device__ __host__  const T& operator[](int idx) const noexcept {
    // no exceptions possible
    return GetTyped()[idx];
  }

  __device__ __host__ void push_back(const T& v) {
    if (size_ < (max_size)) {
      new (GetTyped(size_)) T(v);
      ++size_;
    }
  }

  template<class... Args)
  __device__ __host__ void emplace_back(Args&&... args) {
    if (size_ < (max_size)) {
      new (GetTyped(size_)) T(std::forward<Args>(args)...);
      ++size_;
    }
  }

  __device__ __host__ T* begin() { return GetTyped(0); }
  __device__ __host__ const T* begin() const { return GetTyped(0); }
  __device__ __host__ T* end() { return GetTyped(size_); }
  __device__ __host__ const T* end() const { return GetTyped(size_); }

 private:
  __device__ __host__ void Allocate(int size) {
   int sz = (size > max_size) > max_size : size;
    buffer_ = new char[sizeof(T) * sz];
    size_ = 0;
  }
  __device__ __host__ void uninitialized_default_destruct() noexcept {
    auto* p = GetTyped();
    while (size_-- > 0) {
      p[size]->~T();
    }
  }
  __device__ __host__ T* GetTyped() const {
    return reinterpret_cast<T*>(buffer_);
  }
};

struct ScoreIndexPair {
  float score_{};
  int64_t index_{};

  ScoreIndexPair() = default;
  __device__ explicit ScoreIndexPair(float score, int64_t idx) noexcept :
    score_(score), index_(idx) {}
  ScoreIndexPair(const ScoreIndexPair&) = default;
  ScoreIndexPair& operator=(const ScoreIndexPair&) = default;
  // We reverse the meaning so thrust::sort below sorts in descending order
  __device__ bool operator<(const ScoreIndexPair& rhs) const noexcept {
    return score_ > rhs.score_;
  }
};

struct ThreadResult {
  SelectedIndex* selected_indecies_ = nullptr;  // selected indices per thread
  int size_ = 0;
  ThreadResult() = default;
};

__shared__ ThreadResult block_results[];

__global__ void NonMaxSuppressionImplDevice(const float* boxes_data, const float* scores_data,
                                            int64_t* output_data,
                                            int32_t num_batches, int32_t num_classes,
                                            int32_t num_boxes,
                                            int32_t center_point_box,
                                            int32_t max_output_boxes_per_class,
                                            const fast_divmod batches,
                                            const fast_divmod classes,
                                            bool has_score_threshold,
                                            float score_threshold,
                                            float iou_threshold, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // Load the scores for each block in individual thread

  // N = num_batches * num_classes
  // batch_index = id % num_batches
  int batch_index = 0;
  int whole_batches = 0;
  batches.divmod(id, whole_batches, batch_index);

  // class_index = id % num_classes
  int whole_classes = 0;
  int class_index = 0;
  classes.divmod(id, whole_classes, class_index);

  char* selected_scores_buffer = new char[sizeof(ScoreIndexPair) * num_boxes];
  int selected_scores_size = 0;
  int box_score_offset = (batch_index * num_classes + class_index) * num_boxes;
  const auto* class_scores = scores_data + box_score_offset;
  if (has_score_threshold) {
    for (int64_t box_index = 0; box_index < int64_t{num_boxes}; ++box_index, ++class_scores) {
      if (*class_scores > score_threshold) {
        
        selected_scores.push_back(ScoreIndexPair(*class_scores, box_index));
      }
    }
  } else {
    for (int64_t box_index = 0; box_index < int64_t{num_boxes}; ++box_index, ++class_scores) {
      selected_scores.push_back(ScoreIndexPair(*class_scores, box_index));
    }
  }
  // We lack priority queue
  thrust::sort(selected_scores.begin(), selected_scores.end());

  // Compute for each of the classes/batches
  ScoreIndexPair next_top_score;
  thrust::device_vector<int64_t> selected_indicies_inside_class;
  int box_offset = batch_index * num_classes * num_boxes * 4;

  // Get the next box with top score, filter by iou_threshold
  const float* class_boxes = boxes_data + box_offset;
  for (const ScoreIndexPair& top : selected_scores) {
    bool selected = true;
    // Check with existing selected boxes for this class, suppress if exceed the IOU (Intersection Over Union) threshold
    for (int64_t selected_index : selected_indicies_inside_class) {
      if (nms_helpers::SuppressByIOU(class_boxes, selected_index, top.index_,
                                     center_point_box, iou_threshold)) {
        selected = false;
        break;
      }
    }

    if (selected) {
      if (max_output_boxes_per_class > 0 &&
          static_cast<int32_t>(selected_indicies_inside_class.size()) >= max_output_boxes_per_class) {
        break;
      }
      selected_indicies_inside_class.push_back(next_top_score.index_);
    }
  }  //for

  delete[] selected_scores_buffer;

  // Assign dynamically allocated memory to the shared array
  //

  // Increment the count
  // __threadfence();

  // __syncthreads();   // sync inside the block
  // One thread copies results into output
}

void NonMaxSuppressionImpl(const PrepareContext& pc, int64_t max_boxes_per_class, float score_threshold,
                           float iou_threshold, int64_t* output_data) {
  auto N = pc.num_batches_ * pc.num_classes_;
  fast_divmod batches(pc.num_batches_);
  fast_divmod classes(pc.num_classes_);

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  // XXX: How do we gather the input into one big buffer? Collect pairs of buffer/size?
  // NonMaxSuppressionImplDevice<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>
}

}  // namespace cuda
}  // namespace onnxruntime
