// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/gpu_data_transfer.h"
#include "shared_inc/cuda_call.h"

namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer() {
  // create streams, default is nullptr
  streams_[kCudaStreamDefault] = nullptr;
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyIn], cudaStreamNonBlocking));
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyOut], cudaStreamNonBlocking));
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_Device) const override {
  return true;
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override {
  if (strcmp(src.Location().name, CUDA) != 0 && strcmp(src.Location().name, CUDA_PINNED) != 0 &&
      strcmp(dst.Location().name, CUDA) != 0 && strcmp(dst.Location().name, CUDA_PINNED) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported tensor location: src_location is: ", src.Location().name, " and dst_location is: ", dst.Location().name);
  }

  size_t bytes = src.Size();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  if (strcmp(dst.Location().name, CUDA) == 0) {
    if (strcmp(src.Location().name, CUDA_PINNED) == 0) {
      // copy from pinned memory to GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, streams_[exec_queue_id]));
    } else if (strcmp(src.Location().name, CUDA) == 0) {
      // copying between GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, streams_[kCudaStreamDefault]));
    } else {
      // copy from other CPU memory to GPU, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
    }
  } else if (strcmp(src.Location().name, CUDA) == 0) {
    if (strcmp(dst.Location().name, CUDA_PINNED) == 0) {
      // copying from GPU to pinned memory, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, streams_[exec_queue_id]));
    } else {
      // copying from GPU to CPU memory, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
    }
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

}  // namespace onnxruntime
