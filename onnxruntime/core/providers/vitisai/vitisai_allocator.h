// Copyright (C) Intel Corporation
// Licensed under the MIT License
// #ifdef USE_VITISAI_CPU_ALIGNED
#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"

namespace onnxruntime {

class VitisAIAllocator : public IAllocator {
 public:
  VitisAIAllocator(OrtDevice::DeviceType device_type, OrtDevice::DeviceId device_id, const char* name);
  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

}  // namespace onnxruntime
// #endif
