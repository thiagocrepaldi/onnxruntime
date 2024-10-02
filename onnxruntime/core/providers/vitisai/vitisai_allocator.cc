// Copyright (C) Intel Corporation
// Licensed under the MIT License
// #ifdef USE_VITISAI_CPU_ALIGNED
#include "core/providers/vitisai/vitisai_allocator.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

constexpr size_t default_alignment = 4096;  // Preferred by Vitis AI to be able to zero-copy from CPU into NPU

#ifdef USE_MIMALLOC
void* AllocatorDefaultAlloc(size_t size) {
  //   const size_t alignment = MlasGetPreferredBufferAlignment();  // FROM CPU EP
  const size_t alignment = default_alignment;
  if (size <= 0) return nullptr;
  size += MLAS_SYMM_QGEMM_BUF_OVERRUN;
  void* p;
#if defined(_MSC_VER)
  p = mi_malloc_aligned(size, alignment);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#elif defined(_LIBCPP_SGX_CONFIG)
  p = mi_memalign(alignment, size);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#else
  int ret = mi_posix_memalign(&p, alignment, size);
  if (ret != 0)
    ORT_THROW_EX(std::bad_alloc);
#endif
  return p;
}

void AllocatorDefaultFree(void* p) {
#if defined(_MSC_VER)
  const size_t alignment = MlasGetPreferredBufferAlignment();
  mi_free_aligned(p, alignment);
#else
  mi_free(p);
#endif
}

#else
void* AllocatorDefaultAlloc(size_t size) {
  //   const size_t alignment = MlasGetPreferredBufferAlignment();  // FROM CPU EP
  const size_t alignment = default_alignment;
  if (size <= 0) return nullptr;
  size += MLAS_SYMM_QGEMM_BUF_OVERRUN;
  void* p;
#if _MSC_VER
  p = _aligned_malloc(size, alignment);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#elif defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr)
    ORT_THROW_EX(std::bad_alloc);
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
    ORT_THROW_EX(std::bad_alloc);
#endif
  return p;
}

void AllocatorDefaultFree(void* p) {
#if _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

#endif  // USE_MIMALLOC

VitisAIAllocator::VitisAIAllocator(OrtDevice::DeviceType device_type, OrtDevice::DeviceId device_id, const char* name) :
IAllocator(
    OrtMemoryInfo(
        name,
        OrtAllocatorType::OrtDeviceAllocator,
        OrtDevice(
            device_type,
            OrtDevice::MemType::DEFAULT,
            device_id),
        device_id,
        OrtMemTypeCPUInput
    )
) {}

void* VitisAIAllocator::Alloc(size_t size) {
  return AllocatorDefaultAlloc(size);
}

void VitisAIAllocator::Free(void* p) {
  AllocatorDefaultFree(p);
}

}  // namespace onnxruntime
// #endif
