#pragma once

#include <sys/types.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif
  void* gmlake_malloc(size_t size, int device, cudaStream_t stream);

  void gmlake_free(void* ptr, size_t size, int device, cudaStream_t stream);

  void gmlake_init(int device_count);

  void gmlake_reset();

  void gmlake_memory_fraction(double fraction, int device);

  void* gmlake_base_alloc(void* ptr, size_t* size);

  void gmlake_record_stream(void* ptr, cudaStream_t stream);

  void gmlake_begin_allocate_to_pool(int device, c10::cuda::MempoolId_t mempool_id, bool(*filter)(cudaStream_t));

  void gmlake_end_allocate_to_pool(int device, c10::cuda::MempoolId_t mempool_id);

  void gmlake_release_pool(int device, c10::cuda::MempoolId_t mempool_id);
#ifdef __cplusplus
}
#endif