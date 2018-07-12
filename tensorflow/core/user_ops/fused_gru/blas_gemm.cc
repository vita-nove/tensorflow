#include "blas_gemm.h"

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

#include "blas_gemm.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

#if GOOGLE_CUDA
namespace {
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace
#endif  // GOOGLE_CUDA

namespace functor {
template <typename T>
void TensorCuBlasGemm<T>::operator()(OpKernelContext* ctx, bool transa,
                                     bool transb, uint64 m, uint64 n, uint64 k,
                                     T alpha, const T* a, int lda, const T* b,
                                     int ldb, T beta, T* c, int ldc) {
#if GOOGLE_CUDA
  se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose,
                                 se::blas::Transpose::kTranspose};

  auto a_ptr = AsDeviceMemory(a);
  auto b_ptr = AsDeviceMemory(b);
  auto c_ptr = AsDeviceMemory(c);

  bool blas_launch_status =
      ctx->op_device_context()
          ->stream()
          ->ThenBlasGemm(trans[transa], trans[transb], m, n, k, alpha, a_ptr,
                         lda, b_ptr, ldb, beta, &c_ptr, ldc)
          .ok();
  OP_REQUIRES(ctx, blas_launch_status, errors::Aborted("CuBlasGemm failed!"));
#else
  ctx->SetStatus(errors::InvalidArgument("CuBlasGemm needs CUDA."));
#endif
}

template struct TensorCuBlasGemm<float>;
template struct TensorCuBlasGemm<double>;

}  // end namespace functor
