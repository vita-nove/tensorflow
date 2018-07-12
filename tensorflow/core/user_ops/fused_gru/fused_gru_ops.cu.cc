#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "fused_gru_ops.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

namespace {

// Adds bias, applies non-linearities and gating for r and u.
//
// Launch with a 2D setup such that there is one thread per (example,
// activation) with 'x' governing example index and 'y' governing activation.
//
// Launch with blocks of (batch x 32)
template <typename T>
__global__ void ru_gates(const T* r_u_bar, const T* b_ru, T* r, T* u,
                         const T* h_prev, T* h_prevr,
                         const int batch_size, const int cell_size) {
  const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int act_id = blockIdx.y * blockDim.y + threadIdx.y;

  if (batch_id >= batch_size || act_id >= cell_size) return;

  // The following code assumes the input arrays are of the following
  // shapes and interpretations.
  //
  // 1) 'r_u_bar' is a matrix such that,
  //
  //   cell_size  cell_size
  //  +----------+----------+
  //  |          |          |
  //  |    r     |    u     |  batch_size
  //  |          |          |
  //  +----------+----------+
  //
  // 'gid' is the index assigned to this thread for 'r_u_bar' in the 'i' submatrix.
  //
  // 2) 'b_ru' is a vector such that,
  //
  //   cell_size  cell_size
  //  +----------+----------+
  //  |    r     |    u     |  1
  //  +----------+----------+
  //
  // 'act_id' is the index assigned to this thread for 'b_ru' in the 'i' subvector.
  //
  // 3) 'r' and 'u' are matrices have the form,
  //
  //   cell_size
  //  +----------+
  //  |          |
  //  |    i     |  batch_size
  //  |          |
  //  +----------+
  //
  // 'cid' is the index assigned to this thread.

  const int gid = batch_id * cell_size * 2 + act_id;
  const int cid = batch_id * cell_size + act_id;

  Eigen::internal::scalar_sigmoid_op<T> sigmoid_op;

  // Slice r_u_bar into r, u and apply the sigmoid.
  r[cid] = sigmoid_op(r_u_bar[0 * cell_size + gid] + b_ru[0 * cell_size + act_id]);
  u[cid] = sigmoid_op(r_u_bar[1 * cell_size + gid] + b_ru[1 * cell_size + act_id]);

  // h_prevr = r .* h_prev
  h_prevr[cid] = r[cid] * h_prev[cid];
}

// Adds bias, applies non-linearities and gating for c.
//
// Launch with a 2D setup such that there is one thread per (example,
// activation) with 'x' governing example index and 'y' governing activation.
//
// Launch with blocks of (batch x 32)
template <typename T>
__global__ void c_gate(T* c, const T* b_c, T* h, T* u, const T* h_prev,
                       const int batch_size, const int cell_size) {
  const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int act_id = blockIdx.y * blockDim.y + threadIdx.y;

  if (batch_id >= batch_size || act_id >= cell_size) return;

  // The following code assumes the input arrays are of the following
  // shapes and interpretations.
  //
  // 1) 'b_c' is a vector such that,
  //
  //   cell_size  cell_size
  //  +----------+
  //  |    r     |  1
  //  +----------+
  //
  // 'act_id' is the index assigned to this thread for 'b_ru' in the 'i' subvector.
  //
  // 2) All other matrices have the form,
  //
  //   cell_size
  //  +----------+
  //  |          |
  //  |    i     |  batch_size
  //  |          |
  //  +----------+
  //
  // 'cid' is the index assigned to this thread.

  const int cid = batch_id * cell_size + act_id;
  Eigen::internal::scalar_tanh_op<T> tanh_op;

  // c = tanh(c + b_c)
  c[cid] = tanh_op(c[cid] + b_c[act_id]);

  // h = u * h_prev + c * (1 - u)
  h[cid] = u[cid] * (h_prev[cid] - c[cid]) + c[cid];
}

// Concatenate 'x' and 'h' and copy their contents into 'xh'.
template <typename T>
__global__ void concat_xh(T* xh, const T* x, const T* h_prev,
                          const int batch_size, const int cell_size,
                          const int input_size) {
  // Assumes 'x', 'h', and 'xh' are of the following shape,
  //
  //   input_size  cell_size
  //  +----------+----------+
  //  |          |          |
  //  |    x     |    h     |  batch_size
  //  |          |          |
  //  +----------+----------+
  //
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = input_size + cell_size;

  if (gid >= width * batch_size) return;

  const int output_row = gid / width;
  const int output_col = gid % width;

  if (output_col < input_size) {  // x
    xh[gid] = x[output_row * input_size + output_col];
  } else {  // h
    xh[gid] = h_prev[output_row * cell_size + output_col - input_size];
  }
}

template <typename T>
void GRUBlockCellFpropWithCuda(
    OpKernelContext* ctx, const GPUDevice& d, typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstMatrix h_prev,
    typename TTypes<T>::ConstMatrix w_ru, typename TTypes<T>::ConstMatrix w_c,
    typename TTypes<T>::ConstVec b_ru, typename TTypes<T>::ConstVec b_c,
    typename TTypes<T>::Matrix r_u_bar, typename TTypes<T>::Matrix r,
    typename TTypes<T>::Matrix u, typename TTypes<T>::Matrix c,
    typename TTypes<T>::Matrix h, typename TTypes<T>::Matrix x_h_prev,
    typename TTypes<T>::Matrix x_h_prevr, typename TTypes<T>::Matrix h_prevr,
    int batch_size, int cell_size,
    int input_size) {
  const cudaStream_t& cu_stream = GetCudaStream(ctx);

  // Concat x_h_prev = [x, h_prev].
  //
  // Each block is assigned 128 threads. Good values are in [128, 1024] and are
  // divisible by 32 (the size of a warp). The number of blocks is such that
  // there are enough to process all the data.
  const int block_dim = 128;
  const int grid_dim =
      Eigen::divup(batch_size * (cell_size + input_size), block_dim);
  concat_xh<<< grid_dim, block_dim, 0, cu_stream >>> (
      x_h_prev.data(), x.data(), h_prev.data(), batch_size, cell_size, input_size);

  // r_u_bar = x_h_prev * w_ru
  typename TTypes<T>::ConstMatrix const_x_h_prev(x_h_prev.data(), x_h_prev.dimensions());
  TensorBlasGemm<GPUDevice, T, false>::compute(
      ctx, d, false, false, T(1), const_x_h_prev, w_ru, T(0), r_u_bar);

  // Add bias, apply non-linearities and gating for r and u.
  //
  // Use 2D blocks. The number of threads per block is equal to x * y, where x =
  // min(batch_size, 8) and y = 32. See above for guidance on number of
  // threads.
  dim3 block_dim_2d(std::min(batch_size, 8), 32);
  dim3 grid_dim_2d(Eigen::divup(batch_size, static_cast<int>(block_dim_2d.x)),
                   Eigen::divup(cell_size, static_cast<int>(block_dim_2d.y)));
  ru_gates<T><<<grid_dim_2d, block_dim_2d, 0, cu_stream>>>(
      r_u_bar.data(), b_ru.data(), r.data(), u.data(), h_prev.data(), h_prevr.data(),
          batch_size, cell_size);

  // Concat x_h_prevr = [x, h_prevr]
  concat_xh<<<grid_dim, block_dim, 0, cu_stream>>>(
      x_h_prevr.data(), x.data(), h_prevr.data(), batch_size, cell_size, input_size);

  // c = x_h_prevr * w_c
  typename TTypes<T>::ConstMatrix const_x_h_prevr(x_h_prevr.data(), x_h_prevr.dimensions());
  TensorBlasGemm<GPUDevice, T, false>::compute(
      ctx, d, false, false, T(1), const_x_h_prevr, w_c, T(0), c);

  // Add bias, apply non-linearities and gating for c.
  //
  // Use 2D blocks. The number of threads per block is equal to x * y, where x =
  // min(batch_size, 8) and y = 32. See above for guidance on number of
  // threads.
  c_gate<T><<<grid_dim_2d, block_dim_2d, 0, cu_stream>>>(
      c.data(), b_c.data(), h.data(), u.data(), h_prev.data(), batch_size, cell_size);
}
}

#define DEFINE_GPU_SPECS(T)                                                    \
  template struct TensorZero<GPUDevice, T>;                                    \
  template struct TensorUnalignedZero<GPUDevice, T>;                           \
  template struct TensorCopy<GPUDevice, T>;                                    \
  template struct TensorCopyUnaligned<GPUDevice, T>;                           \
  template struct TensorCopyToUnaligned<GPUDevice, T>;                         \
  template struct TensorAdd<GPUDevice, T>;                                     \
  template <>                                                                  \
  void GRUBlockCellFprop<GPUDevice, T, true /* USE_CUBLAS */>::operator()(     \
      OpKernelContext* ctx, const GPUDevice& d,                                \
      typename TTypes<T>::ConstMatrix x,                                       \
      typename TTypes<T>::ConstMatrix h_prev,                                  \
      typename TTypes<T>::ConstMatrix w_ru,                                    \
      typename TTypes<T>::ConstMatrix w_c,                                     \
      typename TTypes<T>::ConstVec b_ru, typename TTypes<T>::ConstVec b_c,     \
      typename TTypes<T>::Matrix r_u_bar, typename TTypes<T>::Matrix r,        \
      typename TTypes<T>::Matrix u, typename TTypes<T>::Matrix c,              \
      typename TTypes<T>::Matrix h, typename TTypes<T>::Matrix x_h_prev,       \
      typename TTypes<T>::Matrix x_h_prevr,                                    \
      typename TTypes<T>::Matrix h_prevr) {                                    \
    GRUBlockCellFpropWithCuda<T>(ctx, d, x, h_prev, w_ru, w_c, b_ru, b_c,      \
    r_u_bar, r, u, c, h, x_h_prev, x_h_prevr, h_prevr, batch_size_, cell_size_,\
    input_size_);                                                              \
  };                                                                           \
  template struct GRUBlockCellFprop<GPUDevice, T, true>;                       \
  template struct GRUBlockCellBprop<GPUDevice, T, true>;

DEFINE_GPU_SPECS(float);
#undef DEFINE_GPU_SPECS

}  // end namespace functor
#endif  // GOOGLE_CUDA
