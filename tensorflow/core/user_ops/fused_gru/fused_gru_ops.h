#ifndef FUSED_GRU_OPS_H_
#define FUSED_GRU_OPS_H_

#include "blas_gemm.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

namespace functor {

template <typename Device, typename T>
struct TensorZero {
  void operator()(const Device& d, typename TTypes<T>::Flat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorUnalignedZero {
  void operator()(const Device& d, typename TTypes<T>::UnalignedFlat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorCopy {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyUnaligned {
  void operator()(const Device& d, typename TTypes<T>::UnalignedConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyToUnaligned {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::UnalignedFlat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorAdd {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat a,
                  typename TTypes<T>::ConstFlat b, typename TTypes<T>::Flat c) {
    c.device(d) = a + b;
  }
};

template <typename Device, typename T>
struct TensorZeroPadding {
  void operator()(const Device& d, const int64 time_idx,
                  typename TTypes<int64>::ConstVec seq_len,
                  typename TTypes<float>::Vec mask,
                  typename TTypes<float>::Matrix m) {
    // mask is shape [batch_size].
    mask.device(d) = seq_len.constant(time_idx) < seq_len;

    // m_shape is [batch_size, 1].
    Eigen::array<Eigen::DenseIndex, 2> m_shape({m.dimensions()[0], 1});
    // broadcast_shape is [1, units].
    Eigen::array<Eigen::DenseIndex, 2> broadcast_shape({1, m.dimensions()[1]});

    // m is shape [batch_size, units].
    m.device(d) = m * mask.reshape(m_shape).broadcast(broadcast_shape);
  }
};

struct GRUCell {
  GRUCell(const int batch_size, const int input_size, const int cell_size)
      : batch_size_(batch_size),
        input_size_(input_size),
        cell_size_(cell_size) {}

  int batch_size() const { return batch_size_; }

  int input_size() const { return input_size_; }

  int cell_size() const { return cell_size_; }

  inline Eigen::array<Eigen::DenseIndex, 2> x_offset() const {
    return {0, 0};
  }
  inline Eigen::array<Eigen::DenseIndex, 2> x_extent() const {
    return {batch_size_, input_size_};
  }
  inline Eigen::array<Eigen::DenseIndex, 2> h_offset() const {
    return {0, input_size_};
  }
  inline Eigen::array<Eigen::DenseIndex, 2> h_extent() const {
    return {batch_size_, cell_size_};
  }
  inline Eigen::array<Eigen::DenseIndex, 2> ru_r_offset() const {
    return {0, 0};
  }
  inline Eigen::array<Eigen::DenseIndex, 2> ru_u_offset() const {
    return {0, cell_size_};
  }
  inline Eigen::array<Eigen::DenseIndex, 2> cell_extent() const {
    return {batch_size_, cell_size_};
  }
 protected:
  const int batch_size_;
  const int input_size_;
  const int cell_size_;
};

// See fused_gru_ops.cc for CPUDevice implementation and fused_gru_ops.cu.cc for
// GPUDevice implementation.
template <typename Device, typename T, bool USE_CUBLAS>
struct GRUBlockCellFprop : public GRUCell {
  explicit GRUBlockCellFprop(const int batch_size, const int input_size, const int cell_size)
      : GRUCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix h_prev,
      typename TTypes<T>::ConstMatrix w_ru, typename TTypes<T>::ConstMatrix w_c,
      typename TTypes<T>::ConstVec b_ru, typename TTypes<T>::ConstVec b_c,
      typename TTypes<T>::Matrix r_u_bar, typename TTypes<T>::Matrix r,
      typename TTypes<T>::Matrix u, typename TTypes<T>::Matrix c,
      typename TTypes<T>::Matrix h, typename TTypes<T>::Matrix x_h_prev,
      typename TTypes<T>::Matrix x_h_prevr, typename TTypes<T>::Matrix h_prevr);
};

template <typename Device, typename T, bool USE_CUBLAS>
struct GRUBlockCellBprop : public GRUCell {
  GRUBlockCellBprop(const int batch_size, const int input_size, const int cell_size)
      : GRUCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix h_prev,
      typename TTypes<T>::ConstMatrix w_ru, typename TTypes<T>::ConstMatrix w_c,
      typename TTypes<T>::ConstVec b_ru, typename TTypes<T>::ConstVec b_c,
      typename TTypes<T>::ConstMatrix r, typename TTypes<T>::ConstMatrix u,
      typename TTypes<T>::ConstMatrix c, typename TTypes<T>::ConstMatrix d_h,
      typename TTypes<T>::Matrix x_grad, typename TTypes<T>::Matrix h_prev_grad,
      typename TTypes<T>::Matrix d_c_bar,
      typename TTypes<T>::Matrix d_r_bar_u_bar,
      typename TTypes<T>::Matrix d_r_bar, typename TTypes<T>::Matrix d_u_bar,
      typename TTypes<T>::Matrix d_hr,
      typename TTypes<T>::Matrix d_x_comp1_and_h_prev_comp1,
      typename TTypes<T>::Matrix d_x_comp2_and_h_prevr,
      typename TTypes<T>::Matrix x_h_prev, typename TTypes<T>::Matrix x_h_prevr,
      typename TTypes<T>::Matrix w_ru_grad, typename TTypes<T>::Matrix w_c_grad,
      typename TTypes<T>::Vec b_ru_grad, typename TTypes<T>::Vec b_c_grad) {
    // d_c_bar = d_h * (1 - u) * (1 - (c * c))
    d_c_bar.device(d) = (d_h * (u.constant(T(1)) - u)) * (c.constant(T(1)) - c * c);

    // d_u_bar = d_h * (h - c) * (u * (1 - u))
    d_u_bar.device(d) = d_h * (h_prev - c) * u * (u.constant(T(1)) - u);

    // [2nd_component_of_d_x d_h_prevr] = d_c_bar X w_c^T
    typename TTypes<T>::ConstMatrix const_d_c_bar(d_c_bar.data(), d_c_bar.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(ctx, d, false, true, T(1),
                                                   const_d_c_bar, w_c, T(0),
                                                   d_x_comp2_and_h_prevr);

    d_hr.device(d) = d_x_comp2_and_h_prevr.slice(h_offset(), h_extent());
    d_r_bar.device(d) = (d_hr * h_prev * r) * (r.constant(T(1)) - r);

    // d_r_bar_u_bar = concatenate(d_r_bar, d_u_bar) along axis = 1.
    d_r_bar_u_bar.slice(ru_r_offset(), cell_extent()).device(d) = d_r_bar;
    d_r_bar_u_bar.slice(ru_u_offset(), cell_extent()).device(d) = d_u_bar;

    // [1st_component_of_d_x 1st_component_of_d_h_prev] = [d_r_bar d_u_bar] X
    // w_ru^T
    typename TTypes<T>::ConstMatrix const_d_r_bar_u_bar(
        d_r_bar_u_bar.data(), d_r_bar_u_bar.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, false, true, T(1), const_d_r_bar_u_bar, w_ru, T(0),
        d_x_comp1_and_h_prev_comp1);

    // x_grad = d_x_comp1 + d_x_comp2
    x_grad.device(d) = (d_x_comp1_and_h_prev_comp1 + d_x_comp2_and_h_prevr)
        .slice(x_offset(), x_extent());

    // h_prev_grad = d_h_comp1 + d_hr*r + d_h*u
    h_prev_grad.device(d) =
        d_x_comp1_and_h_prev_comp1.slice(h_offset(), h_extent()) +
            (d_hr * r) + (d_h * u);

    // Concat x_h_prev = [x, h_prev].
    x_h_prev.slice(x_offset(), x_extent()).device(d) = x;
    x_h_prev.slice(h_offset(), h_extent()).device(d) = h_prev;
    // d_w_ru = x_h_prev^T * d_r_bar_u_bar
    typename TTypes<T>::ConstMatrix const_x_h_prev(x_h_prev.data(), x_h_prev.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, true, false, T(1), const_x_h_prev, const_d_r_bar_u_bar, T(1), w_ru_grad);
    // d_b_ru = sum of d_r_bar_u_bar along axis = 0
    b_ru_grad.device(d) += d_r_bar_u_bar.sum(Eigen::array<int, 1>({0}));

    // Concat x_h_prevr = [x, h_prev .* r]
    x_h_prevr.slice(x_offset(), x_extent()).device(d) = x;
    x_h_prevr.slice(x_offset(), x_extent()).device(d) = h_prev * r;
    // d_w_c = x_h_prevr^T * d_c_bar
    typename TTypes<T>::ConstMatrix const_x_h_prevr(x_h_prevr.data(), x_h_prevr.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, true, false, T(1), const_x_h_prevr, const_d_c_bar, T(1), w_c_grad);
    // d_b_c = sum of d_c_bar along axis = 0
    b_c_grad.device(d) += d_c_bar.sum(Eigen::array<int, 1>({0}));
  }
};

}  // namespace functor

#endif  // FUSED_GRU_OPS_H
