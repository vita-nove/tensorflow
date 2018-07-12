#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "fused_gru_ops.h"

#include <memory>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
void GRUBlockCellFpropWithEigen(
    const GRUCell& cell, OpKernelContext* ctx,
    const CPUDevice& d, typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstMatrix h_prev,
    typename TTypes<T>::ConstMatrix w_ru, typename TTypes<T>::ConstMatrix w_c,
    typename TTypes<T>::ConstVec b_ru, typename TTypes<T>::ConstVec b_c,
    typename TTypes<T>::Matrix r_u_bar, typename TTypes<T>::Matrix r,
    typename TTypes<T>::Matrix u, typename TTypes<T>::Matrix c,
    typename TTypes<T>::Matrix h, typename TTypes<T>::Matrix x_h_prev,
    typename TTypes<T>::Matrix x_h_prevr, typename TTypes<T>::Matrix h_prevr) {
  // Concat x_h_prev = [x, h_prev].
  x_h_prev.slice(cell.x_offset(), cell.x_extent()).device(d) = x;
  x_h_prev.slice(cell.h_offset(), cell.h_extent()).device(d) = h_prev;

  // r_u_bar = x_h_prev * w_ru + bru
  typename TTypes<T>::ConstMatrix const_x_h_prev(x_h_prev.data(), x_h_prev.dimensions());
  TensorBlasGemm<CPUDevice, T, false>::compute(
      ctx, d, false, false, T(1), const_x_h_prev, w_ru, T(0), r_u_bar);

  // Creating a bias matrix for adding by broadcasting 'b_ru'.
  Eigen::array<Eigen::DenseIndex, 2> broadcast_shape({cell.batch_size(), 1});
  Eigen::array<Eigen::DenseIndex, 2> b_ru_shape({1, b_ru.dimensions()[0]});
  r_u_bar.device(d) += b_ru.reshape(b_ru_shape).broadcast(broadcast_shape);

  // Slice r_u_bar into r, u and apply the sigmoid.
  r.device(d) = (r_u_bar.slice(cell.ru_r_offset(), cell.cell_extent())).sigmoid();
  u.device(d) = (r_u_bar.slice(cell.ru_u_offset(), cell.cell_extent())).sigmoid();

  // Concat x_h_prevr = [x, h_prev .* r]
  x_h_prevr.slice(cell.x_offset(), cell.x_extent()).device(d) = x;
  x_h_prevr.slice(cell.x_offset(), cell.x_extent()).device(d) = h_prev * r;

  // c = tanh(x_h_prevr * w_c + b_c)
  typename TTypes<T>::ConstMatrix const_x_h_prevr(x_h_prevr.data(), x_h_prevr.dimensions());
  TensorBlasGemm<CPUDevice, T, false>::compute(
      ctx, d, false, false, T(1), const_x_h_prevr, w_c, T(0), c);
  Eigen::array<Eigen::DenseIndex, 2> b_c_shape({1, b_c.dimensions()[0]});
  c.device(d) += b_c.reshape(b_c_shape).broadcast(broadcast_shape);
  c.device(d) = c.tanh();

  // h = u * h_prev + c * (1 - u)
  h.device(d) = u * (h_prev - c) + c;
}

#define DEFINE_CPU_SPECS(T)                                                       \
  template <>                                                                     \
  void GRUBlockCellFprop<CPUDevice, T, false /* USE_CUBLAS */>::operator()(       \
      OpKernelContext* ctx, const CPUDevice& d, typename TTypes<T>::ConstMatrix x,\
      typename TTypes<T>::ConstMatrix h_prev,                                     \
      typename TTypes<T>::ConstMatrix w_ru, typename TTypes<T>::ConstMatrix w_c,  \
      typename TTypes<T>::ConstVec b_ru, typename TTypes<T>::ConstVec b_c,        \
      typename TTypes<T>::Matrix r_u_bar, typename TTypes<T>::Matrix r,           \
      typename TTypes<T>::Matrix u, typename TTypes<T>::Matrix c,                 \
      typename TTypes<T>::Matrix h, typename TTypes<T>::Matrix x_h_prev,          \
      typename TTypes<T>::Matrix x_h_prevr, typename TTypes<T>::Matrix h_prevr){  \
    GRUBlockCellFpropWithEigen<T>(                                                \
        *this, ctx, d, x, h_prev, w_ru, w_c, b_ru, b_c, r_u_bar,                  \
        r, u, c, h, x_h_prev, x_h_prevr, h_prevr);                                \
  }                                                                               \
  template struct GRUBlockCellFprop<CPUDevice, T, false /* USE_CUBLAS */>;

DEFINE_CPU_SPECS(float);
#undef DEFINE_CPU_SPECS

}  // namespace functor



namespace {
// This helper class can be used to access timeslices of a 3D tensor. If a slice
// happens to be unaligned (usually because both batch size and number of cells
// are odd - this isn't common) this involves overhead, since data needs to be
// copied. However, if all slices are aligned, the bits aren't copied. In the
// cases where copying is needed, the outputs have to be recopied back.
// At the end of each time step you should call FinishTimeStep which does this,
// and also allows for reuse of temporary tensors.
template <typename Device, typename T>
class SliceHelper {
 public:
  explicit SliceHelper(OpKernelContext* ctx)
      : ctx_(ctx), device_(ctx_->eigen_device<Device>()) {}

  ~SliceHelper() {
    CHECK(copy_out_.empty());
    for (const auto& entry : pool_) {
      CHECK(!entry.second.second);  // nothing is in use
    }
  }

  // Slice through an input tensor. This may copy unaligned slices, but no
  // copying back will be done at the end.
  const Tensor InputSlice(const Tensor& t, int pos, const string& name) {
    Tensor res = UnalignedSlice(t, pos);
    if (res.IsAligned()) {
      return res;
    } else {
      return AlignTensor(res, name);
    }
  }

  // Slice through an output tensor. This may copy unaligned slices, and
  // schedule copying back on destruction.
  Tensor OutputSlice(Tensor* t, int pos, const string& name) {
    Tensor res = UnalignedSlice(*t, pos);
    if (res.IsAligned()) {
      return res;
    } else {
      Tensor aligned = AlignTensor(res, name);
      copy_out_.emplace_back(res, aligned);
      return aligned;
    }
  }

  void FinishTimeStep() {
    for (const auto& p : copy_out_) {
      const Tensor& aligned = p.second;
      Tensor original = p.first;
      // Copy from aligned back to original.
      functor::TensorCopyToUnaligned<Device, T>()(device_, aligned.flat<T>(),
                                                  original.unaligned_flat<T>());
    }
    copy_out_.clear();
    // Mark all entries as not in use.
    for (auto& entry : pool_) {
      entry.second.second = false;
    }
  }

 private:
  // Return a slice at position 'pos'. Result may be unaligned. The resulting
  // tensor always shares data with the source tensor.
  Tensor UnalignedSlice(const Tensor& t, int pos) const {
    Tensor res;
    // CHECK should never fail here, since the number of elements must match
    CHECK(res.CopyFrom(t.Slice(pos, pos + 1), {t.dim_size(1), t.dim_size(2)}));
    return res;
  }

  // Assumes input is not aligned, creates a temporary aligned tensor of the
  // same shape and copies the original tensor's content into it.
  Tensor AlignTensor(const Tensor& t, const string& name) {
    VLOG(1) << "AlignTensor called for " << name << ", shape "
            << t.shape().DebugString()
            << ". This is unnecessary copying. Consider using shapes with even "
            << "sizes";
    Tensor aligned;
    auto found = pool_.find(name);
    if (found != pool_.end()) {  // found in pool
      CHECK(!found->second.second) << "Tensor " << name << " is in use";
      found->second.second = true;  // mark in use
      aligned = found->second.first;
      CHECK(aligned.shape().IsSameSize(t.shape()));
      CHECK_EQ(aligned.dtype(), t.dtype());
    } else {  // allocate a new temporary tensor
      TF_CHECK_OK(ctx_->allocate_temp(t.dtype(), t.shape(), &aligned));
      pool_.emplace(name, std::make_pair(aligned, true));
    }
    functor::TensorCopyUnaligned<Device, T>()(device_, t.unaligned_flat<T>(),
                                              aligned.flat<T>());
    return aligned;
  }

  // Tensors to be copied.
  std::vector<std::pair<Tensor, const Tensor>> copy_out_;
  // A pool of pre-allocated temporary tensors, with an indicator for whether
  // it's in use.
  std::map<string, std::pair<Tensor, bool>> pool_;
  // Op context
  OpKernelContext* ctx_ = nullptr;
  // Device
  const Device& device_;
};
}  // namespace

template <typename Device, typename T, bool USE_CUBLAS>
class BlockGRUOp : public OpKernel {
 public:
  explicit BlockGRUOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    // Grab the input tensors.
    const Tensor* seq_len_max_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("seq_len_max", &seq_len_max_tensor));

    const Tensor* x = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_ru", &w_ru_tensor));

    const Tensor* w_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_c", &w_c_tensor));

    const Tensor* b_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_ru", &b_ru_tensor));

    const Tensor* b_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_c", &b_c_tensor));

    const int64 time_len = x->dim_size(0);
    const int64 batch_size = x->dim_size(1);
    const int64 input_size = x->dim_size(2);
    const int64 cell_size = h_prev_tensor->dim_size(1);

    // Sanity checks for input shapes.

    // Shape of 'x' must be [time_len, batch_size, input_size]
    OP_REQUIRES(ctx, x->dims() == 3, errors::InvalidArgument("x must be 3D"));

    // Shape of 'h' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, h_prev_tensor->dims() == 2,
                errors::InvalidArgument("h_prev must be 2D"));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "h_prev.dims(0) != batch_size: ",
                    h_prev_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ",
                    h_prev_tensor->dim_size(1), " vs. ", cell_size));

    // Shape of 'w_ru' must be [input_size + cell_size, 2 * cell_size]
    OP_REQUIRES(ctx, w_ru_tensor->dims() == 2,
                errors::InvalidArgument("w_ru must be 2D"));
    OP_REQUIRES(ctx, w_ru_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_ru.dim_size(0) != input_size + cell_size: ",
                    w_ru_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_ru_tensor->dim_size(1) == cell_size * 2,
                errors::InvalidArgument(
                    "w_ru.dim_size(1) != cell_size * 2: ",
                    w_ru_tensor->dim_size(1), " vs. ", cell_size * 2));

    // Shape of 'w_c' must be [input_size + cell_size, cell_size]
    OP_REQUIRES(ctx, w_c_tensor->dims() == 2,
                errors::InvalidArgument("w_c must be 2D"));
    OP_REQUIRES(ctx, w_c_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(0) != input_size + cell_size: ",
                    w_c_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_c_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(1) != cell_size: ",
                    w_c_tensor->dim_size(1), " vs. ", cell_size));

    // Shape of 'b_ru' must be [2 * cell_size]
    OP_REQUIRES(ctx, b_ru_tensor->dims() == 1,
                errors::InvalidArgument("b_ru must be 1D"));
    OP_REQUIRES(ctx, b_ru_tensor->dim_size(0) == cell_size * 2,
                errors::InvalidArgument(
                    "b_ru.dim_size(0) != cell_size * 2: ",
                    b_ru_tensor->dim_size(0), " vs. ", cell_size * 2));

    // Shape of 'b_c' must be [cell_size]
    OP_REQUIRES(ctx, b_c_tensor->dims() == 1,
                errors::InvalidArgument("b_c must be 1D"));
    OP_REQUIRES(ctx, b_c_tensor->dim_size(0) == cell_size,
                errors::InvalidArgument(
                    "b_c.dim_size(0) != cell_size: ",
                    b_c_tensor->dim_size(0), " vs. ", cell_size));

    // Create output tensors.
    TensorShape batch_cell_shape({time_len, batch_size, cell_size});

    Tensor* r_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("r", batch_cell_shape, &r_out));

    Tensor* u_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("u", batch_cell_shape, &u_out));

    Tensor* c_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("c", batch_cell_shape, &c_out));

    Tensor* h_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("h", batch_cell_shape, &h_out));

    // Allocate temp tensors.
    Tensor x_h_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, input_size + cell_size}),
        &x_h_prev_tensor));

    Tensor x_h_prevr_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, input_size + cell_size}),
        &x_h_prevr_tensor));

    Tensor r_u_bar_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, 2 * cell_size}),
        &r_u_bar_tensor));

    Tensor h_prevr_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size}),
        &h_prevr_tensor));

    const Device& device = ctx->eigen_device<Device>();

    const int64 seq_len_max = seq_len_max_tensor->scalar<int64>()();
    SliceHelper<Device, T> slicer(ctx);

    for (int64 t = 0; t < seq_len_max; ++t) {
      const Tensor x_tensor = slicer.InputSlice(*x, t, "x");
      const Tensor& h_prev_tensor2 =
          t == 0 ? *h_prev_tensor : slicer.OutputSlice(h_out, t - 1, "h_prev");

      Tensor r_tensor = slicer.OutputSlice(r_out, t, "r_out");
      Tensor u_tensor = slicer.OutputSlice(u_out, t, "u_out");
      Tensor c_tensor = slicer.OutputSlice(c_out, t, "c_out");
      Tensor h_tensor = slicer.OutputSlice(h_out, t, "h_out");

      functor::GRUBlockCellFprop<Device, T, USE_CUBLAS>(batch_size, input_size, cell_size)(
          ctx, device, x_tensor.matrix<T>(), h_prev_tensor2.matrix<T>(),
          w_ru_tensor->matrix<T>(), w_c_tensor->matrix<T>(),
          b_ru_tensor->vec<T>(), b_c_tensor->vec<T>(), r_u_bar_tensor.matrix<T>(),
          r_tensor.matrix<T>(), u_tensor.matrix<T>(), c_tensor.matrix<T>(),
          h_tensor.matrix<T>(), x_h_prev_tensor.matrix<T>(),
          x_h_prevr_tensor.matrix<T>(), h_prevr_tensor.matrix<T>());
      slicer.FinishTimeStep();
    }

    if (seq_len_max < time_len) {
      Tensor h_tensor = h_out->Slice(seq_len_max, time_len);
      functor::TensorUnalignedZero<Device, T>()(
          device, h_tensor.unaligned_flat<float>());
    }
  }
};

// Register the Block GRU cell kernel for CPU.
#define REGISTER_KERNEL(T)                                            \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BlockGRU").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      BlockGRUOp<CPUDevice, T, false>);

REGISTER_KERNEL(float);
#undef REGISTER_KERNEL

template <typename Device, typename T, bool USE_CUBLAS>
class BlockGRUGradOp : public OpKernel {
 public:
  explicit BlockGRUGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    // Grab input tensors.
    const Tensor* seq_len_max_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("seq_len_max", &seq_len_max_tensor));

    const Tensor* x;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_ru", &w_ru_tensor));

    const Tensor* w_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_c", &w_c_tensor));

    const Tensor* b_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_ru", &b_ru_tensor));

    const Tensor* b_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_c", &b_c_tensor));

    const Tensor* r_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("r", &r_out));

    const Tensor* u_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("u", &u_out));

    const Tensor* c_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("c", &c_out));

    const Tensor* h_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h", &h_out));

    const Tensor* d_h = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("d_h", &d_h));

    const int64 time_len = x->dim_size(0);
    const int64 batch_size = x->dim_size(1);
    const int64 input_size = x->dim_size(2);
    const int64 cell_size = h_prev_tensor->dim_size(1);

    // Sanity checks for input shapes.

    // Shape of 'x' must be [time_len, batch_size, input_size]
    OP_REQUIRES(ctx, x->dims() == 3, errors::InvalidArgument("x must be 3D"));

    // Shape of 'h_prev' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, h_prev_tensor->dims() == 2,
                errors::InvalidArgument("h_prev must be 2D"));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "h_prev.dims(0) != batch_size: ",
                    h_prev_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ",
                    h_prev_tensor->dim_size(1), " vs. ", cell_size));

    // Shape of 'w_ru' must be [input_size + cell_size, 2 * cell_size]
    OP_REQUIRES(ctx, w_ru_tensor->dims() == 2,
                errors::InvalidArgument("w_ru must be 2D"));
    OP_REQUIRES(ctx, w_ru_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_ru.dim_size(0) != input_size + cell_size: ",
                    w_ru_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_ru_tensor->dim_size(1) == cell_size * 2,
                errors::InvalidArgument(
                    "w_ru.dim_size(1) != cell_size * 2: ",
                    w_ru_tensor->dim_size(1), " vs. ", cell_size * 2));

    // Shape of 'w_c' must be [input_size + cell_size, cell_size]
    OP_REQUIRES(ctx, w_c_tensor->dims() == 2,
                errors::InvalidArgument("w_c must be 2D"));
    OP_REQUIRES(ctx, w_c_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(0) != input_size + cell_size: ",
                    w_c_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_c_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(1) != cell_size: ",
                    w_c_tensor->dim_size(1), " vs. ", cell_size));

    // Shape of 'b_ru' must be [2 * cell_size]
    OP_REQUIRES(ctx, b_ru_tensor->dims() == 1,
                errors::InvalidArgument("b_ru must be 1D"));
    OP_REQUIRES(ctx, b_ru_tensor->dim_size(0) == cell_size * 2,
                errors::InvalidArgument(
                    "b_ru.dim_size(0) != cell_size * 2: ",
                    b_ru_tensor->dim_size(0), " vs. ", cell_size * 2));

    // Shape of 'b_c' must be [cell_size]
    OP_REQUIRES(ctx, b_c_tensor->dims() == 1,
                errors::InvalidArgument("b_c must be 1D"));
    OP_REQUIRES(ctx, b_c_tensor->dim_size(0) == cell_size,
                errors::InvalidArgument(
                    "b_c.dim_size(0) != cell_size: ",
                    b_c_tensor->dim_size(0), " vs. ", cell_size));

    // Shape of 'r' must be [time_len, batch_size, cell_size]
    OP_REQUIRES(ctx, r_out->dims() == 3, errors::InvalidArgument("r must be 3D"));

    // Shape of 'u' must be [time_len, batch_size, cell_size]
    OP_REQUIRES(ctx, u_out->dims() == 3, errors::InvalidArgument("u must be 3D"));

    // Shape of 'c' must be [time_len, batch_size, cell_size]
    OP_REQUIRES(ctx, c_out->dims() == 3, errors::InvalidArgument("c must be 3D"));

    // Shape of 'h' must be [time_len, batch_size, cell_size]
    OP_REQUIRES(ctx, h_out->dims() == 3, errors::InvalidArgument("h must be 3D"));

    // Create output tensors.
    Tensor* x_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        "x_grad", x->shape(), &x_grad));

    Tensor* h_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
        "h_prev_grad", h_prev_tensor->shape(), &h_prev_grad_tensor));

    Tensor* w_ru_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
        "w_ru_grad", w_ru_tensor->shape(), &w_ru_grad_tensor));

    Tensor* w_c_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
        "w_c_grad", w_c_tensor->shape(), &w_c_grad_tensor));

    Tensor* b_ru_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
        "b_ru_grad", b_ru_tensor->shape(), &b_ru_grad_tensor));

    Tensor* b_c_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
        "b_c_grad", b_c_tensor->shape(), &b_c_grad_tensor));

    // Allocate temp tensors.
    Tensor x_h_prev_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, input_size + cell_size}),
        &x_h_prev_tensor));

    Tensor x_h_prevr_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, input_size + cell_size}),
        &x_h_prevr_tensor));

    Tensor d_c_bar_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size}), &d_c_bar_tensor));

    Tensor d_r_bar_u_bar_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, 2 * cell_size}), &d_r_bar_u_bar_tensor));

    Tensor d_r_bar_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size}), &d_r_bar_tensor));

    Tensor d_u_bar_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size}), &d_u_bar_tensor));

    Tensor d_h_prevr_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size}), &d_h_prevr_tensor));

    Tensor d_x_component_1_h_prev_compenent_1;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, input_size + cell_size}),
        &d_x_component_1_h_prev_compenent_1));

    Tensor d_x_component_2_h_prevr;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, input_size + cell_size}),
        &d_x_component_2_h_prevr));

    Tensor d_h_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
        DataTypeToEnum<T>::v(),
        TensorShape({batch_size, cell_size}), &d_h_tensor));

    const Device& device = ctx->eigen_device<Device>();
    const int64 seq_len_max = seq_len_max_tensor->scalar<int64>()();
    SliceHelper<Device, T> slicer(ctx);

    functor::TensorZero<Device, T>()(device, h_prev_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, d_h_tensor.flat<float>());

    functor::TensorZero<Device, T>()(device, w_ru_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, w_c_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, b_ru_grad_tensor->flat<float>());
    functor::TensorZero<Device, T>()(device, b_c_grad_tensor->flat<float>());

    for (int64 t = seq_len_max - 1; t >= 0; --t) {
      const Tensor& x_tensor = slicer.InputSlice(*x, t, "x");
      const Tensor& h_prev_tensor2 =
          t == 0 ? *h_prev_tensor : slicer.InputSlice(*h_out, t - 1, "h_prev");
      const Tensor& r_tensor = slicer.InputSlice(*r_out, t, "r_out");
      const Tensor& u_tensor = slicer.InputSlice(*u_out, t, "u_out");
      const Tensor& c_tensor = slicer.InputSlice(*c_out, t, "c_out");

      // Combine previous h grad and h grad coming on top.
      const Tensor& const_h_prev_grad_tensor = *h_prev_grad_tensor;
      const Tensor const_d_h_slice = slicer.InputSlice(*d_h, t, "d_h");
      functor::TensorAdd<Device, T>()(
          device, const_h_prev_grad_tensor.flat<T>(),
          const_d_h_slice.flat<T>(), d_h_tensor.flat<T>());
      const Tensor& const_d_h_tensor = d_h_tensor;

      Tensor x_grad_tensor = slicer.OutputSlice(x_grad, t, "x_grad");

      functor::GRUBlockCellBprop<Device, T, USE_CUBLAS>(batch_size, input_size, cell_size)(
          ctx, device, x_tensor.matrix<T>(), h_prev_tensor2.matrix<T>(),
          w_ru_tensor->matrix<T>(), w_c_tensor->matrix<T>(),
          b_ru_tensor->vec<T>(), b_c_tensor->vec<T>(),
          r_tensor.matrix<T>(), u_tensor.matrix<T>(),
          c_tensor.matrix<T>(), const_d_h_tensor.matrix<T>(),
          x_grad_tensor.matrix<T>(), h_prev_grad_tensor->matrix<T>(),
          d_c_bar_tensor.matrix<T>(),
          d_r_bar_u_bar_tensor.matrix<T>(),
          d_r_bar_tensor.matrix<T>(), d_u_bar_tensor.matrix<T>(),
          d_h_prevr_tensor.matrix<T>(),
          d_x_component_1_h_prev_compenent_1.matrix<T>(),
          d_x_component_2_h_prevr.matrix<T>(),
          x_h_prev_tensor.matrix<T>(), x_h_prevr_tensor.matrix<T>(),
          w_ru_grad_tensor->matrix<T>(), w_c_grad_tensor->matrix<T>(),
          b_ru_grad_tensor->vec<T>(), b_c_grad_tensor->vec<T>());

      slicer.FinishTimeStep();
    }
    if (seq_len_max < time_len) {
      Tensor x_grad_tensor = x_grad->Slice(seq_len_max, time_len);
      functor::TensorUnalignedZero<Device, T>()(
          device, x_grad_tensor.unaligned_flat<T>());
    }

  }
};

// Register the gradient kernel for CPU.
#define REGISTER_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BlockGRUGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      BlockGRUGradOp<CPUDevice, T, false>);

REGISTER_KERNEL(float);
#undef REGISTER_KERNEL

// GPU support.
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

// Forward declare the GPU Fprop functor.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void TensorZero<GPUDevice, T>::operator()(const GPUDevice& d,               \
                                            typename TTypes<T>::Flat t);      \
                                                                              \
  template <>                                                                 \
  void TensorUnalignedZero<GPUDevice, T>::operator()(                         \
      const GPUDevice& d, typename TTypes<T>::UnalignedFlat t);               \
                                                                              \
  template <>                                                                 \
  void GRUBlockCellFprop<GPUDevice, T, true>::operator()(                     \
      OpKernelContext* ctx, const GPUDevice& d,                               \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix w_ru,                                   \
      typename TTypes<T>::ConstMatrix w_c, typename TTypes<T>::ConstVec b_ru, \
      typename TTypes<T>::ConstVec b_c, typename TTypes<T>::Matrix r_u_bar,   \
      typename TTypes<T>::Matrix r, typename TTypes<T>::Matrix u,             \
      typename TTypes<T>::Matrix c, typename TTypes<T>::Matrix h,             \
      typename TTypes<T>::Matrix x_h_prev,                                    \
      typename TTypes<T>::Matrix x_h_prevr,                                   \
      typename TTypes<T>::Matrix h_prevr);                                    \
  extern template struct GRUBlockCellFprop<GPUDevice, T, true>;               \
  extern template struct TensorZero<GPUDevice, T>;                            \
  extern template struct TensorUnalignedZero<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

// Register the Block GRU cell kernel for GPU.
#define REGISTER_GPU_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BlockGRU").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      BlockGRUOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
#undef REGISTER_GPU_KERNEL

// Forward declare the GPU Bprop functor.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void TensorCopy<GPUDevice, T>::operator()(const GPUDevice& d,                \
                                            typename TTypes<T>::ConstFlat src, \
                                            typename TTypes<T>::Flat dst);     \
                                                                               \
  template <>                                                                  \
  void TensorCopyUnaligned<GPUDevice, T>::operator()(                          \
      const GPUDevice& d, typename TTypes<T>::UnalignedConstFlat src,          \
      typename TTypes<T>::Flat dst);                                           \
                                                                               \
  template <>                                                                  \
  void TensorCopyToUnaligned<GPUDevice, T>::operator()(                        \
      const GPUDevice& d, typename TTypes<T>::ConstFlat src,                   \
      typename TTypes<T>::UnalignedFlat dst);                                  \
                                                                               \
  template <>                                                                  \
  void TensorAdd<GPUDevice, T>::operator()(                                    \
      const GPUDevice& d, typename TTypes<T>::ConstFlat a,                     \
      typename TTypes<T>::ConstFlat b, typename TTypes<T>::Flat c);            \
                                                                               \
  template <>                                                                  \
  void GRUBlockCellBprop<GPUDevice, T, true>::operator()(                      \
      OpKernelContext* ctx, const GPUDevice& d,                                \
      typename TTypes<T>::ConstMatrix x, typename TTypes<T>::ConstMatrix h,    \
      typename TTypes<T>::ConstMatrix w_ru,                                    \
      typename TTypes<T>::ConstMatrix w_c, typename TTypes<T>::ConstVec b_ru,  \
      typename TTypes<T>::ConstVec b_c, typename TTypes<T>::ConstMatrix r,     \
      typename TTypes<T>::ConstMatrix u, typename TTypes<T>::ConstMatrix c,    \
      typename TTypes<T>::ConstMatrix d_h, typename TTypes<T>::Matrix d_x,     \
      typename TTypes<T>::Matrix d_h_prev, typename TTypes<T>::Matrix d_c_bar, \
      typename TTypes<T>::Matrix d_r_bar_u_bar,                                \
      typename TTypes<T>::Matrix d_r_bar, typename TTypes<T>::Matrix d_u_bar,  \
      typename TTypes<T>::Matrix d_h_prevr,                                    \
      typename TTypes<T>::Matrix d_x_comp1_h_prev_comp1,                       \
      typename TTypes<T>::Matrix d_x_comp2_and_h_prevr,                        \
      typename TTypes<T>::Matrix x_h_prev,                                     \
      typename TTypes<T>::Matrix x_h_prevr,																		 \
      typename TTypes<T>::Matrix w_ru_grad,                                    \
      typename TTypes<T>::Matrix w_c_grad,																		 \
      typename TTypes<T>::Vec b_ru_grad, typename TTypes<T>::Vec b_c_grad); 	 \
      extern template struct TensorCopy<GPUDevice, T>;                         \
      extern template struct TensorAdd<GPUDevice, T>;                          \
      extern template struct GRUBlockCellBprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

// Register the gradient kernel for GPU.
#define REGISTER_GPU_KERNEL(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BlockGRUGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      BlockGRUGradOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA
