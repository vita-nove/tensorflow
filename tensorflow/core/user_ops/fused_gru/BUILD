load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
		name = "_fused_gru_ops.so",
		srcs = [
				"blas_gemm.cc",
				"blas_gemm.h",
				"fused_gru_ops.cc",
				"fused_gru_ops.h",
				"fused_gru_ops_reg.cc",
				],
		gpu_srcs = [
				"blas_gemm.h",
				"fused_gru_ops.cu.cc",
				"fused_gru_ops.h",
				],
		deps = ["//tensorflow/core/kernels:eigen_helpers"],
)
