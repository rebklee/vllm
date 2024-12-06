// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "../gemm_a8w8_common.cuh"

template <typename DEDataType>
torch::Tensor
a8w8_rowwise_256x32x256x128_32x32_1x2_8x32x1_8x32x1_1x8x1x32_8x8x1_1x2_intrawave_v3(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
  // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool pad =  (K % 128 != 0);

  // Dispatch based on whether padding is needed or not.
  if (pad) {
  using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType,
      256,
      32,
      256,
      128,
      32,
      32,
      1,
      2,
      S<8, 32, 1>,
      S<8, 32, 1>,
      S<1, 8, 1, 32>,
      S<8, 8, 1>,
      1,
      2,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
        // Run kernel instance.
  return gemm_a8w8_rowwise_impl<DEDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
  else{
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType,
      256,
      32,
      256,
      128,
      32,
      32,
      1,
      2,
      S<8, 32, 1>,
      S<8, 32, 1>,
      S<1, 8, 1, 32>,
      S<8, 8, 1>,
      1,
      2,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3>;
        // Run kernel instance.
  return gemm_a8w8_rowwise_impl<DEDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template torch::Tensor
a8w8_rowwise_256x32x256x128_32x32_1x2_8x32x1_8x32x1_1x8x1x32_8x8x1_1x2_intrawave_v3<F16>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x32x256x128_32x32_1x2_8x32x1_8x32x1_1x8x1x32_8x8x1_1x2_intrawave_v3<B16>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);
