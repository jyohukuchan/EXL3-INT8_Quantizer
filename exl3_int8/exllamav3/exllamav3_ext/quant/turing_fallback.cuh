#pragma once

#include <ATen/ATen.h>
#include "reconstruct.cuh"
#include "hadamard.cuh"

inline at::Tensor turing_view(int64_t address, at::IntArrayRef shape, const at::TensorOptions& options)
{
    TORCH_CHECK(address, "null device pointer");
    return at::from_blob(reinterpret_cast<void*>(address), shape, options);
}

inline at::Tensor turing_host(const at::Tensor& t)
{
    return t.to(at::kCPU).contiguous();
}

inline at::Tensor turing_linear(const at::Tensor& x, int64_t packed_ptr, int64_t suh_ptr, int64_t svh_ptr,
                                int bits, bool mcg, bool mul1, int width, int stride,
                                at::ScalarType dtype, at::Tensor* had_out = nullptr, bool half_matmul = false)
{
    int in = x.size(-1);
    auto packed = turing_view(packed_ptr, {in / 16, stride / 16, bits * 16}, x.options().dtype(at::kShort));
    auto suh = turing_view(suh_ptr, {in}, x.options());
    auto svh = turing_view(svh_ptr, {width}, x.options());
    auto weight = at::empty({in, width}, x.options());
    reconstruct_slice(weight, packed, bits, mcg, mul1, 0);
    auto had = at::empty_like(x);
    had_r_128(x, had, suh, c10::nullopt, 1.0f);
    if (had_out) had_out->copy_(had);
    auto out = dtype == at::kFloat && !half_matmul ? at::matmul(had.to(at::kFloat), weight.to(at::kFloat)) : at::matmul(had, weight).to(dtype);
    had_r_128(out, out, c10::nullopt, svh, 1.0f);
    return out;
}
