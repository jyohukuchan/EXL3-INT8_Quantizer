#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemm_kernel.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "codebook.cuh"
#include "exl3_gemv.cuh"
#include "exl3_gemv_int8.cuh"
#include "coop_autotune.cuh"
#include "reconstruct.cuh"
#include "hadamard.cuh"
#include "turing_fallback.cuh"
#include <ATen/Functions.h>
#include <set>
#include <vector>

int exl3_gemm_tilesize_k_g[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n_g[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim_g[] = {EXL3_GEMM_BLOCKDIM};

std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

uint64_t roundup_pow2(uint64_t x)
{
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x |= x >> 32;
    return x + 1;
}

uint64_t gemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb
)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) MIN(roundup_pow2(size_m), 16));
    mix((uint64_t) size_k);
    mix((uint64_t) size_n);
    mix((uint64_t) K);
    mix(c_fp32 ? 1ull : 0ull);
    mix((uint64_t) device);
    mix((uint64_t) cc);
    mix((uint64_t) max_num_sms);
    mix((uint64_t) cb);
    return h;
}

uint64_t mgemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb,
    int bszm_in,
    int bszm_out
)
{
    uint64_t h = gemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb);
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) MIN(bszm_in, 24));
    mix((uint64_t) MIN(bszm_out, 24));
    return h;
}

int exl3_gemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);

    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = nullptr;
    if (suh_ptr)
    {

        A_had_ptr = (half*) OPTPTR(A_had);

    }

    const half* svh_ptr = (const half*) OPTPTR(svh);

    int device;
    cudaGetDevice(&device);
    int num_sms = force_num_sms ? force_num_sms : DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    int K = B.size(2) / 16;
    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;

    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    if (cc == CC_OLD)
    {
        TORCH_CHECK(!graph, "exl3_gemm: graph capture is unavailable on devices below sm_80");
        auto a = A.reshape({size_m, size_k});
        auto c = C.reshape({size_m, size_n});
        auto weight = at::empty({size_k, size_n}, A.options());
        reconstruct(weight, B, K, mcg, mul1);
        auto transformed = a;
        if (suh)
        {
            transformed = A_had ? A_had.value().reshape({size_m, size_k}) : at::empty_like(a);
            had_r_128(a, transformed, suh, c10::nullopt, 1.0f);
        }
        if (c_fp32) c.copy_(at::matmul(transformed.to(at::kFloat), weight.to(at::kFloat)));
        else c.copy_(at::matmul(transformed, weight));
        if (svh) had_r_128(c, c, c10::nullopt, svh, 1.0f);
        return 0;
    }
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    if (EXL3_MUL1_MULT != 0xDCD12Du && mul1 && exl3_gemv_int8_enabled())
    {
        if (exl3_gemv_int8(A, B, C, suh, A_had, svh, stream, graph))
            return 0;
    }

    int block_dim;
    int shape_idx;
    fp_exl3_gemm_kernel kernel;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_gemm_A, 0);
            graph->record_param(kernel_ptr, GP_gemm_B_trellis, 1);
            graph->record_param(kernel_ptr, GP_gemm_C, 2);
            graph->record_param(kernel_ptr, GP_gemm_B_suh, 7);
            graph->record_param(kernel_ptr, GP_gemm_A_had, 8);
            graph->record_param(kernel_ptr, GP_gemm_B_svh, 9);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    if (EXL3_MUL1_MULT != 0xDCD12Du && force_shape_idx <= 0 && force_num_sms <= 0)
    {
        void* gemv_kernel = nullptr;
        if (exl3_gemv_try_launch
        (
            kernelArgs, size_m, size_k, size_n, K, cb, c_fp32,
            suh_ptr && A_had_ptr && svh_ptr,
            device, stream, &gemv_kernel, false
        ))
        {
            add_graph_args(gemv_kernel);
            cuda_check(cudaPeekAtLastError());
            return 90;
        }
    }

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = gemm_autotune_hash(MAX(size_m, 2), size_k, size_n, K, c_fp32, device, cc, num_sms, cb);
        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, SMEM_MAX, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
        std::vector<CoopAutotuneCandidate> candidates;
        for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
        {
            if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K)) continue;

            fp_exl3_gemm_kernel candidate_kernel = get_gemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb);
            if (!candidate_kernel) continue;

            int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
            int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
            int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
            int max_candidate_sms = MAX(MIN(max_slices, num_sms), 1);

            candidates.push_back
            ({
                (void*) candidate_kernel,
                exl3_gemm_blockdim_g[candidate_shape_idx],
                max_candidate_sms,
                1,
                max_candidate_sms,
                candidate_shape_idx
            });
        }
        TORCH_CHECK(!candidates.empty(), "exl3_gemm autotune: no compatible kernel shapes");

        tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, SMEM_MAX, stream, (size_t) size_k * size_n);
        if (graph)
        add_graph_args((void*) tuned.kernel);
        cuda_check(cudaPeekAtLastError());
        return tuned.tag;
    }

    kernel = select_exl3_gemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb
    );
    if (!kernel) return 0;

    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }
    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        num_sms,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
)
{
    return exl3_gemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        force_shape_idx,
        mcg,
        mul1,
        force_num_sms,
        nullptr
    );
}

int exl3_mgemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int min_index,
    int max_index,
    int force_num_sms,
    Graph* graph,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs,
    const c10::optional<at::Tensor>& n_stride_list,
    const c10::optional<at::Tensor>& had_src_list,
    int num_had_src
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kLong);
    TORCH_CHECK_DTYPE(suh, kLong);
    TORCH_CHECK_DTYPE(svh, kLong);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK_DIM(A, 3);
    TORCH_CHECK_DIM(B, 1);
    TORCH_CHECK_DIM(suh, 1);
    TORCH_CHECK_DIM(svh, 1);
    TORCH_CHECK_DIM(C, 3);

    TORCH_CHECK_SHAPES(A, 1, C, 1, 1);
    if (!had_src_list) TORCH_CHECK_SHAPES(B, 0, suh, 0, 1);
    TORCH_CHECK_SHAPES(B, 0, svh, 0, 1);

    int bsz = A.size(1);
    int bszm_in = A.size(0);
    int bszm_out = C.size(0);

    const int* size_n_list_ptr = nullptr;
    void** c_list_ptr = nullptr;
    if (size_n_list)
    {
        TORCH_CHECK(c_ptrs, "exl3_mgemm: size_n_list requires c_ptrs");
        TORCH_CHECK_DTYPE(size_n_list.value(), kInt);
        TORCH_CHECK_DTYPE(c_ptrs.value(), kLong);
        TORCH_CHECK(num_tokens == 1 && min_index < 0 && !weights,
                    "exl3_mgemm: per-matrix widths incompatible with multi-token/filtering/weights");
        size_n_list_ptr = (const int*) size_n_list.value().data_ptr();
        c_list_ptr = (void**) c_ptrs.value().data_ptr();
        bszm_out = (int) c_ptrs.value().size(0);
    }
    int bszm = MAX(bszm_in, bszm_out);

    const int* n_stride_list_ptr = nullptr;
    const int* had_src_list_ptr = nullptr;
    if (had_src_list)
    {
        TORCH_CHECK(size_n_list && c_ptrs && n_stride_list,
                    "exl3_mgemm: sliced mode requires size_n_list, c_ptrs and n_stride_list");
        TORCH_CHECK(bszm_in == 1 && !indices && !weights && num_tokens == 1 && min_index < 0,
                    "exl3_mgemm: sliced mode is single-input, unfiltered and single-token");
        TORCH_CHECK_DTYPE(had_src_list.value(), kInt);
        TORCH_CHECK_DTYPE(n_stride_list.value(), kInt);
        TORCH_CHECK(had_src_list.value().size(0) >= bszm_out && n_stride_list.value().size(0) >= bszm_out,
                    "exl3_mgemm: had_src_list / n_stride_list must have one entry per slice");
        TORCH_CHECK(num_had_src > 0 && suh.size(0) >= num_had_src,
                    "exl3_mgemm: sliced mode needs one suh entry per source");
        n_stride_list_ptr = (const int*) n_stride_list.value().data_ptr();
        had_src_list_ptr = (const int*) had_src_list.value().data_ptr();
    }
    else
    {
        TORCH_CHECK(!n_stride_list, "exl3_mgemm: n_stride_list requires had_src_list");
    }

    int64_t had_slabs = had_src_list ? num_had_src : bszm;
    TORCH_CHECK(A_had.numel() >= had_slabs * A.size(1) * A.size(2),
                "exl3_mgemm: A_had must hold bszm * m * k elements");

    const int64_t* indices_ptr = (const int64_t*) OPTPTR(indices);
    const half* weights_ptr = (const half*) OPTPTR(weights);

    if (indices)
    {
        TORCH_CHECK_DIM(indices.value(), 2);
        int num_indices = indices.value().size(1);
        TORCH_CHECK(num_indices <= bszm_in || num_indices <= bszm_out, "mgemm: too many indices for tensor batch");
        if (bszm_in > num_indices) bszm_in = num_indices;
        if (bszm_out > num_indices) bszm_out = num_indices;
    }

    if (weights)
    {
        TORCH_CHECK_DIM(weights.value(), 2);
    }

    int size_m = A.size(1);
    int size_k = A.size(2);
    int size_n = C.size(2);

    int device;
    cudaGetDevice(&device);
    int total_sms = DevCtx::instance().get_num_sms(device);
    int num_sms = force_num_sms ? force_num_sms : total_sms;
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    const half* A_ptr = (const half*) A.data_ptr();
    if (cc == CC_OLD)
    {
        TORCH_CHECK(!graph, "exl3_mgemm: graph capture is unavailable on devices below sm_80");
        auto bp = turing_host(B);
        auto sp = turing_host(suh);
        auto vp = turing_host(svh);
        auto ids = indices ? turing_host(indices.value()).reshape({-1}) : at::Tensor();
        auto ws = weights ? turing_host(weights.value()).to(at::kFloat).reshape({-1}) : at::Tensor();
        auto widths = size_n_list ? turing_host(size_n_list.value()) : at::Tensor();
        auto destinations = c_ptrs ? turing_host(c_ptrs.value()) : at::Tensor();
        auto strides = n_stride_list ? turing_host(n_stride_list.value()) : at::Tensor();
        auto sources = had_src_list ? turing_host(had_src_list.value()) : at::Tensor();
        int total = MAX(bszm_in, bszm_out);
        if (ids.defined()) total = MIN(total, (int) ids.numel());
        std::vector<int> selected;
        std::vector<float> factors;
        for (int j = 0; j < total; ++j)
        {
            int id = ids.defined() ? (int) ids.data_ptr<int64_t>()[j] : j;
            if (min_index >= 0)
            {
                bool active = id >= min_index && id < max_index;
                if (!active && num_tokens == 1) continue;
                id = active ? id - min_index : -1;
            }
            selected.push_back(id);
            if (ws.defined()) factors.push_back(ws.data_ptr<float>()[j]);
        }
        TORCH_CHECK(num_tokens > 0 && (!ws.defined() || selected.size() % num_tokens == 0), "exl3_mgemm: invalid token count");
        std::vector<at::Tensor> results;
        for (int j = 0; j < (int) selected.size(); ++j)
        {
            int id = selected[j];
            if (id < 0) { results.emplace_back(); continue; }
            TORCH_CHECK(id < bp.numel() && id < vp.numel(), "exl3_mgemm: expert index out of range");
            int src = sources.defined() ? sources.data_ptr<int>()[id] : id;
            TORCH_CHECK(src >= 0 && src < sp.numel(), "exl3_mgemm: input scale index out of range");
            int width = widths.defined() ? widths.data_ptr<int>()[id] : size_n;
            int stride = strides.defined() ? strides.data_ptr<int>()[id] : width;
            auto x = A.select(0, A.size(0) == 1 ? 0 : j);
            auto had = A_had.reshape({-1, size_m, size_k}).select(0, sources.defined() ? src : j);
            auto result = turing_linear(x, bp.data_ptr<int64_t>()[id], sp.data_ptr<int64_t>()[src],
                                        vp.data_ptr<int64_t>()[id], K, mcg, mul1, width, stride, C.scalar_type(), &had);
            if (ws.defined()) result.mul_(factors[j]);
            if (destinations.defined())
            {
                auto dest = turing_view(destinations.data_ptr<int64_t>()[id], {size_m, stride}, C.options());
                dest.narrow(1, 0, width).copy_(result);
            }
            else C.select(0, j).copy_(result);
            results.push_back(result);
        }
        if (ws.defined())
        {
            int group = selected.size() / num_tokens;
            for (int t = 0; t < num_tokens; ++t)
            {
                auto acc = at::zeros_like(C.select(0, t));
                for (int j = t * group; j < (t + 1) * group; ++j)
                    if (results[j].defined()) acc.add_(results[j]);
                C.select(0, t).copy_(acc);
            }
        }
        return 0;
    }
    const uintptr_t* B_ptr_ptr = (const uintptr_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();
    const half* A_had_ptr = (const half*) A_had.data_ptr();
    const uintptr_t* suh_ptr_ptr = (const uintptr_t*) suh.data_ptr();
    const uintptr_t* svh_ptr_ptr = (const uintptr_t*) svh.data_ptr();

    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int shape_idx;
    int block_dim;
    fp_exl3_mgemm_kernel kernel;
    int concurrency;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr_ptr,
        (void*)& indices_ptr,
        (void*)& weights_ptr,
        (void*)& bszm_in,
        (void*)& bszm_out,
        (void*)& min_index,
        (void*)& max_index,
        (void*)& num_tokens,
        (void*)& size_n_list_ptr,
        (void*)& c_list_ptr,
        (void*)& n_stride_list_ptr,
        (void*)& had_src_list_ptr,
        (void*)& num_had_src
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_mgemm_A, 0);
            graph->record_param(kernel_ptr, GP_mgemm_C, 2);
            graph->record_param(kernel_ptr, GP_mgemm_indices, 10);
            graph->record_param(kernel_ptr, GP_mgemm_weights, 11);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = mgemm_autotune_hash
        (
            size_m, size_k, size_n, K, c_fp32, device, cc, total_sms, cb, bszm_in, bszm_out
        );
        if (had_src_list) autotune_key ^= 0x9e3779b97f4a7c15ull;

        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, SMEM_MAX, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
        if (!graph)
        {
            std::vector<CoopAutotuneCandidate> candidates;
            for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
            {
                if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K)) continue;

                fp_exl3_mgemm_kernel candidate_kernel = get_mgemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb);
                if (!candidate_kernel) continue;

                int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
                int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
                int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
                int max_candidate_sms = MAX(MIN(max_slices, total_sms), 1);

                candidates.push_back
                ({
                    (void*) candidate_kernel,
                    exl3_gemm_blockdim_g[candidate_shape_idx],
                    max_candidate_sms,
                    bszm,
                    total_sms,
                    candidate_shape_idx
                });
            }
            TORCH_CHECK(!candidates.empty(), "exl3_mgemm autotune: no compatible kernel shapes");

            tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, SMEM_MAX, stream, (size_t) size_k * size_n * bszm);
            add_graph_args((void*) tuned.kernel);

            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
    }

    kernel = select_exl3_mgemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb, bszm_in, bszm_out
    );
    int tilesize_k = exl3_gemm_tilesize_k_g[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n_g[shape_idx];
    int tiles = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
    num_sms = tiles;
    if (num_sms * bszm > total_sms) num_sms = MAX(total_sms / bszm, 1);
    if (num_sms <= total_sms && tiles / num_sms > 48) num_sms = MIN(total_sms, num_sms * 2);
    concurrency = MIN(total_sms / num_sms, bszm);

    dim3 block_grid(num_sms, 1, concurrency);

    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        kernel_attr_set[device].insert((void*) kernel);
    }

    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        block_grid,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms,
    int num_tokens,
    const c10::optional<at::Tensor>& size_n_list,
    const c10::optional<at::Tensor>& c_ptrs,
    const c10::optional<at::Tensor>& n_stride_list,
    const c10::optional<at::Tensor>& had_src_list,
    int num_had_src
)
{
    return exl3_mgemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        indices,
        weights,
        K,
        force_shape_idx,
        mcg_mult,
        mul1_mult,
        min_index,
        max_index,
        force_num_sms,
        nullptr,
        num_tokens,
        size_n_list,
        c_ptrs,
        n_stride_list,
        had_src_list,
        num_had_src
    );
}
