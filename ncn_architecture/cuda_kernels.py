# ncn_architecture/cuda_kernels.py

"""
 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
 To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
 or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

 Original Author: Michael Morgan
 2025.11.25
 Github: https://github.com/Mmorgan-ML
 Email: mmorgankorea@gmail.com
 Twitter: @Mmorgan_ML
"""

import torch
from torch.utils.cpp_extension import load_inline

# --- CUDA SOURCE ---
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// --- HELPER: WARP REDUCTION ---
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// =========================================================================
// KERNEL 1: FUSED RMSNORM (Block Reduction)
// =========================================================================
template <typename T>
__global__ void rmsnorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int d_model,
    float eps
) {
    extern __shared__ float s_mem[]; 
    int tid = threadIdx.x;
    int row_idx = blockIdx.x;
    
    const T* row_input = input + row_idx * d_model;
    T* row_output = output + row_idx * d_model;

    float thread_sum_sq = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]);
        thread_sum_sq += val * val;
    }

    thread_sum_sq = warpReduceSum(thread_sum_sq);

    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) s_mem[warp_id] = thread_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float block_sum_sq = (tid < (blockDim.x / 32)) ? s_mem[tid] : 0.0f;
        block_sum_sq = warpReduceSum(block_sum_sq);
        if (tid == 0) {
            s_mem[0] = rsqrtf(block_sum_sq / d_model + eps);
        }
    }
    __syncthreads();

    float inv_rms = s_mem[0];
    for (int i = tid; i < d_model; i += blockDim.x) {
        row_output[i] = static_cast<T>(static_cast<float>(row_input[i]) * inv_rms * static_cast<float>(weight[i]));
    }
}

// =========================================================================
// KERNEL 2: FUSED NCN ACTUATOR (Split & Coalesce)
// =========================================================================
template <typename T>
__global__ void ncn_actuator_kernel(
    const T* __restrict__ input,
    T* __restrict__ out_gain,
    T* __restrict__ out_precision,
    T* __restrict__ out_gate,
    int total_triplets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_triplets) {
        int base_ptr = idx * 3;
        float raw_g = static_cast<float>(input[base_ptr]);
        float raw_p = static_cast<float>(input[base_ptr + 1]);
        float raw_f = static_cast<float>(input[base_ptr + 2]);

        // 2 * Sigmoid(g) -> Range [0, 2]
        // GUARD: Prevent exp overflow for large negative inputs
        float g_val;
        if (raw_g < -88.0f) g_val = 0.0f;
        else g_val = 2.0f * (1.0f / (1.0f + expf(-raw_g)));

        // Softplus(p) + 0.01
        // GUARD: Linear approximation for large inputs to prevent exp overflow
        float p_val;
        if (raw_p > 20.0f) {
            p_val = raw_p + 0.01f;
        } else {
            p_val = logf(1.0f + expf(raw_p)) + 0.01f;
        }
        
        // CRITICAL FIX: Thermodynamic Limit (Match Paper Section 3.5)
        // Clamp precision to max 4.0 to prevent gradient explosion and Argmax collapse.
        p_val = fminf(p_val, 4.0f);

        // Sigmoid(f)
        // GUARD: Prevent exp overflow
        float f_val;
        if (raw_f < -88.0f) f_val = 0.0f;
        else f_val = 1.0f / (1.0f + expf(-raw_f));

        out_gain[idx]      = static_cast<T>(g_val);
        out_precision[idx] = static_cast<T>(p_val);
        out_gate[idx]      = static_cast<T>(f_val);
    }
}

// =========================================================================
// KERNEL 3: KV CACHE UPDATE (Zero-Copy)
// =========================================================================
template <typename T>
__global__ void kv_update_kernel(
    T* __restrict__ cache,
    const T* __restrict__ new_tok,
    const int* __restrict__ positions,
    int stride_b, int stride_s, int stride_h, int stride_d,
    int batch_size, int num_heads, int head_dim, int cache_seq_len
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int d_idx = threadIdx.x;

    if (b >= batch_size || h >= num_heads) return;
    int pos = positions[b]; 
    if (pos >= cache_seq_len) return;

    for (int d = d_idx; d < head_dim; d += blockDim.x) {
        int src_idx = (b * num_heads * head_dim) + (h * head_dim) + d;
        long long dst_idx = (long long)b * stride_b + (long long)pos * stride_s + 
                            (long long)h * stride_h + (long long)d * stride_d;
        cache[dst_idx] = new_tok[src_idx];
    }
}

// =========================================================================
// KERNEL 4: ONLINE SOFTMAX UPDATE (Fused Cross-Entropy Support)
// =========================================================================
// This kernel updates running Max and Sum-Exp statistics using a new "chunk" of logits.
// It avoids materializing the full (B, S, V) tensor by processing V in chunks.
//
// running_max: (Batch * Seq)
// running_sum: (Batch * Seq)
// chunk_logits: (Batch * Seq, Chunk_Size)
template <typename T>
__global__ void online_softmax_update_kernel(
    float* __restrict__ running_max,
    float* __restrict__ running_sum,
    const T* __restrict__ chunk_logits,
    int num_rows,       // Batch * Seq
    int chunk_size      // Cols in this chunk
) {
    // One block per Row (Token)
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= num_rows) return;

    // 1. Find Max in this Chunk
    float local_max = -1e30f; // Neg infinity
    
    for (int i = tid; i < chunk_size; i += blockDim.x) {
        float val = static_cast<float>(chunk_logits[row * chunk_size + i]);
        if (val > local_max) local_max = val;
    }
    local_max = warpReduceMax(local_max);
    
    // Shared mem for block reduction of Max
    extern __shared__ float s_softmax[];
    if ((tid % 32) == 0) s_softmax[tid / 32] = local_max;
    __syncthreads();
    
    float chunk_max_val = -1e30f;
    if (tid == 0) {
        for (int i = 0; i < (blockDim.x / 32); i++) {
            chunk_max_val = max(chunk_max_val, s_softmax[i]);
        }
        // Update global running max
        float old_max = running_max[row];
        float new_max = max(old_max, chunk_max_val);
        
        // Store temporarily for broadcasting
        s_softmax[0] = old_max;
        s_softmax[1] = new_max;
        
        // Write back global max
        running_max[row] = new_max;
    }
    __syncthreads();
    
    float old_max = s_softmax[0];
    float new_max = s_softmax[1];

    // 2. Compute Sum Exp for this Chunk AND update old sum
    // Formula: sum_new = sum_old * exp(old_max - new_max) + sum(exp(chunk_vals - new_max))
    
    float local_sum = 0.0f;
    for (int i = tid; i < chunk_size; i += blockDim.x) {
        float val = static_cast<float>(chunk_logits[row * chunk_size + i]);
        local_sum += expf(val - new_max);
    }
    local_sum = warpReduceSum(local_sum);
    
    if ((tid % 32) == 0) s_softmax[tid / 32] = local_sum;
    __syncthreads();
    
    if (tid == 0) {
        float chunk_sum = 0.0f;
        for (int i = 0; i < (blockDim.x / 32); i++) chunk_sum += s_softmax[i];
        
        float old_sum = running_sum[row];
        // Scaling factor for the old sum to align with new max
        float scale = expf(old_max - new_max);
        
        running_sum[row] = old_sum * scale + chunk_sum;
    }
}

// =========================================================================
// KERNEL 5: MODULATED ADD (Original/Legacy)
// =========================================================================
template <typename scalar_t>
__global__ void modulated_add_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ gain,
    scalar_t* __restrict__ output,
    int total_elements, int d_model 
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int gain_idx = idx / d_model;
        output[idx] = x[idx] * gain[gain_idx] + residual[idx];
    }
}

template <typename scalar_t>
__global__ void modulated_add_bwd_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ gain,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_residual,
    int total_elements, int d_model
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int gain_idx = idx / d_model;
        grad_x[idx] = grad_output[idx] * gain[gain_idx];
        grad_residual[idx] = grad_output[idx];
    }
}
"""

# --- C++ WRAPPER SOURCE ---
cpp_source = r"""
torch::Tensor launch_rmsnorm(torch::Tensor input, torch::Tensor weight, float eps);
std::vector<torch::Tensor> launch_ncn_actuator(torch::Tensor input);
void launch_kv_update(torch::Tensor cache, torch::Tensor new_tok, torch::Tensor positions);
void launch_online_softmax_update(torch::Tensor running_max, torch::Tensor running_sum, torch::Tensor chunk_logits);
torch::Tensor modulated_add_forward(torch::Tensor x, torch::Tensor residual, torch::Tensor gain);
std::vector<torch::Tensor> modulated_add_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor gain);
"""

# --- CUDA WRAPPERS (Python Binding Logic) ---
cuda_wrapper = r"""

// 1. RMSNorm
torch::Tensor launch_rmsnorm(torch::Tensor input, torch::Tensor weight, float eps) {
    auto output = torch::empty_like(input);
    int batch_tokens = input.numel() / input.size(-1);
    int d_model = input.size(-1);
    int threads = 256;
    if (d_model > 256) threads = 512;
    if (d_model > 512) threads = 1024;
    int shared_mem = 64 * sizeof(float); 

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rmsnorm_kernel", ([&] {
        rmsnorm_kernel<scalar_t><<<batch_tokens, threads, shared_mem>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), d_model, eps
        );
    }));
    return output;
}

// 2. NCN Actuator
std::vector<torch::Tensor> launch_ncn_actuator(torch::Tensor input) {
    int total_elements = input.numel();
    int total_triplets = total_elements / 3;
    auto sizes = input.sizes().vec(); sizes.pop_back();
    auto opts = input.options();
    auto out_gain = torch::empty(sizes, opts);
    auto out_prec = torch::empty(sizes, opts);
    auto out_gate = torch::empty(sizes, opts);
    int threads = 256;
    int blocks = (total_triplets + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "ncn_actuator_kernel", ([&] {
        ncn_actuator_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), out_gain.data_ptr<scalar_t>(), out_prec.data_ptr<scalar_t>(), out_gate.data_ptr<scalar_t>(), total_triplets
        );
    }));
    return {out_gain, out_prec, out_gate};
}

// 3. KV Update
void launch_kv_update(torch::Tensor cache, torch::Tensor new_tok, torch::Tensor positions) {
    int batch_size = new_tok.size(0);
    int num_heads = new_tok.size(1);
    int head_dim = new_tok.size(2);
    int cache_seq_len = cache.size(1);
    dim3 grid(batch_size, num_heads);
    int threads = std::min(1024, head_dim);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(cache.scalar_type(), "kv_update_kernel", ([&] {
        kv_update_kernel<scalar_t><<<grid, threads>>>(
            cache.data_ptr<scalar_t>(), new_tok.data_ptr<scalar_t>(), positions.data_ptr<int>(),
            cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
            batch_size, num_heads, head_dim, cache_seq_len
        );
    }));
}

// 4. Online Softmax Update
void launch_online_softmax_update(torch::Tensor running_max, torch::Tensor running_sum, torch::Tensor chunk_logits) {
    // running_max/sum: (Batch*Seq)
    // chunk_logits: (Batch*Seq, ChunkSize)
    int num_rows = running_max.numel();
    int chunk_size = chunk_logits.size(1);
    
    int threads = 256; // Standard block
    int shared_mem = 32 * sizeof(float); // For Warp reduction buffer

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(chunk_logits.scalar_type(), "online_softmax_update", ([&] {
        online_softmax_update_kernel<scalar_t><<<num_rows, threads, shared_mem>>>(
            running_max.data_ptr<float>(), // Always float accumulator
            running_sum.data_ptr<float>(), // Always float accumulator
            chunk_logits.data_ptr<scalar_t>(),
            num_rows, chunk_size
        );
    }));
}

// 5. Modulated Add (Legacy)
torch::Tensor modulated_add_forward(torch::Tensor x, torch::Tensor residual, torch::Tensor gain) {
    auto output = torch::empty_like(x);
    int total_elements = x.numel();
    int d_model = x.size(-1); 
    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "modulated_add_forward", ([&] {
        modulated_add_fwd_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), gain.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), total_elements, d_model
        );
    }));
    return output;
}

std::vector<torch::Tensor> modulated_add_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor gain) {
    auto grad_x = torch::empty_like(x);
    auto grad_residual = torch::empty_like(x);
    int total_elements = x.numel();
    int d_model = x.size(-1);
    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "modulated_add_backward", ([&] {
        modulated_add_bwd_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(), gain.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(), grad_residual.data_ptr<scalar_t>(), total_elements, d_model
        );
    }));
    return {grad_x, grad_residual};
}
"""

print("Compiling NCN High-Performance Kernels (v5.3 - Stability Fixes)...")
try:
    _ncn_cuda = load_inline(
        name='ncn_fused_ops_v5_3', 
        cpp_sources=cpp_source,
        cuda_sources=cuda_source + cuda_wrapper,
        functions=[
            'launch_rmsnorm', 
            'launch_ncn_actuator', 
            'launch_kv_update',
            'launch_online_softmax_update',
            'modulated_add_forward', 
            'modulated_add_backward'
        ],
        extra_cuda_cflags=[
            '-O3', 
            '--use_fast_math', 
            '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
            '-allow-unsupported-compiler' # <--- ADDED FIX FOR WINDOWS
        ],
        with_cuda=True,
        verbose=False
    )
    print("NCN Kernels Compiled Successfully.")
except Exception as e:
    print(f"KERNEL COMPILATION FAILED: {e}")
    _ncn_cuda = None

# --- PYTHON BINDINGS ---

class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.eps = eps
        # Save original weights (likely FP32) for backward pass
        ctx.save_for_backward(x, weight)
        
        if _ncn_cuda and x.is_cuda:
            # TYPE SAFETY FIX for AMP:
            # During Mixed Precision, activations 'x' are cast to Half, but 'weight' remains Float.
            # The C++ kernel expects input and weight to be of the same template type 'scalar_t'.
            # We explicitly cast weight to match x.dtype (Half) for the forward execution.
            # This does not affect the saved 'weight' for backward (already saved above).
            kernel_weight = weight
            if weight.dtype != x.dtype:
                kernel_weight = weight.to(x.dtype)
            
            return _ncn_cuda.launch_rmsnorm(x, kernel_weight, eps)
            
        # Fallback (Only use if compilation fails, but we want to know if it fails!)
        if _ncn_cuda is None:
            print("WARNING: Using PyTorch Fallback for RMSNorm (CUDA Failed)")
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        with torch.enable_grad():
             x_in = x.detach().requires_grad_(True)
             w_in = weight.detach().requires_grad_(True)
             out = x_in * torch.rsqrt(x_in.pow(2).mean(-1, keepdim=True) + eps) * w_in
             out.backward(grad_output)
        return x_in.grad, w_in.grad, None

class FusedNCNActuator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if _ncn_cuda and x.is_cuda:
            return _ncn_cuda.launch_ncn_actuator(x)
        raise RuntimeError("CUDA Kernel failed to load for NCN Actuator")

    @staticmethod
    def backward(ctx, grad_gain, grad_prec, grad_gate):
        x, = ctx.saved_tensors
        # x is the RAW INPUT
        g, p, f = x.chunk(3, dim=-1)
        
        d_g = d_p = d_f = None
        
        # --- FIXED GAIN MATH ---
        if grad_gain is not None:
             # 1. Compute Sigmoid of raw input
             sig_g = torch.sigmoid(g)
             # 2. Apply derivative: 2 * s * (1 - s)
             d_g = grad_gain * (2.0 * sig_g * (1.0 - sig_g))
             
        if grad_prec is not None:
             # Derivative of Softplus(x) is Sigmoid(x)
             d_p = grad_prec * torch.sigmoid(p)
             
        if grad_gate is not None:
             # Derivative of Sigmoid(x) is s * (1 - s)
             sig_f = torch.sigmoid(f)
             d_f = grad_gate * (sig_f * (1.0 - sig_f))
             
        return torch.cat([d_g, d_p, d_f], dim=-1)

class FusedModulatedAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, gain):
        ctx.save_for_backward(x, gain)
        x = x.contiguous(); residual = residual.contiguous(); gain = gain.contiguous()
        if residual.dtype != x.dtype: residual = residual.to(x.dtype)
        if gain.dtype != x.dtype: gain = gain.to(x.dtype)
        if _ncn_cuda and x.is_cuda:
            return _ncn_cuda.modulated_add_forward(x, residual, gain.view(-1))
        return x * gain + residual

    @staticmethod
    def backward(ctx, grad_output):
        x, gain = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if _ncn_cuda and grad_output.is_cuda:
            grads = _ncn_cuda.modulated_add_backward(grad_output, x, gain.view(-1))
            grad_x, grad_residual = grads[0], grads[1]
            tmp = grad_output * x
            grad_gain = tmp.sum(dim=-1, keepdim=True).view_as(gain)
            return grad_x, grad_residual, grad_gain
        return None, None, None

def rms_norm_cuda(x, weight, eps):
    return FusedRMSNorm.apply(x, weight, eps)

def ncn_actuator_cuda(x):
    return FusedNCNActuator.apply(x)

def kv_cache_update_cuda(cache, new_tok, positions):
    if _ncn_cuda and cache.is_cuda:
        _ncn_cuda.launch_kv_update(cache, new_tok, positions.int())

def fused_modulated_add(x, residual, gain):
    return FusedModulatedAdd.apply(x, residual, gain)

# Helper for Online Softmax (Use in generation loop or custom loss)
def online_softmax_update_cuda(running_max, running_sum, chunk_logits):
    if _ncn_cuda and chunk_logits.is_cuda:
        _ncn_cuda.launch_online_softmax_update(running_max, running_sum, chunk_logits)