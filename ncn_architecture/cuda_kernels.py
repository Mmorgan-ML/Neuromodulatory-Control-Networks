# ncn_architecture/cuda_kernels.py

import torch
from torch.utils.cpp_extension import load_inline
import os

# Define the C++ Source (Templated for FP16/FP32 support)
cpp_source = """
torch::Tensor modulated_add_forward(torch::Tensor x, torch::Tensor residual, torch::Tensor gain);
std::vector<torch::Tensor> modulated_add_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor gain);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// --- TEMPLATED FORWARD KERNEL ---
// Handles both float (FP32) and at::Half (FP16)
// Supports (Batch, Seq, Dim) input with (Batch, Seq, 1) gain
template <typename scalar_t>
__global__ void modulated_add_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ gain,
    scalar_t* __restrict__ output,
    int total_elements,
    int d_model // Size of the last dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // Map flat index to gain index.
        // x shape: (..., D). gain shape: (..., 1).
        // Since gain broadcasts over D, every D elements in x share the same gain value.
        int gain_idx = idx / d_model;
        
        // Math: out = x * gain + residual
        output[idx] = x[idx] * gain[gain_idx] + residual[idx];
    }
}

// --- TEMPLATED BACKWARD KERNEL ---
template <typename scalar_t>
__global__ void modulated_add_bwd_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ gain,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_residual,
    int total_elements,
    int d_model
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int gain_idx = idx / d_model;
        
        // dL/dx = dL/dout * gain
        grad_x[idx] = grad_output[idx] * gain[gain_idx];
        
        // dL/d_residual = dL/dout * 1
        grad_residual[idx] = grad_output[idx];
    }
}

// --- C++ WRAPPERS ---

torch::Tensor modulated_add_forward(torch::Tensor x, torch::Tensor residual, torch::Tensor gain) {
    auto output = torch::empty_like(x);
    
    int total_elements = x.numel();
    int d_model = x.size(-1); 
    
    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "modulated_add_forward", ([&] {
        modulated_add_fwd_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
            gain.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements,
            d_model
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
            grad_output.data_ptr<scalar_t>(),
            gain.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_residual.data_ptr<scalar_t>(),
            total_elements,
            d_model
        );
    }));
    
    return {grad_x, grad_residual};
}
"""

# Compile the module
print("Compiling NCN Custom Kernels (FP16 + Phasic Support)...")
try:
    _ncn_cuda = load_inline(
        name='ncn_fused_ops_v3', # Bump version for Phasic fix
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['modulated_add_forward', 'modulated_add_backward'],
        extra_cuda_cflags=['-allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'],
        with_cuda=True,
        verbose=False
    )
    print("NCN Custom Kernels Compiled Successfully.")
except Exception as e:
    print(f"KERNEL COMPILATION FAILED: {e}")
    _ncn_cuda = None

# --- Autograd Function ---
class FusedModulatedAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, gain):
        # Save for backward
        ctx.save_for_backward(x, gain)
        
        # Ensure contiguous memory
        x = x.contiguous()
        residual = residual.contiguous()
        gain = gain.contiguous() 
        
        # TYPE SAFETY: Ensure all tensors are the same type (FP16 or FP32)
        if residual.dtype != x.dtype:
            residual = residual.to(x.dtype)
        if gain.dtype != x.dtype:
            gain = gain.to(x.dtype)
        
        if _ncn_cuda is not None and x.is_cuda:
            # Flatten gain for C++? No, keep it as is, kernel handles pointers.
            # But we flatten inputs to vector view for easier handling if needed,
            # actually x.data_ptr() handles it. 
            # Just ensure gain is broadcast-compatible logic wise (B, S, 1).
            return _ncn_cuda.modulated_add_forward(x, residual, gain.view(-1))
        else:
            # Fallback
            return x * gain + residual

    @staticmethod
    def backward(ctx, grad_output):
        x, gain = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        if grad_output.dtype != x.dtype:
            grad_output = grad_output.to(x.dtype)
        
        grad_x = grad_residual = grad_gain = None
        
        if _ncn_cuda is not None and grad_output.is_cuda:
            # 1. Element-wise backward pass (Fused)
            grads = _ncn_cuda.modulated_add_backward(grad_output, x, gain.view(-1))
            grad_x, grad_residual = grads[0], grads[1]
            
            # 2. Reduction for gain
            # x shape: (Batch, Seq, Dim)
            # gain shape: (Batch, Seq, 1)
            # dL/dg = sum over Dim of (grad_output * x)
            # Result shape: (Batch, Seq, 1)
            
            # We compute in FP32 to prevent overflow during summation
            # (B, S, D) * (B, S, D) -> (B, S, D) -> sum(-1) -> (B, S) -> view(B, S, 1)
            
            # Optimized PyTorch Summation (faster than writing a custom reduction kernel)
            tmp = grad_output * x
            grad_gain = tmp.sum(dim=-1, keepdim=True).view_as(gain)
            
        else:
            # Fallback
            if ctx.needs_input_grad[0]:
                grad_x = grad_output * gain
            if ctx.needs_input_grad[1]:
                grad_residual = grad_output
            if ctx.needs_input_grad[2]:
                grad_gain = (grad_output * x).sum(dim=-1, keepdim=True)
                
        return grad_x, grad_residual, grad_gain

def fused_modulated_add(x, residual, gain):
    return FusedModulatedAdd.apply(x, residual, gain)