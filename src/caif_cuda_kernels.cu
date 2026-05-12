// Copyright 2026 Eric Malloy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//------------------------------------------------------------------------------
// CAIF - AI Framework
// Custom CUDA kernels for operations not supported by cuDNN
//------------------------------------------------------------------------------
// Disable GNU C++ extensions to avoid rsqrt conflict between CUDA and glibc
// This must be set BEFORE any includes
#undef _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

//------------------------------------------------------------------------------
// RoPE dimension pairing style — must match CAIF_RoPEStyle in caif_cuda_kernels.h
//------------------------------------------------------------------------------
enum CAIF_RoPEStyle
{
  CAIF_ROPE_INTERLEAVED=0,
  CAIF_ROPE_HALF_SPLIT=1
};

//------------------------------------------------------------------------------
// 128-bit vectorized access via int4 blob. int4 = 16 bytes = 4 fp32 / 8 fp16
// / 8 bf16. Templated kernels load as int4, reinterpret as T[lanes], op, store.
//------------------------------------------------------------------------------
template<typename T>
__host__ __device__ constexpr int caif_vec_lanes(){return 16/sizeof(T);}

//------------------------------------------------------------------------------
// caif_load_f / caif_store_f: dtype-agnostic fp32 load/store helpers.
// Used by kernels that compute in fp32 for numerical stability while
// reading/writing storage tensors of any dtype (float/__half/__nv_bfloat16).
//------------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ float caif_load_f(const T &v)
{
  return static_cast<float>(v);
}
template<>
__device__ __forceinline__ float caif_load_f<__half>(const __half &v)
{
  return __half2float(v);
}
template<>
__device__ __forceinline__ float caif_load_f<__nv_bfloat16>(const __nv_bfloat16 &v)
{
  return __bfloat162float(v);
}

template<typename T>
__device__ __forceinline__ T caif_store_f(float v)
{
  return static_cast<T>(v);
}
template<>
__device__ __forceinline__ __half caif_store_f<__half>(float v)
{
  return __float2half(v);
}
template<>
__device__ __forceinline__ __nv_bfloat16 caif_store_f<__nv_bfloat16>(float v)
{
  return __float2bfloat16(v);
}

//------------------------------------------------------------------------------
// caif_atomic_add: dispatch atomicAdd on storage T, with a CAS-loop fallback
// for bf16 below sm_90 (CUDA only added native bf16 atomicAdd on sm_90).
//------------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ void caif_atomic_add(T *addr,T val)
{
  atomicAdd(addr,val);
}

template<>
__device__ __forceinline__ void caif_atomic_add<__nv_bfloat16>(__nv_bfloat16 *addr,
                                                               __nv_bfloat16 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  atomicAdd(addr,val);
#else
  // Word-aligned CAS emulation: pack/unpack the bf16 in the high or low half
  // of the containing 32-bit word.
  uintptr_t addr_int=reinterpret_cast<uintptr_t>(addr);
  unsigned int *base=reinterpret_cast<unsigned int *>(addr_int & ~uintptr_t(3));
  const unsigned int shift=((addr_int & uintptr_t(3))!=0)?16u:0u;
  const float val_f=__bfloat162float(val);
  unsigned int old=*base;
  unsigned int assumed;
  do
  {
    assumed=old;
    const unsigned short cur_bits=static_cast<unsigned short>((assumed>>shift)&0xFFFFu);
    __nv_bfloat16 cur;
    *reinterpret_cast<unsigned short *>(&cur)=cur_bits;
    const __nv_bfloat16 sum=__float2bfloat16(__bfloat162float(cur)+val_f);
    const unsigned short sum_bits=*reinterpret_cast<const unsigned short *>(&sum);
    const unsigned int new_word=(assumed & ~(0xFFFFu<<shift))|
                                (static_cast<unsigned int>(sum_bits)<<shift);
    old=atomicCAS(base,assumed,new_word);
  }while(assumed!=old);
#endif
}

//------------------------------------------------------------------------------
// Vectorized Activation Kernels
// Every thread processes lanes=16/sizeof(T) elements (4 fp32 / 8 fp16 / 8 bf16)
// via a 128-bit int4 load. Tail elements handled by scalar fallback kernel.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// CUDA kernel constants — must match values in caif_constants.h
// (mirrored here because this .cu does not include caif_constants.h to keep
// the CUDA TU independent of host-only headers)
//------------------------------------------------------------------------------
constexpr int g_caif_cuda_block_size=256;
// Per-arch softmax block sizes — mirror of caif_constants.h. The row-reduce
// kernels no longer use inter-warp shared-memory tree reduction; intra-warp
// is __shfl_xor_sync only, cross-warp combine stages (num_warps) floats.
constexpr int g_caif_cuda_softmax_block_size_sm75=128;   // Turing
constexpr int g_caif_cuda_softmax_block_size_sm80=128;   // Ampere (A100)
constexpr int g_caif_cuda_softmax_block_size_sm86=128;   // Ampere (GA10x)
constexpr int g_caif_cuda_softmax_block_size_sm89=128;   // Ada Lovelace
constexpr int g_caif_cuda_softmax_block_size_sm90=128;   // Hopper
constexpr int g_caif_cuda_softmax_block_size_sm120=128;  // Blackwell
constexpr int g_caif_cuda_softmax_block_size_default=128;
constexpr int g_caif_cuda_warp_size=32;
constexpr unsigned g_caif_cuda_warp_full_mask=0xffffffffu;
constexpr int g_caif_cuda_default_shared_memory=49152;
constexpr int g_caif_cuda_max_threads_fallback=1024;

constexpr int SelectSoftmaxBlockSize(const int cc_major,
                                     const int cc_minor)
{
  if(cc_major==7&&cc_minor==5)
  {
    return g_caif_cuda_softmax_block_size_sm75;
  }
  if(cc_major==8&&cc_minor==0)
  {
    return g_caif_cuda_softmax_block_size_sm80;
  }
  if(cc_major==8&&cc_minor==6)
  {
    return g_caif_cuda_softmax_block_size_sm86;
  }
  if(cc_major==8&&cc_minor==9)
  {
    return g_caif_cuda_softmax_block_size_sm89;
  }
  if(cc_major==9)
  {
    return g_caif_cuda_softmax_block_size_sm90;
  }
  if(cc_major==12)
  {
    return g_caif_cuda_softmax_block_size_sm120;
  }
  return g_caif_cuda_softmax_block_size_default;
}

// Cached per-process softmax block size — queries compute capability once
// on first call, reuses thereafter. Single-device; multi-GPU would key by
// device id. Used by launch_attention_softmax{,_backward}.
inline int GetSoftmaxBlockSize()
{
  static const int block_size=[]()
  {
    int dev=0;
    cudaGetDevice(&dev);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props,dev);
    return SelectSoftmaxBlockSize(props.major,props.minor);
  }();
  return block_size;
}

//------------------------------------------------------------------------------
// Warp-level sum reduction using shuffle intrinsics.
// Avoids shared memory entirely. Returns the sum in lane 0.
// All 32 lanes in the warp must call this with their value.
//------------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val)
{
  for(int offset=16;offset>0;offset/=2)
  {
    val+=__shfl_down_sync(0xffffffff,val,offset);
  }
  return val;
}

//------------------------------------------------------------------------------
// Warp-level max reduction using shuffle intrinsics.
// Returns the max in lane 0.
//------------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val)
{
  for(int offset=16;offset>0;offset/=2)
  {
    val=fmaxf(val,__shfl_down_sync(0xffffffff,val,offset));
  }
  return val;
}

//------------------------------------------------------------------------------
// ReLU Forward: f(x) = max(0, x)
// 128-bit int4 load → unpack T[lanes] → op → pack → 128-bit store.
//------------------------------------------------------------------------------
template<typename T>
__global__ void relu_forward_kernel(const T *input,
                                    T *output,
                                    const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      lane[i]=lane[i]>T(0)?lane[i]:T(0);
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void relu_forward_tail_kernel(const T *input,
                                         T *output,
                                         const int offset,
                                         const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=input[idx]>T(0)?input[idx]:T(0);
  }
}

//------------------------------------------------------------------------------
// ReLU Backward: grad = upstream if input > 0, else 0
//------------------------------------------------------------------------------
template<typename T>
__global__ void relu_backward_kernel(const T *grad_output,
                                     const T *input,
                                     T *grad_input,
                                     const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 x=reinterpret_cast<const int4 *>(input)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *x_lane=reinterpret_cast<const T*>(&x);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=x_lane[i]>T(0)?g_lane[i]:T(0);
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void relu_backward_tail_kernel(const T *grad_output,
                                          const T *input,
                                          T *grad_input,
                                          const int offset,
                                          const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    grad_input[idx]=input[idx]>T(0)?grad_output[idx]:T(0);
  }
}

//------------------------------------------------------------------------------
// Sigmoid Forward: f(x) = 1 / (1 + exp(-x))
// Transcendentals compute in fp32 regardless of storage dtype.
//------------------------------------------------------------------------------
template<typename T>
__global__ void sigmoid_forward_kernel(const T *input,
                                       T *output,
                                       const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      lane[i]=T(1.0f/(1.0f+expf(-x)));
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void sigmoid_forward_tail_kernel(const T *input,
                                            T *output,
                                            const int offset,
                                            const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    output[idx]=T(1.0f/(1.0f+expf(-x)));
  }
}

//------------------------------------------------------------------------------
// Sigmoid Backward: grad = upstream * output * (1 - output)
//------------------------------------------------------------------------------
template<typename T>
__global__ void sigmoid_backward_kernel(const T *grad_output,
                                        const T *output,
                                        T *grad_input,
                                        const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 o=reinterpret_cast<const int4 *>(output)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *o_lane=reinterpret_cast<const T*>(&o);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float ov=float(o_lane[i]);
      r_lane[i]=T(gv*ov*(1.0f-ov));
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void sigmoid_backward_tail_kernel(const T *grad_output,
                                             const T *output,
                                             T *grad_input,
                                             const int offset,
                                             const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float ov=float(output[idx]);
    grad_input[idx]=T(gv*ov*(1.0f-ov));
  }
}

//------------------------------------------------------------------------------
// Tanh Forward: f(x) = tanh(x)
//------------------------------------------------------------------------------
template<typename T>
__global__ void tanh_forward_kernel(const T *input,
                                    T *output,
                                    const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      lane[i]=T(tanhf(float(lane[i])));
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void tanh_forward_tail_kernel(const T *input,
                                         T *output,
                                         const int offset,
                                         const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=T(tanhf(float(input[idx])));
  }
}

//------------------------------------------------------------------------------
// Tanh Backward: grad = upstream * (1 - output^2)
//------------------------------------------------------------------------------
template<typename T>
__global__ void tanh_backward_kernel(const T *grad_output,
                                     const T *output,
                                     T *grad_input,
                                     const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 o=reinterpret_cast<const int4 *>(output)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *o_lane=reinterpret_cast<const T*>(&o);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float ov=float(o_lane[i]);
      r_lane[i]=T(gv*(1.0f-ov*ov));
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void tanh_backward_tail_kernel(const T *grad_output,
                                          const T *output,
                                          T *grad_input,
                                          const int offset,
                                          const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float ov=float(output[idx]);
    grad_input[idx]=T(gv*(1.0f-ov*ov));
  }
}

//------------------------------------------------------------------------------
// GELU Forward: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//------------------------------------------------------------------------------
constexpr float g_caif_gelu_sqrt_2_over_pi=0.7978845608f;  // must match caif_constants.h
constexpr float g_caif_gelu_coeff=0.044715f;               // must match caif_constants.h

template<typename T>
__global__ void gelu_forward_kernel(const T *input,
                                    T *output,
                                    const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      const float inner=g_caif_gelu_sqrt_2_over_pi*(x+g_caif_gelu_coeff*x*x*x);
      lane[i]=T(0.5f*x*(1.0f+tanhf(inner)));
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void gelu_forward_tail_kernel(const T *input,
                                         T *output,
                                         const int offset,
                                         const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    const float inner=g_caif_gelu_sqrt_2_over_pi*(x+g_caif_gelu_coeff*x*x*x);
    output[idx]=T(0.5f*x*(1.0f+tanhf(inner)));
  }
}

//------------------------------------------------------------------------------
// GELU Backward
//------------------------------------------------------------------------------
template<typename T>
__global__ void gelu_backward_kernel(const T *grad_output,
                                     const T *input,
                                     T *grad_input,
                                     const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *v_lane=reinterpret_cast<const T*>(&v);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float x=float(v_lane[i]);
      const float inner=g_caif_gelu_sqrt_2_over_pi*(x+g_caif_gelu_coeff*x*x*x);
      const float th=tanhf(inner);
      const float di=g_caif_gelu_sqrt_2_over_pi*(1.0f+3.0f*g_caif_gelu_coeff*x*x);
      r_lane[i]=T(gv*(0.5f*(1.0f+th)+0.5f*x*(1.0f-th*th)*di));
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void gelu_backward_tail_kernel(const T *grad_output,
                                          const T *input,
                                          T *grad_input,
                                          const int offset,
                                          const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float x=float(input[idx]);
    const float inner=g_caif_gelu_sqrt_2_over_pi*(x+g_caif_gelu_coeff*x*x*x);
    const float th=tanhf(inner);
    const float di=g_caif_gelu_sqrt_2_over_pi*(1.0f+3.0f*g_caif_gelu_coeff*x*x);
    grad_input[idx]=T(gv*(0.5f*(1.0f+th)+0.5f*x*(1.0f-th*th)*di));
  }
}

//------------------------------------------------------------------------------
// Swish/SiLU Forward: f(x) = x * sigmoid(x)
//------------------------------------------------------------------------------
template<typename T>
__global__ void swish_forward_kernel(const T *input,
                                     T *output,
                                     const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      lane[i]=T(x/(1.0f+expf(-x)));
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void swish_forward_tail_kernel(const T *input,
                                          T *output,
                                          const int offset,
                                          const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    output[idx]=T(x/(1.0f+expf(-x)));
  }
}

//------------------------------------------------------------------------------
// Swish/SiLU Backward: d/dx(x*sig(x)) = sig(x) * (1 + x * (1 - sig(x)))
// Note: the *output* argument from the forward pass is unused here — the
// closed-form expression recomputes sig(x) directly, which is cheap.
//------------------------------------------------------------------------------
template<typename T>
__global__ void swish_backward_kernel(const T *grad_output,
                                      const T *input,
                                      T *grad_input,
                                      const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *v_lane=reinterpret_cast<const T*>(&v);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float x=float(v_lane[i]);
      const float s=1.0f/(1.0f+expf(-x));
      r_lane[i]=T(gv*s*(1.0f+x*(1.0f-s)));
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void swish_backward_tail_kernel(const T *grad_output,
                                           const T *input,
                                           T *grad_input,
                                           const int offset,
                                           const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float x=float(input[idx]);
    const float s=1.0f/(1.0f+expf(-x));
    grad_input[idx]=T(gv*s*(1.0f+x*(1.0f-s)));
  }
}

//------------------------------------------------------------------------------
// LeakyReLU Forward: f(x) = x if x > 0, else alpha * x
// alpha stays float — scalar kernel arg, cast to T per-lane at use.
//------------------------------------------------------------------------------
template<typename T>
__global__ void leaky_relu_forward_kernel(const T *input,
                                          T *output,
                                          const float alpha,
                                          const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      if(x>0.0f)
      {
        lane[i]=T(x);
      }
      else
      {
        lane[i]=T(alpha*x);
      }
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void leaky_relu_forward_tail_kernel(const T *input,
                                               T *output,
                                               const float alpha,
                                               const int offset,
                                               const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    if(x>0.0f)
    {
      output[idx]=T(x);
    }
    else
    {
      output[idx]=T(alpha*x);
    }
  }
}

//------------------------------------------------------------------------------
// LeakyReLU Backward
//------------------------------------------------------------------------------
template<typename T>
__global__ void leaky_relu_backward_kernel(const T *grad_output,
                                           const T *input,
                                           T *grad_input,
                                           const float alpha,
                                           const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 x=reinterpret_cast<const int4 *>(input)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *x_lane=reinterpret_cast<const T*>(&x);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float xv=float(x_lane[i]);
      if(xv>0.0f)
      {
        r_lane[i]=T(gv);
      }
      else
      {
        r_lane[i]=T(alpha*gv);
      }
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void leaky_relu_backward_tail_kernel(const T *grad_output,
                                                const T *input,
                                                T *grad_input,
                                                const float alpha,
                                                const int offset,
                                                const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float xv=float(input[idx]);
    if(xv>0.0f)
    {
      grad_input[idx]=T(gv);
    }
    else
    {
      grad_input[idx]=T(alpha*gv);
    }
  }
}

//------------------------------------------------------------------------------
// ELU Forward: f(x) = x if x > 0, else alpha * (exp(x) - 1)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elu_forward_kernel(const T *input,
                                   T *output,
                                   const float alpha,
                                   const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 v=reinterpret_cast<const int4 *>(input)[idx];
    T *lane=reinterpret_cast<T*>(&v);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float x=float(lane[i]);
      if(x>0.0f)
      {
        lane[i]=T(x);
      }
      else
      {
        lane[i]=T(alpha*(expf(x)-1.0f));
      }
    }
    reinterpret_cast<int4 *>(output)[idx]=v;
  }
}

template<typename T>
__global__ void elu_forward_tail_kernel(const T *input,
                                        T *output,
                                        const float alpha,
                                        const int offset,
                                        const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float x=float(input[idx]);
    if(x>0.0f)
    {
      output[idx]=T(x);
    }
    else
    {
      output[idx]=T(alpha*(expf(x)-1.0f));
    }
  }
}

//------------------------------------------------------------------------------
// ELU Backward: grad = upstream if input > 0, else upstream * (output + alpha)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elu_backward_kernel(const T *grad_output,
                                    const T *input,
                                    const T *output,
                                    T *grad_input,
                                    const float alpha,
                                    const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 g=reinterpret_cast<const int4 *>(grad_output)[idx];
    int4 x=reinterpret_cast<const int4 *>(input)[idx];
    int4 o=reinterpret_cast<const int4 *>(output)[idx];
    int4 r;
    const T *g_lane=reinterpret_cast<const T*>(&g);
    const T *x_lane=reinterpret_cast<const T*>(&x);
    const T *o_lane=reinterpret_cast<const T*>(&o);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      const float gv=float(g_lane[i]);
      const float xv=float(x_lane[i]);
      const float ov=float(o_lane[i]);
      if(xv>0.0f)
      {
        r_lane[i]=T(gv);
      }
      else
      {
        r_lane[i]=T(gv*(ov+alpha));
      }
    }
    reinterpret_cast<int4 *>(grad_input)[idx]=r;
  }
}

template<typename T>
__global__ void elu_backward_tail_kernel(const T *grad_output,
                                         const T *input,
                                         const T *output,
                                         T *grad_input,
                                         const float alpha,
                                         const int offset,
                                         const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float gv=float(grad_output[idx]);
    const float xv=float(input[idx]);
    if(xv>0.0f)
    {
      grad_input[idx]=T(gv);
    }
    else
    {
      const float ov=float(output[idx]);
      grad_input[idx]=T(gv*(ov+alpha));
    }
  }
}

//------------------------------------------------------------------------------
// Kernel launcher functions (callable from C++)
// All launchers use float4 vectorized path with scalar tail handling.
//------------------------------------------------------------------------------

template<typename T>
void launch_relu_forward(const T *input,
                         T *output,
                         const int n,
                         cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    relu_forward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      input,output,n_vec);
  }
  if(tail>0)
  {
    relu_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_relu_forward<float>(const float*, float*, int, cudaStream_t);
template void launch_relu_forward<__half>(const __half*, __half*, int, cudaStream_t);
template void launch_relu_forward<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, cudaStream_t);

template<typename T>
void launch_relu_backward(const T *grad_output,
                          const T *input,
                          T *grad_input,
                          const int n,
                          cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    relu_backward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      grad_output,input,grad_input,n_vec);
  }
  if(tail>0)
  {
    relu_backward_tail_kernel<T><<<1,tail,0,stream>>>(grad_output,input,grad_input,n_vec*lanes,n);
  }
}
template void launch_relu_backward<float>(const float *,
                                           const float *,
                                           float *,
                                           int,
                                           cudaStream_t);
template void launch_relu_backward<__half>(const __half *,
                                            const __half *,
                                            __half *,
                                            int,
                                            cudaStream_t);
template void launch_relu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                   const __nv_bfloat16 *,
                                                   __nv_bfloat16 *,
                                                   int,
                                                   cudaStream_t);

template<typename T>
void launch_sigmoid_forward(const T *input,
                            T *output,
                            const int n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    sigmoid_forward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      input,output,n_vec);
  }
  if(tail>0)
  {
    sigmoid_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_sigmoid_forward<float>(const float*, float*, int, cudaStream_t);
template void launch_sigmoid_forward<__half>(const __half*, __half*, int, cudaStream_t);
template void launch_sigmoid_forward<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, cudaStream_t);

template<typename T>
void launch_sigmoid_backward(const T *grad_output,
                             const T *output,
                             T *grad_input,
                             const int n,
                             cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    sigmoid_backward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      grad_output,output,grad_input,n_vec);
  }
  if(tail>0)
  {
    sigmoid_backward_tail_kernel<T><<<1,tail,0,stream>>>(
      grad_output,output,grad_input,n_vec*lanes,n);
  }
}
template void launch_sigmoid_backward<float>(const float *,
                                              const float *,
                                              float *,
                                              int,
                                              cudaStream_t);
template void launch_sigmoid_backward<__half>(const __half *,
                                               const __half *,
                                               __half *,
                                               int,
                                               cudaStream_t);
template void launch_sigmoid_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                      const __nv_bfloat16 *,
                                                      __nv_bfloat16 *,
                                                      int,
                                                      cudaStream_t);

template<typename T>
void launch_tanh_forward(const T *input,
                         T *output,
                         const int n,
                         cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    tanh_forward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      input,output,n_vec);
  }
  if(tail>0)
  {
    tanh_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_tanh_forward<float>(const float*, float*, int, cudaStream_t);
template void launch_tanh_forward<__half>(const __half*, __half*, int, cudaStream_t);
template void launch_tanh_forward<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, cudaStream_t);

template<typename T>
void launch_tanh_backward(const T *grad_output,
                          const T *output,
                          T *grad_input,
                          const int n,
                          cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    tanh_backward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      grad_output,output,grad_input,n_vec);
  }
  if(tail>0)
  {
    tanh_backward_tail_kernel<T><<<1,tail,0,stream>>>(
      grad_output,output,grad_input,n_vec*lanes,n);
  }
}
template void launch_tanh_backward<float>(const float *,
                                           const float *,
                                           float *,
                                           int,
                                           cudaStream_t);
template void launch_tanh_backward<__half>(const __half *,
                                            const __half *,
                                            __half *,
                                            int,
                                            cudaStream_t);
template void launch_tanh_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                   const __nv_bfloat16 *,
                                                   __nv_bfloat16 *,
                                                   int,
                                                   cudaStream_t);

template<typename T>
void launch_leaky_relu_forward(const T *input,
                               T *output,
                               const float alpha,
                               const int n,
                               cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    leaky_relu_forward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      input,output,alpha,n_vec);
  }
  if(tail>0)
  {
    leaky_relu_forward_tail_kernel<T><<<1,tail,0,stream>>>(
      input,output,alpha,n_vec*lanes,n);
  }
}
template void launch_leaky_relu_forward<float>(const float *,
                                                float *,
                                                float,
                                                int,
                                                cudaStream_t);
template void launch_leaky_relu_forward<__half>(const __half *,
                                                 __half *,
                                                 float,
                                                 int,
                                                 cudaStream_t);
template void launch_leaky_relu_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        float,
                                                        int,
                                                        cudaStream_t);

template<typename T>
void launch_leaky_relu_backward(const T *grad_output,
                                const T *input,
                                T *grad_input,
                                const float alpha,
                                const int n,
                                cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    leaky_relu_backward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      grad_output,input,grad_input,alpha,n_vec);
  }
  if(tail>0)
  {
    leaky_relu_backward_tail_kernel<T><<<1,tail,0,stream>>>(
      grad_output,input,grad_input,alpha,n_vec*lanes,n);
  }
}
template void launch_leaky_relu_backward<float>(const float*, const float*, float*, float, int, cudaStream_t);
template void launch_leaky_relu_backward<__half>(const __half*, const __half*, __half*, float, int, cudaStream_t);
template void launch_leaky_relu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        const __nv_bfloat16 *,
                                                        __nv_bfloat16 *,
                                                        float,
                                                        int,
                                                        cudaStream_t);

template<typename T>
void launch_gelu_forward(const T *input,
                         T *output,
                         const int n,
                         cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    gelu_forward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      input,output,n_vec);
  }
  if(tail>0)
  {
    gelu_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_gelu_forward<float>(const float*, float*, int, cudaStream_t);
template void launch_gelu_forward<__half>(const __half*, __half*, int, cudaStream_t);
template void launch_gelu_forward<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, cudaStream_t);

template<typename T>
void launch_gelu_backward(const T *grad_output,
                          const T *input,
                          T *grad_input,
                          const int n,
                          cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    gelu_backward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      grad_output,input,grad_input,n_vec);
  }
  if(tail>0)
  {
    gelu_backward_tail_kernel<T><<<1,tail,0,stream>>>(
      grad_output,input,grad_input,n_vec*lanes,n);
  }
}
template void launch_gelu_backward<float>(const float*, const float*, float*, int, cudaStream_t);
template void launch_gelu_backward<__half>(const __half*, const __half*, __half*, int, cudaStream_t);
template void launch_gelu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  int,
                                                  cudaStream_t);

template<typename T>
void launch_swish_forward(const T *input,
                          T *output,
                          const int n,
                          cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    swish_forward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      input,output,n_vec);
  }
  if(tail>0)
  {
    swish_forward_tail_kernel<T><<<1,tail,0,stream>>>(input,output,n_vec*lanes,n);
  }
}
template void launch_swish_forward<float>(const float*, float*, int, cudaStream_t);
template void launch_swish_forward<__half>(const __half*, __half*, int, cudaStream_t);
template void launch_swish_forward<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, cudaStream_t);

// Note: the `output` pointer is kept in the launcher signature for API
// symmetry with other backward launchers; the closed-form backward
// recomputes sigmoid(x) from input and the kernel does not read it.
template<typename T>
void launch_swish_backward(const T *grad_output,
                           const T *input,
                           const T *output,
                           T *grad_input,
                           const int n,
                           cudaStream_t stream)
{
  (void)output;
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    swish_backward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      grad_output,input,grad_input,n_vec);
  }
  if(tail>0)
  {
    swish_backward_tail_kernel<T><<<1,tail,0,stream>>>(
      grad_output,input,grad_input,n_vec*lanes,n);
  }
}
template void launch_swish_backward<float>(const float*, const float*, const float*, float*, int, cudaStream_t);
template void launch_swish_backward<__half>(const __half *,
                                            const __half *,
                                            const __half *,
                                            __half *,
                                            int,
                                            cudaStream_t);
template void launch_swish_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                   const __nv_bfloat16 *,
                                                   const __nv_bfloat16 *,
                                                   __nv_bfloat16 *,
                                                   int,
                                                   cudaStream_t);

template<typename T>
void launch_elu_forward(const T *input,
                        T *output,
                        const float alpha,
                        const int n,
                        cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elu_forward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      input,output,alpha,n_vec);
  }
  if(tail>0)
  {
    elu_forward_tail_kernel<T><<<1,tail,0,stream>>>(
      input,output,alpha,n_vec*lanes,n);
  }
}
template void launch_elu_forward<float>(const float*, float*, float, int, cudaStream_t);
template void launch_elu_forward<__half>(const __half*, __half*, float, int, cudaStream_t);
template void launch_elu_forward<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, float, int, cudaStream_t);

template<typename T>
void launch_elu_backward(const T *grad_output,
                         const T *input,
                         const T *output,
                         T *grad_input,
                         const float alpha,
                         const int n,
                         cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elu_backward_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(
      grad_output,input,output,grad_input,alpha,n_vec);
  }
  if(tail>0)
  {
    elu_backward_tail_kernel<T><<<1,tail,0,stream>>>(
      grad_output,input,output,grad_input,alpha,n_vec*lanes,n);
  }
}
template void launch_elu_backward<float>(const float *,
                                         const float *,
                                         const float *,
                                         float *,
                                         float,
                                         int,
                                         cudaStream_t);
template void launch_elu_backward<__half>(const __half *,
                                          const __half *,
                                          const __half *,
                                          __half *,
                                          float,
                                          int,
                                          cudaStream_t);
template void launch_elu_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                 const __nv_bfloat16 *,
                                                 const __nv_bfloat16 *,
                                                 __nv_bfloat16 *,
                                                 float,
                                                 int,
                                                 cudaStream_t);

//------------------------------------------------------------------------------
// Element-wise Add Kernel (tensor + tensor)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_add_kernel(const T *a,
                                       const T *b,
                                       T *result,
                                       const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 bv=reinterpret_cast<const int4 *>(b)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    const T *b_lane=reinterpret_cast<const T*>(&bv);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])+float(b_lane[i]));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_add_tail_kernel(const T *a,
                                            const T *b,
                                            T *result,
                                            const int offset,
                                            const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])+float(b[idx]));
  }
}

//------------------------------------------------------------------------------
// Element-wise Add Scalar Kernel (tensor + scalar)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_add_scalar_kernel(const T *a,
                                              const float scalar,
                                              T *result,
                                              const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])+scalar);
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_add_scalar_tail_kernel(const T *a,
                                                   const float scalar,
                                                   T *result,
                                                   const int offset,
                                                   const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])+scalar);
  }
}

//------------------------------------------------------------------------------
// Bias Add (2D: broadcast bias over batch rows)
// Scalar-indexed template (col=idx%units), no vectorization.
//------------------------------------------------------------------------------
template<typename T>
__global__ void bias_add_2d_kernel(const T *input,
                                   const T *bias,
                                   T *output,
                                   const int units,
                                   const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int col=idx%units;
    output[idx]=T(float(input[idx])+float(bias[col]));
  }
}

//------------------------------------------------------------------------------
// Bias Gradient (sum over batch rows)
// fp32 accumulator, final cast to T on store.
//------------------------------------------------------------------------------
template<typename T>
__global__ void bias_grad_2d_kernel(const T *grad,
                                    T *bias_grad,
                                    const int batch,
                                    const int units)
{
  const int u=blockIdx.x*blockDim.x+threadIdx.x;
  if(u<units)
  {
    float sum=0.0f;
    for(int b=0;b<batch;++b)
    {
      sum+=float(grad[b*units+u]);
    }
    bias_grad[u]=T(sum);
  }
}

//------------------------------------------------------------------------------
// Element-wise Subtract Kernel (tensor - tensor)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_sub_kernel(const T *a,
                                       const T *b,
                                       T *result,
                                       const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 bv=reinterpret_cast<const int4 *>(b)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    const T *b_lane=reinterpret_cast<const T*>(&bv);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])-float(b_lane[i]));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_sub_tail_kernel(const T *a,
                                            const T *b,
                                            T *result,
                                            const int offset,
                                            const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])-float(b[idx]));
  }
}

//------------------------------------------------------------------------------
// Element-wise Subtract Scalar Kernel (tensor - scalar)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_sub_scalar_kernel(const T *a,
                                              const float scalar,
                                              T *result,
                                              const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])-scalar);
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_sub_scalar_tail_kernel(const T *a,
                                                   const float scalar,
                                                   T *result,
                                                   const int offset,
                                                   const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])-scalar);
  }
}

//------------------------------------------------------------------------------
// Element-wise Multiply Kernel (tensor * tensor)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_mul_kernel(const T *a,
                                       const T *b,
                                       T *result,
                                       const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 bv=reinterpret_cast<const int4 *>(b)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    const T *b_lane=reinterpret_cast<const T*>(&bv);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])*float(b_lane[i]));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_mul_tail_kernel(const T *a,
                                            const T *b,
                                            T *result,
                                            const int offset,
                                            const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])*float(b[idx]));
  }
}

//------------------------------------------------------------------------------
// Element-wise Multiply Scalar Kernel (tensor * scalar)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_mul_scalar_kernel(const T *a,
                                              const float scalar,
                                              T *result,
                                              const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])*scalar);
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_mul_scalar_tail_kernel(const T *a,
                                                   const float scalar,
                                                   T *result,
                                                   const int offset,
                                                   const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])*scalar);
  }
}

//------------------------------------------------------------------------------
// Element-wise Divide Kernel (tensor / tensor)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_div_kernel(const T *a,
                                       const T *b,
                                       T *result,
                                       const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 bv=reinterpret_cast<const int4 *>(b)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    const T *b_lane=reinterpret_cast<const T*>(&bv);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])/float(b_lane[i]));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_div_tail_kernel(const T *a,
                                            const T *b,
                                            T *result,
                                            const int offset,
                                            const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])/float(b[idx]));
  }
}

//------------------------------------------------------------------------------
// Element-wise Divide Scalar Kernel (tensor / scalar)
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_div_scalar_kernel(const T *a,
                                              const float scalar,
                                              T *result,
                                              const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(float(a_lane[i])/scalar);
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_div_scalar_tail_kernel(const T *a,
                                                   const float scalar,
                                                   T *result,
                                                   const int offset,
                                                   const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(float(a[idx])/scalar);
  }
}

//------------------------------------------------------------------------------
// Element-wise Sqrt Kernel
//------------------------------------------------------------------------------
template<typename T>
__global__ void elementwise_sqrt_kernel(const T *a,
                                        T *result,
                                        const int n_vec)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n_vec)
  {
    int4 av=reinterpret_cast<const int4 *>(a)[idx];
    int4 r;
    const T *a_lane=reinterpret_cast<const T*>(&av);
    T *r_lane=reinterpret_cast<T*>(&r);
    #pragma unroll
    for(int i=0;i<lanes;++i)
    {
      r_lane[i]=T(sqrtf(float(a_lane[i])));
    }
    reinterpret_cast<int4 *>(result)[idx]=r;
  }
}

template<typename T>
__global__ void elementwise_sqrt_tail_kernel(const T *a,
                                             T *result,
                                             const int offset,
                                             const int n)
{
  const int idx=offset+blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    result[idx]=T(sqrtf(float(a[idx])));
  }
}

//------------------------------------------------------------------------------
// Reduction Sum Kernel (parallel reduction)
// Uses shared memory for efficient intra-block reduction
//------------------------------------------------------------------------------
template<typename T>
__global__ void reduction_sum_kernel(const T *input,
                                     float *output,
                                     const int n)
{
  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;

  float val=0.0f;
  if(idx<n)
  {
    val=float(input[idx]);
  }

  // Warp shuffle reduction (no shared memory for intra-warp)
  val=warp_reduce_sum(val);

  // Per-warp partial sums via shared memory
  __shared__ float warp_sums[g_caif_cuda_warp_size];
  const int lane=tid&(g_caif_cuda_warp_size-1);
  const int warp_id=tid/g_caif_cuda_warp_size;
  const int num_warps=blockDim.x/g_caif_cuda_warp_size;

  if(lane==0)
  {
    warp_sums[warp_id]=val;
  }
  __syncthreads();

  // First warp reduces the per-warp sums
  if(warp_id==0)
  {
    val=0.0f;
    if(lane<num_warps)
    {
      val=warp_sums[lane];
    }
    val=warp_reduce_sum(val);
  }

  if(tid==0)
  {
    output[blockIdx.x]=val;
  }
}

//------------------------------------------------------------------------------
// Element-wise launcher functions
//------------------------------------------------------------------------------
template<typename T>
void launch_elementwise_add(const T *a,
                            const T *b,
                            T *result,
                            const int n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elementwise_add_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(a,b,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_add_tail_kernel<T><<<1,tail,0,stream>>>(a,b,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_add<float>(const float *,
                                            const float *,
                                            float *,
                                            int,
                                            cudaStream_t);
template void launch_elementwise_add<__half>(const __half *,
                                             const __half *,
                                             __half *,
                                             int,
                                             cudaStream_t);
template void launch_elementwise_add<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int,
                                                    cudaStream_t);

template<typename T>
void launch_elementwise_add_scalar(const T *a,
                                   const float scalar,
                                   T *result,
                                   const int n,
                                   cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elementwise_add_scalar_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(a,scalar,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_add_scalar_tail_kernel<T><<<1,tail,0,stream>>>(a,scalar,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_add_scalar<float>(const float *,
                                                   float,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
template void launch_elementwise_add_scalar<__half>(const __half *,
                                                    float,
                                                    __half *,
                                                    int,
                                                    cudaStream_t);
template void launch_elementwise_add_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           float,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           cudaStream_t);

template<typename T>
void launch_bias_add_2d(const T *input,
                        const T *bias,
                        T *output,
                        const int batch,
                        const int units,
                        cudaStream_t stream)
{
  const int total=batch*units;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  bias_add_2d_kernel<T><<<num_blocks,block_size,0,stream>>>(input,bias,output,units,total);
}
template void launch_bias_add_2d<float>(const float *,
                                        const float *,
                                        float *,
                                        int,
                                        int,
                                        cudaStream_t);
template void launch_bias_add_2d<__half>(const __half *,
                                         const __half *,
                                         __half *,
                                         int,
                                         int,
                                         cudaStream_t);
template void launch_bias_add_2d<__nv_bfloat16>(const __nv_bfloat16 *,
                                                const __nv_bfloat16 *,
                                                __nv_bfloat16 *,
                                                int,
                                                int,
                                                cudaStream_t);

template<typename T>
void launch_bias_grad_2d(const T *grad_output,
                         T *bias_grad,
                         const int batch,
                         const int units,
                         cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(units+block_size-1)/block_size;
  bias_grad_2d_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_output,bias_grad,batch,units);
}
template void launch_bias_grad_2d<float>(const float *,
                                         float *,
                                         int,
                                         int,
                                         cudaStream_t);
template void launch_bias_grad_2d<__half>(const __half *,
                                          __half *,
                                          int,
                                          int,
                                          cudaStream_t);
template void launch_bias_grad_2d<__nv_bfloat16>(const __nv_bfloat16 *,
                                                 __nv_bfloat16 *,
                                                 int,
                                                 int,
                                                 cudaStream_t);

template<typename T>
void launch_elementwise_sub(const T *a,
                            const T *b,
                            T *result,
                            const int n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elementwise_sub_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(a,b,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_sub_tail_kernel<T><<<1,tail,0,stream>>>(a,b,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_sub<float>(const float *,
                                            const float *,
                                            float *,
                                            int,
                                            cudaStream_t);
template void launch_elementwise_sub<__half>(const __half *,
                                             const __half *,
                                             __half *,
                                             int,
                                             cudaStream_t);
template void launch_elementwise_sub<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int,
                                                    cudaStream_t);

template<typename T>
void launch_elementwise_sub_scalar(const T *a,
                                   const float scalar,
                                   T *result,
                                   const int n,
                                   cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elementwise_sub_scalar_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(a,scalar,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_sub_scalar_tail_kernel<T><<<1,tail,0,stream>>>(a,scalar,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_sub_scalar<float>(const float *,
                                                   float,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
template void launch_elementwise_sub_scalar<__half>(const __half *,
                                                    float,
                                                    __half *,
                                                    int,
                                                    cudaStream_t);
template void launch_elementwise_sub_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           float,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           cudaStream_t);

template<typename T>
void launch_elementwise_mul(const T *a,
                            const T *b,
                            T *result,
                            const int n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elementwise_mul_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(a,b,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_mul_tail_kernel<T><<<1,tail,0,stream>>>(a,b,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_mul<float>(const float *,
                                            const float *,
                                            float *,
                                            int,
                                            cudaStream_t);
template void launch_elementwise_mul<__half>(const __half *,
                                             const __half *,
                                             __half *,
                                             int,
                                             cudaStream_t);
template void launch_elementwise_mul<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int,
                                                    cudaStream_t);

template<typename T>
void launch_elementwise_mul_scalar(const T *a,
                                   const float scalar,
                                   T *result,
                                   const int n,
                                   cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elementwise_mul_scalar_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(a,scalar,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_mul_scalar_tail_kernel<T><<<1,tail,0,stream>>>(a,scalar,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_mul_scalar<float>(const float *,
                                                   float,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
template void launch_elementwise_mul_scalar<__half>(const __half *,
                                                    float,
                                                    __half *,
                                                    int,
                                                    cudaStream_t);
template void launch_elementwise_mul_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           float,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           cudaStream_t);

template<typename T>
void launch_elementwise_div(const T *a,
                            const T *b,
                            T *result,
                            const int n,
                            cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elementwise_div_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(a,b,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_div_tail_kernel<T><<<1,tail,0,stream>>>(a,b,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_div<float>(const float *,
                                            const float *,
                                            float *,
                                            int,
                                            cudaStream_t);
template void launch_elementwise_div<__half>(const __half *,
                                             const __half *,
                                             __half *,
                                             int,
                                             cudaStream_t);
template void launch_elementwise_div<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int,
                                                    cudaStream_t);

template<typename T>
void launch_elementwise_div_scalar(const T *a,
                                   const float scalar,
                                   T *result,
                                   const int n,
                                   cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elementwise_div_scalar_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(a,scalar,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_div_scalar_tail_kernel<T><<<1,tail,0,stream>>>(a,scalar,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_div_scalar<float>(const float *,
                                                   float,
                                                   float *,
                                                   int,
                                                   cudaStream_t);
template void launch_elementwise_div_scalar<__half>(const __half *,
                                                    float,
                                                    __half *,
                                                    int,
                                                    cudaStream_t);
template void launch_elementwise_div_scalar<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           float,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           cudaStream_t);

template<typename T>
void launch_elementwise_sqrt(const T *a,
                             T *result,
                             const int n,
                             cudaStream_t stream)
{
  constexpr int lanes=caif_vec_lanes<T>();
  const int n_vec=n/lanes;
  const int tail=n-n_vec*lanes;
  if(n_vec>0)
  {
    const int num_blocks=(n_vec+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
    elementwise_sqrt_kernel<T><<<num_blocks,g_caif_cuda_block_size,0,stream>>>(a,result,n_vec);
  }
  if(tail>0)
  {
    elementwise_sqrt_tail_kernel<T><<<1,tail,0,stream>>>(a,result,n_vec*lanes,n);
  }
}
template void launch_elementwise_sqrt<float>(const float *,
                                             float *,
                                             int,
                                             cudaStream_t);
template void launch_elementwise_sqrt<__half>(const __half *,
                                              __half *,
                                              int,
                                              cudaStream_t);
template void launch_elementwise_sqrt<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     __nv_bfloat16 *,
                                                     int,
                                                     cudaStream_t);

template<typename T>
void launch_reduction_sum(const T *input,
                          float *output,
                          const int n,
                          cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  reduction_sum_kernel<T><<<num_blocks,block_size,0,stream>>>(input,output,n);
}
template void launch_reduction_sum<float>(const float *,
                                          float *,
                                          int,
                                          cudaStream_t);
template void launch_reduction_sum<__half>(const __half *,
                                           float *,
                                           int,
                                           cudaStream_t);
template void launch_reduction_sum<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  float *,
                                                  int,
                                                  cudaStream_t);

//------------------------------------------------------------------------------
// Loss Function Kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Cross Entropy Loss Kernel
// loss_i = -target_i * log(max(epsilon, min(1-epsilon, pred_i)))
//------------------------------------------------------------------------------
template<typename T>
__global__ void cross_entropy_loss_kernel(const T *predictions,
                                          const T *targets,
                                          float *loss,
                                          const float epsilon,
                                          const int batch_size,
                                          const int num_classes)
{
  const int batch_idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(batch_idx<batch_size)
  {
    float sample_loss=0.0f;
    for(int c=0;c<num_classes;++c)
    {
      const int idx=batch_idx*num_classes+c;
      float pred=float(predictions[idx]);
      // Clamp to avoid log(0)
      if(pred<epsilon)
      {
        pred=epsilon;
      }
      else if(pred>1.0f-epsilon)
      {
        pred=1.0f-epsilon;
      }
      const float target=float(targets[idx]);
      if(target>0.0f)
      {
        sample_loss-=target*logf(pred);
      }
    }
    loss[batch_idx]=sample_loss;
  }
}

//------------------------------------------------------------------------------
// Cross Entropy Gradient Kernel
// grad_i = -target_i / max(epsilon, min(1-epsilon, pred_i)) / batch_size
//------------------------------------------------------------------------------
template<typename T>
__global__ void cross_entropy_gradient_kernel(const T *predictions,
                                              const T *targets,
                                              T *gradient,
                                              const float epsilon,
                                              const float batch_size_inv,
                                              const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float pred=float(predictions[idx]);
    // Clamp to avoid division by zero
    if(pred<epsilon)
    {
      pred=epsilon;
    }
    else if(pred>1.0f-epsilon)
    {
      pred=1.0f-epsilon;
    }
    const float target=float(targets[idx]);
    gradient[idx]=T(-target/(pred*batch_size_inv));
  }
}

//------------------------------------------------------------------------------
// Cross Entropy with index targets
//------------------------------------------------------------------------------
template<typename T>
__global__ void cross_entropy_loss_index_kernel(const T *predictions,
                                                const int *target_idx,
                                                float *loss,
                                                const float epsilon,
                                                const int num_classes)
{
  const int b=blockIdx.x*blockDim.x+threadIdx.x;
  if(b>=gridDim.x*blockDim.x)
  {
    return;
  }
  const int cls=target_idx[b];
  if(cls<0 || cls>=num_classes)
  {
    loss[b]=0.0f;
    return;
  }
  const int idx=b*num_classes+cls;
  const float pred=float(predictions[idx]);
  const float clipped=fminf(1.0f-epsilon,fmaxf(epsilon,pred));
  loss[b]=-logf(clipped);
}

template<typename T>
__global__ void cross_entropy_gradient_index_kernel(const T *predictions,
                                                    const int *target_idx,
                                                    T *gradient,
                                                    const float epsilon,
                                                    const float batch_size_inv,
                                                    const int num_classes,
                                                    const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total)
  {
    return;
  }
  const int b=idx/num_classes;
  const int c=idx%num_classes;
  const int target=target_idx[b];
  if(target==c)
  {
    float pred=float(predictions[idx]);
    const float clipped=fminf(1.0f-epsilon,fmaxf(epsilon,pred));
    gradient[idx]=T(-batch_size_inv/clipped);
  }
  else
  {
    gradient[idx]=T(0.0f);
  }
}

//------------------------------------------------------------------------------
// MSE Loss Reduction Kernel
// Computes sum of (pred - target)^2 via parallel reduction to a scalar.
// Output must be pre-zeroed; each block atomically adds its partial sum.
//------------------------------------------------------------------------------
template<typename T>
__global__ void mse_loss_reduce_kernel(const T *predictions,
                                       const T *targets,
                                       float *loss,
                                       const int n)
{
  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;

  float val=0.0f;
  if(idx<n)
  {
    const float diff=float(predictions[idx])-float(targets[idx]);
    val=diff*diff;
  }

  // Warp shuffle reduction
  val=warp_reduce_sum(val);

  // Per-warp partial sums via shared memory
  __shared__ float warp_sums[g_caif_cuda_warp_size];
  const int lane=tid&(g_caif_cuda_warp_size-1);
  const int warp_id=tid/g_caif_cuda_warp_size;
  const int num_warps=blockDim.x/g_caif_cuda_warp_size;

  if(lane==0)
  {
    warp_sums[warp_id]=val;
  }
  __syncthreads();

  // First warp reduces the per-warp sums
  if(warp_id==0)
  {
    val=0.0f;
    if(lane<num_warps)
    {
      val=warp_sums[lane];
    }
    val=warp_reduce_sum(val);
  }

  if(tid==0)
  {
    atomicAdd(loss,val);
  }
}

//------------------------------------------------------------------------------
// MSE Gradient Kernel
// grad = 2 * (pred - target) / n
//------------------------------------------------------------------------------
template<typename T>
__global__ void mse_gradient_kernel(const T *predictions,
                                    const T *targets,
                                    T *gradient,
                                    const float scale,
                                    const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    gradient[idx]=T(scale*(float(predictions[idx])-float(targets[idx])));
  }
}

//------------------------------------------------------------------------------
// Loss Function Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_cross_entropy_loss(const T *predictions,
                               const T *targets,
                               float *loss,
                               const float epsilon,
                               const int batch_size,
                               const int num_classes,
                               cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(batch_size+block_size-1)/block_size;
  cross_entropy_loss_kernel<T><<<num_blocks,block_size,0,stream>>>(predictions,
                                                                   targets,
                                                                   loss,
                                                                   epsilon,
                                                                   batch_size,
                                                                   num_classes);
}

template void launch_cross_entropy_loss<float>(const float *,
                                               const float *,
                                               float *,
                                               float,
                                               int,
                                               int,
                                               cudaStream_t);
template void launch_cross_entropy_loss<__half>(const __half *,
                                                const __half *,
                                                float *,
                                                float,
                                                int,
                                                int,
                                                cudaStream_t);
template void launch_cross_entropy_loss<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       const __nv_bfloat16 *,
                                                       float *,
                                                       float,
                                                       int,
                                                       int,
                                                       cudaStream_t);

template<typename T>
void launch_cross_entropy_gradient(const T *predictions,
                                   const T *targets,
                                   T *gradient,
                                   const float epsilon,
                                   const int batch_size,
                                   const int n,
                                   cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  const float batch_size_inv=1.0f/static_cast<float>(batch_size);
  cross_entropy_gradient_kernel<T><<<num_blocks,block_size,0,stream>>>(predictions,
                                                                       targets,
                                                                       gradient,
                                                                       epsilon,
                                                                       batch_size_inv,
                                                                       n);
}

template void launch_cross_entropy_gradient<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   float,
                                                   int,
                                                   int,
                                                   cudaStream_t);
template void launch_cross_entropy_gradient<__half>(const __half *,
                                                    const __half *,
                                                    __half *,
                                                    float,
                                                    int,
                                                    int,
                                                    cudaStream_t);
template void launch_cross_entropy_gradient<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           float,
                                                           int,
                                                           int,
                                                           cudaStream_t);

template<typename T>
void launch_cross_entropy_loss_index(const T *predictions,
                                     const int *target_indices,
                                     float *loss,
                                     const float epsilon,
                                     const int batch_size,
                                     const int num_classes,
                                     cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(batch_size+block_size-1)/block_size;
  cross_entropy_loss_index_kernel<T><<<num_blocks,block_size,0,stream>>>(predictions,
                                                                         target_indices,
                                                                         loss,
                                                                         epsilon,
                                                                         num_classes);
}

template void launch_cross_entropy_loss_index<float>(const float *,
                                                     const int *,
                                                     float *,
                                                     float,
                                                     int,
                                                     int,
                                                     cudaStream_t);
template void launch_cross_entropy_loss_index<__half>(const __half *,
                                                      const int *,
                                                      float *,
                                                      float,
                                                      int,
                                                      int,
                                                      cudaStream_t);
template void launch_cross_entropy_loss_index<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             const int *,
                                                             float *,
                                                             float,
                                                             int,
                                                             int,
                                                             cudaStream_t);

template<typename T>
void launch_cross_entropy_gradient_index(const T *predictions,
                                         const int *target_indices,
                                         T *gradient,
                                         const float epsilon,
                                         const int batch_size,
                                         const int num_classes,
                                         cudaStream_t stream)
{
  const int total=batch_size*num_classes;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  const float batch_size_inv=1.0f/static_cast<float>(batch_size);
  cross_entropy_gradient_index_kernel<T><<<num_blocks,block_size,0,stream>>>(predictions,
                                                                             target_indices,
                                                                             gradient,
                                                                             epsilon,
                                                                             batch_size_inv,
                                                                             num_classes,
                                                                             total);
}

template void launch_cross_entropy_gradient_index<float>(const float *,
                                                         const int *,
                                                         float *,
                                                         float,
                                                         int,
                                                         int,
                                                         cudaStream_t);
template void launch_cross_entropy_gradient_index<__half>(const __half *,
                                                          const int *,
                                                          __half *,
                                                          float,
                                                          int,
                                                          int,
                                                          cudaStream_t);
template void launch_cross_entropy_gradient_index<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                 const int *,
                                                                 __nv_bfloat16 *,
                                                                 float,
                                                                 int,
                                                                 int,
                                                                 cudaStream_t);

template<typename T>
void launch_mse_loss(const T *predictions,
                     const T *targets,
                     float *loss,
                     const int n,
                     cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  mse_loss_reduce_kernel<T><<<num_blocks,block_size,0,stream>>>(predictions,
                                                                targets,
                                                                loss,
                                                                n);
}

template void launch_mse_loss<float>(const float *,
                                     const float *,
                                     float *,
                                     int,
                                     cudaStream_t);
template void launch_mse_loss<__half>(const __half *,
                                      const __half *,
                                      float *,
                                      int,
                                      cudaStream_t);
template void launch_mse_loss<__nv_bfloat16>(const __nv_bfloat16 *,
                                             const __nv_bfloat16 *,
                                             float *,
                                             int,
                                             cudaStream_t);

template<typename T>
void launch_mse_gradient(const T *predictions,
                         const T *targets,
                         T *gradient,
                         const int n,
                         cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  const float scale=2.0f/static_cast<float>(n);
  mse_gradient_kernel<T><<<num_blocks,block_size,0,stream>>>(predictions,
                                                             targets,
                                                             gradient,
                                                             scale,
                                                             n);
}

template void launch_mse_gradient<float>(const float *,
                                         const float *,
                                         float *,
                                         int,
                                         cudaStream_t);
template void launch_mse_gradient<__half>(const __half *,
                                          const __half *,
                                          __half *,
                                          int,
                                          cudaStream_t);
template void launch_mse_gradient<__nv_bfloat16>(const __nv_bfloat16 *,
                                                 const __nv_bfloat16 *,
                                                 __nv_bfloat16 *,
                                                 int,
                                                 cudaStream_t);

//------------------------------------------------------------------------------
// Fused Adam Optimizer Kernel
// Combines all Adam operations into a single kernel for efficiency:
// AdamW (decoupled weight decay, the Loshchilov & Hutter formulation):
// 1. Update first moment: m = beta1 * m + (1 - beta1) * g
// 2. Update second moment: v = beta2 * v + (1 - beta2) * g^2
// 3. Compute bias-corrected moments
// 4. Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
// 5. Apply decoupled weight decay: param = param - lr * weight_decay * param
//------------------------------------------------------------------------------
// param/grad in storage T (float/__half/__nv_bfloat16); m/v always fp32 master.
// All math is fp32 — only the load/store of param and grad cross precisions.
template<typename T>
__global__ void fused_adam_kernel(T *param,
                                  const T *grad,
                                  float *m,
                                  float *v,
                                  const float lr,
                                  const float beta1,
                                  const float beta2,
                                  const float epsilon,
                                  const float weight_decay,
                                  const float bias_correction1,
                                  const float bias_correction2,
                                  const float grad_clip_val,
                                  const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx]);
    float m_val=m[idx];
    float v_val=v[idx];

    if(isnan(g) || isinf(g))
    {
      g=0.0f;
    }
    else if(g>grad_clip_val)
    {
      g=grad_clip_val;
    }
    else if(g<-grad_clip_val)
    {
      g=-grad_clip_val;
    }

    m_val=beta1*m_val+(1.0f-beta1)*g;
    v_val=beta2*v_val+(1.0f-beta2)*g*g;

    const float m_hat=m_val/bias_correction1;
    const float v_hat=v_val/bias_correction2;

    p=p-lr*m_hat/(sqrtf(v_hat)+epsilon);

    if(weight_decay>0.0f)
    {
      p=p-lr*weight_decay*p;
    }

    param[idx]=caif_store_f<T>(p);
    m[idx]=m_val;
    v[idx]=v_val;
  }
}

//------------------------------------------------------------------------------
// Fused Adam with Gradient Clipping Kernel
// Same as fused_adam but includes gradient clipping
//------------------------------------------------------------------------------
template<typename T>
__global__ void fused_adam_clipped_kernel(T *param,
                                          const T *grad,
                                          float *m,
                                          float *v,
                                          const float lr,
                                          const float beta1,
                                          const float beta2,
                                          const float epsilon,
                                          const float weight_decay,
                                          const float bias_correction1,
                                          const float bias_correction2,
                                          const float grad_scale,
                                          const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx])*grad_scale;
    float m_val=m[idx];
    float v_val=v[idx];

    m_val=beta1*m_val+(1.0f-beta1)*g;
    v_val=beta2*v_val+(1.0f-beta2)*g*g;

    const float m_hat=m_val/bias_correction1;
    const float v_hat=v_val/bias_correction2;

    p=p-lr*m_hat/(sqrtf(v_hat)+epsilon);

    if(weight_decay>0.0f)
    {
      p=p-lr*weight_decay*p;
    }

    param[idx]=caif_store_f<T>(p);
    m[idx]=m_val;
    v[idx]=v_val;
  }
}

//------------------------------------------------------------------------------
// Fused SGD with Momentum Kernel
// Combines momentum update and parameter update
//------------------------------------------------------------------------------
template<typename T>
__global__ void fused_sgd_momentum_kernel(T *param,
                                          const T *grad,
                                          T *velocity,
                                          const float lr,
                                          const float momentum,
                                          const float weight_decay,
                                          const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx]);
    float v=caif_load_f<T>(velocity[idx]);

    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }

    v=momentum*v+g;
    p=p-lr*v;

    param[idx]=caif_store_f<T>(p);
    velocity[idx]=caif_store_f<T>(v);
  }
}

//------------------------------------------------------------------------------
// Fused plain-SGD kernel (no momentum, no velocity buffer)
// param = param - lr * (grad + weight_decay * param)
//------------------------------------------------------------------------------
template<typename T>
__global__ void fused_sgd_kernel(T *param,
                                 const T *grad,
                                 const float lr,
                                 const float weight_decay,
                                 const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx]);
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    param[idx]=caif_store_f<T>(p-lr*g);
  }
}

//------------------------------------------------------------------------------
// Fused RMSprop kernel
// avg_sq = alpha * avg_sq + (1 - alpha) * grad^2
// param  = param - lr * (grad + wd * param) / (sqrt(avg_sq) + epsilon)
//------------------------------------------------------------------------------
template<typename T>
__global__ void fused_rmsprop_kernel(T *param,
                                     const T *grad,
                                     T *avg_sq,
                                     const float lr,
                                     const float alpha,
                                     const float epsilon,
                                     const float weight_decay,
                                     const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx]);
    float a=caif_load_f<T>(avg_sq[idx]);
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    a=alpha*a+(1.0f-alpha)*g*g;
    param[idx]=caif_store_f<T>(p-lr*g/(sqrtf(a)+epsilon));
    avg_sq[idx]=caif_store_f<T>(a);
  }
}

//------------------------------------------------------------------------------
// Fused AdaGrad kernel
// accum = accum + grad^2
// param = param - lr * (grad + wd * param) / (sqrt(accum) + epsilon)
//------------------------------------------------------------------------------
template<typename T>
__global__ void fused_adagrad_kernel(T *param,
                                     const T *grad,
                                     T *accum,
                                     const float lr,
                                     const float epsilon,
                                     const float weight_decay,
                                     const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float p=caif_load_f<T>(param[idx]);
    float g=caif_load_f<T>(grad[idx]);
    float a=caif_load_f<T>(accum[idx]);
    if(weight_decay>0.0f)
    {
      g=g+weight_decay*p;
    }
    a=a+g*g;
    param[idx]=caif_store_f<T>(p-lr*g/(sqrtf(a)+epsilon));
    accum[idx]=caif_store_f<T>(a);
  }
}

//------------------------------------------------------------------------------
// Optimizer Kernel Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_fused_adam(T *param,
                       const T *grad,
                       float *m,
                       float *v,
                       const float lr,
                       const float beta1,
                       const float beta2,
                       const float epsilon,
                       const float weight_decay,
                       const float bias_correction1,
                       const float bias_correction2,
                       const int n,
                       cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  fused_adam_kernel<T><<<num_blocks,block_size,0,stream>>>(param,grad,m,v,lr,beta1,beta2,
                                                           epsilon,weight_decay,
                                                           bias_correction1,bias_correction2,
                                                           1e30f,n);
}

template void launch_fused_adam<float>(float *,
                                       const float *,
                                       float *,
                                       float *,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const float,
                                       const int,
                                       cudaStream_t);
template void launch_fused_adam<__half>(__half *,
                                        const __half *,
                                        float *,
                                        float *,
                                        const float,
                                        const float,
                                        const float,
                                        const float,
                                        const float,
                                        const float,
                                        const float,
                                        const int,
                                        cudaStream_t);
template void launch_fused_adam<__nv_bfloat16>(__nv_bfloat16 *,
                                               const __nv_bfloat16 *,
                                               float *,
                                               float *,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const int,
                                               cudaStream_t);

template<typename T>
void launch_fused_adam_clipped(T *param,
                               const T *grad,
                               float *m,
                               float *v,
                               const float lr,
                               const float beta1,
                               const float beta2,
                               const float epsilon,
                               const float weight_decay,
                               const float bias_correction1,
                               const float bias_correction2,
                               const float grad_scale,
                               const int n,
                               cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  fused_adam_clipped_kernel<T><<<num_blocks,block_size,0,stream>>>(
                                                                 param,grad,m,v,lr,beta1,beta2,
                                                                 epsilon,weight_decay,
                                                                 bias_correction1,bias_correction2,
                                                                 grad_scale,n);
}

template void launch_fused_adam_clipped<float>(float *,
                                               const float *,
                                               float *,
                                               float *,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const float,
                                               const int,
                                               cudaStream_t);
template void launch_fused_adam_clipped<__half>(__half *,
                                                const __half *,
                                                float *,
                                                float *,
                                                const float,
                                                const float,
                                                const float,
                                                const float,
                                                const float,
                                                const float,
                                                const float,
                                                const float,
                                                const int,
                                                cudaStream_t);
template void launch_fused_adam_clipped<__nv_bfloat16>(__nv_bfloat16 *,
                                                       const __nv_bfloat16 *,
                                                       float *,
                                                       float *,
                                                       const float,
                                                       const float,
                                                       const float,
                                                       const float,
                                                       const float,
                                                       const float,
                                                       const float,
                                                       const float,
                                                       const int,
                                                       cudaStream_t);

template<typename T>
void launch_fused_sgd_momentum(T *param,
                               const T *grad,
                               T *velocity,
                               const float lr,
                               const float momentum,
                               const float weight_decay,
                               const int n,
                               cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  fused_sgd_momentum_kernel<T><<<num_blocks,block_size,0,stream>>>(param,grad,velocity,
                                                                   lr,momentum,weight_decay,n);
}

template void launch_fused_sgd_momentum<float>(float *,
                                               const float *,
                                               float *,
                                               const float,
                                               const float,
                                               const float,
                                               const int,
                                               cudaStream_t);
template void launch_fused_sgd_momentum<__half>(__half *,
                                                const __half *,
                                                __half *,
                                                const float,
                                                const float,
                                                const float,
                                                const int,
                                                cudaStream_t);
template void launch_fused_sgd_momentum<__nv_bfloat16>(__nv_bfloat16 *,
                                                       const __nv_bfloat16 *,
                                                       __nv_bfloat16 *,
                                                       const float,
                                                       const float,
                                                       const float,
                                                       const int,
                                                       cudaStream_t);

template<typename T>
void launch_fused_sgd(T *param,
                      const T *grad,
                      const float lr,
                      const float weight_decay,
                      const int n,
                      cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  fused_sgd_kernel<T><<<num_blocks,block_size,0,stream>>>(param,grad,lr,weight_decay,n);
}

template void launch_fused_sgd<float>(float *,
                                      const float *,
                                      const float,
                                      const float,
                                      const int,
                                      cudaStream_t);
template void launch_fused_sgd<__half>(__half *,
                                       const __half *,
                                       const float,
                                       const float,
                                       const int,
                                       cudaStream_t);
template void launch_fused_sgd<__nv_bfloat16>(__nv_bfloat16 *,
                                              const __nv_bfloat16 *,
                                              const float,
                                              const float,
                                              const int,
                                              cudaStream_t);

template<typename T>
void launch_fused_rmsprop(T *param,
                          const T *grad,
                          T *avg_sq,
                          const float lr,
                          const float alpha,
                          const float epsilon,
                          const float weight_decay,
                          const int n,
                          cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  fused_rmsprop_kernel<T><<<num_blocks,block_size,0,stream>>>(param,grad,avg_sq,
                                                              lr,alpha,epsilon,
                                                              weight_decay,n);
}

template void launch_fused_rmsprop<float>(float *,
                                          const float *,
                                          float *,
                                          const float,
                                          const float,
                                          const float,
                                          const float,
                                          const int,
                                          cudaStream_t);
template void launch_fused_rmsprop<__half>(__half *,
                                           const __half *,
                                           __half *,
                                           const float,
                                           const float,
                                           const float,
                                           const float,
                                           const int,
                                           cudaStream_t);
template void launch_fused_rmsprop<__nv_bfloat16>(__nv_bfloat16 *,
                                                  const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  const float,
                                                  const float,
                                                  const float,
                                                  const float,
                                                  const int,
                                                  cudaStream_t);

template<typename T>
void launch_fused_adagrad(T *param,
                          const T *grad,
                          T *accum,
                          const float lr,
                          const float epsilon,
                          const float weight_decay,
                          const int n,
                          cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  fused_adagrad_kernel<T><<<num_blocks,block_size,0,stream>>>(param,grad,accum,
                                                              lr,epsilon,
                                                              weight_decay,n);
}

template void launch_fused_adagrad<float>(float *,
                                          const float *,
                                          float *,
                                          const float,
                                          const float,
                                          const float,
                                          const int,
                                          cudaStream_t);
template void launch_fused_adagrad<__half>(__half *,
                                           const __half *,
                                           __half *,
                                           const float,
                                           const float,
                                           const float,
                                           const int,
                                           cudaStream_t);
template void launch_fused_adagrad<__nv_bfloat16>(__nv_bfloat16 *,
                                                  const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  const float,
                                                  const float,
                                                  const float,
                                                  const int,
                                                  cudaStream_t);

//------------------------------------------------------------------------------
// RMSNorm Forward Kernel
// y[row][col] = x[row][col] / rms(x[row]) * gamma[col]
// rms(x) = sqrt(mean(x^2) + epsilon)
// One block per row, shared memory parallel reduction.
// Templated on activation T ∈ {float, __half, __nv_bfloat16}.
// gamma and rms_cache are always fp32 (standard autocast convention).
// All internal reductions are fp32; input loaded as T then cast.
//------------------------------------------------------------------------------
template<typename T>
__global__ void rmsnorm_forward_kernel(const T *input,
                                        const float *gamma,
                                        T *output,
                                        float *rms_cache,
                                        const float epsilon,
                                        const int rows,
                                        const int dim)
{
  const int row=blockIdx.x;
  if(row>=rows)
  {
    return;
  }

  const T *x=input+row*dim;
  T *y=output+row*dim;
  const int tid=threadIdx.x;

  // Phase 1: Compute sum(x^2) using stride loop
  float local_sum=0.0f;
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float val=float(x[col]);
    local_sum+=val*val;
  }

  // Warp shuffle reduction
  local_sum=warp_reduce_sum(local_sum);

  // Per-warp partial sums via shared memory
  __shared__ float warp_sums[g_caif_cuda_warp_size];
  const int lane=tid&(g_caif_cuda_warp_size-1);
  const int warp_id=tid/g_caif_cuda_warp_size;
  const int num_warps=blockDim.x/g_caif_cuda_warp_size;

  if(lane==0)
  {
    warp_sums[warp_id]=local_sum;
  }
  __syncthreads();

  // First warp reduces the per-warp sums
  if(warp_id==0)
  {
    local_sum=0.0f;
    if(lane<num_warps)
    {
      local_sum=warp_sums[lane];
    }
    local_sum=warp_reduce_sum(local_sum);
  }

  // Broadcast result to all threads via shared memory
  if(tid==0)
  {
    warp_sums[0]=local_sum;
  }
  __syncthreads();

  // Compute RMS value
  const float rms=sqrtf(warp_sums[0]/static_cast<float>(dim)+epsilon);
  if(tid==0)
  {
    rms_cache[row]=rms;
  }

  const float rstd=1.0f/rms;

  // Phase 2: Normalize and scale by gamma
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    y[col]=T(float(x[col])*rstd*gamma[col]);
  }
}

//------------------------------------------------------------------------------
// RMSNorm Backward Kernel
// Computes grad_input and accumulates grad_gamma via atomicAdd.
// grad_gamma must be pre-zeroed before this kernel is launched.
// Templated on T ∈ {float, __half, __nv_bfloat16}. gamma / rms_cache /
// grad_gamma stay fp32 (standard autocast convention).
//------------------------------------------------------------------------------
template<typename T>
__global__ void rmsnorm_backward_kernel(const T *grad_output,
                                         const T *input,
                                         const float *gamma,
                                         const float *rms_cache,
                                         T *grad_input,
                                         float *grad_gamma,
                                         const float epsilon,
                                         const int rows,
                                         const int dim)
{
  const int row=blockIdx.x;
  if(row>=rows)
  {
    return;
  }

  const T *dy=grad_output+row*dim;
  const T *x=input+row*dim;
  T *dx=grad_input+row*dim;
  const int tid=threadIdx.x;

  const float rstd=1.0f/rms_cache[row];

  // Phase 1: Compute sum(dy * gamma * x_hat) where x_hat = x * rstd
  float local_sum=0.0f;
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float x_hat=float(x[col])*rstd;
    local_sum+=float(dy[col])*gamma[col]*x_hat;
  }

  // Warp shuffle reduction
  local_sum=warp_reduce_sum(local_sum);

  __shared__ float warp_sums[g_caif_cuda_warp_size];
  const int lane=tid&(g_caif_cuda_warp_size-1);
  const int warp_id=tid/g_caif_cuda_warp_size;
  const int num_warps=blockDim.x/g_caif_cuda_warp_size;

  if(lane==0)
  {
    warp_sums[warp_id]=local_sum;
  }
  __syncthreads();

  if(warp_id==0)
  {
    local_sum=0.0f;
    if(lane<num_warps)
    {
      local_sum=warp_sums[lane];
    }
    local_sum=warp_reduce_sum(local_sum);
  }

  // Broadcast result
  if(tid==0)
  {
    warp_sums[0]=local_sum;
  }
  __syncthreads();

  const float sum_term=warp_sums[0]/static_cast<float>(dim);

  // Phase 2: Compute grad_input and accumulate grad_gamma
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float x_val=float(x[col]);
    const float dy_val=float(dy[col]);
    const float x_hat=x_val*rstd;
    dx[col]=T(rstd*(dy_val*gamma[col]-x_hat*sum_term));
    atomicAdd(&grad_gamma[col],dy_val*x_hat);
  }
}

//------------------------------------------------------------------------------
// LayerNorm Forward Kernel (Block-per-Row)
// y[row][col] = (x[row][col] - mean) / sqrt(var + eps) * gamma[col] + beta[col]
// One block per row, 256 threads, shared memory parallel reduction.
// Mirrors the RMSNorm architecture for maximum memory bandwidth utilization.
//------------------------------------------------------------------------------
template<typename T>
__global__ void layernorm_forward_kernel(const T *input,
                                             const float *gamma,
                                             const float *beta,
                                             T *output,
                                             float *mean_cache,
                                             float *rstd_cache,
                                             const float epsilon,
                                             const int rows,
                                             const int dim)
{
  const int row=blockIdx.x;
  if(row>=rows)
  {
    return;
  }

  const T *x=input+row*dim;
  T *y=output+row*dim;
  const int tid=threadIdx.x;

  // Phase 1: Compute sum(x) and sum(x^2) using stride loop
  float local_sum=0.0f;
  float local_sum_sq=0.0f;
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float val=float(x[col]);
    local_sum+=val;
    local_sum_sq+=val*val;
  }

  // Warp shuffle reduction for both sums
  local_sum=warp_reduce_sum(local_sum);
  local_sum_sq=warp_reduce_sum(local_sum_sq);

  // Per-warp partial sums via shared memory (two arrays)
  __shared__ float ws_sum[g_caif_cuda_warp_size];
  __shared__ float ws_sum_sq[g_caif_cuda_warp_size];
  const int lane=tid&(g_caif_cuda_warp_size-1);
  const int warp_id=tid/g_caif_cuda_warp_size;
  const int num_warps=blockDim.x/g_caif_cuda_warp_size;

  if(lane==0)
  {
    ws_sum[warp_id]=local_sum;
    ws_sum_sq[warp_id]=local_sum_sq;
  }
  __syncthreads();

  if(warp_id==0)
  {
    local_sum=0.0f;
    local_sum_sq=0.0f;
    if(lane<num_warps)
    {
      local_sum=ws_sum[lane];
      local_sum_sq=ws_sum_sq[lane];
    }
    local_sum=warp_reduce_sum(local_sum);
    local_sum_sq=warp_reduce_sum(local_sum_sq);
  }

  // Broadcast mean and rstd
  __shared__ float s_mean;
  __shared__ float s_rstd;
  if(tid==0)
  {
    const float dim_f=static_cast<float>(dim);
    s_mean=local_sum/dim_f;
    const float variance=local_sum_sq/dim_f-s_mean*s_mean;
    s_rstd=rsqrtf(variance+epsilon);
    mean_cache[row]=s_mean;
    rstd_cache[row]=s_rstd;
  }
  __syncthreads();

  const float mean=s_mean;
  const float rstd=s_rstd;

  // Phase 2: Normalize, scale, and shift
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float x_hat=(float(x[col])-mean)*rstd;
    y[col]=T(x_hat*gamma[col]+beta[col]);
  }
}

//------------------------------------------------------------------------------
// LayerNorm Backward Kernel
// Computes grad_input and accumulates grad_gamma/grad_beta via atomicAdd.
// grad_gamma and grad_beta must be pre-zeroed before this kernel is launched.
// Templated on T ∈ {float, __half, __nv_bfloat16}. gamma / beta / cache /
// grad_gamma / grad_beta stay fp32.
//------------------------------------------------------------------------------
template<typename T>
__global__ void layernorm_backward_kernel(const T *grad_output,
                                           const T *input,
                                           const float *gamma,
                                           const float *mean_cache,
                                           const float *rstd_cache,
                                           T *grad_input,
                                           float *grad_gamma,
                                           float *grad_beta,
                                           const int rows,
                                           const int dim)
{
  const int row=blockIdx.x;
  if(row>=rows)
  {
    return;
  }

  const T *dy=grad_output+row*dim;
  const T *x=input+row*dim;
  T *dx=grad_input+row*dim;
  const int tid=threadIdx.x;

  const float mean=mean_cache[row];
  const float rstd=rstd_cache[row];

  // Phase 1: Compute S1=sum(dy*gamma) and S2=sum(dy*gamma*x_hat)
  float local_s1=0.0f;
  float local_s2=0.0f;
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float x_hat=(float(x[col])-mean)*rstd;
    const float dy_gamma=float(dy[col])*gamma[col];
    local_s1+=dy_gamma;
    local_s2+=dy_gamma*x_hat;
  }

  // Warp shuffle reduction for both sums
  local_s1=warp_reduce_sum(local_s1);
  local_s2=warp_reduce_sum(local_s2);

  __shared__ float ws_s1[g_caif_cuda_warp_size];
  __shared__ float ws_s2[g_caif_cuda_warp_size];
  const int lane=tid&(g_caif_cuda_warp_size-1);
  const int warp_id=tid/g_caif_cuda_warp_size;
  const int num_warps=blockDim.x/g_caif_cuda_warp_size;

  if(lane==0)
  {
    ws_s1[warp_id]=local_s1;
    ws_s2[warp_id]=local_s2;
  }
  __syncthreads();

  if(warp_id==0)
  {
    local_s1=0.0f;
    local_s2=0.0f;
    if(lane<num_warps)
    {
      local_s1=ws_s1[lane];
      local_s2=ws_s2[lane];
    }
    local_s1=warp_reduce_sum(local_s1);
    local_s2=warp_reduce_sum(local_s2);
  }

  // Broadcast results
  if(tid==0)
  {
    ws_s1[0]=local_s1;
    ws_s2[0]=local_s2;
  }
  __syncthreads();

  const float dim_f=static_cast<float>(dim);
  const float s1=ws_s1[0]/dim_f;
  const float s2=ws_s2[0]/dim_f;

  // Phase 2: Compute grad_input, accumulate grad_gamma and grad_beta
  for(int col=tid;col<dim;col+=blockDim.x)
  {
    const float dy_val=float(dy[col]);
    const float x_hat=(float(x[col])-mean)*rstd;
    dx[col]=T(rstd*(dy_val*gamma[col]-s1-x_hat*s2));
    atomicAdd(&grad_gamma[col],dy_val*x_hat);
    atomicAdd(&grad_beta[col],dy_val);
  }
}

//------------------------------------------------------------------------------
// RMSNorm / LayerNorm Kernel Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_rmsnorm_forward(const T *input,
                            const float *gamma,
                            T *output,
                            float *rms_cache,
                            const float epsilon,
                            const int rows,
                            const int dim,
                            cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  rmsnorm_forward_kernel<T><<<rows,block_size,0,stream>>>(input,
                                                          gamma,
                                                          output,
                                                          rms_cache,
                                                          epsilon,
                                                          rows,
                                                          dim);
}
template void launch_rmsnorm_forward<float>(const float *,
                                            const float *,
                                            float *,
                                            float *,
                                            float,
                                            int,
                                            int,
                                            cudaStream_t);
template void launch_rmsnorm_forward<__half>(const __half *,
                                             const float *,
                                             __half *,
                                             float *,
                                             float,
                                             int,
                                             int,
                                             cudaStream_t);
template void launch_rmsnorm_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const float *,
                                                    __nv_bfloat16 *,
                                                    float *,
                                                    float,
                                                    int,
                                                    int,
                                                    cudaStream_t);

template<typename T>
void launch_rmsnorm_backward(const T *grad_output,
                              const T *input,
                              const float *gamma,
                              const float *rms_cache,
                              T *grad_input,
                              float *grad_gamma,
                              const float epsilon,
                              const int rows,
                              const int dim,
                              cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  rmsnorm_backward_kernel<T><<<rows,block_size,0,stream>>>(grad_output,
                                                           input,
                                                           gamma,
                                                           rms_cache,
                                                           grad_input,
                                                           grad_gamma,
                                                           epsilon,
                                                           rows,
                                                           dim);
}
template void launch_rmsnorm_backward<float>(const float *,
                                             const float *,
                                             const float *,
                                             const float *,
                                             float *,
                                             float *,
                                             float,
                                             int,
                                             int,
                                             cudaStream_t);
template void launch_rmsnorm_backward<__half>(const __half *,
                                              const __half *,
                                              const float *,
                                              const float *,
                                              __half *,
                                              float *,
                                              float,
                                              int,
                                              int,
                                              cudaStream_t);
template void launch_rmsnorm_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     const __nv_bfloat16 *,
                                                     const float *,
                                                     const float *,
                                                     __nv_bfloat16 *,
                                                     float *,
                                                     float,
                                                     int,
                                                     int,
                                                     cudaStream_t);

template<typename T>
void launch_layernorm_forward(const T *input,
                              const float *gamma,
                              const float *beta,
                              T *output,
                              float *mean_cache,
                              float *rstd_cache,
                              const float epsilon,
                              const int rows,
                              const int dim,
                              cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  layernorm_forward_kernel<T><<<rows,block_size,0,stream>>>(input,
                                                            gamma,
                                                            beta,
                                                            output,
                                                            mean_cache,
                                                            rstd_cache,
                                                            epsilon,
                                                            rows,
                                                            dim);
}
template void launch_layernorm_forward<float>(const float *,
                                              const float *,
                                              const float *,
                                              float *,
                                              float *,
                                              float *,
                                              float,
                                              int,
                                              int,
                                              cudaStream_t);
template void launch_layernorm_forward<__half>(const __half *,
                                               const float *,
                                               const float *,
                                               __half *,
                                               float *,
                                               float *,
                                               float,
                                               int,
                                               int,
                                               cudaStream_t);
template void launch_layernorm_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                      const float *,
                                                      const float *,
                                                      __nv_bfloat16 *,
                                                      float *,
                                                      float *,
                                                      float,
                                                      int,
                                                      int,
                                                      cudaStream_t);

template<typename T>
void launch_layernorm_backward(const T *grad_output,
                                const T *input,
                                const float *gamma,
                                const float *mean_cache,
                                const float *rstd_cache,
                                T *grad_input,
                                float *grad_gamma,
                                float *grad_beta,
                                const int rows,
                                const int dim,
                                cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  layernorm_backward_kernel<T><<<rows,block_size,0,stream>>>(grad_output,
                                                             input,
                                                             gamma,
                                                             mean_cache,
                                                             rstd_cache,
                                                             grad_input,
                                                             grad_gamma,
                                                             grad_beta,
                                                             rows,
                                                             dim);
}
template void launch_layernorm_backward<float>(const float *,
                                               const float *,
                                               const float *,
                                               const float *,
                                               const float *,
                                               float *,
                                               float *,
                                               float *,
                                               int,
                                               int,
                                               cudaStream_t);
template void launch_layernorm_backward<__half>(const __half *,
                                                const __half *,
                                                const float *,
                                                const float *,
                                                const float *,
                                                __half *,
                                                float *,
                                                float *,
                                                int,
                                                int,
                                                cudaStream_t);
template void launch_layernorm_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       const __nv_bfloat16 *,
                                                       const float *,
                                                       const float *,
                                                       const float *,
                                                       __nv_bfloat16 *,
                                                       float *,
                                                       float *,
                                                       int,
                                                       int,
                                                       cudaStream_t);

//------------------------------------------------------------------------------
// Transpose 0213 Kernel
// Swaps dims 1 and 2 of a logical 4D tensor [batch, dim0, dim1, dim2]
// -> [batch, dim1, dim0, dim2]
// Element-parallel: one thread per element.
//------------------------------------------------------------------------------
template<typename T>
__global__ void transpose_0213_kernel(const T *input,
                                       T *output,
                                       const int dim0,
                                       const int dim1,
                                       const int dim2,
                                       const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    // Decompose flat index into [b, d0, d1, d2]
    const int stride_d0=dim1*dim2;
    const int stride_batch=dim0*stride_d0;
    const int b=idx/stride_batch;
    const int rem=idx%stride_batch;
    const int d0=rem/stride_d0;
    const int rem2=rem%stride_d0;
    const int d1=rem2/dim2;
    const int d2=rem2%dim2;

    // Output index in [b, d1, d0, d2]
    const int out_stride_d1=dim0*dim2;
    const int out_idx=b*dim1*out_stride_d1+
                      d1*out_stride_d1+
                      d0*dim2+
                      d2;
    output[out_idx]=input[idx];
  }
}

//------------------------------------------------------------------------------
// Transpose 0213 Strided Kernel
// Same logical reshuffle as transpose_0213_kernel, but reads from an input
// whose stride along dim0 may be LARGER than dim1*dim2 (i.e. the input is a
// width-slice of a wider packed row). Caller supplies the base pointer
// positioned at column offset 0 of the slice, and input_d0_stride = full row
// width of the packed buffer.
// Input logical view: [batch, dim0, dim1, dim2] with row stride input_d0_stride
// Output layout:      [batch, dim1, dim0, dim2] contiguous
//------------------------------------------------------------------------------
template<typename T>
__global__ void transpose_0213_strided_kernel(const T *input,
                                              T *output,
                                              const int dim0,
                                              const int dim1,
                                              const int dim2,
                                              const int input_d0_stride,
                                              const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    // Decompose flat output index into [b, d0, d1, d2]
    const int stride_d0=dim1*dim2;
    const int stride_batch=dim0*stride_d0;
    const int b=idx/stride_batch;
    const int rem=idx%stride_batch;
    const int d0=rem/stride_d0;
    const int rem2=rem%stride_d0;
    const int d1=rem2/dim2;
    const int d2=rem2%dim2;

    // Input index uses input_d0_stride (> dim1*dim2) between consecutive d0
    const int input_batch_stride=dim0*input_d0_stride;
    const int in_idx=b*input_batch_stride+
                     d0*input_d0_stride+
                     d1*dim2+
                     d2;

    // Output index in [b, d1, d0, d2]
    const int out_stride_d1=dim0*dim2;
    const int out_idx=b*dim1*out_stride_d1+
                      d1*out_stride_d1+
                      d0*dim2+
                      d2;
    output[out_idx]=input[in_idx];
  }
}

//------------------------------------------------------------------------------
// Causal Mask Fill Kernel
// Sets upper triangle (j > i) to -1e9 for each [seq_len, seq_len] matrix.
// Layout: [num_matrices, seq_len, seq_len]
//------------------------------------------------------------------------------
template<typename T>
__global__ void causal_mask_fill_kernel(T *scores,
                                        const int seq_len,
                                        const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int pos=idx%(seq_len*seq_len);
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    if(col>row)
    {
      scores[idx]=T(-1e9f);
    }
  }
}

//------------------------------------------------------------------------------
// Causal Mask Fill with Offset Kernel (for KV-cache)
// Sets positions where col > (row + offset) to -1e9 for rectangular matrices.
// Layout: [num_matrices, query_len, key_len]
// offset = previous cached sequence length
//------------------------------------------------------------------------------
template<typename T>
__global__ void causal_mask_fill_offset_kernel(T *scores,
                                               const int query_len,
                                               const int key_len,
                                               const int offset,
                                               const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int matrix_size=query_len*key_len;
    const int pos=idx%matrix_size;
    const int row=pos/key_len;
    const int col=pos%key_len;
    if(col>(offset+row))
    {
      scores[idx]=T(-1e9f);
    }
  }
}

//------------------------------------------------------------------------------
// Attention Softmax Forward Kernel
// Row-wise softmax with max subtraction for numerical stability.
// One block per row: multi-warp online reduction, normalize.
// Layout: [num_rows, row_len]
//------------------------------------------------------------------------------
// Multi-warp block per row. Intra-warp uses __shfl_xor_sync only. Cross-warp
// combine writes num_warps floats each for (m, s) into shared memory, then
// warp 0 reduces those num_warps values using __shfl_xor_sync again (still
// a pure warp-shuffle reduction — the shared memory is just a staging area
// for warp leaders, not the inter-warp reduction pattern that triggers the
// sm_120 hazard documented in caif_constants.h).
template<typename T>
__global__ void attention_softmax_kernel(const T *input,
                                          T *output,
                                          const int num_rows,
                                          const int row_len)
{
  const int row=blockIdx.x;
  if(row>=num_rows)
  {
    return;
  }

  const T *x=input+row*row_len;
  T *y=output+row*row_len;
  const unsigned mask=g_caif_cuda_warp_full_mask;
  const int lane=threadIdx.x%g_caif_cuda_warp_size;
  const int warp_id=threadIdx.x/g_caif_cuda_warp_size;
  const int num_warps=blockDim.x/g_caif_cuda_warp_size;

  extern __shared__ float smem[];
  // Layout: smem[0..num_warps)          = per-warp max
  //         smem[num_warps..2*num_warps) = per-warp sum
  //         smem[2*num_warps]            = final row_max
  //         smem[2*num_warps+1]          = final row_sum

  // Phase 1: per-thread online (max, sum) scan.
  // Reference: Milakov & Gimelshein, "Online normalizer calculation for
  // softmax" (2018). Invariant per thread: s = sum_{seen} exp(x_i - m).
  float m=-1e30f;
  float s=0.0f;
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    const float v=float(x[col]);
    if(v>m)
    {
      s=s*expf(m-v)+1.0f;
      m=v;
    }
    else
    {
      s+=expf(v-m);
    }
  }

  // Intra-warp combine.
  for(int off=g_caif_cuda_warp_size/2;off>0;off>>=1)
  {
    const float m_other=__shfl_xor_sync(mask,m,off);
    const float s_other=__shfl_xor_sync(mask,s,off);
    if(m_other>m)
    {
      s=s*expf(m-m_other)+s_other;
      m=m_other;
    }
    else
    {
      s+=s_other*expf(m_other-m);
    }
  }

  // Cross-warp combine.
  if(lane==0)
  {
    smem[warp_id]=m;
    smem[num_warps+warp_id]=s;
  }
  __syncthreads();

  if(warp_id==0)
  {
    float fm=-1e30f;
    float fs=0.0f;
    if(lane<num_warps)
    {
      fm=smem[lane];
      fs=smem[num_warps+lane];
    }
    for(int off=num_warps>>1;off>0;off>>=1)
    {
      const float m_other=__shfl_xor_sync(mask,fm,off);
      const float s_other=__shfl_xor_sync(mask,fs,off);
      if(m_other>fm)
      {
        fs=fs*expf(fm-m_other)+s_other;
        fm=m_other;
      }
      else
      {
        fs+=s_other*expf(m_other-fm);
      }
    }
    if(lane==0)
    {
      smem[2*num_warps]=fm;
      smem[2*num_warps+1]=fs;
    }
  }
  __syncthreads();

  const float row_max=smem[2*num_warps];
  const float inv_sum=1.0f/smem[2*num_warps+1];

  // Phase 2: normalize (second and final pass over input).
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    y[col]=T(expf(float(x[col])-row_max)*inv_sum);
  }
}

//------------------------------------------------------------------------------
// Attention Softmax Backward Kernel
// For each row: grad_input[i] = output[i] * (grad_output[i] - dot(grad_output, output))
// One block per row, multi-warp, shuffle-only reductions.
//------------------------------------------------------------------------------
template<typename T>
__global__ void attention_softmax_backward_kernel(const T *grad_output,
                                                    const T *output,
                                                    T *grad_input,
                                                    const int num_rows,
                                                    const int row_len)
{
  const int row=blockIdx.x;
  if(row>=num_rows)
  {
    return;
  }

  const T *dy=grad_output+row*row_len;
  const T *y=output+row*row_len;
  T *dx=grad_input+row*row_len;
  const unsigned mask=g_caif_cuda_warp_full_mask;
  const int lane=threadIdx.x%g_caif_cuda_warp_size;
  const int warp_id=threadIdx.x/g_caif_cuda_warp_size;
  const int num_warps=blockDim.x/g_caif_cuda_warp_size;

  extern __shared__ float smem[];
  // Layout: smem[0..num_warps) = per-warp dot partial
  //         smem[num_warps]    = final row dot

  // Phase 1: dot(grad_output, output) — per-thread partial.
  float local_dot=0.0f;
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    local_dot+=float(dy[col])*float(y[col]);
  }
  // Intra-warp reduce.
  for(int off=g_caif_cuda_warp_size/2;off>0;off>>=1)
  {
    local_dot+=__shfl_xor_sync(mask,local_dot,off);
  }
  // Cross-warp combine.
  if(lane==0)
  {
    smem[warp_id]=local_dot;
  }
  __syncthreads();

  if(warp_id==0)
  {
    float fd=0.0f;
    if(lane<num_warps)
    {
      fd=smem[lane];
    }
    for(int off=num_warps>>1;off>0;off>>=1)
    {
      fd+=__shfl_xor_sync(mask,fd,off);
    }
    if(lane==0)
    {
      smem[num_warps]=fd;
    }
  }
  __syncthreads();
  const float dot_val=smem[num_warps];

  // Phase 2: gradient.
  for(int col=threadIdx.x;col<row_len;col+=blockDim.x)
  {
    const float y_val=float(y[col]);
    const float dy_val=float(dy[col]);
    dx[col]=T(y_val*(dy_val-dot_val));
  }
}

//------------------------------------------------------------------------------
// Causal Mask Gradient Kernel
// Zeros upper triangle (j > i) of gradient scores.
// Layout: [num_matrices, seq_len, seq_len]
//------------------------------------------------------------------------------
template<typename T>
__global__ void causal_mask_grad_kernel(T *grad_scores,
                                         const int seq_len,
                                         const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int pos=idx%(seq_len*seq_len);
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    if(col>row)
    {
      grad_scores[idx]=T(0.0f);
    }
  }
}

//------------------------------------------------------------------------------
// Prefix Mask Fill Kernel
// Allowed if (k <= q) OR (k < prefix_lengths[b]); disallowed positions get -1e9.
// Layout: [num_matrices, seq_len, seq_len] where num_matrices = batch*num_heads.
// prefix_lengths: device array of length batch (int32 per batch element).
//------------------------------------------------------------------------------
template<typename T>
__global__ void prefix_mask_fill_kernel(T *scores,
                                        const uint32_t *prefix_lengths,
                                        const int num_heads,
                                        const int seq_len,
                                        const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int matrix_size=seq_len*seq_len;
    const int matrix_idx=idx/matrix_size;
    const int pos=idx%matrix_size;
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    const int batch_idx=matrix_idx/num_heads;
    const int pfx=prefix_lengths[batch_idx];
    if(col>row && col>=pfx)
    {
      scores[idx]=T(-1e9f);
    }
  }
}

//------------------------------------------------------------------------------
// Prefix Mask Fill with Offset Kernel (for KV-cache / cached inference)
// Query at local row r corresponds to absolute position (offset + r).
// Allowed if col <= (offset + r) OR col < prefix_lengths[b].
//------------------------------------------------------------------------------
template<typename T>
__global__ void prefix_mask_fill_offset_kernel(T *scores,
                                               const uint32_t *prefix_lengths,
                                               const int num_heads,
                                               const int query_len,
                                               const int key_len,
                                               const int offset,
                                               const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int matrix_size=query_len*key_len;
    const int matrix_idx=idx/matrix_size;
    const int pos=idx%matrix_size;
    const int row=pos/key_len;
    const int col=pos%key_len;
    const int batch_idx=matrix_idx/num_heads;
    const int pfx=prefix_lengths[batch_idx];
    if(col>(offset+row) && col>=pfx)
    {
      scores[idx]=T(-1e9f);
    }
  }
}

//------------------------------------------------------------------------------
// Prefix Mask Gradient Kernel
// Zeros disallowed positions (col > row AND col >= prefix_lengths[b]).
//------------------------------------------------------------------------------
template<typename T>
__global__ void prefix_mask_grad_kernel(T *grad_scores,
                                        const uint32_t *prefix_lengths,
                                        const int num_heads,
                                        const int seq_len,
                                        const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int matrix_size=seq_len*seq_len;
    const int matrix_idx=idx/matrix_size;
    const int pos=idx%matrix_size;
    const int row=pos/seq_len;
    const int col=pos%seq_len;
    const int batch_idx=matrix_idx/num_heads;
    const int pfx=prefix_lengths[batch_idx];
    if(col>row && col>=pfx)
    {
      grad_scores[idx]=T(0.0f);
    }
  }
}

//------------------------------------------------------------------------------
// Multi-Head Attention Kernel Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_transpose_0213(const T *input,
                           T *output,
                           const int batch,
                           const int dim0,
                           const int dim1,
                           const int dim2,
                           cudaStream_t stream)
{
  const int total=batch*dim0*dim1*dim2;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  transpose_0213_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                               output,
                                                               dim0,
                                                               dim1,
                                                               dim2,
                                                               total);
}

template void launch_transpose_0213<float>(const float *,float *,int,int,int,int,cudaStream_t);
template void launch_transpose_0213<__half>(const __half *,__half *,int,int,int,int,cudaStream_t);
template void launch_transpose_0213<__nv_bfloat16>(const __nv_bfloat16 *,
                                                   __nv_bfloat16 *,
                                                   int,int,int,int,
                                                   cudaStream_t);

template<typename T>
void launch_transpose_0213_strided(const T *input,
                                   T *output,
                                   const int batch,
                                   const int dim0,
                                   const int dim1,
                                   const int dim2,
                                   const int input_d0_stride,
                                   cudaStream_t stream)
{
  const int total=batch*dim0*dim1*dim2;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  transpose_0213_strided_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                                       output,
                                                                       dim0,
                                                                       dim1,
                                                                       dim2,
                                                                       input_d0_stride,
                                                                       total);
}

template void launch_transpose_0213_strided<float>(const float *,
                                                   float *,
                                                   int,
                                                   int,
                                                   int,
                                                   int,
                                                   int,
                                                   cudaStream_t);
template void launch_transpose_0213_strided<__half>(const __half *,
                                                    __half *,
                                                    int,
                                                    int,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);
template void launch_transpose_0213_strided<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           cudaStream_t);

template<typename T>
void launch_causal_mask_fill(T *scores,
                             const int num_matrices,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int total=num_matrices*seq_len*seq_len;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  causal_mask_fill_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,
                                                                 seq_len,
                                                                 total);
}

template void launch_causal_mask_fill<float>(float *,int,int,cudaStream_t);
template void launch_causal_mask_fill<__half>(__half *,int,int,cudaStream_t);
template void launch_causal_mask_fill<__nv_bfloat16>(__nv_bfloat16 *,int,int,cudaStream_t);

template<typename T>
void launch_causal_mask_fill_offset(T *scores,
                                    const int num_matrices,
                                    const int query_len,
                                    const int key_len,
                                    const int offset,
                                    cudaStream_t stream)
{
  const int total=num_matrices*query_len*key_len;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  causal_mask_fill_offset_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,
                                                                         query_len,
                                                                         key_len,
                                                                         offset,
                                                                         total);
}

template void launch_causal_mask_fill_offset<float>(float *,int,int,int,int,cudaStream_t);
template void launch_causal_mask_fill_offset<__half>(__half *,int,int,int,int,cudaStream_t);
template void launch_causal_mask_fill_offset<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,int,cudaStream_t);

template<typename T>
void launch_attention_softmax(const T *input,
                              T *output,
                              const int num_rows,
                              const int row_len,
                              cudaStream_t stream)
{
  // Multi-warp block per row. Intra-warp reductions use __shfl_xor_sync only;
  // cross-warp combine writes one float per warp to shared memory, reduces
  // those num_warps values in warp 0 via warp-shuffle, and broadcasts the
  // result back through shared memory. Shared-mem layout:
  //   [0..num_warps)       = per-warp max
  //   [num_warps..2*num_warps) = per-warp sum
  //   [2*num_warps]        = final row max
  //   [2*num_warps+1]      = final row sum
  const int block_size=GetSoftmaxBlockSize();
  const int num_warps=block_size/g_caif_cuda_warp_size;
  const size_t shared_mem_size=(2*num_warps+2)*sizeof(float);
  attention_softmax_kernel<T><<<num_rows,block_size,shared_mem_size,stream>>>(
    input,output,num_rows,row_len);
}
template void launch_attention_softmax<float>(const float *,
                                              float *,
                                              int,
                                              int,
                                              cudaStream_t);
template void launch_attention_softmax<__half>(const __half *,
                                               __half *,
                                               int,
                                               int,
                                               cudaStream_t);
template void launch_attention_softmax<__nv_bfloat16>(const __nv_bfloat16 *,
                                                      __nv_bfloat16 *,
                                                      int,
                                                      int,
                                                      cudaStream_t);

template<typename T>
void launch_attention_softmax_backward(const T *grad_output,
                                       const T *output,
                                       T *grad_input,
                                       const int num_rows,
                                       const int row_len,
                                       cudaStream_t stream)
{
  // Multi-warp block per row. See launch_attention_softmax for the layout
  // rationale; backward needs only (num_warps + 1) staging floats (per-warp
  // dot partials + final).
  const int block_size=GetSoftmaxBlockSize();
  const int num_warps=block_size/g_caif_cuda_warp_size;
  const size_t shared_mem_size=(num_warps+1)*sizeof(float);
  attention_softmax_backward_kernel<T><<<num_rows,block_size,shared_mem_size,stream>>>(
    grad_output,output,grad_input,num_rows,row_len);
}
template void launch_attention_softmax_backward<float>(const float *,
                                                       const float *,
                                                       float *,
                                                       int,
                                                       int,
                                                       cudaStream_t);
template void launch_attention_softmax_backward<__half>(const __half *,
                                                        const __half *,
                                                        __half *,
                                                        int,
                                                        int,
                                                        cudaStream_t);
template void launch_attention_softmax_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                               const __nv_bfloat16 *,
                                                               __nv_bfloat16 *,
                                                               int,
                                                               int,
                                                               cudaStream_t);

template<typename T>
void launch_causal_mask_grad(T *grad_scores,
                             const int num_matrices,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int total=num_matrices*seq_len*seq_len;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  causal_mask_grad_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_scores,
                                                                 seq_len,
                                                                 total);
}

template void launch_causal_mask_grad<float>(float *,int,int,cudaStream_t);
template void launch_causal_mask_grad<__half>(__half *,int,int,cudaStream_t);
template void launch_causal_mask_grad<__nv_bfloat16>(__nv_bfloat16 *,int,int,cudaStream_t);

template<typename T>
void launch_prefix_mask_fill(T *scores,
                             const uint32_t *prefix_lengths,
                             const int batch_size,
                             const int num_heads,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int total=batch_size*num_heads*seq_len*seq_len;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  prefix_mask_fill_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,
                                                                  prefix_lengths,
                                                                  num_heads,
                                                                  seq_len,
                                                                  total);
}

template void launch_prefix_mask_fill<float>(float *,const uint32_t *,int,int,int,cudaStream_t);
template void launch_prefix_mask_fill<__half>(__half *,const uint32_t *,int,int,int,cudaStream_t);
template void launch_prefix_mask_fill<__nv_bfloat16>(__nv_bfloat16 *,const uint32_t *,int,int,int,cudaStream_t);

template<typename T>
void launch_prefix_mask_fill_offset(T *scores,
                                    const uint32_t *prefix_lengths,
                                    const int batch_size,
                                    const int num_heads,
                                    const int query_len,
                                    const int key_len,
                                    const int offset,
                                    cudaStream_t stream)
{
  const int total=batch_size*num_heads*query_len*key_len;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  prefix_mask_fill_offset_kernel<T><<<num_blocks,block_size,0,stream>>>(scores,
                                                                         prefix_lengths,
                                                                         num_heads,
                                                                         query_len,
                                                                         key_len,
                                                                         offset,
                                                                         total);
}

template void launch_prefix_mask_fill_offset<float>(float *,const uint32_t *,int,int,int,int,int,cudaStream_t);
template void launch_prefix_mask_fill_offset<__half>(__half *,const uint32_t *,int,int,int,int,int,cudaStream_t);
template void launch_prefix_mask_fill_offset<__nv_bfloat16>(__nv_bfloat16 *,
                                                            const uint32_t *,
                                                            int,int,int,int,int,
                                                            cudaStream_t);

template<typename T>
void launch_prefix_mask_grad(T *grad_scores,
                             const uint32_t *prefix_lengths,
                             const int batch_size,
                             const int num_heads,
                             const int seq_len,
                             cudaStream_t stream)
{
  const int total=batch_size*num_heads*seq_len*seq_len;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  prefix_mask_grad_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_scores,
                                                                  prefix_lengths,
                                                                  num_heads,
                                                                  seq_len,
                                                                  total);
}

template void launch_prefix_mask_grad<float>(float *,const uint32_t *,int,int,int,cudaStream_t);
template void launch_prefix_mask_grad<__half>(__half *,const uint32_t *,int,int,int,cudaStream_t);
template void launch_prefix_mask_grad<__nv_bfloat16>(__nv_bfloat16 *,const uint32_t *,int,int,int,cudaStream_t);

//------------------------------------------------------------------------------
// RoPE Forward Kernel
// Applies rotary position embeddings in-place.
// data: [batch_heads, seq_len, head_dim], head_dim must be even.
// One thread per (batch_head, position, pair_index).
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_forward_kernel(T *data,
                                    const int seq_len,
                                    const int head_dim,
                                    const float base,
                                    const int style,
                                    const int total_pairs)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int half_dim=head_dim/2;
    const int pair_per_row=half_dim;
    const int row=idx/pair_per_row;
    const int pair=idx%pair_per_row;

    const int pos=row%seq_len;

    const float freq_exp=
      2.0f*static_cast<float>(pair)/static_cast<float>(head_dim);
    const float theta=static_cast<float>(pos)/powf(base,freq_exp);
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int row_base=row*head_dim;
    int idx0;
    int idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_dim;
    }
    else
    {
      idx0=row_base+pair*2;
      idx1=row_base+pair*2+1;
    }

    const float x0=float(data[idx0]);
    const float x1=float(data[idx1]);
    data[idx0]=T(x0*cos_t-x1*sin_t);
    data[idx1]=T(x0*sin_t+x1*cos_t);
  }
}

//------------------------------------------------------------------------------
// RoPE Forward with Position Offset Kernel (for KV-cache)
// Same as rope_forward_kernel but adds pos_offset to position calculation.
// Used when processing new tokens that continue from cached positions.
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_forward_offset_kernel(T *data,
                                           const int seq_len,
                                           const int head_dim,
                                           const float base,
                                           const int pos_offset,
                                           const int style,
                                           const int total_pairs)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int half_dim=head_dim/2;
    const int pair_per_row=half_dim;
    const int row=idx/pair_per_row;
    const int pair=idx%pair_per_row;

    const int pos=(row%seq_len)+pos_offset;

    const float freq_exp=
      2.0f*static_cast<float>(pair)/static_cast<float>(head_dim);
    const float theta=static_cast<float>(pos)/powf(base,freq_exp);
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int row_base=row*head_dim;
    int idx0;
    int idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_dim;
    }
    else
    {
      idx0=row_base+pair*2;
      idx1=row_base+pair*2+1;
    }

    const float x0=float(data[idx0]);
    const float x1=float(data[idx1]);
    data[idx0]=T(x0*cos_t-x1*sin_t);
    data[idx1]=T(x0*sin_t+x1*cos_t);
  }
}

//------------------------------------------------------------------------------
// RoPE Backward Kernel
// Applies inverse rotation in-place (swap sin sign).
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_backward_kernel(T *data,
                                     const int seq_len,
                                     const int head_dim,
                                     const float base,
                                     const int style,
                                     const int total_pairs)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int half_dim=head_dim/2;
    const int pair_per_row=half_dim;
    const int row=idx/pair_per_row;
    const int pair=idx%pair_per_row;

    const int pos=row%seq_len;

    const float freq_exp=
      2.0f*static_cast<float>(pair)/static_cast<float>(head_dim);
    const float theta=static_cast<float>(pos)/powf(base,freq_exp);
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int row_base=row*head_dim;
    int idx0;
    int idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_dim;
    }
    else
    {
      idx0=row_base+pair*2;
      idx1=row_base+pair*2+1;
    }

    const float g0=float(data[idx0]);
    const float g1=float(data[idx1]);
    data[idx0]=T(g0*cos_t+g1*sin_t);
    data[idx1]=T(-g0*sin_t+g1*cos_t);
  }
}

//------------------------------------------------------------------------------
// GQA Repeat KV Kernel
// input: [batch * num_kv_heads, seq_len, head_dim]
// output: [batch * num_heads, seq_len, head_dim]
// One thread per output element.
//------------------------------------------------------------------------------
template<typename T>
__global__ void gqa_repeat_kv_kernel(const T *input,
                                     T *output,
                                     const int num_kv_heads,
                                     const int repeat_factor,
                                     const int seq_len,
                                     const int head_dim,
                                     const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int num_heads=num_kv_heads*repeat_factor;
    const int head_stride=seq_len*head_dim;
    const int batch_stride=num_heads*head_stride;

    const int b=idx/batch_stride;
    const int rem=idx%batch_stride;
    const int h=rem/head_stride;
    const int rem2=rem%head_stride;

    const int kv_h=h/repeat_factor;
    const int input_batch_stride=num_kv_heads*head_stride;
    const int in_idx=b*input_batch_stride+kv_h*head_stride+rem2;

    output[idx]=input[in_idx];
  }
}

//------------------------------------------------------------------------------
// GQA Reduce KV Kernel
// input: [batch * num_heads, seq_len, head_dim]
// output: [batch * num_kv_heads, seq_len, head_dim]
// One thread per output element, loops over repeat_factor.
//------------------------------------------------------------------------------
template<typename T>
__global__ void gqa_reduce_kv_kernel(const T *input,
                                     T *output,
                                     const int num_kv_heads,
                                     const int repeat_factor,
                                     const int seq_len,
                                     const int head_dim,
                                     const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int head_stride=seq_len*head_dim;
    const int out_batch_stride=num_kv_heads*head_stride;

    const int b=idx/out_batch_stride;
    const int rem=idx%out_batch_stride;
    const int kv_h=rem/head_stride;
    const int rem2=rem%head_stride;

    const int num_heads=num_kv_heads*repeat_factor;
    const int in_batch_stride=num_heads*head_stride;

    float sum=0.0f;
    for(int r=0;r<repeat_factor;++r)
    {
      const int h=kv_h*repeat_factor+r;
      const int in_idx=b*in_batch_stride+h*head_stride+rem2;
      sum+=float(input[in_idx]);
    }
    output[idx]=T(sum);
  }
}

//------------------------------------------------------------------------------
// RoPE / GQA Kernel Launchers
//------------------------------------------------------------------------------

template<typename T>
void launch_rope_forward(T *data,
                         const int batch_heads,
                         const int seq_len,
                         const int head_dim,
                         const float base,
                         const int style,
                         cudaStream_t stream)
{
  const int total_pairs=batch_heads*seq_len*(head_dim/2);
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_forward_kernel<T><<<num_blocks,block_size,0,stream>>>(
    data,seq_len,head_dim,base,style,total_pairs);
}

template void launch_rope_forward<float>(float *,int,int,int,float,int,cudaStream_t);
template void launch_rope_forward<__half>(__half *,int,int,int,float,int,cudaStream_t);
template void launch_rope_forward<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,float,int,cudaStream_t);

//------------------------------------------------------------------------------
// Partial-rotary RoPE Forward Kernel
// Rotates only the first `rope_dim` dims of each head row. Stride within
// a head is still `head_dim` (the unrotated tail at indices [rope_dim,
// head_dim) is left untouched and stays in place).  Frequencies use
// rope_dim as the divisor (matches HF rotate_half(x[..., :rope_dim])).
// One thread per (batch_head, position, pair_index in [0, rope_dim/2)).
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_forward_partial_kernel(T *data,
                                            const int seq_len,
                                            const int head_dim,
                                            const int rope_dim,
                                            const float base,
                                            const int style,
                                            const int total_pairs)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int half_rope=rope_dim/2;
    const int row=idx/half_rope;
    const int pair=idx%half_rope;

    const int pos=row%seq_len;

    const float freq_exp=
      2.0f*static_cast<float>(pair)/static_cast<float>(rope_dim);
    const float theta=static_cast<float>(pos)/powf(base,freq_exp);
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int row_base=row*head_dim;
    int idx0;
    int idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_rope;
    }
    else
    {
      idx0=row_base+pair*2;
      idx1=row_base+pair*2+1;
    }

    const float x0=float(data[idx0]);
    const float x1=float(data[idx1]);
    data[idx0]=T(x0*cos_t-x1*sin_t);
    data[idx1]=T(x0*sin_t+x1*cos_t);
  }
}

template<typename T>
void launch_rope_forward_partial(T *data,
                                 const int batch_heads,
                                 const int seq_len,
                                 const int head_dim,
                                 const int rope_dim,
                                 const float base,
                                 const int style,
                                 cudaStream_t stream)
{
  const int total_pairs=batch_heads*seq_len*(rope_dim/2);
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_forward_partial_kernel<T><<<num_blocks,block_size,0,stream>>>(
    data,seq_len,head_dim,rope_dim,base,style,total_pairs);
}

template void launch_rope_forward_partial<float>(float *,int,int,int,int,float,int,cudaStream_t);
template void launch_rope_forward_partial<__half>(__half *,int,int,int,int,float,int,cudaStream_t);
template void launch_rope_forward_partial<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,int,float,int,cudaStream_t);

template<typename T>
void launch_rope_forward_offset(T *data,
                                const int batch_heads,
                                const int seq_len,
                                const int head_dim,
                                const float base,
                                const int pos_offset,
                                const int style,
                                cudaStream_t stream)
{
  const int total_pairs=batch_heads*seq_len*(head_dim/2);
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_forward_offset_kernel<T><<<num_blocks,block_size,0,stream>>>(
    data,seq_len,head_dim,base,pos_offset,style,total_pairs);
}

template void launch_rope_forward_offset<float>(float *,int,int,int,float,int,int,cudaStream_t);
template void launch_rope_forward_offset<__half>(__half *,int,int,int,float,int,int,cudaStream_t);
template void launch_rope_forward_offset<__nv_bfloat16>(__nv_bfloat16 *,
                                                        int,int,int,float,
                                                        int,int,cudaStream_t);

template<typename T>
void launch_rope_backward(T *data,
                          const int batch_heads,
                          const int seq_len,
                          const int head_dim,
                          const float base,
                          const int style,
                          cudaStream_t stream)
{
  const int total_pairs=batch_heads*seq_len*(head_dim/2);
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(
    data,seq_len,head_dim,base,style,total_pairs);
}

template void launch_rope_backward<float>(float *,int,int,int,float,int,cudaStream_t);
template void launch_rope_backward<__half>(__half *,int,int,int,float,int,cudaStream_t);
template void launch_rope_backward<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,float,int,cudaStream_t);

//------------------------------------------------------------------------------
// Partial-rotary RoPE Backward Kernel — inverse rotation on first
// `rope_dim` dims of each head row, untouched tail.
//------------------------------------------------------------------------------
template<typename T>
__global__ void rope_backward_partial_kernel(T *data,
                                             const int seq_len,
                                             const int head_dim,
                                             const int rope_dim,
                                             const float base,
                                             const int style,
                                             const int total_pairs)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total_pairs)
  {
    const int half_rope=rope_dim/2;
    const int row=idx/half_rope;
    const int pair=idx%half_rope;

    const int pos=row%seq_len;

    const float freq_exp=
      2.0f*static_cast<float>(pair)/static_cast<float>(rope_dim);
    const float theta=static_cast<float>(pos)/powf(base,freq_exp);
    const float cos_t=cosf(theta);
    const float sin_t=sinf(theta);

    const int row_base=row*head_dim;
    int idx0;
    int idx1;
    if(style==CAIF_ROPE_HALF_SPLIT)
    {
      idx0=row_base+pair;
      idx1=row_base+pair+half_rope;
    }
    else
    {
      idx0=row_base+pair*2;
      idx1=row_base+pair*2+1;
    }

    const float x0=float(data[idx0]);
    const float x1=float(data[idx1]);
    data[idx0]=T(x0*cos_t+x1*sin_t);
    data[idx1]=T(-x0*sin_t+x1*cos_t);
  }
}

template<typename T>
void launch_rope_backward_partial(T *data,
                                  const int batch_heads,
                                  const int seq_len,
                                  const int head_dim,
                                  const int rope_dim,
                                  const float base,
                                  const int style,
                                  cudaStream_t stream)
{
  const int total_pairs=batch_heads*seq_len*(rope_dim/2);
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total_pairs+block_size-1)/block_size;
  rope_backward_partial_kernel<T><<<num_blocks,block_size,0,stream>>>(
    data,seq_len,head_dim,rope_dim,base,style,total_pairs);
}

template void launch_rope_backward_partial<float>(float *,int,int,int,int,float,int,cudaStream_t);
template void launch_rope_backward_partial<__half>(__half *,int,int,int,int,float,int,cudaStream_t);
template void launch_rope_backward_partial<__nv_bfloat16>(__nv_bfloat16 *,int,int,int,int,float,int,cudaStream_t);

template<typename T>
void launch_gqa_repeat_kv(const T *input,
                          T *output,
                          const int batch,
                          const int num_kv_heads,
                          const int repeat_factor,
                          const int seq_len,
                          const int head_dim,
                          cudaStream_t stream)
{
  const int total=batch*num_kv_heads*repeat_factor*seq_len*head_dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  gqa_repeat_kv_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                              output,
                                                              num_kv_heads,
                                                              repeat_factor,
                                                              seq_len,
                                                              head_dim,
                                                              total);
}

template void launch_gqa_repeat_kv<float>(const float *,float *,int,int,int,int,int,cudaStream_t);
template void launch_gqa_repeat_kv<__half>(const __half *,__half *,int,int,int,int,int,cudaStream_t);
template void launch_gqa_repeat_kv<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  int,int,int,int,int,
                                                  cudaStream_t);

template<typename T>
void launch_gqa_reduce_kv(const T *input,
                          T *output,
                          const int batch,
                          const int num_kv_heads,
                          const int repeat_factor,
                          const int seq_len,
                          const int head_dim,
                          cudaStream_t stream)
{
  const int total=batch*num_kv_heads*seq_len*head_dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  gqa_reduce_kv_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                              output,
                                                              num_kv_heads,
                                                              repeat_factor,
                                                              seq_len,
                                                              head_dim,
                                                              total);
}

template void launch_gqa_reduce_kv<float>(const float *,float *,int,int,int,int,int,cudaStream_t);
template void launch_gqa_reduce_kv<__half>(const __half *,__half *,int,int,int,int,int,cudaStream_t);
template void launch_gqa_reduce_kv<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  __nv_bfloat16 *,
                                                  int,int,int,int,int,
                                                  cudaStream_t);

//------------------------------------------------------------------------------
// KV-Cache kernels
//------------------------------------------------------------------------------

/**
 * Kernel to append new K/V data to the cache.
 * new_kv: [batch, new_len, num_kv_heads, head_dim] - contiguous
 * cache: [batch, max_seq_len, num_kv_heads, head_dim] - contiguous
 * Copies new_kv into cache starting at position cache_pos.
 */
template<typename T>
__global__ void kv_cache_append_kernel(const T *new_kv,
                                       T *cache,
                                       const int new_len,
                                       const int cache_pos,
                                       const int max_seq_len,
                                       const int num_kv_heads,
                                       const int head_dim,
                                       const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total)
  {
    return;
  }

  const int kv_size=num_kv_heads*head_dim;
  const int batch_stride=new_len*kv_size;
  const int b=idx/batch_stride;
  const int rem=idx%batch_stride;
  const int new_pos=rem/kv_size;
  const int kv_idx=rem%kv_size;

  const int cache_batch_stride=max_seq_len*kv_size;
  const int cache_dst=b*cache_batch_stride+(cache_pos+new_pos)*kv_size+kv_idx;

  cache[cache_dst]=new_kv[idx];
}

template<typename T>
void launch_kv_cache_append(const T *new_kv,
                            T *cache,
                            const int batch,
                            const int new_len,
                            const int cache_pos,
                            const int max_seq_len,
                            const int num_kv_heads,
                            const int head_dim,
                            cudaStream_t stream)
{
  const int total=batch*new_len*num_kv_heads*head_dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  kv_cache_append_kernel<T><<<num_blocks,block_size,0,stream>>>(new_kv,
                                                                cache,
                                                                new_len,
                                                                cache_pos,
                                                                max_seq_len,
                                                                num_kv_heads,
                                                                head_dim,
                                                                total);
}

template void launch_kv_cache_append<float>(const float *,float *,int,int,int,int,int,int,cudaStream_t);
template void launch_kv_cache_append<__half>(const __half *,__half *,int,int,int,int,int,int,cudaStream_t);
template void launch_kv_cache_append<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    int,int,int,int,int,int,
                                                    cudaStream_t);

//------------------------------------------------------------------------------
// KV-Cache Append Kernel (transposed layout)
// new_kv: [batch_kv_heads, new_len, head_dim]
// cache:  [batch_kv_heads, max_seq_len, head_dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void kv_cache_append_transposed_kernel(const T *new_kv,
                                                  T *cache,
                                                  const int new_len,
                                                  const int cache_pos,
                                                  const int max_seq_len,
                                                  const int head_dim,
                                                  const int total)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total)
  {
    return;
  }

  const int row_size=new_len*head_dim;
  const int bkv=idx/row_size;
  const int rem=idx%row_size;
  const int new_pos=rem/head_dim;
  const int d=rem%head_dim;

  const int cache_row_size=max_seq_len*head_dim;
  const int cache_dst=bkv*cache_row_size+(cache_pos+new_pos)*head_dim+d;

  cache[cache_dst]=new_kv[idx];
}

template<typename T>
void launch_kv_cache_append_transposed(const T *new_kv,
                                       T *cache,
                                       const int batch_kv_heads,
                                       const int new_len,
                                       const int cache_pos,
                                       const int max_seq_len,
                                       const int head_dim,
                                       cudaStream_t stream)
{
  const int total=batch_kv_heads*new_len*head_dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  kv_cache_append_transposed_kernel<T><<<num_blocks,block_size,0,stream>>>(new_kv,
                                                                          cache,
                                                                          new_len,
                                                                          cache_pos,
                                                                          max_seq_len,
                                                                          head_dim,
                                                                          total);
}

template void launch_kv_cache_append_transposed<float>(const float *,float *,
                                                       int,int,int,int,int,
                                                       cudaStream_t);
template void launch_kv_cache_append_transposed<__half>(const __half *,
                                                        __half *,
                                                        int,int,int,int,int,
                                                        cudaStream_t);
template void launch_kv_cache_append_transposed<__nv_bfloat16>(
    const __nv_bfloat16 *,__nv_bfloat16 *,int,int,int,int,int,cudaStream_t);

//------------------------------------------------------------------------------
// Gated Activation device helpers
//------------------------------------------------------------------------------
__device__ float gated_apply_op(float x,int op)
{
  if(op==0)  // Swish: x * sigmoid(x)
  {
    return x/(1.0f+expf(-x));
  }
  if(op==1)  // GELU approximation
  {
    const float sqrt_2_over_pi=0.7978845608f;
    const float coeff=0.044715f;
    const float inner=sqrt_2_over_pi*(x+coeff*x*x*x);
    return 0.5f*x*(1.0f+tanhf(inner));
  }
  if(op==2)  // ReLU
  {
    if(x>0.0f)
    {
      return x;
    }
    return 0.0f;
  }
  if(op==3)  // Sigmoid
  {
    return 1.0f/(1.0f+expf(-x));
  }
  // op==4: Linear (identity)
  return x;
}

__device__ float gated_apply_op_derivative(float x,float activated,int op)
{
  if(op==0)  // Swish derivative: sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
  {              //                 = activated/x + activated*(1-activated/x) when x!=0
    const float sig=1.0f/(1.0f+expf(-x));
    return sig+x*sig*(1.0f-sig);
  }
  if(op==1)  // GELU derivative
  {
    const float sqrt_2_over_pi=0.7978845608f;
    const float coeff=0.044715f;
    const float inner=sqrt_2_over_pi*(x+coeff*x*x*x);
    const float tanh_val=tanhf(inner);
    const float sech2=1.0f-tanh_val*tanh_val;
    const float d_inner=sqrt_2_over_pi*(1.0f+3.0f*coeff*x*x);
    return 0.5f*(1.0f+tanh_val)+0.5f*x*sech2*d_inner;
  }
  if(op==2)  // ReLU derivative
  {
    (void)activated;
    if(x>0.0f)
    {
      return 1.0f;
    }
    return 0.0f;
  }
  if(op==3)  // Sigmoid derivative: sigmoid(x)*(1-sigmoid(x))
  {
    return activated*(1.0f-activated);
  }
  // op==4: Linear derivative
  (void)activated;
  return 1.0f;
}

//------------------------------------------------------------------------------
// Gated Activation Forward Kernel
// output[i] = apply_op(gate_input[i], op) * up_input[i]
//------------------------------------------------------------------------------
template<typename T>
__global__ void gated_activation_forward_kernel(const T *gate_input,
                                                const T *up_input,
                                                T *output,
                                                const int op,
                                                const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float g=float(gate_input[idx]);
    const float u=float(up_input[idx]);
    const float activated=gated_apply_op(g,op);
    output[idx]=T(activated*u);
  }
}

//------------------------------------------------------------------------------
// Gated Activation Backward Kernel
// grad_gate[i] = grad_output[i] * up[i] * d_activate(gate[i])
// grad_up[i]   = grad_output[i] * activate(gate[i])
//------------------------------------------------------------------------------
template<typename T>
__global__ void gated_activation_backward_kernel(const T *grad_output,
                                                 const T *cached_gate_input,
                                                 const T *cached_up_input,
                                                 T *grad_gate,
                                                 T *grad_up,
                                                 const int op,
                                                 const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float g=float(cached_gate_input[idx]);
    const float u=float(cached_up_input[idx]);
    const float go=float(grad_output[idx]);
    const float activated=gated_apply_op(g,op);
    const float d_activated=gated_apply_op_derivative(g,activated,op);
    grad_gate[idx]=T(go*u*d_activated);
    grad_up[idx]=T(go*activated);
  }
}

//------------------------------------------------------------------------------
// Gated Activation Launchers
//------------------------------------------------------------------------------
template<typename T>
void launch_gated_activation_forward(const T *gate_input,
                                     const T *up_input,
                                     T *output,
                                     const int op,
                                     const int n,
                                     cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  gated_activation_forward_kernel<T><<<num_blocks,block_size,0,stream>>>(
    gate_input,up_input,output,op,n);
}

template void launch_gated_activation_forward<float>(const float *,
                                                     const float *,
                                                     float *,
                                                     int,
                                                     int,
                                                     cudaStream_t);
template void launch_gated_activation_forward<__half>(const __half *,
                                                      const __half *,
                                                      __half *,
                                                      int,
                                                      int,
                                                      cudaStream_t);
template void launch_gated_activation_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             int,
                                                             int,
                                                             cudaStream_t);

template<typename T>
void launch_gated_activation_backward(const T *grad_output,
                                      const T *cached_gate_input,
                                      const T *cached_up_input,
                                      T *grad_gate,
                                      T *grad_up,
                                      const int op,
                                      const int n,
                                      cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  gated_activation_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(
    grad_output,cached_gate_input,cached_up_input,grad_gate,grad_up,op,n);
}

template void launch_gated_activation_backward<float>(const float *,
                                                      const float *,
                                                      const float *,
                                                      float *,
                                                      float *,
                                                      int,
                                                      int,
                                                      cudaStream_t);
template void launch_gated_activation_backward<__half>(const __half *,
                                                       const __half *,
                                                       const __half *,
                                                       __half *,
                                                       __half *,
                                                       int,
                                                       int,
                                                       cudaStream_t);
template void launch_gated_activation_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                              const __nv_bfloat16 *,
                                                              const __nv_bfloat16 *,
                                                              __nv_bfloat16 *,
                                                              __nv_bfloat16 *,
                                                              int,
                                                              int,
                                                              cudaStream_t);

//==============================================================================
// Embedding kernels
//==============================================================================

//------------------------------------------------------------------------------
// Embedding Lookup Kernel - Vectorized 2D Grid (uint32 token IDs)
// Grid: (num_tokens, ceil(dim/THREADS_PER_BLOCK/4))
// Each thread loads a float4 (4 elements) - no div/mod needed
//------------------------------------------------------------------------------
template<typename T>
__global__ void embedding_lookup_kernel(const T *__restrict__ table,
                                        const unsigned int *__restrict__ token_ids,
                                        T *__restrict__ output,
                                        const int num_tokens,
                                        const int dim)
{
  const int token_idx=blockIdx.x;
  const int d=blockIdx.y*blockDim.x+threadIdx.x;

  if(token_idx>=num_tokens || d>=dim)
  {
    return;
  }

  const unsigned int token_id=token_ids[token_idx];
  output[token_idx*dim+d]=table[token_id*dim+d];
}

//------------------------------------------------------------------------------
// Embedding Lookup Kernel - 2D Grid (float-encoded token IDs)
//------------------------------------------------------------------------------
template<typename T>
__global__ void embedding_lookup_float_kernel(const T *__restrict__ table,
                                              const float *__restrict__ float_ids,
                                              T *__restrict__ output,
                                              const int num_tokens,
                                              const int dim)
{
  const int token_idx=blockIdx.x;
  const int d=blockIdx.y*blockDim.x+threadIdx.x;

  if(token_idx>=num_tokens || d>=dim)
  {
    return;
  }

  const unsigned int token_id=static_cast<unsigned int>(float_ids[token_idx]);
  output[token_idx*dim+d]=table[token_id*dim+d];
}

//------------------------------------------------------------------------------
// Float-to-uint conversion kernel (runs on GPU, eliminates host roundtrip)
//------------------------------------------------------------------------------
__global__ void float_to_uint_kernel(const float *float_ids,
                                     unsigned int *uint_ids,
                                     const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    uint_ids[idx]=static_cast<unsigned int>(float_ids[idx]);
  }
}

//------------------------------------------------------------------------------
// Embedding Backward Kernel - Vectorized 2D Grid (scatter-add gradients)
// Uses float4 vectorized loads for better bandwidth
//------------------------------------------------------------------------------
template<typename T>
__global__ void embedding_backward_kernel(const T *grad_output,
                                          const unsigned int *token_ids,
                                          T *grad_table,
                                          const int num_tokens,
                                          const int dim)
{
  const int token_idx=blockIdx.x;
  const int d=blockIdx.y*blockDim.x+threadIdx.x;

  if(token_idx>=num_tokens || d>=dim)
  {
    return;
  }

  const unsigned int token_id=token_ids[token_idx];
  atomicAdd(&grad_table[token_id*dim+d],grad_output[token_idx*dim+d]);
}

//------------------------------------------------------------------------------
// Embedding Launchers
//------------------------------------------------------------------------------
template<typename T>
void launch_embedding_lookup(const T *table,
                             const unsigned int *token_ids,
                             T *output,
                             const int num_tokens,
                             const int dim,
                             cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int y_blocks=(dim+block_size-1)/block_size;
  dim3 grid(num_tokens,y_blocks);
  embedding_lookup_kernel<T><<<grid,block_size,0,stream>>>(
    table,token_ids,output,num_tokens,dim);
}

template void launch_embedding_lookup<float>(const float *,
                                             const unsigned int *,
                                             float *,
                                             int,
                                             int,
                                             cudaStream_t);
template void launch_embedding_lookup<__half>(const __half *,
                                              const unsigned int *,
                                              __half *,
                                              int,
                                              int,
                                              cudaStream_t);
template void launch_embedding_lookup<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     const unsigned int *,
                                                     __nv_bfloat16 *,
                                                     int,
                                                     int,
                                                     cudaStream_t);

template<typename T>
void launch_embedding_lookup_float(const T *table,
                                   const float *float_ids,
                                   T *output,
                                   const int num_tokens,
                                   const int dim,
                                   cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int y_blocks=(dim+block_size-1)/block_size;
  dim3 grid(num_tokens,y_blocks);
  embedding_lookup_float_kernel<T><<<grid,block_size,0,stream>>>(
    table,float_ids,output,num_tokens,dim);
}

template void launch_embedding_lookup_float<float>(const float *,
                                                   const float *,
                                                   float *,
                                                   int,
                                                   int,
                                                   cudaStream_t);
template void launch_embedding_lookup_float<__half>(const __half *,
                                                    const float *,
                                                    __half *,
                                                    int,
                                                    int,
                                                    cudaStream_t);
template void launch_embedding_lookup_float<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const float *,
                                                           __nv_bfloat16 *,
                                                           int,
                                                           int,
                                                           cudaStream_t);

void launch_float_to_uint(const float *float_ids,
                           unsigned int *uint_ids,
                           const int n,
                           cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  float_to_uint_kernel<<<num_blocks,block_size,0,stream>>>(
    float_ids,uint_ids,n);
}

template<typename T>
void launch_embedding_backward(const T *grad_output,
                               const unsigned int *token_ids,
                               T *grad_table,
                               const int num_tokens,
                               const int dim,
                               cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int y_blocks=(dim+block_size-1)/block_size;
  dim3 grid(num_tokens,y_blocks);
  embedding_backward_kernel<T><<<grid,block_size,0,stream>>>(
    grad_output,token_ids,grad_table,num_tokens,dim);
}

template void launch_embedding_backward<float>(const float *,
                                               const unsigned int *,
                                               float *,
                                               int,
                                               int,
                                               cudaStream_t);
template void launch_embedding_backward<__half>(const __half *,
                                                const unsigned int *,
                                                __half *,
                                                int,
                                                int,
                                                cudaStream_t);
template void launch_embedding_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                       const unsigned int *,
                                                       __nv_bfloat16 *,
                                                       int,
                                                       int,
                                                       cudaStream_t);

//==============================================================================
// Patch embedding kernels
// Uses caif_atomic_add<T>, caif_load_f<T>, caif_store_f<T> defined at top of file.
//==============================================================================

//------------------------------------------------------------------------------
// Extract Patches Kernel (im2col for non-overlapping patches)
// input:  [batch, height, width, channels] (BHWC)
// output: [batch * num_patches, patch_flat_dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void extract_patches_kernel(const T *input,
                                       T *output,
                                       const int batch,
                                       const int height,
                                       const int width,
                                       const int channels,
                                       const int patch_size,
                                       const int num_patches_h,
                                       const int num_patches_w,
                                       const int patch_flat_dim)
{
  const int total=batch*num_patches_h*num_patches_w*patch_flat_dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int num_patches=num_patches_h*num_patches_w;
    const int flat_pos=idx%patch_flat_dim;
    const int patch_idx=(idx/patch_flat_dim)%num_patches;
    const int b=idx/(patch_flat_dim*num_patches);

    const int ph=patch_idx/num_patches_w;
    const int pw=patch_idx%num_patches_w;

    const int c=flat_pos%channels;
    const int local_pixel=flat_pos/channels;
    const int local_h=local_pixel/patch_size;
    const int local_w=local_pixel%patch_size;

    const int global_h=ph*patch_size+local_h;
    const int global_w=pw*patch_size+local_w;

    const int input_idx=((b*height+global_h)*width+global_w)*channels+c;
    output[idx]=input[input_idx];
  }
}

//------------------------------------------------------------------------------
// Extract Patches Backward Kernel (col2im scatter-add)
//------------------------------------------------------------------------------
template<typename T>
__global__ void extract_patches_backward_kernel(const T *grad_patches,
                                                T *grad_input,
                                                const int batch,
                                                const int height,
                                                const int width,
                                                const int channels,
                                                const int patch_size,
                                                const int num_patches_h,
                                                const int num_patches_w,
                                                const int patch_flat_dim)
{
  const int num_patches=num_patches_h*num_patches_w;
  const int total=batch*num_patches*patch_flat_dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int flat_pos=idx%patch_flat_dim;
    const int patch_idx=(idx/patch_flat_dim)%num_patches;
    const int b=idx/(patch_flat_dim*num_patches);

    const int ph=patch_idx/num_patches_w;
    const int pw=patch_idx%num_patches_w;

    const int c=flat_pos%channels;
    const int local_pixel=flat_pos/channels;
    const int local_h=local_pixel/patch_size;
    const int local_w=local_pixel%patch_size;

    const int global_h=ph*patch_size+local_h;
    const int global_w=pw*patch_size+local_w;

    const int input_idx=((b*height+global_h)*width+global_w)*channels+c;
    caif_atomic_add<T>(&grad_input[input_idx],grad_patches[idx]);
  }
}

//------------------------------------------------------------------------------
// CLS Prepend Kernel
// Prepend CLS token at position 0, shift patches to 1..N
// patches: [batch, num_patches, dim], cls: [1, dim]
// output:  [batch, num_patches+1, dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void cls_prepend_kernel(const T *patches,
                                   const T *cls_token,
                                   T *output,
                                   const int batch,
                                   const int num_patches,
                                   const int dim)
{
  const int out_seq=num_patches+1;
  const int total=batch*out_seq*dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int d=idx%dim;
    const int s=(idx/dim)%out_seq;
    const int b=idx/(dim*out_seq);

    if(s==0)
    {
      output[idx]=cls_token[d];
    }
    else
    {
      output[idx]=patches[(b*num_patches+(s-1))*dim+d];
    }
  }
}

//------------------------------------------------------------------------------
// CLS Gradient Extract Kernel
// Split CLS gradient from patch gradients
// grad_output: [batch, num_patches+1, dim]
// grad_cls: [1, dim] (summed over batch), grad_patches: [batch, num_patches, dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void cls_grad_extract_kernel(const T *grad_output,
                                        T *grad_cls,
                                        T *grad_patches,
                                        const int batch,
                                        const int num_patches,
                                        const int dim)
{
  const int out_seq=num_patches+1;
  const int total=batch*out_seq*dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int d=idx%dim;
    const int s=(idx/dim)%out_seq;
    const int b=idx/(dim*out_seq);

    if(s==0)
    {
      caif_atomic_add<T>(&grad_cls[d],grad_output[idx]);
    }
    else
    {
      grad_patches[(b*num_patches+(s-1))*dim+d]=grad_output[idx];
    }
  }
}

//------------------------------------------------------------------------------
// Patch Embedding Launchers
//------------------------------------------------------------------------------
template<typename T>
void launch_extract_patches(const T *input,
                            T *output,
                            const int batch,
                            const int height,
                            const int width,
                            const int channels,
                            const int patch_size,
                            const int num_patches_h,
                            const int num_patches_w,
                            const int patch_flat_dim,
                            cudaStream_t stream)
{
  const int n=batch*num_patches_h*num_patches_w*patch_flat_dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  extract_patches_kernel<T><<<num_blocks,block_size,0,stream>>>(
    input,output,batch,height,width,channels,
    patch_size,num_patches_h,num_patches_w,patch_flat_dim);
}

template void launch_extract_patches<float>(const float *,
                                            float *,
                                            const int,const int,const int,const int,
                                            const int,const int,const int,const int,
                                            cudaStream_t);
template void launch_extract_patches<__half>(const __half *,
                                             __half *,
                                             const int,const int,const int,const int,
                                             const int,const int,const int,const int,
                                             cudaStream_t);
template void launch_extract_patches<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    __nv_bfloat16 *,
                                                    const int,const int,const int,const int,
                                                    const int,const int,const int,const int,
                                                    cudaStream_t);

template<typename T>
void launch_extract_patches_backward(const T *grad_patches,
                                     T *grad_input,
                                     const int batch,
                                     const int height,
                                     const int width,
                                     const int channels,
                                     const int patch_size,
                                     const int num_patches_h,
                                     const int num_patches_w,
                                     const int patch_flat_dim,
                                     cudaStream_t stream)
{
  const int n=batch*num_patches_h*num_patches_w*patch_flat_dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  extract_patches_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(
    grad_patches,grad_input,batch,height,width,channels,
    patch_size,num_patches_h,num_patches_w,patch_flat_dim);
}

template void launch_extract_patches_backward<float>(const float *,
                                                     float *,
                                                     const int,const int,const int,const int,
                                                     const int,const int,const int,const int,
                                                     cudaStream_t);
template void launch_extract_patches_backward<__half>(const __half *,
                                                      __half *,
                                                      const int,const int,const int,const int,
                                                      const int,const int,const int,const int,
                                                      cudaStream_t);
template void launch_extract_patches_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             const int,const int,const int,const int,
                                                             const int,const int,const int,const int,
                                                             cudaStream_t);

template<typename T>
void launch_cls_prepend(const T *patches,
                        const T *cls_token,
                        T *output,
                        const int batch,
                        const int num_patches,
                        const int dim,
                        cudaStream_t stream)
{
  const int out_seq=num_patches+1;
  const int n=batch*out_seq*dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  cls_prepend_kernel<T><<<num_blocks,block_size,0,stream>>>(
    patches,cls_token,output,batch,num_patches,dim);
}

template void launch_cls_prepend<float>(const float *,const float *,float *,
                                        const int,const int,const int,cudaStream_t);
template void launch_cls_prepend<__half>(const __half *,const __half *,__half *,
                                         const int,const int,const int,cudaStream_t);
template void launch_cls_prepend<__nv_bfloat16>(const __nv_bfloat16 *,
                                                const __nv_bfloat16 *,
                                                __nv_bfloat16 *,
                                                const int,const int,const int,cudaStream_t);

template<typename T>
void launch_cls_grad_extract(const T *grad_output,
                             T *grad_cls,
                             T *grad_patches,
                             const int batch,
                             const int num_patches,
                             const int dim,
                             cudaStream_t stream)
{
  const int out_seq=num_patches+1;
  const int n=batch*out_seq*dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  cls_grad_extract_kernel<T><<<num_blocks,block_size,0,stream>>>(
    grad_output,grad_cls,grad_patches,batch,num_patches,dim);
}

template void launch_cls_grad_extract<float>(const float *,float *,float *,
                                             const int,const int,const int,cudaStream_t);
template void launch_cls_grad_extract<__half>(const __half *,__half *,__half *,
                                              const int,const int,const int,cudaStream_t);
template void launch_cls_grad_extract<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     __nv_bfloat16 *,
                                                     __nv_bfloat16 *,
                                                     const int,const int,const int,cudaStream_t);

//==============================================================================
// Positional encoding kernels
//==============================================================================

//------------------------------------------------------------------------------
// Add Positional Encoding Kernel
// output[b,s,d] = input[b,s,d] + pe_table[s,d]
//------------------------------------------------------------------------------
template<typename T>
__global__ void add_positional_encoding_kernel(const T *input,
                                               const T *pe_table,
                                               T *output,
                                               const int batch,
                                               const int seq_len,
                                               const int dim)
{
  const int total=batch*seq_len*dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int d=idx%dim;
    const int s=(idx/dim)%seq_len;
    const float v=static_cast<float>(input[idx])+static_cast<float>(pe_table[s*dim+d]);
    output[idx]=static_cast<T>(v);
  }
}

//------------------------------------------------------------------------------
// PE Table Backward Kernel
// grad_table[s,d] = sum_b grad_output[b,s,d]
// One thread per (s,d) pair, loops over batch. Sum accumulates in float
// for numerical stability at fp16/bf16.
//------------------------------------------------------------------------------
template<typename T>
__global__ void pe_table_backward_kernel(const T *grad_output,
                                         T *grad_table,
                                         const int batch,
                                         const int seq_len,
                                         const int dim)
{
  const int total=seq_len*dim;
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<total)
  {
    const int s=idx/dim;
    const int d=idx%dim;
    float sum=0.0f;
    for(int b=0;b<batch;++b)
    {
      sum+=static_cast<float>(grad_output[(b*seq_len+s)*dim+d]);
    }
    grad_table[idx]=static_cast<T>(sum);
  }
}

//------------------------------------------------------------------------------
// Positional Encoding Launchers
//------------------------------------------------------------------------------
template<typename T>
void launch_add_positional_encoding(const T *input,
                                    const T *pe_table,
                                    T *output,
                                    const int batch,
                                    const int seq_len,
                                    const int dim,
                                    cudaStream_t stream)
{
  const int n=batch*seq_len*dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  add_positional_encoding_kernel<T><<<num_blocks,block_size,0,stream>>>(
    input,pe_table,output,batch,seq_len,dim);
}
template void launch_add_positional_encoding<float>(const float *,
                                                    const float *,
                                                    float *,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);
template void launch_add_positional_encoding<__half>(const __half *,
                                                     const __half *,
                                                     __half *,
                                                     int,
                                                     int,
                                                     int,
                                                     cudaStream_t);
template void launch_add_positional_encoding<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            int,
                                                            int,
                                                            int,
                                                            cudaStream_t);

template<typename T>
void launch_pe_table_backward(const T *grad_output,
                              T *grad_table,
                              const int batch,
                              const int seq_len,
                              const int dim,
                              cudaStream_t stream)
{
  const int n=seq_len*dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  pe_table_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(
    grad_output,grad_table,batch,seq_len,dim);
}
template void launch_pe_table_backward<float>(const float *,
                                              float *,
                                              int,
                                              int,
                                              int,
                                              cudaStream_t);
template void launch_pe_table_backward<__half>(const __half *,
                                               __half *,
                                               int,
                                               int,
                                               int,
                                               cudaStream_t);
template void launch_pe_table_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                      __nv_bfloat16 *,
                                                      int,
                                                      int,
                                                      int,
                                                      cudaStream_t);

//==============================================================================
// Cross-entropy loss from logits (numerically stable for language modeling)
//==============================================================================

//------------------------------------------------------------------------------
// Cross-Entropy Logits Forward Kernel
// One block per position. Each block:
// 1. Finds max logit (for stability)
// 2. Computes log-sum-exp
// 3. Returns loss = -logits[target] + log_sum_exp
//------------------------------------------------------------------------------
template<typename T>
__global__ void cross_entropy_logits_forward_kernel(const T *logits,
                                                     const float *targets,
                                                     float *losses,
                                                     const int vocab_size,
                                                     const int ignore_index)
{
  const int pos=blockIdx.x;
  const int tid=threadIdx.x;
  const int target_id=static_cast<int>(targets[pos]);

  // Check for ignore index
  if(target_id==ignore_index)
  {
    if(tid==0)
    {
      losses[pos]=0.0f;
    }
    return;
  }

  // Shared memory for reduction
  extern __shared__ float shared[];
  float *s_max=shared;
  float *s_sum=shared+blockDim.x;

  // Pointer to this position's logits
  const T *pos_logits=logits+pos*vocab_size;

  // Step 1: Find max logit (parallel reduction)
  float local_max=-1e30f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float val=float(pos_logits[v]);
    if(val>local_max)
    {
      local_max=val;
    }
  }
  s_max[tid]=local_max;
  __syncthreads();

  // Reduce to find global max
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      if(s_max[tid+stride]>s_max[tid])
      {
        s_max[tid]=s_max[tid+stride];
      }
    }
    __syncthreads();
  }
  const float max_logit=s_max[0];

  // Step 2: Compute sum of exp(logit - max)
  float local_sum=0.0f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    local_sum+=expf(float(pos_logits[v])-max_logit);
  }
  s_sum[tid]=local_sum;
  __syncthreads();

  // Reduce to find total sum
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
    }
    __syncthreads();
  }

  // Step 3: Compute loss = log_sum_exp - target_logit (always >= 0 mathematically)
  if(tid==0)
  {
    const float log_sum_exp=max_logit+logf(fmaxf(1.0f,s_sum[0]));
    const float target_logit=float(pos_logits[target_id]);
    const float raw_loss=log_sum_exp-target_logit;
    // Propagate NaN instead of masking it; clamp tiny negative fp errors to 0
    if(isnan(raw_loss))
    {
      losses[pos]=raw_loss;
    }
    else
    {
      losses[pos]=fmaxf(0.0f,raw_loss);
    }
  }
}

//------------------------------------------------------------------------------
// Cross-Entropy Logits Backward Kernel
// grad[i,j] = softmax(logits)[i,j] - (j == target[i] ? 1 : 0)
// One block per position.
//------------------------------------------------------------------------------
template<typename T>
__global__ void cross_entropy_logits_backward_kernel(const T *logits,
                                                      const float *targets,
                                                      T *grad,
                                                      const int vocab_size,
                                                      const int ignore_index,
                                                      const float scale)
{
  const int pos=blockIdx.x;
  const int tid=threadIdx.x;
  const int target_id=static_cast<int>(targets[pos]);

  // Shared memory for reduction
  extern __shared__ float shared[];
  float *s_max=shared;
  float *s_sum=shared+blockDim.x;

  // Pointer to this position's logits and grad
  const T *pos_logits=logits+pos*vocab_size;
  T *pos_grad=grad+pos*vocab_size;

  // Check for ignore index - zero gradient
  if(target_id==ignore_index)
  {
    for(int v=tid;v<vocab_size;v+=blockDim.x)
    {
      pos_grad[v]=T(0.0f);
    }
    return;
  }

  // Step 1: Find max logit (parallel reduction)
  float local_max=-1e30f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float val=float(pos_logits[v]);
    if(val>local_max)
    {
      local_max=val;
    }
  }
  s_max[tid]=local_max;
  __syncthreads();

  // Reduce to find global max
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      if(s_max[tid+stride]>s_max[tid])
      {
        s_max[tid]=s_max[tid+stride];
      }
    }
    __syncthreads();
  }
  const float max_logit=s_max[0];

  // Step 2: Compute sum of exp(logit - max)
  float local_sum=0.0f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    local_sum+=expf(float(pos_logits[v])-max_logit);
  }
  s_sum[tid]=local_sum;
  __syncthreads();

  // Reduce to find total sum
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
    }
    __syncthreads();
  }
  const float sum_exp=s_sum[0];

  // Step 3: Compute gradient: softmax - one_hot
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float softmax_val=expf(float(pos_logits[v])-max_logit)/sum_exp;
    float g=softmax_val;
    if(v==target_id)
    {
      g-=1.0f;
    }
    pos_grad[v]=T(g*scale);
  }
}

//------------------------------------------------------------------------------
// Cross-Entropy Reduce Mean Kernel
// Sums losses and divides by valid count (excluding ignore_index)
//------------------------------------------------------------------------------
__global__ void cross_entropy_reduce_mean_kernel(const float *losses,
                                                  const float *targets,
                                                  float *output,
                                                  const int n,
                                                  const int ignore_index)
{
  extern __shared__ float shared[];
  float *s_sum=shared;
  float *s_count=shared+blockDim.x;

  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+tid;

  float local_sum=0.0f;
  float local_count=0.0f;

  if(idx<n)
  {
    const int target_id=static_cast<int>(targets[idx]);
    if(target_id!=ignore_index)
    {
      local_sum=losses[idx];
      local_count=1.0f;
    }
  }

  s_sum[tid]=local_sum;
  s_count[tid]=local_count;
  __syncthreads();

  // Reduce within block
  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
      s_count[tid]+=s_count[tid+stride];
    }
    __syncthreads();
  }

  // Atomic add to global output
  if(tid==0)
  {
    atomicAdd(&output[0],s_sum[0]);
    atomicAdd(&output[1],s_count[0]);
  }
}

//------------------------------------------------------------------------------
// Cross-Entropy Logits Launchers
//------------------------------------------------------------------------------
template<typename T>
void launch_cross_entropy_logits_forward(const T *logits,
                                         const float *targets,
                                         float *losses,
                                         const int n,
                                         const int vocab_size,
                                         const int ignore_index,
                                         cudaStream_t stream)
{
  // One block per position, use enough threads to cover vocab
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=n;
  const size_t shared_size=2*block_size*sizeof(float);
  cross_entropy_logits_forward_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(
    logits,targets,losses,vocab_size,ignore_index);
}

template void launch_cross_entropy_logits_forward<float>(const float *,
                                                         const float *,
                                                         float *,
                                                         int,
                                                         int,
                                                         int,
                                                         cudaStream_t);
template void launch_cross_entropy_logits_forward<__half>(const __half *,
                                                          const float *,
                                                          float *,
                                                          int,
                                                          int,
                                                          int,
                                                          cudaStream_t);
template void launch_cross_entropy_logits_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                 const float *,
                                                                 float *,
                                                                 int,
                                                                 int,
                                                                 int,
                                                                 cudaStream_t);

template<typename T>
void launch_cross_entropy_logits_backward(const T *logits,
                                          const float *targets,
                                          T *grad,
                                          const int n,
                                          const int vocab_size,
                                          const int ignore_index,
                                          const float scale,
                                          cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=n;
  const size_t shared_size=2*block_size*sizeof(float);
  cross_entropy_logits_backward_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(
    logits,targets,grad,vocab_size,ignore_index,scale);
}

template void launch_cross_entropy_logits_backward<float>(const float *,
                                                          const float *,
                                                          float *,
                                                          int,
                                                          int,
                                                          int,
                                                          float,
                                                          cudaStream_t);
template void launch_cross_entropy_logits_backward<__half>(const __half *,
                                                           const float *,
                                                           __half *,
                                                           int,
                                                           int,
                                                           int,
                                                           float,
                                                           cudaStream_t);
template void launch_cross_entropy_logits_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  const float *,
                                                                  __nv_bfloat16 *,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  float,
                                                                  cudaStream_t);

void launch_cross_entropy_reduce_mean(const float *losses,
                                      const float *targets,
                                      float *output,
                                      const int n,
                                      const int ignore_index,
                                      cudaStream_t stream)
{
  // Output should be pre-zeroed (stores sum and count)
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  const size_t shared_size=2*block_size*sizeof(float);
  cross_entropy_reduce_mean_kernel<<<num_blocks,block_size,shared_size,stream>>>(
    losses,targets,output,n,ignore_index);
}

//------------------------------------------------------------------------------
// Cross-Entropy Fused Kernels
// Fuses forward loss + backward gradient into a single pass over logits.
// Eliminates host roundtrip for valid_count and halves logits memory reads.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Count valid (non-ignored) positions. Writes count to *count_out (pre-zeroed).
//------------------------------------------------------------------------------
__global__ void cross_entropy_count_valid_kernel(const float *targets,
                                                  float *count_out,
                                                  const int n,
                                                  const int ignore_index)
{
  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+tid;

  float local_count=0.0f;
  if(idx<n)
  {
    const int target_id=static_cast<int>(targets[idx]);
    if(target_id!=ignore_index)
    {
      local_count=1.0f;
    }
  }

  local_count=warp_reduce_sum(local_count);

  if((tid%g_caif_cuda_warp_size)==0 && local_count>0.0f)
  {
    atomicAdd(count_out,local_count);
  }
}

//------------------------------------------------------------------------------
// Fused forward+backward: one block per position.
// Reads logits once (3 passes: max, sum_exp, gradient).
// Reads valid_count from device memory (set by count kernel on same stream).
// Outputs per-position loss AND scaled gradient.
//------------------------------------------------------------------------------
template<typename T>
__global__ void cross_entropy_fused_loss_grad_kernel(const T *logits,
                                                      const float *targets,
                                                      float *losses,
                                                      T *grad,
                                                      const float *valid_count,
                                                      const int vocab_size,
                                                      const int ignore_index)
{
  const int pos=blockIdx.x;
  const int tid=threadIdx.x;
  const int target_id=static_cast<int>(targets[pos]);

  const T *pos_logits=logits+pos*vocab_size;
  T *pos_grad=grad+pos*vocab_size;

  // Ignored position: zero loss and gradient
  if(target_id==ignore_index)
  {
    if(tid==0)
    {
      losses[pos]=0.0f;
    }
    for(int v=tid;v<vocab_size;v+=blockDim.x)
    {
      pos_grad[v]=T(0.0f);
    }
    return;
  }

  // Read valid count from device memory and compute gradient scale
  const float count=*valid_count;
  float scale=0.0f;
  if(count>=1.0f)
  {
    scale=1.0f/count;
  }

  // Pass 1: Find max logit (warp shuffle reduction)
  float local_max=-1e30f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    const float val=float(pos_logits[v]);
    if(val>local_max)
    {
      local_max=val;
    }
  }
  local_max=warp_reduce_max(local_max);
  __shared__ float ws_max[g_caif_cuda_warp_size];
  const int warp_id=tid/g_caif_cuda_warp_size;
  const int lane_id=tid%g_caif_cuda_warp_size;
  if(lane_id==0)
  {
    ws_max[warp_id]=local_max;
  }
  __syncthreads();
  if(tid<g_caif_cuda_warp_size)
  {
    const int num_warps=blockDim.x/g_caif_cuda_warp_size;
    float v=-1e30f;
    if(tid<num_warps)
    {
      v=ws_max[tid];
    }
    v=warp_reduce_max(v);
    ws_max[0]=v;
  }
  __syncthreads();
  const float max_logit=ws_max[0];

  // Pass 2: Compute sum of exp(logit - max) (warp shuffle reduction)
  float local_sum=0.0f;
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    local_sum+=expf(float(pos_logits[v])-max_logit);
  }
  local_sum=warp_reduce_sum(local_sum);
  __shared__ float ws_sum[g_caif_cuda_warp_size];
  if(lane_id==0)
  {
    ws_sum[warp_id]=local_sum;
  }
  __syncthreads();
  if(tid<g_caif_cuda_warp_size)
  {
    const int num_warps=blockDim.x/g_caif_cuda_warp_size;
    float v=0.0f;
    if(tid<num_warps)
    {
      v=ws_sum[tid];
    }
    v=warp_reduce_sum(v);
    ws_sum[0]=v;
  }
  __syncthreads();
  const float sum_exp=ws_sum[0];

  // Write per-position loss (thread 0 only)
  if(tid==0)
  {
    const float log_sum_exp=max_logit+logf(fmaxf(1.0f,sum_exp));
    const float target_logit=float(pos_logits[target_id]);
    const float raw_loss=log_sum_exp-target_logit;
    if(isnan(raw_loss)==true)
    {
      losses[pos]=raw_loss;
    }
    else
    {
      losses[pos]=fmaxf(0.0f,raw_loss);
    }
  }

  // Pass 3: Compute scaled gradient = (softmax - one_hot) * scale
  for(int v=tid;v<vocab_size;v+=blockDim.x)
  {
    float g=expf(float(pos_logits[v])-max_logit)/sum_exp;
    if(v==target_id)
    {
      g-=1.0f;
    }
    pos_grad[v]=T(g*scale);
  }
}

//------------------------------------------------------------------------------
// Sum per-position losses (excluding ignored). Writes to *output (pre-zeroed).
//------------------------------------------------------------------------------
__global__ void cross_entropy_sum_losses_kernel(const float *losses,
                                                 const float *targets,
                                                 float *output,
                                                 const int n,
                                                 const int ignore_index)
{
  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+tid;

  float local_sum=0.0f;
  if(idx<n)
  {
    const int target_id=static_cast<int>(targets[idx]);
    if(target_id!=ignore_index)
    {
      local_sum=losses[idx];
    }
  }

  local_sum=warp_reduce_sum(local_sum);

  if((tid%g_caif_cuda_warp_size)==0 && local_sum!=0.0f)
  {
    atomicAdd(output,local_sum);
  }
}

//------------------------------------------------------------------------------
// Fused cross-entropy launcher: count + fused loss/grad + sum losses.
// All 3 kernels on same stream, no host sync between them.
// result[0] = valid_count, result[1] = loss_sum (must be pre-zeroed).
//------------------------------------------------------------------------------
template<typename T>
void launch_cross_entropy_fused(const T *logits,
                                const float *targets,
                                float *losses,
                                T *grad,
                                float *result,
                                const int n,
                                const int vocab_size,
                                const int ignore_index,
                                cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;

  // Kernel 1: count valid positions → result[0]
  const int count_blocks=(n+block_size-1)/block_size;
  cross_entropy_count_valid_kernel<<<count_blocks,block_size,0,stream>>>(
    targets,&result[0],n,ignore_index);

  // Kernel 2: fused forward+backward (reads result[0] for scale)
  cross_entropy_fused_loss_grad_kernel<T><<<n,block_size,0,stream>>>(
    logits,targets,losses,grad,&result[0],vocab_size,ignore_index);

  // Kernel 3: sum per-position losses → result[1]
  const int sum_blocks=(n+block_size-1)/block_size;
  cross_entropy_sum_losses_kernel<<<sum_blocks,block_size,0,stream>>>(
    losses,targets,&result[1],n,ignore_index);
}

template void launch_cross_entropy_fused<float>(const float *,
                                                const float *,
                                                float *,
                                                float *,
                                                float *,
                                                int,
                                                int,
                                                int,
                                                cudaStream_t);
template void launch_cross_entropy_fused<__half>(const __half *,
                                                 const float *,
                                                 float *,
                                                 __half *,
                                                 float *,
                                                 int,
                                                 int,
                                                 int,
                                                 cudaStream_t);
template void launch_cross_entropy_fused<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        const float *,
                                                        float *,
                                                        __nv_bfloat16 *,
                                                        float *,
                                                        int,
                                                        int,
                                                        int,
                                                        cudaStream_t);

// SiLU backward kernels removed 2026-05-02: superseded by templated
// `launch_swish_backward<T>` (see line ~1235). The fp32-only
// `launch_silu_backward` had zero callers in src/tests/benchmarks.

//------------------------------------------------------------------------------
// Sum Axis 0 Kernel (sum over batch)
// input: [batch, dim], output: [dim]
//------------------------------------------------------------------------------
template<typename T>
__global__ void sum_axis0_kernel(const T *input,
                                 float *output,
                                 const int batch,
                                 const int dim)
{
  const int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(d<dim)
  {
    float sum=0.0f;
    for(int b=0;b<batch;++b)
    {
      sum+=float(input[b*dim+d]);
    }
    output[d]=sum;
  }
}

template<typename T>
void launch_sum_axis0(const T *input,
                      float *output,
                      const int batch,
                      const int dim,
                      cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(dim+block_size-1)/block_size;
  sum_axis0_kernel<T><<<num_blocks,block_size,0,stream>>>(input,output,batch,dim);
}
template void launch_sum_axis0<float>(const float *,
                                      float *,
                                      int,
                                      int,
                                      cudaStream_t);
template void launch_sum_axis0<__half>(const __half *,
                                       float *,
                                       int,
                                       int,
                                       cudaStream_t);
template void launch_sum_axis0<__nv_bfloat16>(const __nv_bfloat16 *,
                                              float *,
                                              int,
                                              int,
                                              cudaStream_t);

//------------------------------------------------------------------------------
// Sum Axis 1 Kernel (sum over dim)
// input: [batch, dim], output: [batch]
//------------------------------------------------------------------------------
template<typename T>
__global__ void sum_axis1_kernel(const T *input,
                                 float *output,
                                 const int batch,
                                 const int dim)
{
  const int b=blockIdx.x*blockDim.x+threadIdx.x;
  if(b<batch)
  {
    float sum=0.0f;
    for(int d=0;d<dim;++d)
    {
      sum+=float(input[b*dim+d]);
    }
    output[b]=sum;
  }
}

template<typename T>
void launch_sum_axis1(const T *input,
                      float *output,
                      const int batch,
                      const int dim,
                      cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(batch+block_size-1)/block_size;
  sum_axis1_kernel<T><<<num_blocks,block_size,0,stream>>>(input,output,batch,dim);
}
template void launch_sum_axis1<float>(const float *,
                                      float *,
                                      int,
                                      int,
                                      cudaStream_t);
template void launch_sum_axis1<__half>(const __half *,
                                       float *,
                                       int,
                                       int,
                                       cudaStream_t);
template void launch_sum_axis1<__nv_bfloat16>(const __nv_bfloat16 *,
                                              float *,
                                              int,
                                              int,
                                              cudaStream_t);

//------------------------------------------------------------------------------
// Sum of Squares Kernel
// Computes sum(x[i]^2) over all n elements.
// Uses block-level reduction with atomicAdd to a single output float.
// Caller must zero the output before launch.
//------------------------------------------------------------------------------
template<typename T>
__global__ void sum_of_squares_kernel(const T *input,
                                       float *output,
                                       const int n)
{
  extern __shared__ float shared[];
  const int tid=threadIdx.x;
  const int idx=blockIdx.x*blockDim.x+tid;

  float local_sum=0.0f;
  if(idx<n)
  {
    const float val=float(input[idx]);
    if(isnan(val)==false && isinf(val)==false)
    {
      local_sum=val*val;
    }
  }
  shared[tid]=local_sum;
  __syncthreads();

  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      shared[tid]+=shared[tid+stride];
    }
    __syncthreads();
  }

  if(tid==0)
  {
    atomicAdd(output,shared[0]);
  }
}

template<typename T>
void launch_sum_of_squares(const T *input,
                           float *output,
                           const int n,
                           cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(n+block_size-1)/block_size;
  sum_of_squares_kernel<T><<<num_blocks,block_size,
    block_size*sizeof(float),stream>>>(input,output,n);
}
template void launch_sum_of_squares<float>(const float *,
                                           float *,
                                           int,
                                           cudaStream_t);
template void launch_sum_of_squares<__half>(const __half *,
                                            float *,
                                            int,
                                            cudaStream_t);
template void launch_sum_of_squares<__nv_bfloat16>(const __nv_bfloat16 *,
                                                   float *,
                                                   int,
                                                   cudaStream_t);

//------------------------------------------------------------------------------
// Log-Sum-Exp Kernel
// input: [batch, dim], output: [batch]
// output[b] = log(sum_d(exp(input[b,d])))
//------------------------------------------------------------------------------
template<typename T>
__global__ void logsumexp_kernel(const T *input,
                                 float *output,
                                 const int batch,
                                 const int dim)
{
  const int b=blockIdx.x;
  if(b>=batch)
  {
    return;
  }

  extern __shared__ float shared[];
  float *s_max=shared;
  float *s_sum=shared+blockDim.x;
  const int tid=threadIdx.x;
  const T *row=input+b*dim;

  // Find max
  float local_max=-1e30f;
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    local_max=fmaxf(local_max,float(row[d]));
  }
  s_max[tid]=local_max;
  __syncthreads();

  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_max[tid]=fmaxf(s_max[tid],s_max[tid+stride]);
    }
    __syncthreads();
  }
  const float max_val=s_max[0];

  // Sum exp(x - max)
  float local_sum=0.0f;
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    local_sum+=expf(float(row[d])-max_val);
  }
  s_sum[tid]=local_sum;
  __syncthreads();

  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
    }
    __syncthreads();
  }

  if(tid==0)
  {
    output[b]=max_val+logf(s_sum[0]);
  }
}

template<typename T>
void launch_logsumexp(const T *input,
                      float *output,
                      const int batch,
                      const int dim,
                      cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=batch;
  const size_t shared_size=2*block_size*sizeof(float);
  logsumexp_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(input,output,
                                                                      batch,dim);
}
template void launch_logsumexp<float>(const float *,
                                      float *,
                                      int,
                                      int,
                                      cudaStream_t);
template void launch_logsumexp<__half>(const __half *,
                                       float *,
                                       int,
                                       int,
                                       cudaStream_t);
template void launch_logsumexp<__nv_bfloat16>(const __nv_bfloat16 *,
                                              float *,
                                              int,
                                              int,
                                              cudaStream_t);

//------------------------------------------------------------------------------
// Normalize Rows Kernel
// input: [batch, dim], output: [batch, dim]
// output[b,d] = input[b,d] / sum_d(input[b,:])
//------------------------------------------------------------------------------
template<typename T>
__global__ void normalize_rows_kernel(const T *input,
                                      T *output,
                                      const int batch,
                                      const int dim)
{
  const int b=blockIdx.x;
  if(b>=batch)
  {
    return;
  }

  extern __shared__ float s_sum[];
  const int tid=threadIdx.x;
  const T *row_in=input+b*dim;
  T *row_out=output+b*dim;

  // Sum the row
  float local_sum=0.0f;
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    local_sum+=float(row_in[d]);
  }
  s_sum[tid]=local_sum;
  __syncthreads();

  for(int stride=blockDim.x/2;stride>0;stride>>=1)
  {
    if(tid<stride)
    {
      s_sum[tid]+=s_sum[tid+stride];
    }
    __syncthreads();
  }
  const float sum=fmaxf(s_sum[0],1e-10f);
  const float inv_sum=1.0f/sum;

  // Normalize
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    row_out[d]=T(float(row_in[d])*inv_sum);
  }
}

template<typename T>
void launch_normalize_rows(const T *input,
                           T *output,
                           const int batch,
                           const int dim,
                           cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=batch;
  const size_t shared_size=block_size*sizeof(float);
  normalize_rows_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(input,output,
                                                                           batch,dim);
}
template void launch_normalize_rows<float>(const float *,
                                           float *,
                                           int,
                                           int,
                                           cudaStream_t);
template void launch_normalize_rows<__half>(const __half *,
                                            __half *,
                                            int,
                                            int,
                                            cudaStream_t);
template void launch_normalize_rows<__nv_bfloat16>(const __nv_bfloat16 *,
                                                   __nv_bfloat16 *,
                                                   int,
                                                   int,
                                                   cudaStream_t);

//------------------------------------------------------------------------------
// NormalizeRows Backward Jacobian, gather variant (top-k rows)
//   Given full softmax probs [T,E] and topk indices [T,K] (stored float in CAIF):
//     p[k]  = probs[t, indices[t,k]]
//     s     = sum_k p[k]
//     w[k]  = p[k]/s
//     dot   = sum_k w[k]*grad_w[k]
//     grad_p_topk[t,k] = (grad_w[t,k] - dot)/s
// This avoids caching `w_norm` and `row_sum` in forward — all Jacobian work
// runs here from the probs/indices caches that already exist for the softmax
// backward. One warp per token, top_k <= 32.
//------------------------------------------------------------------------------
template<typename T>
__global__ void normalize_rows_backward_topk_gather_kernel(const T *grad_w,
                                                           const T *probs,
                                                           const int *indices,
                                                           T *grad_p_topk,
                                                           const int num_tokens,
                                                           const int num_experts,
                                                           const int top_k)
{
  const int t=blockIdx.x;
  if(t>=num_tokens)
  {
    return;
  }
  const int tid=threadIdx.x;

  float p_k=0.0f;
  float grad_w_k=0.0f;
  if(tid<top_k)
  {
    const int expert_idx=indices[t*top_k+tid];
    p_k=float(probs[t*num_experts+expert_idx]);
    grad_w_k=float(grad_w[t*top_k+tid]);
  }

  float s=p_k;
  for(int offset=16;offset>0;offset>>=1)
  {
    s+=__shfl_xor_sync(0xffffffff,s,offset);
  }
  s=fmaxf(s,1e-12f);
  const float inv_s=1.0f/s;
  const float w_k=p_k*inv_s;

  float dot=w_k*grad_w_k;
  for(int offset=16;offset>0;offset>>=1)
  {
    dot+=__shfl_xor_sync(0xffffffff,dot,offset);
  }

  if(tid<top_k)
  {
    grad_p_topk[t*top_k+tid]=T((grad_w_k-dot)*inv_s);
  }
}

template<typename T>
void launch_normalize_rows_backward_topk_gather(const T *grad_w,
                                                const T *probs,
                                                const int *indices,
                                                T *grad_p_topk,
                                                const int num_tokens,
                                                const int num_experts,
                                                const int top_k,
                                                cudaStream_t stream)
{
  normalize_rows_backward_topk_gather_kernel<T><<<num_tokens,32,0,stream>>>(grad_w,
                                                                            probs,
                                                                            indices,
                                                                            grad_p_topk,
                                                                            num_tokens,
                                                                            num_experts,
                                                                            top_k);
}
template void launch_normalize_rows_backward_topk_gather<float>(const float *,
                                                                const float *,
                                                                const int *,
                                                                float *,
                                                                int,
                                                                int,
                                                                int,
                                                                cudaStream_t);
template void launch_normalize_rows_backward_topk_gather<__half>(const __half *,
                                                                 const __half *,
                                                                 const int *,
                                                                 __half *,
                                                                 int,
                                                                 int,
                                                                 int,
                                                                 cudaStream_t);
template void launch_normalize_rows_backward_topk_gather<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                        const __nv_bfloat16 *,
                                                                        const int *,
                                                                        __nv_bfloat16 *,
                                                                        int,
                                                                        int,
                                                                        int,
                                                                        cudaStream_t);

//------------------------------------------------------------------------------
// GatherTopKValues — out[t,k] = scores[t, indices[t,k]]
//   Phase 1b of the SigmoidNoauxTc gating path: selection happens on
//   bias-corrected sigmoid scores host-side via AddBias + TopK, but the
//   combine weights must be the ORIGINAL (uncorrected) sigmoid scores at
//   the chosen indices to match HF DeepSeek-V2 / GLM-4-MoE
//   `topk_method=noaux_tc` (transformers/.../modeling_glm4_moe.py
//   `router_logits.gather(1, topk_indices)`).
//
//   One block per token, top_k threads (top_k <= 32 — pre-checked
//   host-side).  Each thread reads one (token, k) gather and writes
//   one element.  No reductions needed.
//------------------------------------------------------------------------------
template<typename T>
__global__ void gather_topk_values_kernel(const T *scores,
                                           const int *indices,
                                           T *out,
                                           const int num_tokens,
                                           const int num_experts,
                                           const int top_k)
{
  const int t=blockIdx.x;
  if(t>=num_tokens)
  {
    return;
  }
  const int tid=threadIdx.x;
  if(tid<top_k)
  {
    const int expert_idx=indices[t*top_k+tid];
    out[t*top_k+tid]=scores[t*num_experts+expert_idx];
  }
}

template<typename T>
void launch_gather_topk_values(const T *scores,
                                const int *indices,
                                T *out,
                                const int num_tokens,
                                const int num_experts,
                                const int top_k,
                                cudaStream_t stream)
{
  gather_topk_values_kernel<T><<<num_tokens,32,0,stream>>>(scores,
                                                            indices,
                                                            out,
                                                            num_tokens,
                                                            num_experts,
                                                            top_k);
}
template void launch_gather_topk_values<float>(const float *,
                                                const int *,
                                                float *,
                                                int,
                                                int,
                                                int,
                                                cudaStream_t);
template void launch_gather_topk_values<__half>(const __half *,
                                                 const int *,
                                                 __half *,
                                                 int,
                                                 int,
                                                 int,
                                                 cudaStream_t);
template void launch_gather_topk_values<__nv_bfloat16>(const __nv_bfloat16 *,
                                                        const int *,
                                                        __nv_bfloat16 *,
                                                        int,
                                                        int,
                                                        int,
                                                        cudaStream_t);

//------------------------------------------------------------------------------
// Top-K Kernel (simple implementation)
// input: [batch, dim], indices: [batch, k], values: [batch, k]
//------------------------------------------------------------------------------
template<typename T>
__global__ void topk_kernel(const T *input,
                            int *indices,
                            T *values,
                            const int batch,
                            const int dim,
                            const int k)
{
  const int b=blockIdx.x;
  if(b>=batch)
  {
    return;
  }

  const T *row=input+b*dim;
  int *out_idx=indices+b*k;
  T *out_val=values+b*k;

  // Simple selection sort for top-k (good enough for small k)
  // Mark selected indices with -inf. Compare in fp32 in shared memory so the
  // sentinel works for fp16 storage (fp16 cannot represent -1e30).
  extern __shared__ float s_data[];
  float *temp=s_data;
  const int tid=threadIdx.x;

  // Copy to shared memory (convert T -> float)
  for(int d=tid;d<dim;d+=blockDim.x)
  {
    temp[d]=float(row[d]);
  }
  __syncthreads();

  // Only thread 0 does the selection (simple but correct)
  if(tid==0)
  {
    for(int i=0;i<k;++i)
    {
      float max_val=-1e30f;
      int max_idx=0;
      for(int d=0;d<dim;++d)
      {
        if(temp[d]>max_val)
        {
          max_val=temp[d];
          max_idx=d;
        }
      }
      out_idx[i]=max_idx;
      out_val[i]=T(max_val);
      temp[max_idx]=-1e30f;  // Mark as selected
    }
  }
}

template<typename T>
void launch_topk(const T *input,
                 int *indices,
                 T *values,
                 const int batch,
                 const int dim,
                 const int k,
                 cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=batch;
  const size_t shared_size=dim*sizeof(float);
  topk_kernel<T><<<num_blocks,block_size,shared_size,stream>>>(input,indices,values,
                                                                 batch,dim,k);
}
template void launch_topk<float>(const float *,
                                 int *,
                                 float *,
                                 int,
                                 int,
                                 int,
                                 cudaStream_t);
template void launch_topk<__half>(const __half *,
                                  int *,
                                  __half *,
                                  int,
                                  int,
                                  int,
                                  cudaStream_t);
template void launch_topk<__nv_bfloat16>(const __nv_bfloat16 *,
                                         int *,
                                         __nv_bfloat16 *,
                                         int,
                                         int,
                                         int,
                                         cudaStream_t);

//------------------------------------------------------------------------------
// Scatter Add Kernel
// output[b, indices[b,k]] += values[b,k]
// values: [batch, k], indices: [batch, k], output: [batch, dim]
// Uses caif_atomic_add<T> defined above (see Patch embedding kernels section).
//------------------------------------------------------------------------------
template<typename T>
__global__ void scatter_add_kernel(const T *values,
                                   const int *indices,
                                   T *output,
                                   const int batch,
                                   const int k,
                                   const int dim)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=batch*k;
  if(idx<total)
  {
    const int b=idx/k;
    const int target_idx=indices[idx];
    if(target_idx>=0 && target_idx<dim)
    {
      caif_atomic_add<T>(&output[b*dim+target_idx],values[idx]);
    }
  }
}

template<typename T>
void launch_scatter_add(const T *values,
                        const int *indices,
                        T *output,
                        const int batch,
                        const int k,
                        const int dim,
                        cudaStream_t stream)
{
  const int total=batch*k;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  scatter_add_kernel<T><<<num_blocks,block_size,0,stream>>>(values,indices,output,
                                                              batch,k,dim);
}
template void launch_scatter_add<float>(const float *,
                                        const int *,
                                        float *,
                                        int,
                                        int,
                                        int,
                                        cudaStream_t);
template void launch_scatter_add<__half>(const __half *,
                                         const int *,
                                         __half *,
                                         int,
                                         int,
                                         int,
                                         cudaStream_t);
template void launch_scatter_add<__nv_bfloat16>(const __nv_bfloat16 *,
                                                const int *,
                                                __nv_bfloat16 *,
                                                int,
                                                int,
                                                int,
                                                cudaStream_t);

//------------------------------------------------------------------------------
// MoE Dispatch Kernel
// Gathers tokens to expert-specific buffers based on routing indices
// expert_indices: [num_tokens, top_k] - which experts each token routes to
// expert_offsets: [num_experts+1] - cumulative token counts per expert
// dispatch_map: [num_tokens, top_k] - position within expert buffer for each assignment
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_dispatch_kernel(const T *input,
                                    const int *expert_indices,
                                    const int *dispatch_map,
                                    T *expert_buffer,
                                    const int *expert_offsets,
                                    const int num_tokens,
                                    const int dim,
                                    const int top_k)
{
  // Each thread handles one dimension of one token-expert assignment
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int total_assignments=num_tokens*top_k*dim;

  if(tid<total_assignments)
  {
    const int d=tid%dim;
    const int assignment_idx=(tid/dim);
    const int token_idx=assignment_idx/top_k;
    const int k_idx=assignment_idx%top_k;

    const int expert_idx=expert_indices[token_idx*top_k+k_idx];
    const int pos_in_expert=dispatch_map[token_idx*top_k+k_idx];

    if(expert_idx>=0 && pos_in_expert>=0)
    {
      const int expert_start=expert_offsets[expert_idx];
      const int dest_idx=(expert_start+pos_in_expert)*dim+d;
      expert_buffer[dest_idx]=input[token_idx*dim+d];
    }
  }
}

template<typename T>
void launch_moe_dispatch(const T *input,
                         const int *expert_indices,
                         const int *dispatch_map,
                         T *expert_buffer,
                         const int *expert_offsets,
                         const int num_tokens,
                         const int dim,
                         const int top_k,
                         cudaStream_t stream)
{
  const int total=num_tokens*top_k*dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  moe_dispatch_kernel<T><<<num_blocks,block_size,0,stream>>>(input,
                                                              expert_indices,
                                                              dispatch_map,
                                                              expert_buffer,
                                                              expert_offsets,
                                                              num_tokens,
                                                              dim,
                                                              top_k);
}

template void launch_moe_dispatch<float>(const float *,
                                          const int *,
                                          const int *,
                                          float *,
                                          const int *,
                                          const int,
                                          const int,
                                          const int,
                                          cudaStream_t);
template void launch_moe_dispatch<__half>(const __half *,
                                           const int *,
                                           const int *,
                                           __half *,
                                           const int *,
                                           const int,
                                           const int,
                                           const int,
                                           cudaStream_t);
template void launch_moe_dispatch<__nv_bfloat16>(const __nv_bfloat16 *,
                                                  const int *,
                                                  const int *,
                                                  __nv_bfloat16 *,
                                                  const int *,
                                                  const int,
                                                  const int,
                                                  const int,
                                                  cudaStream_t);

//------------------------------------------------------------------------------
// MoE Combine Kernel
// Scatters expert outputs back to token positions with routing weights
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_combine_kernel(const T *expert_buffer,
                                   const int *expert_indices,
                                   const T *expert_weights,
                                   const int *dispatch_map,
                                   const int *expert_offsets,
                                   T *output,
                                   const int num_tokens,
                                   const int dim,
                                   const int top_k)
{
  // Each thread handles one dimension of one token
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=num_tokens*dim;

  if(tid<total)
  {
    const int token_idx=tid/dim;
    const int d=tid%dim;

    // fp32 accumulator regardless of T — reduces precision loss when
    // top_k expert outputs are summed at fp16 / bf16.
    float sum=0.0f;

    for(int k=0;k<top_k;++k)
    {
      const int expert_idx=expert_indices[token_idx*top_k+k];
      const float weight=static_cast<float>(expert_weights[token_idx*top_k+k]);
      const int pos_in_expert=dispatch_map[token_idx*top_k+k];

      if(expert_idx>=0 && pos_in_expert>=0)
      {
        const int expert_start=expert_offsets[expert_idx];
        const int src_idx=(expert_start+pos_in_expert)*dim+d;
        sum+=weight*static_cast<float>(expert_buffer[src_idx]);
      }
    }

    output[tid]=static_cast<T>(sum);
  }
}

template<typename T>
void launch_moe_combine(const T *expert_buffer,
                        const int *expert_indices,
                        const T *expert_weights,
                        const int *dispatch_map,
                        const int *expert_offsets,
                        T *output,
                        const int num_tokens,
                        const int dim,
                        const int top_k,
                        cudaStream_t stream)
{
  const int total=num_tokens*dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  moe_combine_kernel<T><<<num_blocks,block_size,0,stream>>>(expert_buffer,
                                                             expert_indices,
                                                             expert_weights,
                                                             dispatch_map,
                                                             expert_offsets,
                                                             output,
                                                             num_tokens,
                                                             dim,
                                                             top_k);
}

template void launch_moe_combine<float>(const float *,
                                         const int *,
                                         const float *,
                                         const int *,
                                         const int *,
                                         float *,
                                         const int,
                                         const int,
                                         const int,
                                         cudaStream_t);
template void launch_moe_combine<__half>(const __half *,
                                          const int *,
                                          const __half *,
                                          const int *,
                                          const int *,
                                          __half *,
                                          const int,
                                          const int,
                                          const int,
                                          cudaStream_t);
template void launch_moe_combine<__nv_bfloat16>(const __nv_bfloat16 *,
                                                 const int *,
                                                 const __nv_bfloat16 *,
                                                 const int *,
                                                 const int *,
                                                 __nv_bfloat16 *,
                                                 const int,
                                                 const int,
                                                 const int,
                                                 cudaStream_t);

//------------------------------------------------------------------------------
// MoE Combine Backward — grad_expert_buffer path
//
// Forward (for reference):
//   output[t,d] = sum_k W[t,k] * B[(off[e(t,k)] + pos(t,k))*dim + d]
//
// Backward (this kernel) parallelizes over (t, k, d) and writes exactly
// one slot of grad_expert_buffer per (t,k,d). No atomics needed because
// each slot is touched by exactly one (t,k).
//
//   grad_B[(off[e]+pos)*dim + d] = W[t,k] * grad_output[t,d]
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_combine_backward_grad_expert_kernel(const T *grad_output,
                                                        const int *expert_indices,
                                                        const T *expert_weights,
                                                        const int *dispatch_map,
                                                        const int *expert_offsets,
                                                        T *grad_expert_buffer,
                                                        const int num_tokens,
                                                        const int dim,
                                                        const int top_k)
{
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=num_tokens*top_k*dim;

  if(tid<total)
  {
    const int d=tid%dim;
    const int assignment_idx=tid/dim;
    const int token_idx=assignment_idx/top_k;
    const int k_idx=assignment_idx%top_k;

    const int expert_idx=expert_indices[token_idx*top_k+k_idx];
    const int pos_in_expert=dispatch_map[token_idx*top_k+k_idx];

    if(expert_idx>=0 && pos_in_expert>=0)
    {
      const float w=static_cast<float>(expert_weights[token_idx*top_k+k_idx]);
      const int dst_idx=(expert_offsets[expert_idx]+pos_in_expert)*dim+d;
      const float g=static_cast<float>(grad_output[token_idx*dim+d]);
      grad_expert_buffer[dst_idx]=static_cast<T>(w*g);
    }
  }
}

template<typename T>
void launch_moe_combine_backward_grad_expert(const T *grad_output,
                                              const int *expert_indices,
                                              const T *expert_weights,
                                              const int *dispatch_map,
                                              const int *expert_offsets,
                                              T *grad_expert_buffer,
                                              const int num_tokens,
                                              const int dim,
                                              const int top_k,
                                              cudaStream_t stream)
{
  const int total=num_tokens*top_k*dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  moe_combine_backward_grad_expert_kernel<T>
                  <<<num_blocks,block_size,0,stream>>>(grad_output,
                                                        expert_indices,
                                                        expert_weights,
                                                        dispatch_map,
                                                        expert_offsets,
                                                        grad_expert_buffer,
                                                        num_tokens,
                                                        dim,
                                                        top_k);
}

template void launch_moe_combine_backward_grad_expert<float>(const float *,
                                                              const int *,
                                                              const float *,
                                                              const int *,
                                                              const int *,
                                                              float *,
                                                              const int,
                                                              const int,
                                                              const int,
                                                              cudaStream_t);
template void launch_moe_combine_backward_grad_expert<__half>(const __half *,
                                                               const int *,
                                                               const __half *,
                                                               const int *,
                                                               const int *,
                                                               __half *,
                                                               const int,
                                                               const int,
                                                               const int,
                                                               cudaStream_t);
template void launch_moe_combine_backward_grad_expert<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                      const int *,
                                                                      const __nv_bfloat16 *,
                                                                      const int *,
                                                                      const int *,
                                                                      __nv_bfloat16 *,
                                                                      const int,
                                                                      const int,
                                                                      const int,
                                                                      cudaStream_t);

//------------------------------------------------------------------------------
// MoE Combine Backward — grad_weights path
//
// grad_W[t,k] = sum_d grad_output[t,d] * expert_buffer[(off[e]+pos)*dim + d]
//
// One thread per (t, k) pair, serial reduction over d.
//------------------------------------------------------------------------------
// One block per (t,k). Threads in the block cooperatively reduce the
// dot product over dim. Dim ranges from hundreds to several thousand in
// real workloads; serial reduction per (t,k) left the GPU starved at
// prod scale (only num_tokens*top_k active threads). Block reduction
// restores parallelism proportional to dim.
// One block per (t,k). Threads in the block cooperatively reduce the
// dot product over dim. Dim ranges from hundreds to several thousand in
// real workloads; serial reduction per (t,k) left the GPU starved at
// prod scale (only num_tokens*top_k active threads). Block reduction
// restores parallelism proportional to dim.
#define MOE_GRAD_WEIGHTS_BLOCK_SIZE 256
template<typename T>
__global__ void moe_combine_backward_grad_weights_block_kernel(const T *grad_output,
                                                                const T *expert_buffer,
                                                                const int *expert_indices,
                                                                const int *dispatch_map,
                                                                const int *expert_offsets,
                                                                T *grad_weights,
                                                                const int num_tokens,
                                                                const int dim,
                                                                const int top_k)
{
  const int assignment_idx=blockIdx.x;
  const int token_idx=assignment_idx/top_k;
  const int k_idx=assignment_idx%top_k;

  if(token_idx>=num_tokens)
  {
    return;
  }

  const int expert_idx=expert_indices[token_idx*top_k+k_idx];
  const int pos_in_expert=dispatch_map[token_idx*top_k+k_idx];

  float partial=0.0f;
  if(expert_idx>=0 && pos_in_expert>=0)
  {
    const int src_base=(expert_offsets[expert_idx]+pos_in_expert)*dim;
    const int grad_base=token_idx*dim;
    for(int d=threadIdx.x;d<dim;d+=MOE_GRAD_WEIGHTS_BLOCK_SIZE)
    {
      partial+=static_cast<float>(grad_output[grad_base+d])
               *static_cast<float>(expert_buffer[src_base+d]);
    }
  }

  __shared__ float sdata[MOE_GRAD_WEIGHTS_BLOCK_SIZE];
  sdata[threadIdx.x]=partial;
  __syncthreads();

  for(int stride=MOE_GRAD_WEIGHTS_BLOCK_SIZE/2;stride>0;stride>>=1)
  {
    if(threadIdx.x<stride)
    {
      sdata[threadIdx.x]+=sdata[threadIdx.x+stride];
    }
    __syncthreads();
  }

  if(threadIdx.x==0)
  {
    grad_weights[assignment_idx]=static_cast<T>(sdata[0]);
  }
}

template<typename T>
void launch_moe_combine_backward_grad_weights(const T *grad_output,
                                               const T *expert_buffer,
                                               const int *expert_indices,
                                               const int *dispatch_map,
                                               const int *expert_offsets,
                                               T *grad_weights,
                                               const int num_tokens,
                                               const int dim,
                                               const int top_k,
                                               cudaStream_t stream)
{
  const int num_blocks=num_tokens*top_k;
  moe_combine_backward_grad_weights_block_kernel<T>
    <<<num_blocks,MOE_GRAD_WEIGHTS_BLOCK_SIZE,0,stream>>>(grad_output,
                                                           expert_buffer,
                                                           expert_indices,
                                                           dispatch_map,
                                                           expert_offsets,
                                                           grad_weights,
                                                           num_tokens,
                                                           dim,
                                                           top_k);
}

template void launch_moe_combine_backward_grad_weights<float>(const float *,
                                                               const float *,
                                                               const int *,
                                                               const int *,
                                                               const int *,
                                                               float *,
                                                               const int,
                                                               const int,
                                                               const int,
                                                               cudaStream_t);
template void launch_moe_combine_backward_grad_weights<__half>(const __half *,
                                                                const __half *,
                                                                const int *,
                                                                const int *,
                                                                const int *,
                                                                __half *,
                                                                const int,
                                                                const int,
                                                                const int,
                                                                cudaStream_t);
template void launch_moe_combine_backward_grad_weights<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                       const __nv_bfloat16 *,
                                                                       const int *,
                                                                       const int *,
                                                                       const int *,
                                                                       __nv_bfloat16 *,
                                                                       const int,
                                                                       const int,
                                                                       const int,
                                                                       cudaStream_t);

//------------------------------------------------------------------------------
// MoE Dispatch Backward
//
// Forward wrote input[t,d] to slot (off[e(t,k)]+pos(t,k))*dim + d for each k.
// Backward sums contributions from all k:
//
//   grad_input[t,d] = sum_k grad_expert_buffer[(off[e(t,k)]+pos(t,k))*dim + d]
//
// One thread per (t, d). Sums over k serially.
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_dispatch_backward_kernel(const T *grad_expert_buffer,
                                              const int *expert_indices,
                                              const int *dispatch_map,
                                              const int *expert_offsets,
                                              T *grad_input,
                                              const int num_tokens,
                                              const int dim,
                                              const int top_k)
{
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=num_tokens*dim;

  if(tid<total)
  {
    const int token_idx=tid/dim;
    const int d=tid%dim;

    float sum=0.0f;
    for(int k=0;k<top_k;++k)
    {
      const int expert_idx=expert_indices[token_idx*top_k+k];
      const int pos_in_expert=dispatch_map[token_idx*top_k+k];
      if(expert_idx>=0 && pos_in_expert>=0)
      {
        const int src_idx=(expert_offsets[expert_idx]+pos_in_expert)*dim+d;
        sum+=static_cast<float>(grad_expert_buffer[src_idx]);
      }
    }
    grad_input[tid]=static_cast<T>(sum);
  }
}

template<typename T>
void launch_moe_dispatch_backward(const T *grad_expert_buffer,
                                   const int *expert_indices,
                                   const int *dispatch_map,
                                   const int *expert_offsets,
                                   T *grad_input,
                                   const int num_tokens,
                                   const int dim,
                                   const int top_k,
                                   cudaStream_t stream)
{
  const int total=num_tokens*dim;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  moe_dispatch_backward_kernel<T><<<num_blocks,block_size,0,stream>>>(grad_expert_buffer,
                                                                       expert_indices,
                                                                       dispatch_map,
                                                                       expert_offsets,
                                                                       grad_input,
                                                                       num_tokens,
                                                                       dim,
                                                                       top_k);
}

template void launch_moe_dispatch_backward<float>(const float *,
                                                   const int *,
                                                   const int *,
                                                   const int *,
                                                   float *,
                                                   const int,
                                                   const int,
                                                   const int,
                                                   cudaStream_t);
template void launch_moe_dispatch_backward<__half>(const __half *,
                                                    const int *,
                                                    const int *,
                                                    const int *,
                                                    __half *,
                                                    const int,
                                                    const int,
                                                    const int,
                                                    cudaStream_t);
template void launch_moe_dispatch_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                           const int *,
                                                           const int *,
                                                           const int *,
                                                           __nv_bfloat16 *,
                                                           const int,
                                                           const int,
                                                           const int,
                                                           cudaStream_t);

//------------------------------------------------------------------------------
// MoE Top-K Gating Kernel
// Fused softmax + top-k selection for router
// Input: router_logits [num_tokens, num_experts] dtype T
// Internal accumulation is fp32 for numerical stability — softmax-over-
// experts wants the same fp32 reduction guarantees as the rest of the
// codebase even when StorageT is fp16/bf16.
// Output: expert_indices Int32 (always); expert_weights / router_probs fp32
// (router output drives expert dispatch which keeps its own StorageT for
// the actual activation pipeline).
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_topk_gating_kernel(const T *router_logits,
                                       int *expert_indices,
                                       float *expert_weights,
                                       float *router_probs,
                                       const int num_tokens,
                                       const int num_experts,
                                       const int top_k)
{
  // Each block handles one token
  const int token_idx=blockIdx.x;
  if(token_idx>=num_tokens)
  {
    return;
  }

  extern __shared__ float shared[];
  float *logits_shared=shared;
  float *probs_shared=shared+num_experts;
  int *top_indices=reinterpret_cast<int*>(shared+2*num_experts);
  float *top_values=reinterpret_cast<float*>(shared+2*num_experts+top_k);

  const int tid=threadIdx.x;

  // Load logits to shared memory, upcasting to fp32 for the reduction.
  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    logits_shared[e]=caif_load_f<T>(router_logits[token_idx*num_experts+e]);
  }
  __syncthreads();

  // Find max for numerical stability (thread 0)
  float max_val=-1e30f;
  if(tid==0)
  {
    for(int e=0;e<num_experts;++e)
    {
      if(logits_shared[e]>max_val)
      {
        max_val=logits_shared[e];
      }
    }
  }
  __syncthreads();

  // Broadcast max_val using shared memory
  if(tid==0)
  {
    probs_shared[0]=max_val;
  }
  __syncthreads();
  max_val=probs_shared[0];

  // Compute exp(logits - max) and sum
  float local_sum=0.0f;
  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    float exp_val=expf(logits_shared[e]-max_val);
    probs_shared[e]=exp_val;
    local_sum+=exp_val;
  }

  // Reduce sum across threads (simple reduction for small num_experts)
  __shared__ float sum_shared[256];
  sum_shared[tid]=local_sum;
  __syncthreads();

  if(tid==0)
  {
    float total_sum=0.0f;
    for(int i=0;i<blockDim.x && i<num_experts;++i)
    {
      total_sum+=sum_shared[i];
    }
    sum_shared[0]=total_sum;
  }
  __syncthreads();

  float total_sum=sum_shared[0];

  // Normalize to get probabilities
  for(int e=tid;e<num_experts;e+=blockDim.x)
  {
    probs_shared[e]/=total_sum;
    router_probs[token_idx*num_experts+e]=probs_shared[e];
  }
  __syncthreads();

  // Top-k selection (thread 0 does sequential selection for simplicity)
  if(tid==0)
  {
    for(int k=0;k<top_k;++k)
    {
      float max_prob=-1.0f;
      int max_idx=-1;

      for(int e=0;e<num_experts;++e)
      {
        // Check if already selected
        bool already_selected=false;
        for(int j=0;j<k;++j)
        {
          if(top_indices[j]==e)
          {
            already_selected=true;
            break;
          }
        }

        if(already_selected==false && probs_shared[e]>max_prob)
        {
          max_prob=probs_shared[e];
          max_idx=e;
        }
      }

      top_indices[k]=max_idx;
      top_values[k]=max_prob;
    }

    // Renormalize top-k weights to sum to 1
    float topk_sum=0.0f;
    for(int k=0;k<top_k;++k)
    {
      topk_sum+=top_values[k];
    }

    for(int k=0;k<top_k;++k)
    {
      expert_indices[token_idx*top_k+k]=top_indices[k];
      expert_weights[token_idx*top_k+k]=top_values[k]/topk_sum;
    }
  }
}

template<typename T>
void launch_moe_topk_gating(const T *router_logits,
                            int *expert_indices,
                            float *expert_weights,
                            float *router_probs,
                            const int num_tokens,
                            const int num_experts,
                            const int top_k,
                            cudaStream_t stream)
{
  // One block per token, enough threads to cover num_experts
  const int threads_per_block=min(256,((num_experts+31)/32)*32);
  const int shared_size=(2*num_experts)*sizeof(float)+(top_k)*sizeof(int)+(top_k)*sizeof(float);
  moe_topk_gating_kernel<T><<<num_tokens,threads_per_block,shared_size,stream>>>(router_logits,
                                                                                  expert_indices,
                                                                                  expert_weights,
                                                                                  router_probs,
                                                                                  num_tokens,
                                                                                  num_experts,
                                                                                  top_k);
}

template void launch_moe_topk_gating<float>(const float *,
                                            int *,
                                            float *,
                                            float *,
                                            int,
                                            int,
                                            int,
                                            cudaStream_t);
template void launch_moe_topk_gating<__half>(const __half *,
                                             int *,
                                             float *,
                                             float *,
                                             int,
                                             int,
                                             int,
                                             cudaStream_t);
template void launch_moe_topk_gating<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    int *,
                                                    float *,
                                                    float *,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);

//------------------------------------------------------------------------------
// MoE Build Dispatch Map Kernel
// Builds the dispatch_map that tracks position of each token within expert buffers
// Also computes expert_offsets (cumulative counts)
//------------------------------------------------------------------------------
__global__ void moe_count_per_expert_kernel(const int *expert_indices,
                                            int *expert_counts,
                                            const int num_tokens,
                                            const int num_experts,
                                            const int top_k,
                                            const int capacity_per_expert)
{
  // Simple atomic counting - each thread handles one token
  const int token_idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(token_idx>=num_tokens)
  {
    return;
  }

  for(int k=0;k<top_k;++k)
  {
    const int expert_idx=expert_indices[token_idx*top_k+k];
    if(expert_idx>=0 && expert_idx<num_experts)
    {
      const int old_count=atomicAdd(&expert_counts[expert_idx],1);
      // Capacity enforcement happens at dispatch time
      (void)old_count;
    }
  }
}

void launch_moe_count_per_expert(const int *expert_indices,
                                 int *expert_counts,
                                 const int num_tokens,
                                 const int num_experts,
                                 const int top_k,
                                 const int capacity_per_expert,
                                 cudaStream_t stream)
{
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(num_tokens+block_size-1)/block_size;
  moe_count_per_expert_kernel<<<num_blocks,block_size,0,stream>>>(expert_indices,
                                                                    expert_counts,
                                                                    num_tokens,
                                                                    num_experts,
                                                                    top_k,
                                                                    capacity_per_expert);
}

//------------------------------------------------------------------------------
// ST-MoE router z-loss gradient add kernel
// grad_logits[t,e] += logsumexp_scaled[t] * probs[t,e]
// logsumexp_scaled is pre-multiplied host-side by (2*z_loss_weight/N).
//------------------------------------------------------------------------------
template<typename T>
__global__ void moe_z_loss_grad_kernel(const T *logsumexp_scaled,
                                       const T *probs,
                                       T *grad_logits,
                                       const int num_tokens,
                                       const int num_experts)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=num_tokens*num_experts;
  if(idx>=total)
  {
    return;
  }
  const int t=idx/num_experts;
  // fp32 accumulate; cast on store keeps fp16/bf16 from rounding inside the
  // multiply-add.
  const float lse=static_cast<float>(logsumexp_scaled[t]);
  const float p=static_cast<float>(probs[idx]);
  const float prev=static_cast<float>(grad_logits[idx]);
  grad_logits[idx]=static_cast<T>(prev+lse*p);
}

template<typename T>
void launch_moe_z_loss_grad(const T *logsumexp_scaled,
                            const T *probs,
                            T *grad_logits,
                            const int num_tokens,
                            const int num_experts,
                            cudaStream_t stream)
{
  const int total=num_tokens*num_experts;
  const int block_size=g_caif_cuda_block_size;
  const int num_blocks=(total+block_size-1)/block_size;
  moe_z_loss_grad_kernel<T><<<num_blocks,block_size,0,stream>>>(logsumexp_scaled,
                                                                probs,
                                                                grad_logits,
                                                                num_tokens,
                                                                num_experts);
}

template void launch_moe_z_loss_grad<float>(const float *,
                                             const float *,
                                             float *,
                                             const int,
                                             const int,
                                             cudaStream_t);
template void launch_moe_z_loss_grad<__half>(const __half *,
                                              const __half *,
                                              __half *,
                                              const int,
                                              const int,
                                              cudaStream_t);
template void launch_moe_z_loss_grad<__nv_bfloat16>(const __nv_bfloat16 *,
                                                     const __nv_bfloat16 *,
                                                     __nv_bfloat16 *,
                                                     const int,
                                                     const int,
                                                     cudaStream_t);

// end of former extern "C" block — functions now have C++ linkage for dtype templates

//------------------------------------------------------------------------------
// FlashAttention-2 Forward Kernel
// Implements tiled attention with online softmax to avoid O(n²) memory
// References: https://arxiv.org/abs/2307.08691
//------------------------------------------------------------------------------

// Legacy block sizes removed — forward uses g_caif_fa_fwd_bc, backward uses g_fa_bwd_*

// Block reduction for max (across all warps)
__device__ __forceinline__ float block_reduce_max(float val,float *shared_mem,int tid,int block_size)
{
  const int warp_id=tid/32;
  const int lane_id=tid%32;
  const int num_warps=(block_size+31)/32;

  val=warp_reduce_max(val);

  if(lane_id==0)
  {
    shared_mem[warp_id]=val;
  }
  __syncthreads();

  if(tid<num_warps)
  {
    val=shared_mem[tid];
  }
  else
  {
    val=-INFINITY;
  }

  if(warp_id==0)
  {
    val=warp_reduce_max(val);
  }

  return val;
}

// Block reduction for sum
__device__ __forceinline__ float block_reduce_sum(float val,float *shared_mem,int tid,int block_size)
{
  const int warp_id=tid/32;
  const int lane_id=tid%32;
  const int num_warps=(block_size+31)/32;

  val=warp_reduce_sum(val);

  if(lane_id==0)
  {
    shared_mem[warp_id]=val;
  }
  __syncthreads();

  if(tid<num_warps)
  {
    val=shared_mem[tid];
  }
  else
  {
    val=0.0f;
  }

  if(warp_id==0)
  {
    val=warp_reduce_sum(val);
  }

  return val;
}

//------------------------------------------------------------------------------
// FlashAttention-2 Forward — TF32 Tensor Core Kernel (sm_80+)
//
// Uses nvcuda::wmma 16x16x8 TF32 tiles for the two matmuls per KV block:
//   S = Q @ K^T  (scores)
//   O += softmax(S) @ V  (output accumulation)
//
// Template: D=head_dim, BR=Q rows/block, BC=KV tile cols
// Grid: (batch_heads, ceil(seq_len/BR))
// Block: n_warps*32 where n_warps = (BR/16) * (BC/16) / tiles_per_warp_s
//
// Shared memory: Q_tile[BR*(D+2)] + KV_buf[BC*(D+2)] + S_tile[BR*(BC+2)]
//                + row_max[BR] + row_sum[BR]  (padded strides for bank conflicts)
// O accumulator and S scores live in wmma register fragments.
// Softmax computed in-register with cross-warp reduce via S_tile[0..NW*16].
//------------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Only compile TC kernel for sm_80+
#define CAIF_HAS_TC_FLASH 1
#else
#define CAIF_HAS_TC_FLASH 0
#endif

using namespace nvcuda;

//------------------------------------------------------------------------------
// Async copy helpers (cp.async, sm_80+) — global→shared bypassing L1
//------------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_f4(void *dst_shared,
                                            const void *src_global)
{
  const uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               ::"r"(dst),"l"(src_global));
}

__device__ __forceinline__ void cp_async_f2(void *dst_shared,
                                            const void *src_global)
{
  const uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
               ::"r"(dst),"l"(src_global));
}

__device__ __forceinline__ void cp_async_commit()
{
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait()
{
  asm volatile("cp.async.wait_group 0;\n");
}

template<typename T,int D,int BR,int BC,int NW>
__global__ void flash_attention_forward_tc_kernel(const T *__restrict__ Q,
                                                  const T *__restrict__ K,
                                                  const T *__restrict__ V,
                                                  T *__restrict__ O,
                                                  float *__restrict__ L,
                                                  const int seq_len,
                                                  const float scale,
                                                  const int causal,
                                                  const uint32_t *__restrict__ prefix_lens,
                                                  const int num_heads,
                                                  const int num_kv_heads)
{
#if CAIF_HAS_TC_FLASH
  const int bh=blockIdx.x;
  int pfx=0;
  if(prefix_lens!=nullptr)
  {
    pfx=prefix_lens[bh/num_heads];
  }
  // Native GQA: Q/O index by the full bh, K/V index by the KV head group.
  // For MHA (num_kv_heads == num_heads) this is the identity map.
  const int bh_kv=bh*num_kv_heads/num_heads;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  const int lane_id=tid%32;
  const int warp_id=tid/32;

  constexpr int n_warps=NW;
  constexpr int tiles_m=BR/16;
  constexpr int tiles_n_s=BC/16;
  constexpr int tiles_n_o=D/16;
  constexpr int block_threads=n_warps*32;

  // Smem stride padding: +2 eliminates bank conflicts for wmma loads.
  // stride % 32 == 2 → each of 16 rows within a wmma tile maps to a
  // distinct pair of banks, giving zero conflicts on column-parallel access.
  constexpr int d_pad=D+2;
  constexpr int bc_pad=BC+2;
  constexpr int d_f2=D/2;
  constexpr int d_pad_f2=d_pad/2;

  // Shared memory layout (O is in registers, not smem)
  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *KV_buf=Q_tile+BR*d_pad;
  float *S_tile=KV_buf+BC*d_pad;
  float *row_max_arr=S_tile+BR*bc_pad;
  float *row_sum_arr=row_max_arr+BR;

  // Batch-head pointers. Q/O follow the full bh (one per query head);
  // K/V follow bh_kv so native GQA avoids materializing a repeat-expanded
  // KV tensor. For MHA this reduces to bh_kv == bh.
  const T *Q_bh=Q+bh*seq_len*D;
  const T *K_bh=K+bh_kv*seq_len*D;
  const T *V_bh=V+bh_kv*seq_len*D;
  T *O_bh=O+bh*seq_len*D;
  float *L_bh=L+bh*seq_len;

  const int q_start=q_block_idx*BR;

  // Warp grouping for register-based softmax.
  // Each M-tile group's warps collectively cover all N S-tiles, enabling
  // in-register softmax with cross-warp reduce (no S_tile smem round-trip).
  constexpr int warps_per_m=n_warps/tiles_m;
  constexpr int s_tiles_pw=(tiles_n_s>=warps_per_m)*(tiles_n_s/warps_per_m);
  constexpr int o_tiles_pw=(tiles_n_o>=warps_per_m)*(tiles_n_o/warps_per_m);
  // Array size must be >=1 for CUDA; loops guard on tile count
  constexpr int s_arr=s_tiles_pw+(!s_tiles_pw);
  constexpr int o_arr=o_tiles_pw+(!o_tiles_pw);
  const int m_idx=warp_id/warps_per_m;
  const int group_warp=warp_id%warps_per_m;
  const int n_start_s=group_warp*s_tiles_pw;
  const int n_start_o=group_warp*o_tiles_pw;
  const int group_base=m_idx*warps_per_m;

  // Persistent O accumulators in wmma registers
  wmma::fragment<wmma::accumulator,16,16,8,float> o_frags[o_arr];
  for(int t=0;t<o_tiles_pw;++t)
  {
    wmma::fill_fragment(o_frags[t],0.0f);
  }

  // Cooperative load Q_tile[BR, d_pad] from global memory (padded stride)
  const int valid_q_rows=min(BR,seq_len-q_start);
  {
    if constexpr(sizeof(T)==4)
    {
      const int valid_q_f2=max(valid_q_rows,0)*d_f2;
      constexpr int total_q_f2=BR*d_f2;
      const float2 *Q_src2=reinterpret_cast<const float2 *>(Q_bh+q_start*D);
      float2 *Q_dst2=reinterpret_cast<float2 *>(Q_tile);
      for(int i=tid;i<valid_q_f2;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        Q_dst2[row*d_pad_f2+f2c]=Q_src2[i];
      }
      const float2 zero2=make_float2(0.0f,0.0f);
      for(int i=valid_q_f2+tid;i<total_q_f2;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        Q_dst2[row*d_pad_f2+f2c]=zero2;
      }
    }
    else
    {
      const int valid_q=max(valid_q_rows,0)*D;
      constexpr int total_q=BR*D;
      for(int i=tid;i<valid_q;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        Q_tile[row*d_pad+col]=float(Q_bh[(q_start+row)*D+col]);
      }
      for(int i=valid_q+tid;i<total_q;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        Q_tile[row*d_pad+col]=0.0f;
      }
    }
  }

  // Init row_max/row_sum
  for(int i=tid;i<BR;i+=block_threads)
  {
    row_max_arr[i]=-INFINITY;
    row_sum_arr[i]=0.0f;
  }
  __syncthreads();

  // Number of KV blocks
  int num_kv_blocks=(seq_len+BC-1)/BC;
  if(causal==1 && prefix_lens==nullptr)
  {
    int max_q=q_start+BR-1;
    if(max_q>=seq_len)
    {
      max_q=seq_len-1;
    }
    num_kv_blocks=min(num_kv_blocks,(max_q/BC)+1);
  }

  constexpr int kv_f2=BC*d_f2;

  // Pipeline: prefetch K[0]
  float2 *KV_dst2=reinterpret_cast<float2 *>(KV_buf);
  if(num_kv_blocks>0)
  {
    if constexpr(sizeof(T)==4)
    {
      const int kv0_valid=min(BC,seq_len)*d_f2;
      const float2 *K0_src2=reinterpret_cast<const float2 *>(K_bh);
      for(int i=tid;i<kv0_valid;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        cp_async_f2(&KV_dst2[row*d_pad_f2+f2c],&K0_src2[i]);
      }
      const float2 zero2=make_float2(0.0f,0.0f);
      for(int i=kv0_valid+tid;i<kv_f2;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        KV_dst2[row*d_pad_f2+f2c]=zero2;
      }
    }
    else
    {
      const int kv0_valid=min(BC,seq_len)*D;
      constexpr int kv_total=BC*D;
      for(int i=tid;i<kv0_valid;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        KV_buf[row*d_pad+col]=float(K_bh[row*D+col]);
      }
      for(int i=kv0_valid+tid;i<kv_total;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        KV_buf[row*d_pad+col]=0.0f;
      }
    }
    cp_async_commit();
  }

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*BC;
    const int valid_kv_rows=min(BC,seq_len-kv_start);
    const int valid_kv_f2=valid_kv_rows*d_f2;

    // PHASE 1: Wait for K, compute S = Q @ K^T in wmma registers
    cp_async_wait();
    __syncthreads();

    wmma::fragment<wmma::accumulator,16,16,8,float> s_accs[s_arr];
    for(int t=0;t<s_tiles_pw;++t)
    {
      const int n=n_start_s+t;
      wmma::fill_fragment(s_accs[t],0.0f);
      for(int k=0;k<D/8;++k)
      {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> k_frag;
        wmma::load_matrix_sync(q_frag,&Q_tile[m_idx*16*d_pad+k*8],d_pad);
        wmma::load_matrix_sync(k_frag,&KV_buf[n*16*d_pad+k*8],d_pad);
        wmma::mma_sync(s_accs[t],q_frag,k_frag,s_accs[t]);
      }
      for(int i=0;i<s_accs[t].num_elements;++i)
      {
        s_accs[t].x[i]*=scale;
      }
    }

    // Sync: all warps done reading KV_buf before V overwrites it
    __syncthreads();

    // Async V load into KV_buf (overlapped with softmax)
    {
      if constexpr(sizeof(T)==4)
      {
        const float2 *V_src2=reinterpret_cast<const float2 *>(V_bh+kv_start*D);
        for(int i=tid;i<valid_kv_f2;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          cp_async_f2(&KV_dst2[row*d_pad_f2+f2c],&V_src2[i]);
        }
        const float2 zero2=make_float2(0.0f,0.0f);
        for(int i=valid_kv_f2+tid;i<kv_f2;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          KV_dst2[row*d_pad_f2+f2c]=zero2;
        }
      }
      else
      {
        const int valid_kv=valid_kv_rows*D;
        constexpr int kv_total=BC*D;
        for(int i=tid;i<valid_kv;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=float(V_bh[(kv_start+row)*D+col]);
        }
        for(int i=valid_kv+tid;i<kv_total;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=0.0f;
        }
      }
      cp_async_commit();
    }

    // PHASE 2: Register-based online softmax on s_accs
    // Fragment layout (stable sm_80-sm_120):
    //   elements {0,1,4,5} → local_row = lane_id/4       (row_lo)
    //   elements {2,3,6,7} → local_row = lane_id/4 + 8   (row_hi)
    //   elem col: 0→(lane_id%4)*2, 1→+1, 4→+8, 5→+9 (same for 2/3/6/7)
    {
      const int row_lo=m_idx*16+(lane_id/4);
      const int row_hi=m_idx*16+(lane_id/4)+8;
      const int global_q_lo=q_start+row_lo;
      const int global_q_hi=q_start+row_hi;

      // Apply causal + boundary masks in registers
      for(int t=0;t<s_tiles_pw;++t)
      {
        const int n=n_start_s+t;
        const int bc0=kv_start+n*16+(lane_id%4)*2;
        const int bc1=bc0+1;
        const int bc2=kv_start+n*16+(lane_id%4)*2+8;
        const int bc3=bc2+1;

        // Prefix-LM: allowed iff (k<=q) OR (k<pfx). Plain causal: k<=q.
        // With prefix_lens==nullptr we reduce exactly to the causal-only rule.
        if(causal==1)
        {
          if(bc0>global_q_lo && bc0>=pfx)
          {
            s_accs[t].x[0]=-INFINITY;
          }
          if(bc1>global_q_lo && bc1>=pfx)
          {
            s_accs[t].x[1]=-INFINITY;
          }
          if(bc0>global_q_hi && bc0>=pfx)
          {
            s_accs[t].x[2]=-INFINITY;
          }
          if(bc1>global_q_hi && bc1>=pfx)
          {
            s_accs[t].x[3]=-INFINITY;
          }
          if(bc2>global_q_lo && bc2>=pfx)
          {
            s_accs[t].x[4]=-INFINITY;
          }
          if(bc3>global_q_lo && bc3>=pfx)
          {
            s_accs[t].x[5]=-INFINITY;
          }
          if(bc2>global_q_hi && bc2>=pfx)
          {
            s_accs[t].x[6]=-INFINITY;
          }
          if(bc3>global_q_hi && bc3>=pfx)
          {
            s_accs[t].x[7]=-INFINITY;
          }
        }
        // K boundary mask: zero-padded K positions beyond seq_len
        // must be masked to -inf so they don't participate in
        // softmax. Causal mask catches these implicitly; non-causal
        // does not.
        if(bc0>=seq_len)
        {
          s_accs[t].x[0]=-INFINITY;
          s_accs[t].x[2]=-INFINITY;
        }
        if(bc1>=seq_len)
        {
          s_accs[t].x[1]=-INFINITY;
          s_accs[t].x[3]=-INFINITY;
        }
        if(bc2>=seq_len)
        {
          s_accs[t].x[4]=-INFINITY;
          s_accs[t].x[6]=-INFINITY;
        }
        if(bc3>=seq_len)
        {
          s_accs[t].x[5]=-INFINITY;
          s_accs[t].x[7]=-INFINITY;
        }
        if(global_q_lo>=seq_len)
        {
          s_accs[t].x[0]=-INFINITY;
          s_accs[t].x[1]=-INFINITY;
          s_accs[t].x[4]=-INFINITY;
          s_accs[t].x[5]=-INFINITY;
        }
        if(global_q_hi>=seq_len)
        {
          s_accs[t].x[2]=-INFINITY;
          s_accs[t].x[3]=-INFINITY;
          s_accs[t].x[6]=-INFINITY;
          s_accs[t].x[7]=-INFINITY;
        }
      }

      // Local max across this warp's S tiles
      float max_lo=-INFINITY;
      float max_hi=-INFINITY;
      for(int t=0;t<s_tiles_pw;++t)
      {
        max_lo=fmaxf(max_lo,fmaxf(s_accs[t].x[0],s_accs[t].x[1]));
        max_lo=fmaxf(max_lo,fmaxf(s_accs[t].x[4],s_accs[t].x[5]));
        max_hi=fmaxf(max_hi,fmaxf(s_accs[t].x[2],s_accs[t].x[3]));
        max_hi=fmaxf(max_hi,fmaxf(s_accs[t].x[6],s_accs[t].x[7]));
      }

      // Reduce max within 4-thread row group (lane_id%4 groups)
      max_lo=fmaxf(max_lo,__shfl_xor_sync(0xffffffff,max_lo,1));
      max_lo=fmaxf(max_lo,__shfl_xor_sync(0xffffffff,max_lo,2));
      max_hi=fmaxf(max_hi,__shfl_xor_sync(0xffffffff,max_hi,1));
      max_hi=fmaxf(max_hi,__shfl_xor_sync(0xffffffff,max_hi,2));

      // Cross-warp max reduce via S_tile[0..NW*16] temporary
      float *reduce_buf=S_tile;
      if(lane_id%4==0)
      {
        reduce_buf[warp_id*16+(lane_id/4)]=max_lo;
        reduce_buf[warp_id*16+(lane_id/4)+8]=max_hi;
      }
      __syncthreads();

      float full_max_lo=-INFINITY;
      float full_max_hi=-INFINITY;
      for(int w=group_base;w<group_base+warps_per_m;++w)
      {
        full_max_lo=fmaxf(full_max_lo,reduce_buf[w*16+(lane_id/4)]);
        full_max_hi=fmaxf(full_max_hi,reduce_buf[w*16+(lane_id/4)+8]);
      }

      // Online correction factor
      const float old_max_lo=row_max_arr[row_lo];
      const float old_max_hi=row_max_arr[row_hi];
      const float new_max_lo=fmaxf(old_max_lo,full_max_lo);
      const float new_max_hi=fmaxf(old_max_hi,full_max_hi);
      const float corr_lo=__expf(old_max_lo-new_max_lo);
      const float corr_hi=__expf(old_max_hi-new_max_hi);

      // Compute exp(S - new_max) in place, accumulate local sum
      float sum_lo=0.0f;
      float sum_hi=0.0f;
      for(int t=0;t<s_tiles_pw;++t)
      {
        s_accs[t].x[0]=__expf(s_accs[t].x[0]-new_max_lo);
        sum_lo+=s_accs[t].x[0];
        s_accs[t].x[1]=__expf(s_accs[t].x[1]-new_max_lo);
        sum_lo+=s_accs[t].x[1];
        s_accs[t].x[4]=__expf(s_accs[t].x[4]-new_max_lo);
        sum_lo+=s_accs[t].x[4];
        s_accs[t].x[5]=__expf(s_accs[t].x[5]-new_max_lo);
        sum_lo+=s_accs[t].x[5];
        s_accs[t].x[2]=__expf(s_accs[t].x[2]-new_max_hi);
        sum_hi+=s_accs[t].x[2];
        s_accs[t].x[3]=__expf(s_accs[t].x[3]-new_max_hi);
        sum_hi+=s_accs[t].x[3];
        s_accs[t].x[6]=__expf(s_accs[t].x[6]-new_max_hi);
        sum_hi+=s_accs[t].x[6];
        s_accs[t].x[7]=__expf(s_accs[t].x[7]-new_max_hi);
        sum_hi+=s_accs[t].x[7];
      }

      // Reduce sum within 4-thread row group
      sum_lo+=__shfl_xor_sync(0xffffffff,sum_lo,1);
      sum_lo+=__shfl_xor_sync(0xffffffff,sum_lo,2);
      sum_hi+=__shfl_xor_sync(0xffffffff,sum_hi,1);
      sum_hi+=__shfl_xor_sync(0xffffffff,sum_hi,2);

      // Cross-warp sum reduce
      if(lane_id%4==0)
      {
        reduce_buf[warp_id*16+(lane_id/4)]=sum_lo;
        reduce_buf[warp_id*16+(lane_id/4)+8]=sum_hi;
      }
      __syncthreads();

      float full_sum_lo=0.0f;
      float full_sum_hi=0.0f;
      for(int w=group_base;w<group_base+warps_per_m;++w)
      {
        full_sum_lo+=reduce_buf[w*16+(lane_id/4)];
        full_sum_hi+=reduce_buf[w*16+(lane_id/4)+8];
      }

      // Update row state (one warp per group writes)
      if(group_warp==0 && lane_id%4==0)
      {
        row_sum_arr[row_lo]=corr_lo*row_sum_arr[row_lo]+full_sum_lo;
        row_sum_arr[row_hi]=corr_hi*row_sum_arr[row_hi]+full_sum_hi;
        row_max_arr[row_lo]=new_max_lo;
        row_max_arr[row_hi]=new_max_hi;
      }

      // Rescale O fragments by correction (all O tiles share this warp's m_idx)
      for(int t=0;t<o_tiles_pw;++t)
      {
        o_frags[t].x[0]*=corr_lo;
        o_frags[t].x[1]*=corr_lo;
        o_frags[t].x[2]*=corr_hi;
        o_frags[t].x[3]*=corr_hi;
        o_frags[t].x[4]*=corr_lo;
        o_frags[t].x[5]*=corr_lo;
        o_frags[t].x[6]*=corr_hi;
        o_frags[t].x[7]*=corr_hi;
      }
    }

    // Barrier: S_tile doubles as the cross-warp sum/max reduce buffer
    // above; without this sync a fast warp can begin the wmma store below
    // and overwrite reduce_buf slots that a slower warp in the same
    // m_idx group is still reading. Latent at NW=2/4 (warps near
    // lockstep), reliably corrupts output at NW=8.
    __syncthreads();

    // Store exp(S) to S_tile for Phase 3 (single write, padded stride)
    for(int t=0;t<s_tiles_pw;++t)
    {
      const int n=n_start_s+t;
      wmma::store_matrix_sync(
        &S_tile[m_idx*16*bc_pad+n*16],s_accs[t],bc_pad,wmma::mem_row_major);
    }

    // Wait for V + ensure S_tile writes visible
    cp_async_wait();
    __syncthreads();

    // PHASE 3: Accumulate O += softmax(S) @ V using tensor cores (padded strides)
    for(int t=0;t<o_tiles_pw;++t)
    {
      const int n=n_start_o+t;
      for(int k=0;k<BC/8;++k)
      {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> s_frag;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> v_frag;
        wmma::load_matrix_sync(s_frag,&S_tile[m_idx*16*bc_pad+k*8],bc_pad);
        wmma::load_matrix_sync(v_frag,&KV_buf[k*8*d_pad+n*16],d_pad);
        wmma::mma_sync(o_frags[t],s_frag,v_frag,o_frags[t]);
      }
    }
    __syncthreads();

    // Pipeline: prefetch K[next] into KV_buf
    if(kv_block+1<num_kv_blocks)
    {
      const int next_start=(kv_block+1)*BC;
      if constexpr(sizeof(T)==4)
      {
        const int next_valid=min(BC,seq_len-next_start)*d_f2;
        const float2 *K_next2=reinterpret_cast<const float2 *>(K_bh+next_start*D);
        for(int i=tid;i<next_valid;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          cp_async_f2(&KV_dst2[row*d_pad_f2+f2c],&K_next2[i]);
        }
        const float2 zero2=make_float2(0.0f,0.0f);
        for(int i=next_valid+tid;i<kv_f2;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          KV_dst2[row*d_pad_f2+f2c]=zero2;
        }
      }
      else
      {
        const int next_valid=min(BC,seq_len-next_start)*D;
        constexpr int kv_total=BC*D;
        for(int i=tid;i<next_valid;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=float(K_bh[(next_start+row)*D+col]);
        }
        for(int i=next_valid+tid;i<kv_total;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=0.0f;
        }
      }
      cp_async_commit();
    }
  }

  // Final: store O fragments to smem (reuse Q_tile, padded stride), normalize, write
  __syncthreads();
  float *O_smem=Q_tile;
  for(int t=0;t<o_tiles_pw;++t)
  {
    const int n=n_start_o+t;
    wmma::store_matrix_sync(&O_smem[m_idx*16*d_pad+n*16],o_frags[t],d_pad,wmma::mem_row_major);
  }
  __syncthreads();

  for(int i=tid;i<BR*D;i+=block_threads)
  {
    const int row=i/D;
    const int col=i-row*D;
    const int global_row=q_start+row;
    if(global_row<seq_len)
    {
      float inv_l=0.0f;
      if(row_sum_arr[row]>0.0f)
      {
        inv_l=1.0f/row_sum_arr[row];
      }
      O_bh[global_row*D+col]=T(O_smem[row*d_pad+col]*inv_l);
    }
  }

  // Write logsumexp (one per Q row)
  for(int r=tid;r<BR;r+=block_threads)
  {
    const int global_row=q_start+r;
    if(global_row<seq_len)
    {
      L_bh[global_row]=row_max_arr[r]+logf(row_sum_arr[r]+1e-10f);
    }
  }
#endif  // CAIF_HAS_TC_FLASH
}

template<typename T,int D,int BR,int BC,int NW>
static void launch_fa_fwd_tc(const T *Q,
                             const T *K,
                             const T *V,
                             T *O,
                             float *L,
                             const int batch_heads,
                             const int seq_len,
                             const float scale,
                             const int causal,
                             const uint32_t *prefix_lens,
                             const int num_heads,
                             const int num_kv_heads,
                             cudaStream_t stream)
{
  const int num_q_blocks=(seq_len+BR-1)/BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(NW*32);
  constexpr size_t smem_size=(BR*(D+2)+BC*(D+2)+BR*(BC+2)+2*BR)*sizeof(float);

  if(smem_size>g_caif_cuda_default_shared_memory)
  {
    cudaFuncSetAttribute(
      (void *)flash_attention_forward_tc_kernel<T,D,BR,BC,NW>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  }
  flash_attention_forward_tc_kernel<T,D,BR,BC,NW>
    <<<grid,block,smem_size,stream>>>(Q,
                                       K,
                                       V,
                                       O,
                                       L,
                                       seq_len,
                                       scale,
                                       causal,
                                       prefix_lens,
                                       num_heads,
                                       num_kv_heads);
}

//------------------------------------------------------------------------------
// FlashAttention-2 Forward Kernel — Warp-Per-Row (Memory-Efficient Fallback)
//
// Each warp (32 threads) processes one Q row. Lanes parallelize across
// head_dim D. BR warps per block = BR Q rows per block.
//
// Grid: (batch_heads, ceil(seq_len / BR))
// Block: (BR * 32) threads
//
// Template: D = head_dim, BR = Q rows per block
// KV tile size is g_caif_fa_fwd_bc (constexpr below).
//
// Q lives in registers (no Q tile in shared memory).
// Two-pass score computation eliminates S_local register array:
//   Pass 1: dot products via warp reduce, find row_max
//   Pass 2: recompute dots, compute exp, accumulate V
constexpr int g_caif_fa_fwd_bc=64;  // K/V block size for forward kernel — must match caif_constants.h

template<typename T,int D,int BR>
__global__ void flash_attention_forward_kernel(const T *__restrict__ Q,
                                               const T *__restrict__ K,
                                               const T *__restrict__ V,
                                               T *__restrict__ O,
                                               float *__restrict__ L,
                                               const int seq_len,
                                               const float scale,
                                               const int causal,
                                               const uint32_t *__restrict__ prefix_lens,
                                               const int num_heads,
                                               const int num_kv_heads)
{
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  const int warp_id=tid/32;
  const int lane_id=tid%32;

  const int q_row=q_block_idx*BR+warp_id;
  const bool q_valid=(q_row<seq_len);

  int pfx=0;
  if(prefix_lens!=nullptr)
  {
    pfx=prefix_lens[bh/num_heads];
  }

  // Native GQA: map the Q head index onto its KV head group so the kernel
  // can attend against a [batch*num_kv_heads, seq, D] KV tensor without the
  // caller materializing a repeat-expanded copy. When num_kv_heads equals
  // num_heads this collapses to the MHA identity (bh_kv == bh).
  const int bh_kv=bh*num_kv_heads/num_heads;

  // Shared memory: K tile + V tile (Q is in registers)
  extern __shared__ float smem[];
  float *K_tile=smem;
  float *V_tile=smem+g_caif_fa_fwd_bc*D;

  // Batch-head pointers
  const T *Q_bh=Q+bh*seq_len*D;
  const T *K_bh=K+bh_kv*seq_len*D;
  const T *V_bh=V+bh_kv*seq_len*D;
  T *O_bh=O+bh*seq_len*D;
  float *L_bh=L+bh*seq_len;

  // Load Q row into registers — each lane holds ceil(D/32) elements
  constexpr int elems=(D+31)/32;
  float q_reg[elems];
  for(int dd=lane_id,e=0;dd<D;dd+=32,++e)
  {
    if(q_valid==true)
    {
      q_reg[e]=float(Q_bh[q_row*D+dd]);
    }
    else
    {
      q_reg[e]=0.0f;
    }
  }

  // Output accumulators in registers
  float m_i=-INFINITY;
  float l_i=0.0f;
  float o_reg[elems];
  for(int e=0;e<elems;++e)
  {
    o_reg[e]=0.0f;
  }

  // Number of KV blocks — uniform across block for cooperative loads
  int num_kv_blocks=(seq_len+g_caif_fa_fwd_bc-1)/g_caif_fa_fwd_bc;
  if(causal==1 && prefix_lens==nullptr)
  {
    int max_q=q_block_idx*BR+BR-1;
    if(max_q>=seq_len)
    {
      max_q=seq_len-1;
    }
    num_kv_blocks=min(num_kv_blocks,(max_q/g_caif_fa_fwd_bc)+1);
  }

  const int block_threads=BR*32;

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*g_caif_fa_fwd_bc;

    // Cooperative K/V tile load — all threads in block participate
    const int tile_elems=g_caif_fa_fwd_bc*D;
    for(int i=tid;i<tile_elems;i+=block_threads)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;
      if(global_row<seq_len)
      {
        K_tile[row*D+col]=float(K_bh[global_row*D+col]);
        V_tile[row*D+col]=float(V_bh[global_row*D+col]);
      }
      else
      {
        K_tile[row*D+col]=0.0f;
        V_tile[row*D+col]=0.0f;
      }
    }
    __syncthreads();

    // Single pass: compute scores, find max, then exp + accumulate V
    float s_local[g_caif_fa_fwd_bc];
    float row_max=-INFINITY;
    int num_valid=0;

    for(int j=0;j<g_caif_fa_fwd_bc;++j)
    {
      const int k_row=kv_start+j;
      if(k_row>=seq_len)
      {
        break;
      }
      // Prefix-LM: allowed iff (k<=q) OR (k<pfx). Plain causal: k<=q.
      bool masked=false;
      if(prefix_lens!=nullptr)
      {
        masked=(k_row>q_row) && (k_row>=pfx);
      }
      else if(causal==1)
      {
        masked=(k_row>q_row);
        if(masked==true)
        {
          break;
        }
      }

      float dot=0.0f;
      for(int dd=lane_id,e=0;dd<D;dd+=32,++e)
      {
        dot+=q_reg[e]*K_tile[j*D+dd];
      }
      // Warp reduce sum
      for(int offset=16;offset>0;offset/=2)
      {
        dot+=__shfl_down_sync(0xffffffff,dot,offset);
      }
      // Broadcast from lane 0
      dot=__shfl_sync(0xffffffff,dot,0);
      if(masked==true)
      {
        s_local[j]=-INFINITY;
      }
      else
      {
        s_local[j]=dot*scale;
        if(s_local[j]>row_max)
        {
          row_max=s_local[j];
        }
      }
      num_valid=j+1;
    }

    // Online softmax rescale
    const float m_new=fmaxf(m_i,row_max);
    const float scale_old=expf(m_i-m_new);
    for(int e=0;e<elems;++e)
    {
      o_reg[e]*=scale_old;
    }

    // Exp + V accumulation from stored scores
    float block_sum=0.0f;
    for(int j=0;j<num_valid;++j)
    {
      const float p=expf(s_local[j]-m_new);
      block_sum+=p;
      for(int dd=lane_id,e=0;dd<D;dd+=32,++e)
      {
        o_reg[e]+=p*V_tile[j*D+dd];
      }
    }

    l_i=scale_old*l_i+block_sum;
    m_i=m_new;
    __syncthreads();
  }

  // Final normalization and output
  if(q_valid==true)
  {
    float inv_l=0.0f;
    if(l_i>0.0f)
    {
      inv_l=1.0f/l_i;
    }
    for(int dd=lane_id,e=0;dd<D;dd+=32,++e)
    {
      O_bh[q_row*D+dd]=T(o_reg[e]*inv_l);
    }
    // Only lane 0 writes logsumexp (one value per Q row)
    if(lane_id==0)
    {
      L_bh[q_row]=m_i+logf(l_i+1e-10f);
    }
  }
}

// Helper: launch a specific <D,BR> instantiation with opt-in shared memory
template<typename T,int D,int BR>
static void launch_fa_fwd(const T *Q,
                          const T *K,
                          const T *V,
                          T *O,
                          float *L,
                          const int batch_heads,
                          const int seq_len,
                          const float scale,
                          const int causal,
                          const uint32_t *prefix_lens,
                          const int num_heads,
                          const int num_kv_heads,
                          cudaStream_t stream)
{
  const int num_q_blocks=(seq_len+BR-1)/BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(BR*32);
  const size_t smem_size=2*g_caif_fa_fwd_bc*D*sizeof(float);

  if(smem_size>g_caif_cuda_default_shared_memory)
  {
    cudaFuncSetAttribute(
      (void *)flash_attention_forward_kernel<T,D,BR>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  }
  flash_attention_forward_kernel<T,D,BR>
    <<<grid,block,smem_size,stream>>>(Q,
                                       K,
                                       V,
                                       O,
                                       L,
                                       seq_len,
                                       scale,
                                       causal,
                                       prefix_lens,
                                       num_heads,
                                       num_kv_heads);
}

//------------------------------------------------------------------------------
// Adaptive TC/scalar dispatch for flash attention forward.
// Computes smem from formula for each (BR,BC) candidate and picks the
// largest BC that fits in the GPU's optin shared memory limit.
//------------------------------------------------------------------------------
static constexpr size_t fa_tc_smem(int d,int br,int bc)
{
  // Padded strides (+2) to eliminate smem bank conflicts for wmma loads.
  // Q_tile[BR*(D+2)] + KV_buf[BC*(D+2)] + S_tile[BR*(BC+2)] + row_max/sum[2*BR]
  const int d_pad=d+2;
  const int bc_pad=bc+2;
  return static_cast<size_t>(br*d_pad+bc*d_pad+br*bc_pad+2*br)*sizeof(float);
}

static constexpr size_t fa_scalar_smem(int d)
{
  return static_cast<size_t>(2*g_caif_fa_fwd_bc*d)*sizeof(float);
}

// Compute optimal number of warps for a given TC tile configuration.
// max_tiles: largest of S-tiles and O-tiles that warps must cover.
// blocks_from_smem: how many blocks the SM can hold from shared memory alone.
// If 2+ blocks fit, halve warps per block to double occupancy.
static int fa_tc_optimal_nw(int max_tiles,
                            int blocks_from_smem,
                            int tiles_m,
                            int tiles_n_s,
                            int tiles_n_o)
{
  int nw;
  if(blocks_from_smem>=2 && max_tiles>=4)
  {
    nw=max_tiles/2;
    if(nw<2)
    {
      nw=2;
    }
  }
  else if(max_tiles>8)
  {
    nw=8;
  }
  else if(max_tiles<2)
  {
    nw=2;
  }
  else
  {
    nw=max_tiles;
  }

  // Ensure register softmax constraints:
  // nw % tiles_m == 0, tiles_n_s and tiles_n_o divisible by warps_per_m
  while(nw>2)
  {
    if(nw%tiles_m==0)
    {
      const int wpm=nw/tiles_m;
      if(tiles_n_s%wpm==0 && tiles_n_o%wpm==0)
      {
        break;
      }
    }
    nw/=2;
  }

  return nw;
}

// Dispatch helper that selects NW from the computed optimal value.
// Instantiates NW=4 and NW=8; runtime picks the closer one.
template<typename T,int D,int BR,int BC>
static void dispatch_fa_fwd_tc_nw(const T *Q,
                                  const T *K,
                                  const T *V,
                                  T *O,
                                  float *L,
                                  const int batch_heads,
                                  const int seq_len,
                                  const float scale,
                                  const int causal,
                                  const uint32_t *prefix_lens,
                                  const int num_heads,
                                  const int num_kv_heads,
                                  cudaStream_t stream,
                                  const int nw)
{
  if(nw<=2)
  {
    launch_fa_fwd_tc<T,D,BR,BC,2>(Q,
                                K,
                                V,
                                O,
                                L,
                                batch_heads,
                                seq_len,
                                scale,
                                causal,
                                prefix_lens,
                                num_heads,
                                num_kv_heads,
                                stream);
  }
  else if(nw<=4)
  {
    launch_fa_fwd_tc<T,D,BR,BC,4>(Q,
                                K,
                                V,
                                O,
                                L,
                                batch_heads,
                                seq_len,
                                scale,
                                causal,
                                prefix_lens,
                                num_heads,
                                num_kv_heads,
                                stream);
  }
  else
  {
    launch_fa_fwd_tc<T,D,BR,BC,8>(Q,
                                K,
                                V,
                                O,
                                L,
                                batch_heads,
                                seq_len,
                                scale,
                                causal,
                                prefix_lens,
                                num_heads,
                                num_kv_heads,
                                stream);
  }
}

template<typename T,int D>
static void dispatch_fa_fwd_tc(const T *Q,
                               const T *K,
                               const T *V,
                               T *O,
                               float *L,
                               const int batch_heads,
                               const int seq_len,
                               const float scale,
                               const int causal,
                               const uint32_t *prefix_lens,
                               const int num_heads,
                               const int num_kv_heads,
                               cudaStream_t stream,
                               const int cc_major,
                               const size_t smem_limit,
                               const size_t smem_per_sm,
                               const int max_threads)
{
  // TC path (sm_80+): try largest BC first, then largest BR.
  // For each candidate, compute smem and optimal NW from GPU properties.
  if(cc_major>=8)
  {
    // Candidate tile sizes in priority order (largest BC first)
    constexpr int br_opts[]={32,16,32,16};
    constexpr int bc_opts[]={128,128,64,64};

    for(int c=0;c<4;++c)
    {
      const int br=br_opts[c];
      const int bc=bc_opts[c];
      const size_t smem=fa_tc_smem(D,br,bc);
      if(smem>smem_limit)
      {
        continue;
      }

      const int tiles_s=(br/16)*(bc/16);
      const int tiles_o=(br/16)*(D/16);
      int max_tiles=tiles_s;
      if(tiles_o>tiles_s)
      {
        max_tiles=tiles_o;
      }
      const int blocks_from_smem=static_cast<int>(smem_per_sm/smem);
      const int tiles_m=br/16;
      const int tiles_n_s=bc/16;
      const int tiles_n_o=D/16;
      const int nw=fa_tc_optimal_nw(max_tiles,blocks_from_smem,tiles_m,tiles_n_s,tiles_n_o);

      // Dispatch to the matching (BR,BC) template with computed NW
      if(br==32 && bc==128)
      {
        dispatch_fa_fwd_tc_nw<T,D,32,128>(Q,
                                        K,
                                        V,
                                        O,
                                        L,
                                        batch_heads,
                                        seq_len,
                                        scale,
                                        causal,
                                        prefix_lens,
                                        num_heads,
                                        num_kv_heads,
                                        stream,
                                        nw);
        return;
      }
      if(br==16 && bc==128)
      {
        dispatch_fa_fwd_tc_nw<T,D,16,128>(Q,
                                        K,
                                        V,
                                        O,
                                        L,
                                        batch_heads,
                                        seq_len,
                                        scale,
                                        causal,
                                        prefix_lens,
                                        num_heads,
                                        num_kv_heads,
                                        stream,
                                        nw);
        return;
      }
      if(br==32 && bc==64)
      {
        dispatch_fa_fwd_tc_nw<T,D,32,64>(Q,
                                       K,
                                       V,
                                       O,
                                       L,
                                       batch_heads,
                                       seq_len,
                                       scale,
                                       causal,
                                       prefix_lens,
                                       num_heads,
                                       num_kv_heads,
                                       stream,
                                       nw);
        return;
      }
      if(br==16 && bc==64)
      {
        dispatch_fa_fwd_tc_nw<T,D,16,64>(Q,
                                       K,
                                       V,
                                       O,
                                       L,
                                       batch_heads,
                                       seq_len,
                                       scale,
                                       causal,
                                       prefix_lens,
                                       num_heads,
                                       num_kv_heads,
                                       stream,
                                       nw);
        return;
      }
    }
  }

  // Scalar warp-per-row fallback (all architectures)
  if(fa_scalar_smem(D)<=smem_limit && max_threads>=256)
  {
    launch_fa_fwd<T,D,8>(Q,
                       K,
                       V,
                       O,
                       L,
                       batch_heads,
                       seq_len,
                       scale,
                       causal,
                       prefix_lens,
                       num_heads,
                       num_kv_heads,
                       stream);
  }
  else if(fa_scalar_smem(D)<=smem_limit && max_threads>=128)
  {
    launch_fa_fwd<T,D,4>(Q,
                       K,
                       V,
                       O,
                       L,
                       batch_heads,
                       seq_len,
                       scale,
                       causal,
                       prefix_lens,
                       num_heads,
                       num_kv_heads,
                       stream);
  }
  else
  {
    launch_fa_fwd<T,D,2>(Q,
                       K,
                       V,
                       O,
                       L,
                       batch_heads,
                       seq_len,
                       scale,
                       causal,
                       prefix_lens,
                       num_heads,
                       num_kv_heads,
                       stream);
  }
}

// Shared dispatch body for forward launchers (causal + prefix variants).
template<typename T>
static void dispatch_flash_fwd(const T *Q,
                               const T *K,
                               const T *V,
                               T *O,
                               float *L,
                               const int batch_heads,
                               const int seq_len,
                               const int head_dim,
                               const float scale,
                               const int causal,
                               const uint32_t *prefix_lens,
                               const int num_heads,
                               const int num_kv_heads,
                               cudaStream_t stream)
{
  int device_id=0;
  cudaGetDevice(&device_id);
  int max_smem_optin=g_caif_cuda_default_shared_memory;
  cudaDeviceGetAttribute(&max_smem_optin,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device_id);
  int smem_per_sm_int=g_caif_cuda_default_shared_memory;
  cudaDeviceGetAttribute(&smem_per_sm_int,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                         device_id);
  int max_threads=g_caif_cuda_max_threads_fallback;
  cudaDeviceGetAttribute(&max_threads,
                         cudaDevAttrMaxThreadsPerBlock,
                         device_id);
  int cc_major=0;
  cudaDeviceGetAttribute(&cc_major,
                         cudaDevAttrComputeCapabilityMajor,
                         device_id);

  const size_t smem_limit=static_cast<size_t>(max_smem_optin);
  const size_t smem_per_sm=static_cast<size_t>(smem_per_sm_int);

  switch(head_dim)
  {
    case 32:
      dispatch_fa_fwd_tc<T,32>(Q,
                             K,
                             V,
                             O,
                             L,
                             batch_heads,
                             seq_len,
                             scale,
                             causal,
                             prefix_lens,
                             num_heads,
                             num_kv_heads,
                             stream,
                             cc_major,
                             smem_limit,
                             smem_per_sm,
                             max_threads);
      break;
    case 64:
      dispatch_fa_fwd_tc<T,64>(Q,
                             K,
                             V,
                             O,
                             L,
                             batch_heads,
                             seq_len,
                             scale,
                             causal,
                             prefix_lens,
                             num_heads,
                             num_kv_heads,
                             stream,
                             cc_major,
                             smem_limit,
                             smem_per_sm,
                             max_threads);
      break;
    case 80:
      dispatch_fa_fwd_tc<T,80>(Q,
                             K,
                             V,
                             O,
                             L,
                             batch_heads,
                             seq_len,
                             scale,
                             causal,
                             prefix_lens,
                             num_heads,
                             num_kv_heads,
                             stream,
                             cc_major,
                             smem_limit,
                             smem_per_sm,
                             max_threads);
      break;
    case 96:
      dispatch_fa_fwd_tc<T,96>(Q,
                             K,
                             V,
                             O,
                             L,
                             batch_heads,
                             seq_len,
                             scale,
                             causal,
                             prefix_lens,
                             num_heads,
                             num_kv_heads,
                             stream,
                             cc_major,
                             smem_limit,
                             smem_per_sm,
                             max_threads);
      break;
    case 128:
      dispatch_fa_fwd_tc<T,128>(Q,
                              K,
                              V,
                              O,
                              L,
                              batch_heads,
                              seq_len,
                              scale,
                              causal,
                              prefix_lens,
                              num_heads,
                              num_kv_heads,
                              stream,
                              cc_major,
                              smem_limit,
                              smem_per_sm,
                              max_threads);
      break;
    default:
      fprintf(stderr,
              "FATAL: flash_attention_forward unsupported head_dim=%d"
              " (supported: 32,64,80,96,128). Use standard attention.\n",
              head_dim);
      abort();
  }
}

// Launch wrappers must have C linkage for header declaration
// (former extern "C" block — C++ linkage used for dtype templates)

// Launch wrapper for FlashAttention forward (causal / no-prefix path).
template<typename T>
void launch_flash_attention_forward(const T *Q,
                                    const T *K,
                                    const T *V,
                                    T *O,
                                    float *L,
                                    const int batch_heads,
                                    const int seq_len,
                                    const int head_dim,
                                    const float scale,
                                    const int causal,
                                    const int num_heads,
                                    const int num_kv_heads,
                                    cudaStream_t stream)
{
  dispatch_flash_fwd<T>(Q,
                     K,
                     V,
                     O,
                     L,
                     batch_heads,
                     seq_len,
                     head_dim,
                     scale,
                     causal,
                     nullptr,
                     num_heads,
                     num_kv_heads,
                     stream);
}
template void launch_flash_attention_forward<float>(const float *,
                                                    const float *,
                                                    const float *,
                                                    float *,
                                                    float *,
                                                    int,
                                                    int,
                                                    int,
                                                    float,
                                                    int,
                                                    int,
                                                    int,
                                                    cudaStream_t);
template void launch_flash_attention_forward<__half>(const __half *,
                                                     const __half *,
                                                     const __half *,
                                                     __half *,
                                                     float *,
                                                     int,
                                                     int,
                                                     int,
                                                     float,
                                                     int,
                                                     int,
                                                     int,
                                                     cudaStream_t);
template void launch_flash_attention_forward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                            const __nv_bfloat16 *,
                                                            const __nv_bfloat16 *,
                                                            __nv_bfloat16 *,
                                                            float *,
                                                            int,
                                                            int,
                                                            int,
                                                            float,
                                                            int,
                                                            int,
                                                            int,
                                                            cudaStream_t);

// Launch wrapper for FlashAttention forward with prefix-LM mask.
// prefix_lens: device pointer, length = batch_size (int32 per batch).
// num_heads: used to map blockIdx.x (batch_heads) back to the batch index.
// Allowed iff (k<=q) OR (k<prefix_lens[batch]).
template<typename T>
void launch_flash_attention_forward_prefix(const T *Q,
                                           const T *K,
                                           const T *V,
                                           T *O,
                                           float *L,
                                           const uint32_t *prefix_lens,
                                           const int batch_size,
                                           const int num_heads,
                                           const int num_kv_heads,
                                           const int seq_len,
                                           const int head_dim,
                                           const float scale,
                                           cudaStream_t stream)
{
  // Prefix-LM always uses causal+prefix masking internally (causal=1).
  dispatch_flash_fwd<T>(Q,
                     K,
                     V,
                     O,
                     L,
                     batch_size*num_heads,
                     seq_len,
                     head_dim,
                     scale,
                     1,
                     prefix_lens,
                     num_heads,
                     num_kv_heads,
                     stream);
}
template void launch_flash_attention_forward_prefix<float>(const float *,
                                                           const float *,
                                                           const float *,
                                                           float *,
                                                           float *,
                                                           const uint32_t *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           float,
                                                           cudaStream_t);
template void launch_flash_attention_forward_prefix<__half>(const __half *,
                                                            const __half *,
                                                            const __half *,
                                                            __half *,
                                                            float *,
                                                            const uint32_t *,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            float,
                                                            cudaStream_t);
template void launch_flash_attention_forward_prefix<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   __nv_bfloat16 *,
                                                                   float *,
                                                                   const uint32_t *,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   float,
                                                                   cudaStream_t);

// end of former extern "C" block

//------------------------------------------------------------------------------
// FlashAttention-2 Backward: Precompute Di = dot(dO, O) per row
//------------------------------------------------------------------------------
template<typename T,int D>
__global__ void flash_attention_precompute_di_kernel(const T *__restrict__ dO,
                                                      const T *__restrict__ O,
                                                      float *__restrict__ Di,
                                                      const int seq_len)
{
  const int bh=blockIdx.x;
  const int row=blockIdx.y*blockDim.x+threadIdx.x;

  if(row>=seq_len)
  {
    return;
  }

  const T *dO_row=dO+bh*seq_len*D+row*D;
  const T *O_row=O+bh*seq_len*D+row*D;

  float sum=0.0f;
  for(int d=0;d<D;++d)
  {
    sum+=float(dO_row[d])*float(O_row[d]);
  }
  Di[bh*seq_len+row]=sum;
}

//------------------------------------------------------------------------------
// FlashAttention-2 Backward Kernel - dK/dV (optimized)
// Uses precomputed Di, template-sized register arrays, 128 thread blocks
//------------------------------------------------------------------------------
constexpr int g_caif_fa_bwd_br=64;     // Q tile rows for dK/dV kernel — must match caif_constants.h
constexpr int g_caif_fa_bwd_bc=128;    // K/V block size (threads per block) — must match caif_constants.h
constexpr int g_caif_fa_bwd_dq_br=128; // Q block size for dQ kernel — must match caif_constants.h
constexpr int g_caif_fa_bwd_dq_bc=64;  // K/V tile rows for dQ kernel — must match caif_constants.h

template<typename T,int D>
__global__ void flash_attention_backward_kernel(const T *__restrict__ Q,
                                                 const T *__restrict__ K,
                                                 const T *__restrict__ V,
                                                 const T *__restrict__ dO,
                                                 const float *__restrict__ L,
                                                 const float *__restrict__ Di,
                                                 T *__restrict__ dK,
                                                 T *__restrict__ dV,
                                                 const int seq_len,
                                                 const float scale,
                                                 const int causal,
                                                 const uint32_t *__restrict__ prefix_lens,
                                                 const int num_heads)
{
  const int bh=blockIdx.x;
  const int kv_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  int pfx=0;
  if(prefix_lens!=nullptr)
  {
    pfx=prefix_lens[bh/num_heads];
  }

  const int kv_row=kv_block_idx*g_caif_fa_bwd_bc+tid;
  const int active=(kv_row<seq_len);

  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *dO_tile=smem+g_caif_fa_bwd_br*D;
  float *L_tile=smem+2*g_caif_fa_bwd_br*D;
  float *Di_tile=smem+2*g_caif_fa_bwd_br*D+g_caif_fa_bwd_br;

  const T *Q_bh=Q+bh*seq_len*D;
  const T *K_bh=K+bh*seq_len*D;
  const T *V_bh=V+bh*seq_len*D;
  const T *dO_bh=dO+bh*seq_len*D;
  const float *L_bh=L+bh*seq_len;
  const float *Di_bh=Di+bh*seq_len;
  T *dK_bh=dK+bh*seq_len*D;
  T *dV_bh=dV+bh*seq_len*D;

  // Load K and V rows into registers (sized exactly to D)
  float K_row[D];
  float V_row[D];
  if(active)
  {
    for(int d=0;d<D;++d)
    {
      K_row[d]=float(K_bh[kv_row*D+d]);
      V_row[d]=float(V_bh[kv_row*D+d]);
    }
  }

  float dK_acc[D];
  float dV_acc[D];
  for(int d=0;d<D;++d)
  {
    dK_acc[d]=0.0f;
    dV_acc[d]=0.0f;
  }

  const int num_q_blocks=(seq_len+g_caif_fa_bwd_br-1)/g_caif_fa_bwd_br;

  int start_q_block=0;
  if(causal && active && prefix_lens==nullptr)
  {
    start_q_block=kv_row/g_caif_fa_bwd_br;
  }
  // Prefix mode: a KV row with k<pfx is attended to by every Q row, so we
  // can't skip early Q blocks. Iterate all blocks, mask per-pair below.

  for(int q_block=start_q_block;q_block<num_q_blocks;++q_block)
  {
    const int q_start=q_block*g_caif_fa_bwd_br;

    // Cooperatively load Q tile and dO tile (all threads participate)
    for(int i=tid;i<g_caif_fa_bwd_br*D;i+=g_caif_fa_bwd_bc)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=q_start+row;

      if(global_row<seq_len)
      {
        Q_tile[row*D+col]=float(Q_bh[global_row*D+col]);
        dO_tile[row*D+col]=float(dO_bh[global_row*D+col]);
      }
      else
      {
        Q_tile[row*D+col]=0.0f;
        dO_tile[row*D+col]=0.0f;
      }
    }

    // Load L and precomputed Di
    if(tid<g_caif_fa_bwd_br)
    {
      const int global_row=q_start+tid;
      if(global_row<seq_len)
      {
        L_tile[tid]=L_bh[global_row];
        Di_tile[tid]=Di_bh[global_row];
      }
      else
      {
        L_tile[tid]=-INFINITY;
        Di_tile[tid]=0.0f;
      }
    }
    __syncthreads();

    if(active)
    {
      for(int qi=0;qi<g_caif_fa_bwd_br;++qi)
      {
        const int q_row=q_start+qi;

        // Prefix-LM: allowed iff (kv<=q) OR (kv<pfx). Plain causal: kv<=q.
        // pfx==0 when prefix_lens is null, reducing exactly to causal.
        if(causal && kv_row>q_row && kv_row>=pfx)
        {
          continue;
        }

        if(q_row>=seq_len)
        {
          continue;
        }

        float dot=0.0f;
        for(int d=0;d<D;++d)
        {
          dot+=Q_tile[qi*D+d]*K_row[d];
        }
        const float S_val=dot*scale;
        const float P_val=expf(fminf(S_val-L_tile[qi],0.0f));

        for(int d=0;d<D;++d)
        {
          dV_acc[d]+=P_val*dO_tile[qi*D+d];
        }

        float dP=0.0f;
        for(int d=0;d<D;++d)
        {
          dP+=dO_tile[qi*D+d]*V_row[d];
        }

        const float dS=P_val*(dP-Di_tile[qi])*scale;

        for(int d=0;d<D;++d)
        {
          dK_acc[d]+=dS*Q_tile[qi*D+d];
        }
      }
    }
    __syncthreads();
  }

  if(active)
  {
    for(int d=0;d<D;++d)
    {
      dK_bh[kv_row*D+d]=T(dK_acc[d]);
      dV_bh[kv_row*D+d]=T(dV_acc[d]);
    }
  }
}

//------------------------------------------------------------------------------
// FlashAttention-2 Backward Kernel - dQ (optimized)
// Uses precomputed Di, template-sized register arrays, 128 thread blocks
//------------------------------------------------------------------------------
template<typename T,int D>
__global__ void flash_attention_backward_dq_kernel(const T *__restrict__ Q,
                                                    const T *__restrict__ K,
                                                    const T *__restrict__ V,
                                                    const T *__restrict__ dO,
                                                    const float *__restrict__ L,
                                                    const float *__restrict__ Di,
                                                    T *__restrict__ dQ,
                                                    const int seq_len,
                                                    const float scale,
                                                    const int causal,
                                                    const uint32_t *__restrict__ prefix_lens,
                                                    const int num_heads)
{
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  int pfx=0;
  if(prefix_lens!=nullptr)
  {
    pfx=prefix_lens[bh/num_heads];
  }

  const int q_row=q_block_idx*g_caif_fa_bwd_dq_br+tid;
  const int active=(q_row<seq_len);

  extern __shared__ float smem[];
  float *K_tile=smem;
  float *V_tile=smem+g_caif_fa_bwd_dq_bc*D;

  const T *Q_bh=Q+bh*seq_len*D;
  const T *K_bh=K+bh*seq_len*D;
  const T *V_bh=V+bh*seq_len*D;
  const T *dO_bh=dO+bh*seq_len*D;
  const float *L_bh=L+bh*seq_len;
  const float *Di_bh=Di+bh*seq_len;
  T *dQ_bh=dQ+bh*seq_len*D;

  // Load Q row and dO row (sized exactly to D)
  float Q_row_reg[D];
  float dO_row[D];
  float L_val=0.0f;
  float Di_val=0.0f;
  if(active)
  {
    for(int d=0;d<D;++d)
    {
      Q_row_reg[d]=float(Q_bh[q_row*D+d]);
      dO_row[d]=float(dO_bh[q_row*D+d]);
    }
    L_val=L_bh[q_row];
    Di_val=Di_bh[q_row];
  }

  float dQ_acc[D];
  for(int d=0;d<D;++d)
  {
    dQ_acc[d]=0.0f;
  }

  int num_kv_blocks=(seq_len+g_caif_fa_bwd_dq_bc-1)/g_caif_fa_bwd_dq_bc;
  if(causal && active && prefix_lens==nullptr)
  {
    num_kv_blocks=min(num_kv_blocks,(q_row/g_caif_fa_bwd_dq_bc)+1);
  }

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*g_caif_fa_bwd_dq_bc;

    // Load K and V tiles cooperatively (all threads participate)
    for(int i=tid;i<g_caif_fa_bwd_dq_bc*D;i+=g_caif_fa_bwd_dq_br)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;

      if(global_row<seq_len)
      {
        K_tile[row*D+col]=float(K_bh[global_row*D+col]);
        V_tile[row*D+col]=float(V_bh[global_row*D+col]);
      }
      else
      {
        K_tile[row*D+col]=0.0f;
        V_tile[row*D+col]=0.0f;
      }
    }
    __syncthreads();

    if(active)
    {
      for(int j=0;j<g_caif_fa_bwd_dq_bc;++j)
      {
        const int k_row=kv_start+j;

        if(causal && k_row>q_row && k_row>=pfx)
        {
          continue;
        }

        if(k_row>=seq_len)
        {
          continue;
        }

        float dot=0.0f;
        for(int d=0;d<D;++d)
        {
          dot+=Q_row_reg[d]*K_tile[j*D+d];
        }
        const float S_val=dot*scale;
        const float P_val=expf(fminf(S_val-L_val,0.0f));

        float dP=0.0f;
        for(int d=0;d<D;++d)
        {
          dP+=dO_row[d]*V_tile[j*D+d];
        }

        const float dS=P_val*(dP-Di_val)*scale;

        for(int d=0;d<D;++d)
        {
          dQ_acc[d]+=dS*K_tile[j*D+d];
        }
      }
    }
    __syncthreads();
  }

  if(active)
  {
    for(int d=0;d<D;++d)
    {
      dQ_bh[q_row*D+d]=T(dQ_acc[d]);
    }
  }
}

template<typename T>
static void dispatch_flash_bwd(const T *Q,
                               const T *K,
                               const T *V,
                               const T *O,
                               const T *dO,
                               const float *L,
                               T *dQ,
                               T *dK,
                               T *dV,
                               const int batch_heads,
                               const int seq_len,
                               const int head_dim,
                               const float scale,
                               const int causal,
                               const uint32_t *prefix_lens,
                               const int num_heads,
                               cudaStream_t stream)
{
  // Allocate temporary Di buffer [batch_heads, seq_len]
  float *Di_buf=nullptr;
  cudaMallocAsync(reinterpret_cast<void **>(&Di_buf),
                  static_cast<size_t>(batch_heads)*seq_len*sizeof(float),stream);

  // Kernel 0: Precompute Di = dot(dO, O) for each row
  {
    const int block_size=g_caif_cuda_block_size;
    const int rows_per_grid=(seq_len+block_size-1)/block_size;
    dim3 grid(batch_heads,rows_per_grid);

    switch(head_dim)
    {
      case 32:
        flash_attention_precompute_di_kernel<T,32><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case 64:
        flash_attention_precompute_di_kernel<T,64><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case 80:
        flash_attention_precompute_di_kernel<T,80><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case 96:
        flash_attention_precompute_di_kernel<T,96><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      case 128:
        flash_attention_precompute_di_kernel<T,128><<<grid,block_size,0,stream>>>(dO,O,Di_buf,seq_len);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 1: Compute dK and dV (128 threads/block)
  {
    const int num_kv_blocks=(seq_len+g_caif_fa_bwd_bc-1)/g_caif_fa_bwd_bc;
    dim3 grid(batch_heads,num_kv_blocks);
    dim3 block(g_caif_fa_bwd_bc);
    const size_t smem_size=(2*g_caif_fa_bwd_br*head_dim+2*g_caif_fa_bwd_br)*sizeof(float);

    switch(head_dim)
    {
      case 32:
        flash_attention_backward_kernel<T,32>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      case 64:
        flash_attention_backward_kernel<T,64>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      case 80:
        cudaFuncSetAttribute((void *)flash_attention_backward_kernel<T,80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_kernel<T,80>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      case 96:
        cudaFuncSetAttribute((void *)flash_attention_backward_kernel<T,96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_kernel<T,96>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      case 128:
        cudaFuncSetAttribute((void *)flash_attention_backward_kernel<T,128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_kernel<T,128>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dK,
                                            dV,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 2: Compute dQ (128 threads/block)
  {
    const int num_q_blocks=(seq_len+g_caif_fa_bwd_dq_br-1)/g_caif_fa_bwd_dq_br;
    dim3 grid(batch_heads,num_q_blocks);
    dim3 block(g_caif_fa_bwd_dq_br);
    const size_t smem_size=2*g_caif_fa_bwd_dq_bc*head_dim*sizeof(float);

    switch(head_dim)
    {
      case 32:
        flash_attention_backward_dq_kernel<T,32>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      case 64:
        flash_attention_backward_dq_kernel<T,64>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      case 80:
        cudaFuncSetAttribute((void *)flash_attention_backward_dq_kernel<T,80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_dq_kernel<T,80>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      case 96:
        cudaFuncSetAttribute((void *)flash_attention_backward_dq_kernel<T,96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_dq_kernel<T,96>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      case 128:
        cudaFuncSetAttribute((void *)flash_attention_backward_dq_kernel<T,128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_dq_kernel<T,128>
          <<<grid,block,smem_size,stream>>>(Q,
                                            K,
                                            V,
                                            dO,
                                            L,
                                            Di_buf,
                                            dQ,
                                            seq_len,
                                            scale,
                                            causal,
                                            prefix_lens,
                                            num_heads);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward_dq unsupported head_dim=%d\n",
                head_dim);
        abort();
        break;
    }
  }

  // Free temporary Di buffer (enqueue on stream for proper ordering)
  cudaFreeAsync(Di_buf,stream);
}

// (former extern "C" block — C++ linkage used for dtype templates)

// Launch wrapper for FlashAttention backward (causal / non-causal)
template<typename T>
void launch_flash_attention_backward(const T *Q,
                                     const T *K,
                                     const T *V,
                                     const T *O,
                                     const T *dO,
                                     const float *L,
                                     T *dQ,
                                     T *dK,
                                     T *dV,
                                     const int batch_heads,
                                     const int seq_len,
                                     const int head_dim,
                                     const float scale,
                                     const int causal,
                                     cudaStream_t stream)
{
  dispatch_flash_bwd<T>(Q,
                     K,
                     V,
                     O,
                     dO,
                     L,
                     dQ,
                     dK,
                     dV,
                     batch_heads,
                     seq_len,
                     head_dim,
                     scale,
                     causal,
                     nullptr,
                     0,
                     stream);
}
template void launch_flash_attention_backward<float>(const float *,
                                                     const float *,
                                                     const float *,
                                                     const float *,
                                                     const float *,
                                                     const float *,
                                                     float *,
                                                     float *,
                                                     float *,
                                                     int,
                                                     int,
                                                     int,
                                                     float,
                                                     int,
                                                     cudaStream_t);
template void launch_flash_attention_backward<__half>(const __half *,
                                                      const __half *,
                                                      const __half *,
                                                      const __half *,
                                                      const __half *,
                                                      const float *,
                                                      __half *,
                                                      __half *,
                                                      __half *,
                                                      int,
                                                      int,
                                                      int,
                                                      float,
                                                      int,
                                                      cudaStream_t);
template void launch_flash_attention_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             const __nv_bfloat16 *,
                                                             const float *,
                                                             __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             __nv_bfloat16 *,
                                                             int,
                                                             int,
                                                             int,
                                                             float,
                                                             int,
                                                             cudaStream_t);

// Launch wrapper for FlashAttention backward with prefix-LM mask
template<typename T>
void launch_flash_attention_backward_prefix(const T *Q,
                                            const T *K,
                                            const T *V,
                                            const T *O,
                                            const T *dO,
                                            const float *L,
                                            T *dQ,
                                            T *dK,
                                            T *dV,
                                            const uint32_t *prefix_lens,
                                            const int batch_size,
                                            const int num_heads,
                                            const int seq_len,
                                            const int head_dim,
                                            const float scale,
                                            cudaStream_t stream)
{
  dispatch_flash_bwd<T>(Q,
                     K,
                     V,
                     O,
                     dO,
                     L,
                     dQ,
                     dK,
                     dV,
                     batch_size*num_heads,
                     seq_len,
                     head_dim,
                     scale,
                     1,
                     prefix_lens,
                     num_heads,
                     stream);
}
template void launch_flash_attention_backward_prefix<float>(const float *,
                                                            const float *,
                                                            const float *,
                                                            const float *,
                                                            const float *,
                                                            const float *,
                                                            float *,
                                                            float *,
                                                            float *,
                                                            const uint32_t *,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            float,
                                                            cudaStream_t);
template void launch_flash_attention_backward_prefix<__half>(const __half *,
                                                             const __half *,
                                                             const __half *,
                                                             const __half *,
                                                             const __half *,
                                                             const float *,
                                                             __half *,
                                                             __half *,
                                                             __half *,
                                                             const uint32_t *,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             float,
                                                             cudaStream_t);
template void launch_flash_attention_backward_prefix<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const __nv_bfloat16 *,
                                                                    const float *,
                                                                    __nv_bfloat16 *,
                                                                    __nv_bfloat16 *,
                                                                    __nv_bfloat16 *,
                                                                    const uint32_t *,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    float,
                                                                    cudaStream_t);

// end of former extern "C" block (flash attention backward)

//==============================================================================
// FlashAttention-2 Cross-Attention Kernels
//
// Identical algorithm to self-attention but Q has q_seq_len while K/V have
// kv_seq_len (different lengths). No causal mask — decoder attends to all
// encoder positions.
//==============================================================================

//------------------------------------------------------------------------------
// Cross-Attention Forward — Tensor Core Kernel
//------------------------------------------------------------------------------
template<typename T,int D,int BR,int BC,int NW>
__global__ void flash_attention_forward_cross_tc_kernel(const T *__restrict__ Q,
                                                        const T *__restrict__ K,
                                                        const T *__restrict__ V,
                                                        T *__restrict__ O,
                                                        float *__restrict__ L,
                                                        const int q_seq_len,
                                                        const int kv_seq_len,
                                                        const float scale)
{
#if CAIF_HAS_TC_FLASH
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  const int lane_id=tid%32;
  const int warp_id=tid/32;

  constexpr int n_warps=NW;
  constexpr int tiles_m=BR/16;
  constexpr int tiles_n_s=BC/16;
  constexpr int tiles_n_o=D/16;
  constexpr int block_threads=n_warps*32;

  constexpr int d_pad=D+2;
  constexpr int bc_pad=BC+2;
  constexpr int d_f2=D/2;
  constexpr int d_pad_f2=d_pad/2;

  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *KV_buf=Q_tile+BR*d_pad;
  float *S_tile=KV_buf+BC*d_pad;
  float *row_max_arr=S_tile+BR*bc_pad;
  float *row_sum_arr=row_max_arr+BR;

  const T *Q_bh=Q+bh*q_seq_len*D;
  const T *K_bh=K+bh*kv_seq_len*D;
  const T *V_bh=V+bh*kv_seq_len*D;
  T *O_bh=O+bh*q_seq_len*D;
  float *L_bh=L+bh*q_seq_len;

  const int q_start=q_block_idx*BR;

  constexpr int warps_per_m=n_warps/tiles_m;
  constexpr int s_tiles_pw=(tiles_n_s>=warps_per_m)*(tiles_n_s/warps_per_m);
  constexpr int o_tiles_pw=(tiles_n_o>=warps_per_m)*(tiles_n_o/warps_per_m);
  constexpr int s_arr=s_tiles_pw+(!s_tiles_pw);
  constexpr int o_arr=o_tiles_pw+(!o_tiles_pw);
  const int m_idx=warp_id/warps_per_m;
  const int group_warp=warp_id%warps_per_m;
  const int n_start_s=group_warp*s_tiles_pw;
  const int n_start_o=group_warp*o_tiles_pw;
  const int group_base=m_idx*warps_per_m;

  wmma::fragment<wmma::accumulator,16,16,8,float> o_frags[o_arr];
  for(int t=0;t<o_tiles_pw;++t)
  {
    wmma::fill_fragment(o_frags[t],0.0f);
  }

  // Load Q_tile from Q (q_seq_len bounded)
  const int valid_q_rows=min(BR,q_seq_len-q_start);
  {
    if constexpr(sizeof(T)==4)
    {
      const int valid_q_f2=max(valid_q_rows,0)*d_f2;
      constexpr int total_q_f2=BR*d_f2;
      const float2 *Q_src2=reinterpret_cast<const float2 *>(Q_bh+q_start*D);
      float2 *Q_dst2=reinterpret_cast<float2 *>(Q_tile);
      for(int i=tid;i<valid_q_f2;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        Q_dst2[row*d_pad_f2+f2c]=Q_src2[i];
      }
      const float2 zero2=make_float2(0.0f,0.0f);
      for(int i=valid_q_f2+tid;i<total_q_f2;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        Q_dst2[row*d_pad_f2+f2c]=zero2;
      }
    }
    else
    {
      const int valid_q=max(valid_q_rows,0)*D;
      constexpr int total_q=BR*D;
      for(int i=tid;i<valid_q;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        Q_tile[row*d_pad+col]=float(Q_bh[(q_start+row)*D+col]);
      }
      for(int i=valid_q+tid;i<total_q;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        Q_tile[row*d_pad+col]=0.0f;
      }
    }
  }

  for(int i=tid;i<BR;i+=block_threads)
  {
    row_max_arr[i]=-INFINITY;
    row_sum_arr[i]=0.0f;
  }
  __syncthreads();

  // KV blocks iterate over kv_seq_len (no causal limit)
  const int num_kv_blocks=(kv_seq_len+BC-1)/BC;
  constexpr int kv_f2=BC*d_f2;

  // Pipeline: prefetch K[0]
  float2 *KV_dst2=reinterpret_cast<float2 *>(KV_buf);
  if(num_kv_blocks>0)
  {
    if constexpr(sizeof(T)==4)
    {
      const int kv0_valid=min(BC,kv_seq_len)*d_f2;
      const float2 *K0_src2=reinterpret_cast<const float2 *>(K_bh);
      for(int i=tid;i<kv0_valid;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        cp_async_f2(&KV_dst2[row*d_pad_f2+f2c],&K0_src2[i]);
      }
      const float2 zero2=make_float2(0.0f,0.0f);
      for(int i=kv0_valid+tid;i<kv_f2;i+=block_threads)
      {
        const int row=i/d_f2;
        const int f2c=i-row*d_f2;
        KV_dst2[row*d_pad_f2+f2c]=zero2;
      }
    }
    else
    {
      const int kv0_valid=min(BC,kv_seq_len)*D;
      constexpr int kv_total=BC*D;
      for(int i=tid;i<kv0_valid;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        KV_buf[row*d_pad+col]=float(K_bh[row*D+col]);
      }
      for(int i=kv0_valid+tid;i<kv_total;i+=block_threads)
      {
        const int row=i/D;
        const int col=i-row*D;
        KV_buf[row*d_pad+col]=0.0f;
      }
    }
    cp_async_commit();
  }

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*BC;
    const int valid_kv_rows=min(BC,kv_seq_len-kv_start);
    const int valid_kv_f2=valid_kv_rows*d_f2;

    // PHASE 1: Wait for K, compute S = Q @ K^T
    cp_async_wait();
    __syncthreads();

    wmma::fragment<wmma::accumulator,16,16,8,float> s_accs[s_arr];
    for(int t=0;t<s_tiles_pw;++t)
    {
      const int n=n_start_s+t;
      wmma::fill_fragment(s_accs[t],0.0f);
      for(int k=0;k<D/8;++k)
      {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> k_frag;
        wmma::load_matrix_sync(q_frag,&Q_tile[m_idx*16*d_pad+k*8],d_pad);
        wmma::load_matrix_sync(k_frag,&KV_buf[n*16*d_pad+k*8],d_pad);
        wmma::mma_sync(s_accs[t],q_frag,k_frag,s_accs[t]);
      }
      for(int i=0;i<s_accs[t].num_elements;++i)
      {
        s_accs[t].x[i]*=scale;
      }
    }

    __syncthreads();

    // Async V load (overlapped with softmax)
    {
      if constexpr(sizeof(T)==4)
      {
        const float2 *V_src2=reinterpret_cast<const float2 *>(V_bh+kv_start*D);
        for(int i=tid;i<valid_kv_f2;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          cp_async_f2(&KV_dst2[row*d_pad_f2+f2c],&V_src2[i]);
        }
        const float2 zero2=make_float2(0.0f,0.0f);
        for(int i=valid_kv_f2+tid;i<kv_f2;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          KV_dst2[row*d_pad_f2+f2c]=zero2;
        }
      }
      else
      {
        const int valid_kv=valid_kv_rows*D;
        constexpr int kv_total=BC*D;
        for(int i=tid;i<valid_kv;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=float(V_bh[(kv_start+row)*D+col]);
        }
        for(int i=valid_kv+tid;i<kv_total;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=0.0f;
        }
      }
      cp_async_commit();
    }

    // PHASE 2: Register-based online softmax — no causal mask, only boundary masks
    {
      const int row_lo=m_idx*16+(lane_id/4);
      const int row_hi=m_idx*16+(lane_id/4)+8;
      const int global_q_lo=q_start+row_lo;
      const int global_q_hi=q_start+row_hi;

      // Boundary masks only (no causal)
      for(int t=0;t<s_tiles_pw;++t)
      {
        const int n=n_start_s+t;
        const int bc0=kv_start+n*16+(lane_id%4)*2;
        const int bc1=bc0+1;
        const int bc2=kv_start+n*16+(lane_id%4)*2+8;
        const int bc3=bc2+1;

        // Mask out-of-bounds KV positions
        if(bc0>=kv_seq_len) { s_accs[t].x[0]=-INFINITY; s_accs[t].x[2]=-INFINITY; }
        if(bc1>=kv_seq_len) { s_accs[t].x[1]=-INFINITY; s_accs[t].x[3]=-INFINITY; }
        if(bc2>=kv_seq_len) { s_accs[t].x[4]=-INFINITY; s_accs[t].x[6]=-INFINITY; }
        if(bc3>=kv_seq_len) { s_accs[t].x[5]=-INFINITY; s_accs[t].x[7]=-INFINITY; }

        // Mask out-of-bounds Q positions
        if(global_q_lo>=q_seq_len)
        {
          s_accs[t].x[0]=-INFINITY;
          s_accs[t].x[1]=-INFINITY;
          s_accs[t].x[4]=-INFINITY;
          s_accs[t].x[5]=-INFINITY;
        }
        if(global_q_hi>=q_seq_len)
        {
          s_accs[t].x[2]=-INFINITY;
          s_accs[t].x[3]=-INFINITY;
          s_accs[t].x[6]=-INFINITY;
          s_accs[t].x[7]=-INFINITY;
        }
      }

      float max_lo=-INFINITY;
      float max_hi=-INFINITY;
      for(int t=0;t<s_tiles_pw;++t)
      {
        max_lo=fmaxf(max_lo,fmaxf(s_accs[t].x[0],s_accs[t].x[1]));
        max_lo=fmaxf(max_lo,fmaxf(s_accs[t].x[4],s_accs[t].x[5]));
        max_hi=fmaxf(max_hi,fmaxf(s_accs[t].x[2],s_accs[t].x[3]));
        max_hi=fmaxf(max_hi,fmaxf(s_accs[t].x[6],s_accs[t].x[7]));
      }

      max_lo=fmaxf(max_lo,__shfl_xor_sync(0xffffffff,max_lo,1));
      max_lo=fmaxf(max_lo,__shfl_xor_sync(0xffffffff,max_lo,2));
      max_hi=fmaxf(max_hi,__shfl_xor_sync(0xffffffff,max_hi,1));
      max_hi=fmaxf(max_hi,__shfl_xor_sync(0xffffffff,max_hi,2));

      float *reduce_buf=S_tile;
      if(lane_id%4==0)
      {
        reduce_buf[warp_id*16+(lane_id/4)]=max_lo;
        reduce_buf[warp_id*16+(lane_id/4)+8]=max_hi;
      }
      __syncthreads();

      float full_max_lo=-INFINITY;
      float full_max_hi=-INFINITY;
      for(int w=group_base;w<group_base+warps_per_m;++w)
      {
        full_max_lo=fmaxf(full_max_lo,reduce_buf[w*16+(lane_id/4)]);
        full_max_hi=fmaxf(full_max_hi,reduce_buf[w*16+(lane_id/4)+8]);
      }

      const float old_max_lo=row_max_arr[row_lo];
      const float old_max_hi=row_max_arr[row_hi];
      const float new_max_lo=fmaxf(old_max_lo,full_max_lo);
      const float new_max_hi=fmaxf(old_max_hi,full_max_hi);
      const float corr_lo=__expf(old_max_lo-new_max_lo);
      const float corr_hi=__expf(old_max_hi-new_max_hi);

      float sum_lo=0.0f;
      float sum_hi=0.0f;
      for(int t=0;t<s_tiles_pw;++t)
      {
        s_accs[t].x[0]=__expf(s_accs[t].x[0]-new_max_lo);
        sum_lo+=s_accs[t].x[0];
        s_accs[t].x[1]=__expf(s_accs[t].x[1]-new_max_lo);
        sum_lo+=s_accs[t].x[1];
        s_accs[t].x[4]=__expf(s_accs[t].x[4]-new_max_lo);
        sum_lo+=s_accs[t].x[4];
        s_accs[t].x[5]=__expf(s_accs[t].x[5]-new_max_lo);
        sum_lo+=s_accs[t].x[5];
        s_accs[t].x[2]=__expf(s_accs[t].x[2]-new_max_hi);
        sum_hi+=s_accs[t].x[2];
        s_accs[t].x[3]=__expf(s_accs[t].x[3]-new_max_hi);
        sum_hi+=s_accs[t].x[3];
        s_accs[t].x[6]=__expf(s_accs[t].x[6]-new_max_hi);
        sum_hi+=s_accs[t].x[6];
        s_accs[t].x[7]=__expf(s_accs[t].x[7]-new_max_hi);
        sum_hi+=s_accs[t].x[7];
      }

      sum_lo+=__shfl_xor_sync(0xffffffff,sum_lo,1);
      sum_lo+=__shfl_xor_sync(0xffffffff,sum_lo,2);
      sum_hi+=__shfl_xor_sync(0xffffffff,sum_hi,1);
      sum_hi+=__shfl_xor_sync(0xffffffff,sum_hi,2);

      if(lane_id%4==0)
      {
        reduce_buf[warp_id*16+(lane_id/4)]=sum_lo;
        reduce_buf[warp_id*16+(lane_id/4)+8]=sum_hi;
      }
      __syncthreads();

      float full_sum_lo=0.0f;
      float full_sum_hi=0.0f;
      for(int w=group_base;w<group_base+warps_per_m;++w)
      {
        full_sum_lo+=reduce_buf[w*16+(lane_id/4)];
        full_sum_hi+=reduce_buf[w*16+(lane_id/4)+8];
      }

      if(group_warp==0 && lane_id%4==0)
      {
        row_sum_arr[row_lo]=corr_lo*row_sum_arr[row_lo]+full_sum_lo;
        row_sum_arr[row_hi]=corr_hi*row_sum_arr[row_hi]+full_sum_hi;
        row_max_arr[row_lo]=new_max_lo;
        row_max_arr[row_hi]=new_max_hi;
      }

      for(int t=0;t<o_tiles_pw;++t)
      {
        o_frags[t].x[0]*=corr_lo;
        o_frags[t].x[1]*=corr_lo;
        o_frags[t].x[2]*=corr_hi;
        o_frags[t].x[3]*=corr_hi;
        o_frags[t].x[4]*=corr_lo;
        o_frags[t].x[5]*=corr_lo;
        o_frags[t].x[6]*=corr_hi;
        o_frags[t].x[7]*=corr_hi;
      }
    }

    // Store exp(S) to S_tile for Phase 3
    for(int t=0;t<s_tiles_pw;++t)
    {
      const int n=n_start_s+t;
      wmma::store_matrix_sync(
        &S_tile[m_idx*16*bc_pad+n*16],s_accs[t],bc_pad,wmma::mem_row_major);
    }

    cp_async_wait();
    __syncthreads();

    // PHASE 3: O += softmax(S) @ V
    for(int t=0;t<o_tiles_pw;++t)
    {
      const int n=n_start_o+t;
      for(int k=0;k<BC/8;++k)
      {
        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> s_frag;
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> v_frag;
        wmma::load_matrix_sync(s_frag,&S_tile[m_idx*16*bc_pad+k*8],bc_pad);
        wmma::load_matrix_sync(v_frag,&KV_buf[k*8*d_pad+n*16],d_pad);
        wmma::mma_sync(o_frags[t],s_frag,v_frag,o_frags[t]);
      }
    }
    __syncthreads();

    // Pipeline: prefetch K[next]
    if(kv_block+1<num_kv_blocks)
    {
      const int next_start=(kv_block+1)*BC;
      if constexpr(sizeof(T)==4)
      {
        const int next_valid=min(BC,kv_seq_len-next_start)*d_f2;
        const float2 *K_next2=reinterpret_cast<const float2 *>(K_bh+next_start*D);
        for(int i=tid;i<next_valid;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          cp_async_f2(&KV_dst2[row*d_pad_f2+f2c],&K_next2[i]);
        }
        const float2 zero2=make_float2(0.0f,0.0f);
        for(int i=next_valid+tid;i<kv_f2;i+=block_threads)
        {
          const int row=i/d_f2;
          const int f2c=i-row*d_f2;
          KV_dst2[row*d_pad_f2+f2c]=zero2;
        }
      }
      else
      {
        const int next_valid=min(BC,kv_seq_len-next_start)*D;
        constexpr int kv_total=BC*D;
        for(int i=tid;i<next_valid;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=float(K_bh[(next_start+row)*D+col]);
        }
        for(int i=next_valid+tid;i<kv_total;i+=block_threads)
        {
          const int row=i/D;
          const int col=i-row*D;
          KV_buf[row*d_pad+col]=0.0f;
        }
      }
      cp_async_commit();
    }
  }

  // Final: normalize and write O
  __syncthreads();
  float *O_smem=Q_tile;
  for(int t=0;t<o_tiles_pw;++t)
  {
    const int n=n_start_o+t;
    wmma::store_matrix_sync(&O_smem[m_idx*16*d_pad+n*16],o_frags[t],d_pad,wmma::mem_row_major);
  }
  __syncthreads();

  for(int i=tid;i<BR*D;i+=block_threads)
  {
    const int row=i/D;
    const int col=i-row*D;
    const int global_row=q_start+row;
    if(global_row<q_seq_len)
    {
      float inv_l=0.0f;
      if(row_sum_arr[row]>0.0f)
      {
        inv_l=1.0f/row_sum_arr[row];
      }
      O_bh[global_row*D+col]=T(O_smem[row*d_pad+col]*inv_l);
    }
  }

  for(int r=tid;r<BR;r+=block_threads)
  {
    const int global_row=q_start+r;
    if(global_row<q_seq_len)
    {
      L_bh[global_row]=row_max_arr[r]+logf(row_sum_arr[r]+1e-10f);
    }
  }
#endif  // CAIF_HAS_TC_FLASH
}

template<typename T,int D,int BR,int BC,int NW>
static void launch_fa_fwd_cross_tc(const T *Q,
                                   const T *K,
                                   const T *V,
                                   T *O,
                                   float *L,
                                   const int batch_heads,
                                   const int q_seq_len,
                                   const int kv_seq_len,
                                   const float scale,
                                   cudaStream_t stream)
{
  const int num_q_blocks=(q_seq_len+BR-1)/BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(NW*32);
  constexpr size_t smem_size=(BR*(D+2)+BC*(D+2)+BR*(BC+2)+2*BR)*sizeof(float);

  if(smem_size>g_caif_cuda_default_shared_memory)
  {
    cudaFuncSetAttribute(
      (void *)flash_attention_forward_cross_tc_kernel<T,D,BR,BC,NW>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  }
  flash_attention_forward_cross_tc_kernel<T,D,BR,BC,NW>
    <<<grid,block,smem_size,stream>>>(Q,K,V,O,L,q_seq_len,kv_seq_len,scale);
}

//------------------------------------------------------------------------------
// Cross-Attention Forward — Warp-Per-Row Scalar Fallback
//------------------------------------------------------------------------------
template<typename T,int D,int BR>
__global__ void flash_attention_forward_cross_kernel(const T *__restrict__ Q,
                                                     const T *__restrict__ K,
                                                     const T *__restrict__ V,
                                                     T *__restrict__ O,
                                                     float *__restrict__ L,
                                                     const int q_seq_len,
                                                     const int kv_seq_len,
                                                     const float scale)
{
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;
  const int warp_id=tid/32;
  const int lane_id=tid%32;

  const int q_row=q_block_idx*BR+warp_id;
  const bool q_valid=(q_row<q_seq_len);

  extern __shared__ float smem[];
  float *K_tile=smem;
  float *V_tile=smem+g_caif_fa_fwd_bc*D;

  const T *Q_bh=Q+bh*q_seq_len*D;
  const T *K_bh=K+bh*kv_seq_len*D;
  const T *V_bh=V+bh*kv_seq_len*D;
  T *O_bh=O+bh*q_seq_len*D;
  float *L_bh=L+bh*q_seq_len;

  constexpr int elems=(D+31)/32;
  float q_reg[elems];
  for(int dd=lane_id,e=0;dd<D;dd+=32,++e)
  {
    if(q_valid==true)
    {
      q_reg[e]=float(Q_bh[q_row*D+dd]);
    }
    else
    {
      q_reg[e]=0.0f;
    }
  }

  float m_i=-INFINITY;
  float l_i=0.0f;
  float o_reg[elems];
  for(int e=0;e<elems;++e)
  {
    o_reg[e]=0.0f;
  }

  // No causal limit — iterate all KV blocks
  const int num_kv_blocks=(kv_seq_len+g_caif_fa_fwd_bc-1)/g_caif_fa_fwd_bc;
  const int block_threads=BR*32;

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*g_caif_fa_fwd_bc;

    const int tile_elems=g_caif_fa_fwd_bc*D;
    for(int i=tid;i<tile_elems;i+=block_threads)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;
      if(global_row<kv_seq_len)
      {
        K_tile[row*D+col]=float(K_bh[global_row*D+col]);
        V_tile[row*D+col]=float(V_bh[global_row*D+col]);
      }
      else
      {
        K_tile[row*D+col]=0.0f;
        V_tile[row*D+col]=0.0f;
      }
    }
    __syncthreads();

    float s_local[g_caif_fa_fwd_bc];
    float row_max=-INFINITY;
    int num_valid=0;

    for(int j=0;j<g_caif_fa_fwd_bc;++j)
    {
      const int k_row=kv_start+j;
      if(k_row>=kv_seq_len)
      {
        break;
      }

      float dot=0.0f;
      for(int dd=lane_id,e=0;dd<D;dd+=32,++e)
      {
        dot+=q_reg[e]*K_tile[j*D+dd];
      }
      for(int offset=16;offset>0;offset/=2)
      {
        dot+=__shfl_down_sync(0xffffffff,dot,offset);
      }
      dot=__shfl_sync(0xffffffff,dot,0);
      s_local[j]=dot*scale;
      if(s_local[j]>row_max)
      {
        row_max=s_local[j];
      }
      num_valid=j+1;
    }

    const float m_new=fmaxf(m_i,row_max);
    const float scale_old=expf(m_i-m_new);
    for(int e=0;e<elems;++e)
    {
      o_reg[e]*=scale_old;
    }

    float block_sum=0.0f;
    for(int j=0;j<num_valid;++j)
    {
      const float p=expf(s_local[j]-m_new);
      block_sum+=p;
      for(int dd=lane_id,e=0;dd<D;dd+=32,++e)
      {
        o_reg[e]+=p*V_tile[j*D+dd];
      }
    }

    l_i=scale_old*l_i+block_sum;
    m_i=m_new;
    __syncthreads();
  }

  if(q_valid==true)
  {
    float inv_l=0.0f;
    if(l_i>0.0f)
    {
      inv_l=1.0f/l_i;
    }
    for(int dd=lane_id,e=0;dd<D;dd+=32,++e)
    {
      O_bh[q_row*D+dd]=T(o_reg[e]*inv_l);
    }
    if(lane_id==0)
    {
      L_bh[q_row]=m_i+logf(l_i+1e-10f);
    }
  }
}

template<typename T,int D,int BR>
static void launch_fa_fwd_cross(const T *Q,
                                const T *K,
                                const T *V,
                                T *O,
                                float *L,
                                const int batch_heads,
                                const int q_seq_len,
                                const int kv_seq_len,
                                const float scale,
                                cudaStream_t stream)
{
  const int num_q_blocks=(q_seq_len+BR-1)/BR;
  dim3 grid(batch_heads,num_q_blocks);
  dim3 block(BR*32);
  const size_t smem_size=2*g_caif_fa_fwd_bc*D*sizeof(float);

  if(smem_size>g_caif_cuda_default_shared_memory)
  {
    cudaFuncSetAttribute(
      (void *)flash_attention_forward_cross_kernel<T,D,BR>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  }
  flash_attention_forward_cross_kernel<T,D,BR>
    <<<grid,block,smem_size,stream>>>(Q,K,V,O,L,q_seq_len,kv_seq_len,scale);
}

//------------------------------------------------------------------------------
// Cross-Attention Forward — Adaptive TC/Scalar Dispatch
//------------------------------------------------------------------------------
template<typename T,int D,int BR,int BC>
static void dispatch_fa_fwd_cross_tc_nw(const T *Q,
                                        const T *K,
                                        const T *V,
                                        T *O,
                                        float *L,
                                        const int batch_heads,
                                        const int q_seq_len,
                                        const int kv_seq_len,
                                        const float scale,
                                        cudaStream_t stream,
                                        const int nw)
{
  if(nw<=2)
  {
    launch_fa_fwd_cross_tc<T,D,BR,BC,2>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream);
  }
  else if(nw<=4)
  {
    launch_fa_fwd_cross_tc<T,D,BR,BC,4>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream);
  }
  else
  {
    launch_fa_fwd_cross_tc<T,D,BR,BC,8>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream);
  }
}

template<typename T,int D>
static void dispatch_fa_fwd_cross(const T *Q,
                                  const T *K,
                                  const T *V,
                                  T *O,
                                  float *L,
                                  const int batch_heads,
                                  const int q_seq_len,
                                  const int kv_seq_len,
                                  const float scale,
                                  cudaStream_t stream,
                                  const int cc_major,
                                  const size_t smem_limit,
                                  const size_t smem_per_sm,
                                  const int max_threads)
{
  if(cc_major>=8)
  {
    constexpr int br_opts[]={32,16,32,16};
    constexpr int bc_opts[]={128,128,64,64};

    for(int c=0;c<4;++c)
    {
      const int br=br_opts[c];
      const int bc=bc_opts[c];
      const size_t smem=fa_tc_smem(D,br,bc);
      if(smem>smem_limit)
      {
        continue;
      }

      const int tiles_s=(br/16)*(bc/16);
      const int tiles_o=(br/16)*(D/16);
      int max_tiles=tiles_s;
      if(tiles_o>tiles_s)
      {
        max_tiles=tiles_o;
      }
      const int blocks_from_smem=static_cast<int>(smem_per_sm/smem);
      const int tiles_m=br/16;
      const int tiles_n_s=bc/16;
      const int tiles_n_o=D/16;
      const int nw=fa_tc_optimal_nw(max_tiles,blocks_from_smem,tiles_m,tiles_n_s,tiles_n_o);

      if(br==32 && bc==128)
      {
        dispatch_fa_fwd_cross_tc_nw<T,D,32,128>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream,nw);
        return;
      }
      if(br==16 && bc==128)
      {
        dispatch_fa_fwd_cross_tc_nw<T,D,16,128>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream,nw);
        return;
      }
      if(br==32 && bc==64)
      {
        dispatch_fa_fwd_cross_tc_nw<T,D,32,64>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream,nw);
        return;
      }
      if(br==16 && bc==64)
      {
        dispatch_fa_fwd_cross_tc_nw<T,D,16,64>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream,nw);
        return;
      }
    }
  }

  if(fa_scalar_smem(D)<=smem_limit && max_threads>=256)
  {
    launch_fa_fwd_cross<T,D,8>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream);
  }
  else if(fa_scalar_smem(D)<=smem_limit && max_threads>=128)
  {
    launch_fa_fwd_cross<T,D,4>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream);
  }
  else
  {
    launch_fa_fwd_cross<T,D,2>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream);
  }
}

// (former extern "C" block — C++ linkage used for dtype templates)

template<typename T>
void launch_flash_attention_forward_cross(const T *Q,
                                          const T *K,
                                          const T *V,
                                          T *O,
                                          float *L,
                                          const int batch_heads,
                                          const int q_seq_len,
                                          const int kv_seq_len,
                                          const int head_dim,
                                          const float scale,
                                          cudaStream_t stream)
{
  int device_id=0;
  cudaGetDevice(&device_id);
  int max_smem_optin=g_caif_cuda_default_shared_memory;
  cudaDeviceGetAttribute(&max_smem_optin,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device_id);
  int smem_per_sm_int=g_caif_cuda_default_shared_memory;
  cudaDeviceGetAttribute(&smem_per_sm_int,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                         device_id);
  int max_threads=g_caif_cuda_max_threads_fallback;
  cudaDeviceGetAttribute(&max_threads,
                         cudaDevAttrMaxThreadsPerBlock,
                         device_id);
  int cc_major=0;
  cudaDeviceGetAttribute(&cc_major,
                         cudaDevAttrComputeCapabilityMajor,
                         device_id);

  const size_t smem_limit=static_cast<size_t>(max_smem_optin);
  const size_t smem_per_sm=static_cast<size_t>(smem_per_sm_int);

  switch(head_dim)
  {
    case 32:
      dispatch_fa_fwd_cross<T,32>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream,cc_major,smem_limit,smem_per_sm,max_threads);
      break;
    case 64:
      dispatch_fa_fwd_cross<T,64>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream,cc_major,smem_limit,smem_per_sm,max_threads);
      break;
    case 80:
      dispatch_fa_fwd_cross<T,80>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream,cc_major,smem_limit,smem_per_sm,max_threads);
      break;
    case 96:
      dispatch_fa_fwd_cross<T,96>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream,cc_major,smem_limit,smem_per_sm,max_threads);
      break;
    case 128:
      dispatch_fa_fwd_cross<T,128>(Q,K,V,O,L,batch_heads,q_seq_len,kv_seq_len,scale,stream,cc_major,smem_limit,smem_per_sm,max_threads);
      break;
    default:
      fprintf(stderr,
              "FATAL: flash_attention_forward_cross unsupported head_dim=%d"
              " (supported: 32,64,80,96,128). Use standard attention.\n",
              head_dim);
      abort();
  }
}
template void launch_flash_attention_forward_cross<float>(const float *,
                                                          const float *,
                                                          const float *,
                                                          float *,
                                                          float *,
                                                          int,
                                                          int,
                                                          int,
                                                          int,
                                                          float,
                                                          cudaStream_t);
template void launch_flash_attention_forward_cross<__half>(const __half *,
                                                           const __half *,
                                                           const __half *,
                                                           __half *,
                                                           float *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           float,
                                                           cudaStream_t);
template void launch_flash_attention_forward_cross<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                  const __nv_bfloat16 *,
                                                                  const __nv_bfloat16 *,
                                                                  __nv_bfloat16 *,
                                                                  float *,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  int,
                                                                  float,
                                                                  cudaStream_t);

// end of former extern "C" block (cross-attention forward)

//------------------------------------------------------------------------------
// Cross-Attention Backward — Precompute Di = dot(dO, O) per Q row
//------------------------------------------------------------------------------
template<typename T,int D>
__global__ void flash_attention_precompute_di_cross_kernel(const T *__restrict__ dO,
                                                            const T *__restrict__ O,
                                                            float *__restrict__ Di,
                                                            const int q_seq_len)
{
  const int bh=blockIdx.x;
  const int row=blockIdx.y*blockDim.x+threadIdx.x;

  if(row>=q_seq_len)
  {
    return;
  }

  const T *dO_row=dO+bh*q_seq_len*D+row*D;
  const T *O_row=O+bh*q_seq_len*D+row*D;

  float sum=0.0f;
  for(int d=0;d<D;++d)
  {
    sum+=float(dO_row[d])*float(O_row[d]);
  }
  Di[bh*q_seq_len+row]=sum;
}

//------------------------------------------------------------------------------
// Cross-Attention Backward — dK/dV kernel
// Each thread owns one K/V row (kv_seq_len), tiles over Q rows (q_seq_len)
//------------------------------------------------------------------------------
constexpr int g_caif_fa_bwd_cross_br=64;
constexpr int g_caif_fa_bwd_cross_bc=128;
constexpr int g_caif_fa_bwd_cross_dq_br=128;
constexpr int g_caif_fa_bwd_cross_dq_bc=64;

template<typename T,int D>
__global__ void flash_attention_backward_cross_kernel(const T *__restrict__ Q,
                                                       const T *__restrict__ K,
                                                       const T *__restrict__ V,
                                                       const T *__restrict__ dO,
                                                       const float *__restrict__ L,
                                                       const float *__restrict__ Di,
                                                       T *__restrict__ dK,
                                                       T *__restrict__ dV,
                                                       const int q_seq_len,
                                                       const int kv_seq_len,
                                                       const float scale)
{
  const int bh=blockIdx.x;
  const int kv_block_idx=blockIdx.y;
  const int tid=threadIdx.x;

  const int kv_row=kv_block_idx*g_caif_fa_bwd_cross_bc+tid;
  const int active=(kv_row<kv_seq_len);

  extern __shared__ float smem[];
  float *Q_tile=smem;
  float *dO_tile=smem+g_caif_fa_bwd_cross_br*D;
  float *L_tile=smem+2*g_caif_fa_bwd_cross_br*D;
  float *Di_tile=smem+2*g_caif_fa_bwd_cross_br*D+g_caif_fa_bwd_cross_br;

  const T *Q_bh=Q+bh*q_seq_len*D;
  const T *K_bh=K+bh*kv_seq_len*D;
  const T *V_bh=V+bh*kv_seq_len*D;
  const T *dO_bh=dO+bh*q_seq_len*D;
  const float *L_bh=L+bh*q_seq_len;
  const float *Di_bh=Di+bh*q_seq_len;
  T *dK_bh=dK+bh*kv_seq_len*D;
  T *dV_bh=dV+bh*kv_seq_len*D;

  float K_row[D];
  float V_row[D];
  if(active)
  {
    for(int d=0;d<D;++d)
    {
      K_row[d]=float(K_bh[kv_row*D+d]);
      V_row[d]=float(V_bh[kv_row*D+d]);
    }
  }

  float dK_acc[D];
  float dV_acc[D];
  for(int d=0;d<D;++d)
  {
    dK_acc[d]=0.0f;
    dV_acc[d]=0.0f;
  }

  // Iterate all Q blocks (no causal skip)
  const int num_q_blocks=(q_seq_len+g_caif_fa_bwd_cross_br-1)/g_caif_fa_bwd_cross_br;

  for(int q_block=0;q_block<num_q_blocks;++q_block)
  {
    const int q_start=q_block*g_caif_fa_bwd_cross_br;

    for(int i=tid;i<g_caif_fa_bwd_cross_br*D;i+=g_caif_fa_bwd_cross_bc)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=q_start+row;

      if(global_row<q_seq_len)
      {
        Q_tile[row*D+col]=float(Q_bh[global_row*D+col]);
        dO_tile[row*D+col]=float(dO_bh[global_row*D+col]);
      }
      else
      {
        Q_tile[row*D+col]=0.0f;
        dO_tile[row*D+col]=0.0f;
      }
    }

    if(tid<g_caif_fa_bwd_cross_br)
    {
      const int global_row=q_start+tid;
      if(global_row<q_seq_len)
      {
        L_tile[tid]=L_bh[global_row];
        Di_tile[tid]=Di_bh[global_row];
      }
      else
      {
        L_tile[tid]=-INFINITY;
        Di_tile[tid]=0.0f;
      }
    }
    __syncthreads();

    if(active)
    {
      for(int qi=0;qi<g_caif_fa_bwd_cross_br;++qi)
      {
        const int q_row=q_start+qi;

        if(q_row>=q_seq_len)
        {
          continue;
        }

        float dot=0.0f;
        for(int d=0;d<D;++d)
        {
          dot+=Q_tile[qi*D+d]*K_row[d];
        }
        const float S_val=dot*scale;
        const float P_val=expf(fminf(S_val-L_tile[qi],0.0f));

        for(int d=0;d<D;++d)
        {
          dV_acc[d]+=P_val*dO_tile[qi*D+d];
        }

        float dP=0.0f;
        for(int d=0;d<D;++d)
        {
          dP+=dO_tile[qi*D+d]*V_row[d];
        }

        const float dS=P_val*(dP-Di_tile[qi])*scale;

        for(int d=0;d<D;++d)
        {
          dK_acc[d]+=dS*Q_tile[qi*D+d];
        }
      }
    }
    __syncthreads();
  }

  if(active)
  {
    for(int d=0;d<D;++d)
    {
      dK_bh[kv_row*D+d]=T(dK_acc[d]);
      dV_bh[kv_row*D+d]=T(dV_acc[d]);
    }
  }
}

//------------------------------------------------------------------------------
// Cross-Attention Backward — dQ kernel
// Each thread owns one Q row (q_seq_len), tiles over K/V rows (kv_seq_len)
//------------------------------------------------------------------------------
template<typename T,int D>
__global__ void flash_attention_backward_cross_dq_kernel(const T *__restrict__ Q,
                                                          const T *__restrict__ K,
                                                          const T *__restrict__ V,
                                                          const T *__restrict__ dO,
                                                          const float *__restrict__ L,
                                                          const float *__restrict__ Di,
                                                          T *__restrict__ dQ,
                                                          const int q_seq_len,
                                                          const int kv_seq_len,
                                                          const float scale)
{
  const int bh=blockIdx.x;
  const int q_block_idx=blockIdx.y;
  const int tid=threadIdx.x;

  const int q_row=q_block_idx*g_caif_fa_bwd_cross_dq_br+tid;
  const int active=(q_row<q_seq_len);

  extern __shared__ float smem[];
  float *K_tile=smem;
  float *V_tile=smem+g_caif_fa_bwd_cross_dq_bc*D;

  const T *Q_bh=Q+bh*q_seq_len*D;
  const T *K_bh=K+bh*kv_seq_len*D;
  const T *V_bh=V+bh*kv_seq_len*D;
  const T *dO_bh=dO+bh*q_seq_len*D;
  const float *L_bh=L+bh*q_seq_len;
  const float *Di_bh=Di+bh*q_seq_len;
  T *dQ_bh=dQ+bh*q_seq_len*D;

  float Q_row_reg[D];
  float dO_row[D];
  float L_val=0.0f;
  float Di_val=0.0f;
  if(active)
  {
    for(int d=0;d<D;++d)
    {
      Q_row_reg[d]=float(Q_bh[q_row*D+d]);
      dO_row[d]=float(dO_bh[q_row*D+d]);
    }
    L_val=L_bh[q_row];
    Di_val=Di_bh[q_row];
  }

  float dQ_acc[D];
  for(int d=0;d<D;++d)
  {
    dQ_acc[d]=0.0f;
  }

  // Iterate all KV blocks (no causal limit)
  const int num_kv_blocks=(kv_seq_len+g_caif_fa_bwd_cross_dq_bc-1)/g_caif_fa_bwd_cross_dq_bc;

  for(int kv_block=0;kv_block<num_kv_blocks;++kv_block)
  {
    const int kv_start=kv_block*g_caif_fa_bwd_cross_dq_bc;

    for(int i=tid;i<g_caif_fa_bwd_cross_dq_bc*D;i+=g_caif_fa_bwd_cross_dq_br)
    {
      const int row=i/D;
      const int col=i%D;
      const int global_row=kv_start+row;

      if(global_row<kv_seq_len)
      {
        K_tile[row*D+col]=float(K_bh[global_row*D+col]);
        V_tile[row*D+col]=float(V_bh[global_row*D+col]);
      }
      else
      {
        K_tile[row*D+col]=0.0f;
        V_tile[row*D+col]=0.0f;
      }
    }
    __syncthreads();

    if(active)
    {
      for(int j=0;j<g_caif_fa_bwd_cross_dq_bc;++j)
      {
        const int k_row=kv_start+j;

        if(k_row>=kv_seq_len)
        {
          continue;
        }

        float dot=0.0f;
        for(int d=0;d<D;++d)
        {
          dot+=Q_row_reg[d]*K_tile[j*D+d];
        }
        const float S_val=dot*scale;
        const float P_val=expf(fminf(S_val-L_val,0.0f));

        float dP=0.0f;
        for(int d=0;d<D;++d)
        {
          dP+=dO_row[d]*V_tile[j*D+d];
        }

        const float dS=P_val*(dP-Di_val)*scale;

        for(int d=0;d<D;++d)
        {
          dQ_acc[d]+=dS*K_tile[j*D+d];
        }
      }
    }
    __syncthreads();
  }

  if(active)
  {
    for(int d=0;d<D;++d)
    {
      dQ_bh[q_row*D+d]=T(dQ_acc[d]);
    }
  }
}

// (former extern "C" block — C++ linkage used for dtype templates)

template<typename T>
void launch_flash_attention_backward_cross(const T *Q,
                                           const T *K,
                                           const T *V,
                                           const T *O,
                                           const T *dO,
                                           const float *L,
                                           T *dQ,
                                           T *dK,
                                           T *dV,
                                           const int batch_heads,
                                           const int q_seq_len,
                                           const int kv_seq_len,
                                           const int head_dim,
                                           const float scale,
                                           cudaStream_t stream)
{
  // Allocate temporary Di buffer [batch_heads, q_seq_len]
  float *Di_buf=nullptr;
  cudaMallocAsync(reinterpret_cast<void **>(&Di_buf),
                  static_cast<size_t>(batch_heads)*q_seq_len*sizeof(float),stream);

  // Kernel 0: Precompute Di = dot(dO, O) for each Q row
  {
    const int block_size=g_caif_cuda_block_size;
    const int rows_per_grid=(q_seq_len+block_size-1)/block_size;
    dim3 grid(batch_heads,rows_per_grid);

    switch(head_dim)
    {
      case 32:
        flash_attention_precompute_di_cross_kernel<T,32><<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      case 64:
        flash_attention_precompute_di_cross_kernel<T,64><<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      case 80:
        flash_attention_precompute_di_cross_kernel<T,80><<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      case 96:
        flash_attention_precompute_di_cross_kernel<T,96><<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      case 128:
        flash_attention_precompute_di_cross_kernel<T,128><<<grid,block_size,0,stream>>>(dO,O,Di_buf,q_seq_len);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward_cross unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 1: Compute dK and dV (128 threads/block, tiles over q_seq_len)
  {
    const int num_kv_blocks=(kv_seq_len+g_caif_fa_bwd_cross_bc-1)/g_caif_fa_bwd_cross_bc;
    dim3 grid(batch_heads,num_kv_blocks);
    dim3 block(g_caif_fa_bwd_cross_bc);
    const size_t smem_size=(2*g_caif_fa_bwd_cross_br*head_dim+2*g_caif_fa_bwd_cross_br)*sizeof(float);

    switch(head_dim)
    {
      case 32:
        flash_attention_backward_cross_kernel<T,32>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,q_seq_len,kv_seq_len,scale);
        break;
      case 64:
        flash_attention_backward_cross_kernel<T,64>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,q_seq_len,kv_seq_len,scale);
        break;
      case 80:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_kernel<T,80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_kernel<T,80>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,q_seq_len,kv_seq_len,scale);
        break;
      case 96:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_kernel<T,96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_kernel<T,96>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,q_seq_len,kv_seq_len,scale);
        break;
      case 128:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_kernel<T,128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_kernel<T,128>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dK,dV,q_seq_len,kv_seq_len,scale);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward_cross unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  // Kernel 2: Compute dQ (128 threads/block, tiles over kv_seq_len)
  {
    const int num_q_blocks=(q_seq_len+g_caif_fa_bwd_cross_dq_br-1)/g_caif_fa_bwd_cross_dq_br;
    dim3 grid(batch_heads,num_q_blocks);
    dim3 block(g_caif_fa_bwd_cross_dq_br);
    const size_t smem_size=2*g_caif_fa_bwd_cross_dq_bc*head_dim*sizeof(float);

    switch(head_dim)
    {
      case 32:
        flash_attention_backward_cross_dq_kernel<T,32>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,q_seq_len,kv_seq_len,scale);
        break;
      case 64:
        flash_attention_backward_cross_dq_kernel<T,64>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,q_seq_len,kv_seq_len,scale);
        break;
      case 80:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_dq_kernel<T,80>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_dq_kernel<T,80>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,q_seq_len,kv_seq_len,scale);
        break;
      case 96:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_dq_kernel<T,96>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_dq_kernel<T,96>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,q_seq_len,kv_seq_len,scale);
        break;
      case 128:
        cudaFuncSetAttribute((void *)flash_attention_backward_cross_dq_kernel<T,128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_size));
        flash_attention_backward_cross_dq_kernel<T,128>
          <<<grid,block,smem_size,stream>>>(Q,K,V,dO,L,Di_buf,dQ,q_seq_len,kv_seq_len,scale);
        break;
      default:
        fprintf(stderr,
                "FATAL: flash_attention_backward_cross_dq unsupported head_dim=%d\n",
                head_dim);
        abort();
    }
  }

  cudaFreeAsync(Di_buf,stream);
}
template void launch_flash_attention_backward_cross<float>(const float *,
                                                           const float *,
                                                           const float *,
                                                           const float *,
                                                           const float *,
                                                           const float *,
                                                           float *,
                                                           float *,
                                                           float *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           float,
                                                           cudaStream_t);
template void launch_flash_attention_backward_cross<__half>(const __half *,
                                                            const __half *,
                                                            const __half *,
                                                            const __half *,
                                                            const __half *,
                                                            const float *,
                                                            __half *,
                                                            __half *,
                                                            __half *,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            float,
                                                            cudaStream_t);
template void launch_flash_attention_backward_cross<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   const __nv_bfloat16 *,
                                                                   const float *,
                                                                   __nv_bfloat16 *,
                                                                   __nv_bfloat16 *,
                                                                   __nv_bfloat16 *,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   float,
                                                                   cudaStream_t);

// end of former extern "C" block (cross-attention backward)

// (former extern "C" block — C++ linkage used for dtype templates)

//------------------------------------------------------------------------------
// Data type conversion kernels (FP32 <-> FP16, FP32 <-> BF16)
//------------------------------------------------------------------------------
// g_caif_cuda_block_size removed — uses g_caif_cuda_block_size

__global__ void convert_fp32_to_fp16_kernel(const float *input,
                                            __half *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__float2half(input[idx]);
  }
}

void launch_convert_fp32_to_fp16(const float *input,
                                 void *output,
                                 int n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  convert_fp32_to_fp16_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(input,static_cast<__half*>(output),n);
}

__global__ void convert_fp16_to_fp32_kernel(const __half *input,
                                            float *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__half2float(input[idx]);
  }
}

void launch_convert_fp16_to_fp32(const void *input,
                                 float *output,
                                 int n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  convert_fp16_to_fp32_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(
      static_cast<const __half*>(input),output,n);
}

__global__ void convert_fp32_to_bf16_kernel(const float *input,
                                            __nv_bfloat16 *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__float2bfloat16(input[idx]);
  }
}

void launch_convert_fp32_to_bf16(const float *input,
                                 void *output,
                                 int n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  convert_fp32_to_bf16_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(
      input,static_cast<__nv_bfloat16*>(output),n);
}

__global__ void convert_bf16_to_fp32_kernel(const __nv_bfloat16 *input,
                                            float *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=__bfloat162float(input[idx]);
  }
}

void launch_convert_bf16_to_fp32(const void *input,
                                 float *output,
                                 int n,
                                 cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  convert_bf16_to_fp32_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(
      static_cast<const __nv_bfloat16*>(input),output,n);
}

//------------------------------------------------------------------------------
// INT8 conversion kernels
//------------------------------------------------------------------------------

__global__ void convert_fp32_to_int8_kernel(const float *input,
                                            int8_t *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    float val=input[idx];
    if(val>127.0f)
    {
      val=127.0f;
    }
    else if(val<-127.0f)
    {
      val=-127.0f;
    }
    output[idx]=static_cast<int8_t>(rintf(val));
  }
}

void launch_convert_fp32_to_int8(const float *input,
                                  void *output,
                                  int n,
                                  cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  convert_fp32_to_int8_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(
      input,static_cast<int8_t*>(output),n);
}

__global__ void convert_int8_to_fp32_kernel(const int8_t *input,
                                            float *output,
                                            const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=static_cast<float>(input[idx]);
  }
}

void launch_convert_int8_to_fp32(const void *input,
                                  float *output,
                                  int n,
                                  cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  convert_int8_to_fp32_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(
      static_cast<const int8_t*>(input),output,n);
}

//------------------------------------------------------------------------------
// INT4 quantization kernels (symmetric per-group with FP16 scales)
//------------------------------------------------------------------------------

__global__ void dequantize_int4_kernel(const uint8_t *packed_data,
                                       const __half *scales,
                                       float *output,
                                       const int num_elements,
                                       const int group_size)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<num_elements)
  {
    const int byte_idx=idx/2;
    const uint8_t packed=packed_data[byte_idx];

    int int4_val;
    if((idx&1)==0)
    {
      int4_val=packed&0x0F;
    }
    else
    {
      int4_val=(packed>>4)&0x0F;
    }

    // Sign extend: if bit 3 is set, value is negative
    if((int4_val&0x08)!=0)
    {
      int4_val|=static_cast<int>(0xFFFFFFF0);
    }

    const int group_idx=idx/group_size;
    const float scale=__half2float(scales[group_idx]);
    output[idx]=static_cast<float>(int4_val)*scale;
  }
}

void launch_dequantize_int4(const void *packed_data,
                             const void *scales,
                             float *output,
                             int num_elements,
                             int group_size,
                             cudaStream_t stream)
{
  if(num_elements<=0)
  {
    return;
  }
  const int grid=(num_elements+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  dequantize_int4_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(
      static_cast<const uint8_t*>(packed_data),
      static_cast<const __half*>(scales),
      output,
      num_elements,
      group_size);
}

__global__ void quantize_to_int4_kernel(const float *input,
                                        uint8_t *packed_output,
                                        __half *scales_output,
                                        const int num_elements,
                                        const int group_size)
{
  const int group_idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int num_groups=(num_elements+group_size-1)/group_size;
  if(group_idx>=num_groups)
  {
    return;
  }

  const int group_start=group_idx*group_size;
  int group_end=group_start+group_size;
  if(group_end>num_elements)
  {
    group_end=num_elements;
  }

  // Find max absolute value in group
  float max_abs=0.0f;
  for(int i=group_start;i<group_end;++i)
  {
    float abs_val=fabsf(input[i]);
    if(abs_val>max_abs)
    {
      max_abs=abs_val;
    }
  }

  // Compute scale: scale = max_abs / 7.0f
  float scale;
  if(max_abs==0.0f)
  {
    scale=0.0f;
  }
  else
  {
    scale=max_abs/7.0f;
  }
  scales_output[group_idx]=__float2half(scale);

  // Quantize and pack 2 elements per byte
  float inv_scale;
  if(scale==0.0f)
  {
    inv_scale=0.0f;
  }
  else
  {
    inv_scale=1.0f/scale;
  }

  for(int i=group_start;i<group_end;i+=2)
  {
    float v0=input[i]*inv_scale;
    if(v0>7.0f)
    {
      v0=7.0f;
    }
    else if(v0<-7.0f)
    {
      v0=-7.0f;
    }
    int q0=static_cast<int>(rintf(v0))&0x0F;

    int q1=0;
    if(i+1<group_end)
    {
      float v1=input[i+1]*inv_scale;
      if(v1>7.0f)
      {
        v1=7.0f;
      }
      else if(v1<-7.0f)
      {
        v1=-7.0f;
      }
      q1=static_cast<int>(rintf(v1))&0x0F;
    }

    packed_output[i/2]=static_cast<uint8_t>(q0|(q1<<4));
  }
}

void launch_quantize_to_int4(const float *input,
                              void *packed_output,
                              void *scales_output,
                              int num_elements,
                              int group_size,
                              cudaStream_t stream)
{
  if(num_elements<=0)
  {
    return;
  }
  const int num_groups=(num_elements+group_size-1)/group_size;
  const int grid=(num_groups+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  quantize_to_int4_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(
      input,
      static_cast<uint8_t*>(packed_output),
      static_cast<__half*>(scales_output),
      num_elements,
      group_size);
}

//------------------------------------------------------------------------------
// INT8 scaled quantization kernels (symmetric, per-tensor and per-channel)
//
// Per-tensor scheme:
//   scale = max(abs(x)) / 127.0f, stored as a single float
//   q[i]  = round(x[i] / scale), clamped to [-127, 127]
//   x'[i] = q[i] * scale
//
// Per-channel scheme (on last dim, interpreted as the output-channel axis):
//   scale[c] = max over rows of abs(x[:, c]) / 127.0f
//   q[r, c]  = round(x[r, c] / scale[c])
//   x'[r, c] = q[r, c] * scale[c]
//
// Weight tensors stored as [in_features, out_features] get per-channel on the
// out axis; activation tensors typically use per-tensor.
//------------------------------------------------------------------------------

// Accumulator buffer layout: scale_out[0] holds max(|x|) as a non-negative
// float. We atomicMax on its int-reinterpretation — valid because IEEE 754
// positive-float bit patterns preserve ordering when compared as int.
__global__ void compute_int8_per_tensor_scale_kernel(const float *input,
                                                      float *scale_out,
                                                      const int n)
{
  float local_max=0.0f;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)
  {
    const float v=fabsf(input[i]);
    if(v>local_max)
    {
      local_max=v;
    }
  }
  if(local_max>0.0f)
  {
    atomicMax(reinterpret_cast<int*>(scale_out),__float_as_int(local_max));
  }
}

__global__ void finalize_int8_per_tensor_scale_kernel(float *scale_out)
{
  if(threadIdx.x==0&&blockIdx.x==0)
  {
    const float max_abs=scale_out[0];
    scale_out[0]=(max_abs>0.0f)?max_abs/127.0f:1.0f;
  }
}

__global__ void quantize_int8_per_tensor_kernel(const float *input,
                                                 int8_t *output,
                                                 const float *scale,
                                                 const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    const float s=scale[0];
    const float inv_s=(s>0.0f)?1.0f/s:0.0f;
    float v=input[idx]*inv_s;
    if(v>127.0f)
    {
      v=127.0f;
    }
    else if(v<-127.0f)
    {
      v=-127.0f;
    }
    output[idx]=static_cast<int8_t>(rintf(v));
  }
}

__global__ void dequantize_int8_per_tensor_kernel(const int8_t *input,
                                                   float *output,
                                                   const float *scale,
                                                   const int n)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n)
  {
    output[idx]=static_cast<float>(input[idx])*scale[0];
  }
}

void launch_quantize_int8_per_tensor(const float *input,
                                      void *output,
                                      void *scale,
                                      int n,
                                      cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  // Zero the scale buffer (used as an atomicMax accumulator).
  cudaMemsetAsync(scale,0,sizeof(float),stream);
  const int block=g_caif_cuda_block_size;
  const int grid=(n+block-1)/block;
  const int cap_grid=(grid<1024)?grid:1024;
  compute_int8_per_tensor_scale_kernel<<<cap_grid,block,0,stream>>>(
      input,static_cast<float*>(scale),n);
  finalize_int8_per_tensor_scale_kernel<<<1,1,0,stream>>>(
      static_cast<float*>(scale));
  quantize_int8_per_tensor_kernel<<<grid,block,0,stream>>>(
      input,static_cast<int8_t*>(output),static_cast<const float*>(scale),n);
}

void launch_dequantize_int8_per_tensor(const void *input,
                                        float *output,
                                        const void *scale,
                                        int n,
                                        cudaStream_t stream)
{
  if(n<=0)
  {
    return;
  }
  const int grid=(n+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  dequantize_int8_per_tensor_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(
      static_cast<const int8_t*>(input),
      output,
      static_cast<const float*>(scale),
      n);
}

__global__ void compute_int8_per_channel_scale_kernel(const float *input,
                                                       float *scales,
                                                       const int rows,
                                                       const int cols)
{
  const int col=blockIdx.x*blockDim.x+threadIdx.x;
  if(col>=cols)
  {
    return;
  }
  float max_abs=0.0f;
  for(int r=0;r<rows;++r)
  {
    const float v=fabsf(input[r*cols+col]);
    if(v>max_abs)
    {
      max_abs=v;
    }
  }
  scales[col]=(max_abs>0.0f)?max_abs/127.0f:1.0f;
}

__global__ void quantize_int8_per_channel_kernel(const float *input,
                                                  int8_t *output,
                                                  const float *scales,
                                                  const int rows,
                                                  const int cols)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=rows*cols;
  if(idx<total)
  {
    const int col=idx%cols;
    const float s=scales[col];
    const float inv_s=(s>0.0f)?1.0f/s:0.0f;
    float v=input[idx]*inv_s;
    if(v>127.0f)
    {
      v=127.0f;
    }
    else if(v<-127.0f)
    {
      v=-127.0f;
    }
    output[idx]=static_cast<int8_t>(rintf(v));
  }
}

__global__ void dequantize_int8_per_channel_kernel(const int8_t *input,
                                                    float *output,
                                                    const float *scales,
                                                    const int rows,
                                                    const int cols)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=rows*cols;
  if(idx<total)
  {
    const int col=idx%cols;
    output[idx]=static_cast<float>(input[idx])*scales[col];
  }
}

void launch_quantize_int8_per_channel(const float *input,
                                       void *output,
                                       void *scales,
                                       int rows,
                                       int cols,
                                       cudaStream_t stream)
{
  if(rows<=0||cols<=0)
  {
    return;
  }
  const int block=g_caif_cuda_block_size;
  const int scale_grid=(cols+block-1)/block;
  compute_int8_per_channel_scale_kernel<<<scale_grid,block,0,stream>>>(
      input,static_cast<float*>(scales),rows,cols);
  const int total=rows*cols;
  const int grid=(total+block-1)/block;
  quantize_int8_per_channel_kernel<<<grid,block,0,stream>>>(
      input,static_cast<int8_t*>(output),
      static_cast<const float*>(scales),rows,cols);
}

void launch_dequantize_int8_per_channel(const void *input,
                                         float *output,
                                         const void *scales,
                                         int rows,
                                         int cols,
                                         cudaStream_t stream)
{
  if(rows<=0||cols<=0)
  {
    return;
  }
  const int total=rows*cols;
  const int grid=(total+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  dequantize_int8_per_channel_kernel<<<grid,g_caif_cuda_block_size,0,stream>>>(
      static_cast<const int8_t*>(input),output,
      static_cast<const float*>(scales),rows,cols);
}

//------------------------------------------------------------------------------
// Tensor slice and concatenation kernels
//------------------------------------------------------------------------------
// g_caif_cuda_block_size removed — uses g_caif_cuda_block_size

template<typename T>
__global__ void slice_last_dim_kernel(const T *input,
                                      T *output,
                                      const int rows,
                                      const int in_cols,
                                      const int col_start,
                                      const int out_cols)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=rows*out_cols;
  if(idx<total)
  {
    const int row=idx/out_cols;
    const int col=idx%out_cols;
    output[idx]=input[row*in_cols+col_start+col];
  }
}

template<typename T>
void launch_slice_last_dim(const T *input,
                           T *output,
                           int rows,
                           int in_cols,
                           int col_start,
                           int out_cols,
                           cudaStream_t stream)
{
  const int total=rows*out_cols;
  if(total<=0)
  {
    return;
  }
  const int grid=(total+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  slice_last_dim_kernel<T><<<grid,g_caif_cuda_block_size,0,stream>>>(input,
                                                                    output,
                                                                    rows,
                                                                    in_cols,
                                                                    col_start,
                                                                    out_cols);
}

template void launch_slice_last_dim<float>(const float *input,
                                           float *output,
                                           int rows,
                                           int in_cols,
                                           int col_start,
                                           int out_cols,
                                           cudaStream_t stream);
template void launch_slice_last_dim<__half>(const __half *input,
                                            __half *output,
                                            int rows,
                                            int in_cols,
                                            int col_start,
                                            int out_cols,
                                            cudaStream_t stream);
template void launch_slice_last_dim<__nv_bfloat16>(const __nv_bfloat16 *input,
                                                   __nv_bfloat16 *output,
                                                   int rows,
                                                   int in_cols,
                                                   int col_start,
                                                   int out_cols,
                                                   cudaStream_t stream);

template<typename T>
__global__ void slice_last_dim_backward_kernel(const T *grad_output,
                                               T *grad_input,
                                               const int rows,
                                               const int in_cols,
                                               const int col_start,
                                               const int out_cols)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=rows*out_cols;
  if(idx<total)
  {
    const int row=idx/out_cols;
    const int col=idx%out_cols;
    grad_input[row*in_cols+col_start+col]+=grad_output[idx];
  }
}

template<typename T>
void launch_slice_last_dim_backward(const T *grad_output,
                                    T *grad_input,
                                    int rows,
                                    int in_cols,
                                    int col_start,
                                    int out_cols,
                                    cudaStream_t stream)
{
  const int total=rows*out_cols;
  if(total<=0)
  {
    return;
  }
  const int grid=(total+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  slice_last_dim_backward_kernel<T><<<grid,g_caif_cuda_block_size,0,stream>>>(grad_output,
                                                                              grad_input,
                                                                              rows,
                                                                              in_cols,
                                                                              col_start,
                                                                              out_cols);
}

template void launch_slice_last_dim_backward<float>(const float *grad_output,
                                                    float *grad_input,
                                                    int rows,
                                                    int in_cols,
                                                    int col_start,
                                                    int out_cols,
                                                    cudaStream_t stream);
template void launch_slice_last_dim_backward<__half>(const __half *grad_output,
                                                     __half *grad_input,
                                                     int rows,
                                                     int in_cols,
                                                     int col_start,
                                                     int out_cols,
                                                     cudaStream_t stream);
template void launch_slice_last_dim_backward<__nv_bfloat16>(const __nv_bfloat16 *grad_output,
                                                            __nv_bfloat16 *grad_input,
                                                            int rows,
                                                            int in_cols,
                                                            int col_start,
                                                            int out_cols,
                                                            cudaStream_t stream);

template<typename T>
__global__ void concat_last_dim_kernel(const T *a,
                                       const T *b,
                                       T *output,
                                       const int rows,
                                       const int cols_a,
                                       const int cols_b)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total_cols=cols_a+cols_b;
  const int total=rows*total_cols;
  if(idx<total)
  {
    const int row=idx/total_cols;
    const int col=idx%total_cols;
    if(col<cols_a)
    {
      output[idx]=a[row*cols_a+col];
    }
    else
    {
      output[idx]=b[row*cols_b+(col-cols_a)];
    }
  }
}

template<typename T>
void launch_concat_last_dim(const T *a,
                            const T *b,
                            T *output,
                            int rows,
                            int cols_a,
                            int cols_b,
                            cudaStream_t stream)
{
  const int total=rows*(cols_a+cols_b);
  if(total<=0)
  {
    return;
  }
  const int grid=(total+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  concat_last_dim_kernel<T><<<grid,g_caif_cuda_block_size,0,stream>>>(a,
                                                                     b,
                                                                     output,
                                                                     rows,
                                                                     cols_a,
                                                                     cols_b);
}

template void launch_concat_last_dim<float>(const float *a,
                                            const float *b,
                                            float *output,
                                            int rows,
                                            int cols_a,
                                            int cols_b,
                                            cudaStream_t stream);
template void launch_concat_last_dim<__half>(const __half *a,
                                             const __half *b,
                                             __half *output,
                                             int rows,
                                             int cols_a,
                                             int cols_b,
                                             cudaStream_t stream);
template void launch_concat_last_dim<__nv_bfloat16>(const __nv_bfloat16 *a,
                                                    const __nv_bfloat16 *b,
                                                    __nv_bfloat16 *output,
                                                    int rows,
                                                    int cols_a,
                                                    int cols_b,
                                                    cudaStream_t stream);

//------------------------------------------------------------------------------
// Relative Position Bias (T5-style)
//------------------------------------------------------------------------------

__device__ int relative_position_bucket(int relative_position,
                                        int bidirectional,
                                        int num_buckets,
                                        int max_distance)
{
  int ret=0;
  int n=-relative_position;

  if(bidirectional!=0)
  {
    num_buckets=num_buckets/2;
    if(n<0)
    {
      ret=num_buckets;
      n=-n;
    }
  }
  else
  {
    if(n<0)
    {
      n=0;
    }
  }

  const int max_exact=num_buckets/2;
  int val;

  if(n<max_exact)
  {
    val=n;
  }
  else
  {
    const float log_ratio=logf(static_cast<float>(n)/static_cast<float>(max_exact))/
                          logf(static_cast<float>(max_distance)/static_cast<float>(max_exact));
    val=max_exact+static_cast<int>(log_ratio*static_cast<float>(num_buckets-max_exact));
    if(val>num_buckets-1)
    {
      val=num_buckets-1;
    }
  }

  ret=ret+val;
  return ret;
}

template<typename T>
__global__ void relative_position_bias_forward_kernel(const float *embedding,
                                                      T *output,
                                                      const int num_heads,
                                                      const int q_len,
                                                      const int k_len,
                                                      const int num_buckets,
                                                      const int max_distance,
                                                      const int bidirectional)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=num_heads*q_len*k_len;
  if(idx>=total)
  {
    return;
  }

  const int h=idx/(q_len*k_len);
  const int rem=idx%(q_len*k_len);
  const int q=rem/k_len;
  const int k=rem%k_len;

  const int rel_pos=k-q;
  const int bucket=relative_position_bucket(rel_pos,bidirectional,num_buckets,max_distance);

  output[idx]=static_cast<T>(embedding[h*num_buckets+bucket]);
}

template<typename T>
__global__ void relative_position_bias_backward_kernel(const T *grad_output,
                                                       float *grad_embedding,
                                                       const int num_heads,
                                                       const int q_len,
                                                       const int k_len,
                                                       const int num_buckets,
                                                       const int max_distance,
                                                       const int bidirectional)
{
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int total=num_heads*q_len*k_len;
  if(idx>=total)
  {
    return;
  }

  const int h=idx/(q_len*k_len);
  const int rem=idx%(q_len*k_len);
  const int q=rem/k_len;
  const int k=rem%k_len;

  const int rel_pos=k-q;
  const int bucket=relative_position_bucket(rel_pos,bidirectional,num_buckets,max_distance);

  atomicAdd(&grad_embedding[h*num_buckets+bucket],static_cast<float>(grad_output[idx]));
}

template<typename T>
void launch_relative_position_bias_forward(const float *embedding,
                                           T *output,
                                           int num_heads,
                                           int q_len,
                                           int k_len,
                                           int num_buckets,
                                           int max_distance,
                                           int bidirectional,
                                           cudaStream_t stream)
{
  const int total=num_heads*q_len*k_len;
  if(total<=0)
  {
    return;
  }
  const int grid=(total+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  relative_position_bias_forward_kernel<T>
    <<<grid,g_caif_cuda_block_size,0,stream>>>(embedding,
                                                output,
                                                num_heads,
                                                q_len,
                                                k_len,
                                                num_buckets,
                                                max_distance,
                                                bidirectional);
}

template void launch_relative_position_bias_forward<float>(const float *,
                                                           float *,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           cudaStream_t);
template void launch_relative_position_bias_forward<__half>(const float *,
                                                            __half *,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            cudaStream_t);
template void launch_relative_position_bias_forward<__nv_bfloat16>(const float *,
                                                                   __nv_bfloat16 *,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   int,
                                                                   cudaStream_t);

template<typename T>
void launch_relative_position_bias_backward(const T *grad_output,
                                            float *grad_embedding,
                                            int num_heads,
                                            int q_len,
                                            int k_len,
                                            int num_buckets,
                                            int max_distance,
                                            int bidirectional,
                                            cudaStream_t stream)
{
  const int total=num_heads*q_len*k_len;
  if(total<=0)
  {
    return;
  }
  const int grid=(total+g_caif_cuda_block_size-1)/g_caif_cuda_block_size;
  relative_position_bias_backward_kernel<T>
    <<<grid,g_caif_cuda_block_size,0,stream>>>(grad_output,
                                                grad_embedding,
                                                num_heads,
                                                q_len,
                                                k_len,
                                                num_buckets,
                                                max_distance,
                                                bidirectional);
}

template void launch_relative_position_bias_backward<float>(const float *,
                                                            float *,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            int,
                                                            cudaStream_t);
template void launch_relative_position_bias_backward<__half>(const __half *,
                                                             float *,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             int,
                                                             cudaStream_t);
template void launch_relative_position_bias_backward<__nv_bfloat16>(const __nv_bfloat16 *,
                                                                    float *,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    int,
                                                                    cudaStream_t);

// end of former extern "C" block

