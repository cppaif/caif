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

/**
 * @file aif_convolution_ops.cpp
 * @brief Implementation of high-performance convolution operations
 */

#include "caif_convolution_ops.h"
#include "caif_framework.h"
#include <cstring>

using namespace instance;

void CAIF_ConvolutionOps::Conv2DForward(
                                       const float *input,
                                       const float *kernel,
                                       float *output,
                                       const uint32_t batch,
                                       const uint32_t in_h,
                                       const uint32_t in_w,
                                       const uint32_t in_c,
                                       const uint32_t out_c,
                                       const uint32_t k_h,
                                       const uint32_t k_w,
                                       const uint32_t stride_h,
                                       const uint32_t stride_w,
                                       const uint32_t pad_h,
                                       const uint32_t pad_w
                                      )
{
  try
  {
    const uint32_t out_h=(in_h+2*pad_h-k_h)/stride_h+1;
    const uint32_t out_w=(in_w+2*pad_w-k_w)/stride_w+1;
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(4)
    #endif
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t oh=0;oh<out_h;++oh)
      {
        for(uint32_t ow=0;ow<out_w;++ow)
        {
          for(uint32_t oc=0;oc<out_c;++oc)
          {
            float sum=0.0f;
            
            for(uint32_t kh=0;kh<k_h;++kh)
            {
              for(uint32_t kw=0;kw<k_w;++kw)
              {
                const int ih=static_cast<int>(oh*stride_h+kh)-static_cast<int>(pad_h);
                const int iw=static_cast<int>(ow*stride_w+kw)-static_cast<int>(pad_w);
                
                if(ih>=0&&ih<static_cast<int>(in_h)&&iw>=0&&iw<static_cast<int>(in_w))
                {
                  for(uint32_t ic=0;ic<in_c;++ic)
                  {
                    const size_t in_idx=static_cast<size_t>(b)*in_h*in_w*in_c+
                                        static_cast<size_t>(ih)*in_w*in_c+
                                        static_cast<size_t>(iw)*in_c+ic;
                    const size_t k_idx=static_cast<size_t>(kh)*k_w*in_c*out_c+
                                       static_cast<size_t>(kw)*in_c*out_c+
                                       static_cast<size_t>(ic)*out_c+oc;
                    sum+=input[in_idx]*kernel[k_idx];
                  }
                }
              }
            }
            
            const size_t out_idx=static_cast<size_t>(b)*out_h*out_w*out_c+
                                 static_cast<size_t>(oh)*out_w*out_c+
                                 static_cast<size_t>(ow)*out_c+oc;
            output[out_idx]=sum;
          }
        }
      }
    }
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_ConvolutionOps::Conv2DBackwardInput(
                                             const float *grad_output,
                                             const float *kernel,
                                             float *grad_input,
                                             const uint32_t batch,
                                             const uint32_t in_h,
                                             const uint32_t in_w,
                                             const uint32_t in_c,
                                             const uint32_t out_h,
                                             const uint32_t out_w,
                                             const uint32_t out_c,
                                             const uint32_t k_h,
                                             const uint32_t k_w,
                                             const uint32_t stride_h,
                                             const uint32_t stride_w,
                                             const uint32_t pad_h,
                                             const uint32_t pad_w
                                            )
{
  try
  {
    const size_t in_size=static_cast<size_t>(batch)*in_h*in_w*in_c;
    std::memset(grad_input,0,in_size*sizeof(float));
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(4)
    #endif
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t oh=0;oh<out_h;++oh)
      {
        for(uint32_t ow=0;ow<out_w;++ow)
        {
          for(uint32_t oc=0;oc<out_c;++oc)
          {
            const size_t out_idx=static_cast<size_t>(b)*out_h*out_w*out_c+
                                 static_cast<size_t>(oh)*out_w*out_c+
                                 static_cast<size_t>(ow)*out_c+oc;
            const float g=grad_output[out_idx];
            
            for(uint32_t kh=0;kh<k_h;++kh)
            {
              for(uint32_t kw=0;kw<k_w;++kw)
              {
                const int ih=static_cast<int>(oh*stride_h+kh)-static_cast<int>(pad_h);
                const int iw=static_cast<int>(ow*stride_w+kw)-static_cast<int>(pad_w);
                
                if(ih>=0&&ih<static_cast<int>(in_h)&&iw>=0&&iw<static_cast<int>(in_w))
                {
                  for(uint32_t ic=0;ic<in_c;++ic)
                  {
                    const size_t in_idx=static_cast<size_t>(b)*in_h*in_w*in_c+
                                        static_cast<size_t>(ih)*in_w*in_c+
                                        static_cast<size_t>(iw)*in_c+ic;
                    const size_t k_idx=static_cast<size_t>(kh)*k_w*in_c*out_c+
                                       static_cast<size_t>(kw)*in_c*out_c+
                                       static_cast<size_t>(ic)*out_c+oc;
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    grad_input[in_idx]+=g*kernel[k_idx];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_ConvolutionOps::Conv2DBackwardWeight(
                                              const float *input,
                                              const float *grad_output,
                                              float *grad_kernel,
                                              const uint32_t batch,
                                              const uint32_t in_h,
                                              const uint32_t in_w,
                                              const uint32_t in_c,
                                              const uint32_t out_h,
                                              const uint32_t out_w,
                                              const uint32_t out_c,
                                              const uint32_t k_h,
                                              const uint32_t k_w,
                                              const uint32_t stride_h,
                                              const uint32_t stride_w,
                                              const uint32_t pad_h,
                                              const uint32_t pad_w
                                             )
{
  try
  {
    const size_t k_size=static_cast<size_t>(k_h)*k_w*in_c*out_c;
    std::memset(grad_kernel,0,k_size*sizeof(float));
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(4)
    #endif
    for(uint32_t kh=0;kh<k_h;++kh)
    {
      for(uint32_t kw=0;kw<k_w;++kw)
      {
        for(uint32_t ic=0;ic<in_c;++ic)
        {
          for(uint32_t oc=0;oc<out_c;++oc)
          {
            float sum=0.0f;
            
            for(uint32_t b=0;b<batch;++b)
            {
              for(uint32_t oh=0;oh<out_h;++oh)
              {
                for(uint32_t ow=0;ow<out_w;++ow)
                {
                  const int ih=static_cast<int>(oh*stride_h+kh)-static_cast<int>(pad_h);
                  const int iw=static_cast<int>(ow*stride_w+kw)-static_cast<int>(pad_w);
                  
                  if(ih>=0&&ih<static_cast<int>(in_h)&&iw>=0&&iw<static_cast<int>(in_w))
                  {
                    const size_t in_idx=static_cast<size_t>(b)*in_h*in_w*in_c+
                                        static_cast<size_t>(ih)*in_w*in_c+
                                        static_cast<size_t>(iw)*in_c+ic;
                    const size_t out_idx=static_cast<size_t>(b)*out_h*out_w*out_c+
                                         static_cast<size_t>(oh)*out_w*out_c+
                                         static_cast<size_t>(ow)*out_c+oc;
                    sum+=input[in_idx]*grad_output[out_idx];
                  }
                }
              }
            }
            
            const size_t k_idx=static_cast<size_t>(kh)*k_w*in_c*out_c+
                               static_cast<size_t>(kw)*in_c*out_c+
                               static_cast<size_t>(ic)*out_c+oc;
            grad_kernel[k_idx]=sum;
          }
        }
      }
    }
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ConvolutionOps::Conv2DForward(
                                             const CAIF_Tensor &input,
                                             const CAIF_Tensor &kernel,
                                             const uint32_t stride_h,
                                             const uint32_t stride_w,
                                             const uint32_t pad_h,
                                             const uint32_t pad_w
                                            )
{
  try
  {
    const auto &in_shape=input.Shape();
    const auto &k_shape=kernel.Shape();
    
    if(in_shape.size()!=4||k_shape.size()!=4)
    {
      THROW_CAIFE("Conv2D requires 4D tensors");
    }
    
    const uint32_t batch=in_shape[0];
    const uint32_t in_h=in_shape[1];
    const uint32_t in_w=in_shape[2];
    const uint32_t in_c=in_shape[3];
    
    const uint32_t k_h=k_shape[0];
    const uint32_t k_w=k_shape[1];
    const uint32_t out_c=k_shape[3];
    
    const uint32_t out_h=(in_h+2*pad_h-k_h)/stride_h+1;
    const uint32_t out_w=(in_w+2*pad_w-k_w)/stride_w+1;
    
    CAIF_Tensor output(input.Framework(),{batch,out_h,out_w,out_c},input.Type());
    
    Conv2DForward(
                  input.ConstData<float>(),
                  kernel.ConstData<float>(),
                  output.MutableData<float>(),
                  batch,
                  in_h,
                  in_w,
                  in_c,
                  out_c,
                  k_h,
                  k_w,
                  stride_h,
                  stride_w,
                  pad_h,
                  pad_w
                 );
    
    return output;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_ConvolutionOps::Conv2DForward(
                                             CAIF_Framework &framework,
                                             const CAIF_Tensor &input,
                                             const CAIF_Tensor &kernel,
                                             const uint32_t stride_h,
                                             const uint32_t stride_w,
                                             const uint32_t pad_h,
                                             const uint32_t pad_w
                                            )
{
  try
  {
    (void)framework;
    // For GPU backends, could use cuDNN here
    // For now, use CPU implementation for all backends
    return Conv2DForward(input,kernel,stride_h,stride_w,pad_h,pad_w);
  }
  CCAIF_CATCH_BLOCK()
}

