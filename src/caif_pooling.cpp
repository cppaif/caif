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
 * @file aif_pooling.cpp
 * @brief Implementation of CAIF_Tensor convenience wrappers for pooling operations
 */

#include "caif_pooling.h"
#include "caif_framework.h"
#include "caif_constants.h"
#include <cstring>

using namespace instance;

void CAIF_Pooling::MaxPool2D(
                            const float *input,
                            float *output,
                            uint32_t *indices,
                            const uint32_t batch,
                            const uint32_t in_h,
                            const uint32_t in_w,
                            const uint32_t channels,
                            const uint32_t pool_size,
                            const uint32_t stride,
                            const uint32_t padding
                           )
{
  try
  {
    const uint32_t out_h=(in_h+2*padding-pool_size)/stride+1;
    const uint32_t out_w=(in_w+2*padding-pool_size)/stride+1;
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(4)
    #endif
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t oh=0;oh<out_h;++oh)
      {
        for(uint32_t ow=0;ow<out_w;++ow)
        {
          for(uint32_t c=0;c<channels;++c)
          {
            float max_val=-std::numeric_limits<float>::infinity();
            uint32_t max_idx=0;
            
            for(uint32_t kh=0;kh<pool_size;++kh)
            {
              for(uint32_t kw=0;kw<pool_size;++kw)
              {
                const int ih=static_cast<int>(oh*stride+kh)-static_cast<int>(padding);
                const int iw=static_cast<int>(ow*stride+kw)-static_cast<int>(padding);
                
                if(ih>=0&&ih<static_cast<int>(in_h)&&iw>=0&&iw<static_cast<int>(in_w))
                {
                  const size_t in_idx=static_cast<size_t>(b)*in_h*in_w*channels+
                                      static_cast<size_t>(ih)*in_w*channels+
                                      static_cast<size_t>(iw)*channels+c;
                  if(input[in_idx]>max_val)
                  {
                    max_val=input[in_idx];
                    max_idx=static_cast<uint32_t>(in_idx);
                  }
                }
              }
            }
            
            const size_t out_idx=static_cast<size_t>(b)*out_h*out_w*channels+
                                 static_cast<size_t>(oh)*out_w*channels+
                                 static_cast<size_t>(ow)*channels+c;
            output[out_idx]=max_val;
            if(indices!=nullptr)
            {
              indices[out_idx]=max_idx;
            }
          }
        }
      }
    }
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_Pooling::MaxPool2DBackward(
                                    const float *grad_output,
                                    const uint32_t *indices,
                                    float *grad_input,
                                    const uint32_t batch,
                                    const uint32_t in_h,
                                    const uint32_t in_w,
                                    const uint32_t out_h,
                                    const uint32_t out_w,
                                    const uint32_t channels
                                   )
{
  try
  {
    const size_t in_size=static_cast<size_t>(batch)*in_h*in_w*channels;
    std::memset(grad_input,0,in_size*sizeof(float));
    
    const size_t out_size=static_cast<size_t>(batch)*out_h*out_w*channels;
    for(size_t i=0;i<out_size;++i)
    {
      grad_input[indices[i]]+=grad_output[i];
    }
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_Pooling::AvgPool2D(
                            const float *input,
                            float *output,
                            const uint32_t batch,
                            const uint32_t in_h,
                            const uint32_t in_w,
                            const uint32_t channels,
                            const uint32_t pool_size,
                            const uint32_t stride,
                            const uint32_t padding
                           )
{
  try
  {
    const uint32_t out_h=(in_h+2*padding-pool_size)/stride+1;
    const uint32_t out_w=(in_w+2*padding-pool_size)/stride+1;
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(4)
    #endif
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t oh=0;oh<out_h;++oh)
      {
        for(uint32_t ow=0;ow<out_w;++ow)
        {
          for(uint32_t c=0;c<channels;++c)
          {
            float sum=0.0f;
            uint32_t count=0;
            
            for(uint32_t kh=0;kh<pool_size;++kh)
            {
              for(uint32_t kw=0;kw<pool_size;++kw)
              {
                const int ih=static_cast<int>(oh*stride+kh)-static_cast<int>(padding);
                const int iw=static_cast<int>(ow*stride+kw)-static_cast<int>(padding);
                
                if(ih>=0&&ih<static_cast<int>(in_h)&&iw>=0&&iw<static_cast<int>(in_w))
                {
                  const size_t in_idx=static_cast<size_t>(b)*in_h*in_w*channels+
                                      static_cast<size_t>(ih)*in_w*channels+
                                      static_cast<size_t>(iw)*channels+c;
                  sum+=input[in_idx];
                  ++count;
                }
              }
            }
            
            const size_t out_idx=static_cast<size_t>(b)*out_h*out_w*channels+
                                 static_cast<size_t>(oh)*out_w*channels+
                                 static_cast<size_t>(ow)*channels+c;
            if(count>0)
            {
              output[out_idx]=sum/static_cast<float>(count);
            }
            else
            {
              output[out_idx]=0.0f;
            }
          }
        }
      }
    }
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_Pooling::AvgPool2DBackward(
                                    const float *grad_output,
                                    float *grad_input,
                                    const uint32_t batch,
                                    const uint32_t in_h,
                                    const uint32_t in_w,
                                    const uint32_t out_h,
                                    const uint32_t out_w,
                                    const uint32_t channels,
                                    const uint32_t pool_size,
                                    const uint32_t stride,
                                    const uint32_t padding
                                   )
{
  try
  {
    const size_t in_size=static_cast<size_t>(batch)*in_h*in_w*channels;
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
          for(uint32_t c=0;c<channels;++c)
          {
            uint32_t count=0;
            for(uint32_t kh=0;kh<pool_size;++kh)
            {
              for(uint32_t kw=0;kw<pool_size;++kw)
              {
                const int ih=static_cast<int>(oh*stride+kh)-static_cast<int>(padding);
                const int iw=static_cast<int>(ow*stride+kw)-static_cast<int>(padding);
                if(ih>=0&&ih<static_cast<int>(in_h)&&iw>=0&&iw<static_cast<int>(in_w))
                {
                  ++count;
                }
              }
            }
            
            if(count>0)
            {
              const size_t out_idx=static_cast<size_t>(b)*out_h*out_w*channels+
                                   static_cast<size_t>(oh)*out_w*channels+
                                   static_cast<size_t>(ow)*channels+c;
              const float grad_per_elem=grad_output[out_idx]/static_cast<float>(count);
              
              for(uint32_t kh=0;kh<pool_size;++kh)
              {
                for(uint32_t kw=0;kw<pool_size;++kw)
                {
                  const int ih=static_cast<int>(oh*stride+kh)-static_cast<int>(padding);
                  const int iw=static_cast<int>(ow*stride+kw)-static_cast<int>(padding);
                  if(ih>=0&&ih<static_cast<int>(in_h)&&iw>=0&&iw<static_cast<int>(in_w))
                  {
                    const size_t in_idx=static_cast<size_t>(b)*in_h*in_w*channels+
                                        static_cast<size_t>(ih)*in_w*channels+
                                        static_cast<size_t>(iw)*channels+c;
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    grad_input[in_idx]+=grad_per_elem;
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

//==============================================================================
// CAIF_Tensor convenience wrappers
//==============================================================================

CAIF_Tensor CAIF_Pooling::MaxPool2D(
                                  const CAIF_Tensor &input,
                                  const uint32_t pool_size,
                                  const uint32_t stride,
                                  const uint32_t padding,
                                  CAIF_Tensor *indices
                                 )
{
  try
  {
    const auto &shape=input.Shape();
    if(shape.size()!=4)
    {
      THROW_CAIFE("MaxPool2D requires 4D input (NHWC)");
    }
    
    const uint32_t batch=shape[0];
    const uint32_t in_h=shape[1];
    const uint32_t in_w=shape[2];
    const uint32_t channels=shape[3];
    
    const uint32_t out_h=(in_h+2*padding-pool_size)/stride+1;
    const uint32_t out_w=(in_w+2*padding-pool_size)/stride+1;
    
    CAIF_Tensor output(input.Framework(),{batch,out_h,out_w,channels},input.Type());
    
    uint32_t *idx_ptr=nullptr;
    if(indices!=nullptr)
    {
      *indices=CAIF_Tensor(input.Framework(),{batch,out_h,out_w,channels},
                          CAIF_DataType::CAIF_DataType_e::UInt32);
      idx_ptr=indices->MutableData<uint32_t>();
    }
    
    MaxPool2D(
              input.ConstData<float>(),
              output.MutableData<float>(),
              idx_ptr,
              batch,
              in_h,
              in_w,
              channels,
              pool_size,
              stride,
              padding
             );
    
    return output;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Pooling::MaxPool2DBackward(
                                          const CAIF_Tensor &grad_output,
                                          const CAIF_Tensor &indices,
                                          const CAIF_Tensor &input,
                                          const uint32_t /*pool_size*/,
                                          const uint32_t /*stride*/,
                                          const uint32_t /*padding*/
                                         )
{
  try
  {
    // Note: pool_size, stride, padding are encoded in indices - not needed here
    const auto &in_shape=input.Shape();
    const auto &out_shape=grad_output.Shape();
    
    CAIF_Tensor grad_input(input.Framework(),in_shape,input.Type());
    
    MaxPool2DBackward(
                      grad_output.ConstData<float>(),
                      indices.ConstData<uint32_t>(),
                      grad_input.MutableData<float>(),
                      in_shape[0],
                      in_shape[1],
                      in_shape[2],
                      out_shape[1],
                      out_shape[2],
                      in_shape[3]
                     );
    
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Pooling::AvgPool2D(
                                  const CAIF_Tensor &input,
                                  const uint32_t pool_size,
                                  const uint32_t stride,
                                  const uint32_t padding
                                 )
{
  try
  {
    const auto &shape=input.Shape();
    if(shape.size()!=4)
    {
      THROW_CAIFE("AvgPool2D requires 4D input (NHWC)");
    }
    
    const uint32_t batch=shape[0];
    const uint32_t in_h=shape[1];
    const uint32_t in_w=shape[2];
    const uint32_t channels=shape[3];
    
    const uint32_t out_h=(in_h+2*padding-pool_size)/stride+1;
    const uint32_t out_w=(in_w+2*padding-pool_size)/stride+1;
    
    CAIF_Tensor output(input.Framework(),{batch,out_h,out_w,channels},input.Type());
    
    AvgPool2D(
              input.ConstData<float>(),
              output.MutableData<float>(),
              batch,
              in_h,
              in_w,
              channels,
              pool_size,
              stride,
              padding
             );
    
    return output;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_Tensor CAIF_Pooling::AvgPool2DBackward(
                                          const CAIF_Tensor &grad_output,
                                          const std::vector<uint32_t> &input_shape,
                                          const uint32_t pool_size,
                                          const uint32_t stride,
                                          const uint32_t padding
                                         )
{
  try
  {
    const auto &out_shape=grad_output.Shape();
    
    CAIF_Tensor grad_input(grad_output.Framework(),input_shape,grad_output.Type());
    
    AvgPool2DBackward(
                      grad_output.ConstData<float>(),
                      grad_input.MutableData<float>(),
                      input_shape[0],
                      input_shape[1],
                      input_shape[2],
                      out_shape[1],
                      out_shape[2],
                      input_shape[3],
                      pool_size,
                      stride,
                      padding
                     );
    
    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

