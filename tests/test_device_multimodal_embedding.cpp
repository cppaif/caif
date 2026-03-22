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
// Tests for multi-modal embedding layers
//------------------------------------------------------------------------------
#include "caif_device_tabular_embedding.h"
#include "caif_device_spectrogram_embedding.h"
#include "caif_device_vit_model.h"
#include "caif_host_tensor.h"
#include "caif_cuda_stream.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace instance;

constexpr float g_tolerance=1e-3f;
constexpr float g_grad_tolerance=5e-2f;
constexpr uint32_t g_test_seed_1=42;
constexpr uint32_t g_test_seed_2=123;
constexpr uint32_t g_test_seed_3=456;
constexpr uint32_t g_test_seed_4=789;
constexpr uint32_t g_test_seed_5=101;
constexpr uint32_t g_test_seed_6=202;
constexpr uint32_t g_test_seed_7=303;
constexpr uint32_t g_test_seed_8=404;
constexpr float g_finite_diff_eps=1e-3f;

//------------------------------------------------------------------------------
// Helper: Convert CAIF_HostTensor to std::vector<float>
//------------------------------------------------------------------------------
static std::vector<float> ToVector(const CAIF_HostTensor &host_tensor)
{
  size_t n=host_tensor.TotalElements();
  std::vector<float> result(n);
  const float *ptr=host_tensor.Data();
  for(size_t i=0;i<n;++i)
  {
    result[i]=ptr[i];
  }
  return result;
}

//------------------------------------------------------------------------------
// Helper: CPU reference for linear projection
//------------------------------------------------------------------------------
static std::vector<float> CpuLinearProject(const std::vector<float> &input,
                                           const std::vector<float> &weight,
                                           const std::vector<float> &bias,
                                           uint32_t batch,
                                           uint32_t seq_len,
                                           uint32_t in_dim,
                                           uint32_t out_dim)
{
  std::vector<float> output(batch*seq_len*out_dim,0.0f);

  for(uint32_t b=0;b<batch;++b)
  {
    for(uint32_t s=0;s<seq_len;++s)
    {
      for(uint32_t o=0;o<out_dim;++o)
      {
        float sum=bias[o];
        for(uint32_t i=0;i<in_dim;++i)
        {
          sum+=input[b*seq_len*in_dim+s*in_dim+i]*weight[i*out_dim+o];
        }
        output[b*seq_len*out_dim+s*out_dim+o]=sum;
      }
    }
  }
  return output;
}

//------------------------------------------------------------------------------
// Test: TabularEmbedding forward
//------------------------------------------------------------------------------
static void TestTabularEmbeddingForward()
{
  std::cout<<"TestTabularEmbeddingForward... "<<std::flush;

  CAIF_CudaStream stream;

  constexpr uint32_t g_num_features=8;
  constexpr uint32_t g_dim=16;
  constexpr uint32_t g_batch=2;

  CAIF_DeviceTabularEmbedding::Config_t config;
  config.num_features=g_num_features;
  config.dim=g_dim;

  CAIF_DeviceTabularEmbedding embed(config,stream);

  // Create input [batch, num_features]
  std::vector<float> input_data(g_batch*g_num_features);
  std::mt19937 rng(g_test_seed_1);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  for(auto &v:input_data)
  {
    v=dist(rng);
  }

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                        {g_batch,g_num_features},
                                                        stream);
  CAIF_DeviceTensor output=embed.Forward(input,false);

  // Check output shape: [batch, 1, dim]
  const auto &shape=output.Shape();
  assert(shape.size()==3);
  assert(shape[0]==g_batch);
  assert(shape[1]==1);
  assert(shape[2]==g_dim);

  // Verify with CPU reference
  std::vector<float> w_host=ToVector(embed.ParameterTensor(0).ToHost());
  std::vector<float> b_host=ToVector(embed.ParameterTensor(1).ToHost());

  std::vector<float> expected=CpuLinearProject(input_data,w_host,b_host,
                                               g_batch,1,g_num_features,g_dim);

  std::vector<float> output_host=ToVector(output.ToHost());
  for(size_t i=0;i<expected.size();++i)
  {
    float diff=std::fabs(output_host[i]-expected[i]);
    if(diff>g_tolerance)
    {
      std::cerr<<"Mismatch at "<<i<<": "<<output_host[i]<<" vs "<<expected[i]<<std::endl;
      assert(false);
    }
  }

  std::cout<<"PASSED"<<std::endl;
}

//------------------------------------------------------------------------------
// Test: TabularEmbedding 3D input
//------------------------------------------------------------------------------
static void TestTabularEmbedding3D()
{
  std::cout<<"TestTabularEmbedding3D... "<<std::flush;

  CAIF_CudaStream stream;

  constexpr uint32_t g_num_features=4;
  constexpr uint32_t g_dim=8;
  constexpr uint32_t g_batch=2;
  constexpr uint32_t g_seq_len=3;

  CAIF_DeviceTabularEmbedding::Config_t config;
  config.num_features=g_num_features;
  config.dim=g_dim;

  CAIF_DeviceTabularEmbedding embed(config,stream);

  // Create input [batch, seq_len, num_features]
  std::vector<float> input_data(g_batch*g_seq_len*g_num_features);
  std::mt19937 rng(g_test_seed_2);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  for(auto &v:input_data)
  {
    v=dist(rng);
  }

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                        {g_batch,g_seq_len,g_num_features},
                                                        stream);
  CAIF_DeviceTensor output=embed.Forward(input,false);

  // Check output shape: [batch, seq_len, dim]
  const auto &shape=output.Shape();
  assert(shape.size()==3);
  assert(shape[0]==g_batch);
  assert(shape[1]==g_seq_len);
  assert(shape[2]==g_dim);

  std::cout<<"PASSED"<<std::endl;
}

//------------------------------------------------------------------------------
// Test: TabularEmbedding gradient check
//------------------------------------------------------------------------------
static void TestTabularEmbeddingGradient()
{
  std::cout<<"TestTabularEmbeddingGradient... "<<std::flush;

  CAIF_CudaStream stream;

  constexpr uint32_t g_num_features=4;
  constexpr uint32_t g_dim=6;
  constexpr uint32_t g_batch=2;

  CAIF_DeviceTabularEmbedding::Config_t config;
  config.num_features=g_num_features;
  config.dim=g_dim;

  CAIF_DeviceTabularEmbedding embed(config,stream);

  std::vector<float> input_data(g_batch*g_num_features);
  std::mt19937 rng(g_test_seed_3);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  for(auto &v:input_data)
  {
    v=dist(rng);
  }

  // Forward and backward
  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                        {g_batch,g_num_features},
                                                        stream);
  CAIF_DeviceTensor output=embed.Forward(input,true);

  // Create grad_output (ones)
  std::vector<float> grad_out_data(g_batch*1*g_dim,1.0f);
  CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(grad_out_data.data(),
                                                              {g_batch,1,g_dim},
                                                              stream);

  embed.ZeroGradients();
  CAIF_DeviceTensor grad_input=embed.Backward(grad_output);

  // Finite difference check for input
  std::vector<float> grad_input_host=ToVector(grad_input.ToHost());
  for(uint32_t i=0;i<g_batch*g_num_features;++i)
  {
    std::vector<float> perturbed=input_data;
    perturbed[i]+=g_finite_diff_eps;
    CAIF_DeviceTensor input_plus=CAIF_DeviceTensor::FromHostData(perturbed.data(),
                                                               {g_batch,g_num_features},
                                                               stream);
    CAIF_DeviceTensor out_plus=embed.Forward(input_plus,false);
    std::vector<float> out_plus_host=ToVector(out_plus.ToHost());

    perturbed[i]=input_data[i]-g_finite_diff_eps;
    CAIF_DeviceTensor input_minus=CAIF_DeviceTensor::FromHostData(perturbed.data(),
                                                                {g_batch,g_num_features},
                                                                stream);
    CAIF_DeviceTensor out_minus=embed.Forward(input_minus,false);
    std::vector<float> out_minus_host=ToVector(out_minus.ToHost());

    float numerical_grad=0.0f;
    for(size_t j=0;j<out_plus_host.size();++j)
    {
      numerical_grad+=(out_plus_host[j]-out_minus_host[j])/(2.0f*g_finite_diff_eps);
    }

    float diff=std::fabs(grad_input_host[i]-numerical_grad);
    if(diff>g_grad_tolerance)
    {
      std::cerr<<"Gradient mismatch at input["<<i<<"]: "
               <<grad_input_host[i]<<" vs "<<numerical_grad<<std::endl;
      assert(false);
    }
  }

  std::cout<<"PASSED"<<std::endl;
}

//------------------------------------------------------------------------------
// Test: SpectrogramEmbedding forward
//------------------------------------------------------------------------------
static void TestSpectrogramEmbeddingForward()
{
  std::cout<<"TestSpectrogramEmbeddingForward... "<<std::flush;

  CAIF_CudaStream stream;

  constexpr uint32_t g_freq_bins=80;
  constexpr uint32_t g_dim=64;
  constexpr uint32_t g_batch=2;
  constexpr uint32_t g_time_frames=10;

  CAIF_DeviceSpectrogramEmbedding::Config_t config;
  config.freq_bins=g_freq_bins;
  config.dim=g_dim;
  config.use_cls_token=false;

  CAIF_DeviceSpectrogramEmbedding embed(config,stream);

  // Create input [batch, time_frames, freq_bins]
  std::vector<float> input_data(g_batch*g_time_frames*g_freq_bins);
  std::mt19937 rng(g_test_seed_4);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  for(auto &v:input_data)
  {
    v=dist(rng);
  }

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                        {g_batch,g_time_frames,g_freq_bins},
                                                        stream);
  CAIF_DeviceTensor output=embed.Forward(input,false);

  // Check output shape: [batch, time_frames, dim]
  const auto &shape=output.Shape();
  assert(shape.size()==3);
  assert(shape[0]==g_batch);
  assert(shape[1]==g_time_frames);
  assert(shape[2]==g_dim);

  std::cout<<"PASSED"<<std::endl;
}

//------------------------------------------------------------------------------
// Test: SpectrogramEmbedding with CLS token
//------------------------------------------------------------------------------
static void TestSpectrogramEmbeddingWithCLS()
{
  std::cout<<"TestSpectrogramEmbeddingWithCLS... "<<std::flush;

  CAIF_CudaStream stream;

  constexpr uint32_t g_freq_bins=40;
  constexpr uint32_t g_dim=32;
  constexpr uint32_t g_batch=2;
  constexpr uint32_t g_time_frames=5;

  CAIF_DeviceSpectrogramEmbedding::Config_t config;
  config.freq_bins=g_freq_bins;
  config.dim=g_dim;
  config.use_cls_token=true;

  CAIF_DeviceSpectrogramEmbedding embed(config,stream);

  // Create input [batch, time_frames, freq_bins]
  std::vector<float> input_data(g_batch*g_time_frames*g_freq_bins);
  std::mt19937 rng(g_test_seed_5);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  for(auto &v:input_data)
  {
    v=dist(rng);
  }

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                        {g_batch,g_time_frames,g_freq_bins},
                                                        stream);
  CAIF_DeviceTensor output=embed.Forward(input,false);

  // Check output shape: [batch, time_frames+1, dim] (CLS token prepended)
  const auto &shape=output.Shape();
  assert(shape.size()==3);
  assert(shape[0]==g_batch);
  assert(shape[1]==g_time_frames+1);  // +1 for CLS
  assert(shape[2]==g_dim);

  // Verify CLS token is the same across batches
  std::vector<float> output_host=ToVector(output.ToHost());
  std::vector<float> cls_token=ToVector(embed.ParameterTensor(2).ToHost());

  for(uint32_t b=0;b<g_batch;++b)
  {
    for(uint32_t d=0;d<g_dim;++d)
    {
      float out_val=output_host[b*(g_time_frames+1)*g_dim+d];  // Position 0 of each batch
      float diff=std::fabs(out_val-cls_token[d]);
      if(diff>g_tolerance)
      {
        std::cerr<<"CLS token mismatch at batch "<<b<<" dim "<<d<<std::endl;
        assert(false);
      }
    }
  }

  std::cout<<"PASSED"<<std::endl;
}

//------------------------------------------------------------------------------
// Test: SpectrogramEmbedding gradient
//------------------------------------------------------------------------------
static void TestSpectrogramEmbeddingGradient()
{
  std::cout<<"TestSpectrogramEmbeddingGradient... "<<std::flush;

  CAIF_CudaStream stream;

  constexpr uint32_t g_freq_bins=8;
  constexpr uint32_t g_dim=6;
  constexpr uint32_t g_batch=2;
  constexpr uint32_t g_time_frames=3;

  CAIF_DeviceSpectrogramEmbedding::Config_t config;
  config.freq_bins=g_freq_bins;
  config.dim=g_dim;
  config.use_cls_token=false;

  CAIF_DeviceSpectrogramEmbedding embed(config,stream);

  std::vector<float> input_data(g_batch*g_time_frames*g_freq_bins);
  std::mt19937 rng(g_test_seed_6);
  std::uniform_real_distribution<float> dist(-1.0f,1.0f);
  for(auto &v:input_data)
  {
    v=dist(rng);
  }

  // Forward and backward
  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                        {g_batch,g_time_frames,g_freq_bins},
                                                        stream);
  CAIF_DeviceTensor output=embed.Forward(input,true);

  // Create grad_output (ones)
  std::vector<float> grad_out_data(g_batch*g_time_frames*g_dim,1.0f);
  CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(grad_out_data.data(),
                                                              {g_batch,g_time_frames,g_dim},
                                                              stream);

  embed.ZeroGradients();
  CAIF_DeviceTensor grad_input=embed.Backward(grad_output);

  // Just check shapes for now
  const auto &grad_shape=grad_input.Shape();
  assert(grad_shape.size()==3);
  assert(grad_shape[0]==g_batch);
  assert(grad_shape[1]==g_time_frames);
  assert(grad_shape[2]==g_freq_bins);

  std::cout<<"PASSED"<<std::endl;
}

//------------------------------------------------------------------------------
// Test: ViT model construction and forward
//------------------------------------------------------------------------------
static void TestViTModelForward()
{
  std::cout<<"TestViTModelForward... "<<std::flush;

  CAIF_CudaStream stream;

  constexpr uint32_t g_image_height=32;
  constexpr uint32_t g_image_width=32;
  constexpr uint32_t g_channels=3;
  constexpr uint32_t g_patch_size=8;
  constexpr uint32_t g_dim=64;
  constexpr uint32_t g_num_layers=2;
  constexpr uint32_t g_num_heads=4;
  constexpr uint32_t g_ffn_dim=128;
  constexpr uint32_t g_num_classes=10;
  constexpr uint32_t g_batch=2;
  constexpr float g_dropout_rate=0.0f;
  constexpr float g_rope_base=10000.0f;

  CAIF_DeviceViTModel::Config_t config;
  config.image_height=g_image_height;
  config.image_width=g_image_width;
  config.channels=g_channels;
  config.patch_size=g_patch_size;
  config.dim=g_dim;
  config.num_layers=g_num_layers;
  config.num_heads=g_num_heads;
  config.ffn_hidden_dim=g_ffn_dim;
  config.num_classes=g_num_classes;
  config.dropout_rate=g_dropout_rate;
  config.use_rope=false;
  config.rope_base=g_rope_base;

  CAIF_DeviceViTModel vit(config,stream);

  // Check computed properties
  constexpr uint32_t g_expected_num_patches=(g_image_height/g_patch_size)*(g_image_width/g_patch_size);
  assert(vit.NumPatches()==g_expected_num_patches);
  assert(vit.SequenceLength()==g_expected_num_patches+1);  // +1 for CLS

  // Create input [batch, H, W, C]
  std::vector<float> input_data(g_batch*g_image_height*g_image_width*g_channels);
  std::mt19937 rng(g_test_seed_7);
  std::uniform_real_distribution<float> dist(0.0f,1.0f);
  for(auto &v:input_data)
  {
    v=dist(rng);
  }

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                        {g_batch,g_image_height,g_image_width,g_channels},
                                                        stream);
  CAIF_DeviceTensor output=vit.Forward(input,false);

  // Check output shape: [batch, num_classes]
  const auto &shape=output.Shape();
  assert(shape.size()==2);
  assert(shape[0]==g_batch);
  assert(shape[1]==g_num_classes);

  // Verify output contains valid values (not NaN/Inf)
  std::vector<float> output_host=ToVector(output.ToHost());
  for(const auto &v:output_host)
  {
    assert(std::isfinite(v)==true);
  }

  std::cout<<"PASSED"<<std::endl;
}

//------------------------------------------------------------------------------
// Test: ViT model backward
//------------------------------------------------------------------------------
static void TestViTModelBackward()
{
  std::cout<<"TestViTModelBackward... "<<std::flush;

  CAIF_CudaStream stream;

  constexpr uint32_t g_image_height=16;
  constexpr uint32_t g_image_width=16;
  constexpr uint32_t g_channels=3;
  constexpr uint32_t g_patch_size=8;
  constexpr uint32_t g_dim=32;
  constexpr uint32_t g_num_layers=1;
  constexpr uint32_t g_num_heads=2;
  constexpr uint32_t g_ffn_dim=64;
  constexpr uint32_t g_num_classes=5;
  constexpr uint32_t g_batch=2;
  constexpr float g_dropout_rate=0.0f;
  constexpr float g_rope_base=10000.0f;

  CAIF_DeviceViTModel::Config_t config;
  config.image_height=g_image_height;
  config.image_width=g_image_width;
  config.channels=g_channels;
  config.patch_size=g_patch_size;
  config.dim=g_dim;
  config.num_layers=g_num_layers;
  config.num_heads=g_num_heads;
  config.ffn_hidden_dim=g_ffn_dim;
  config.num_classes=g_num_classes;
  config.dropout_rate=g_dropout_rate;
  config.use_rope=false;
  config.rope_base=g_rope_base;

  CAIF_DeviceViTModel vit(config,stream);

  // Create input
  std::vector<float> input_data(g_batch*g_image_height*g_image_width*g_channels);
  std::mt19937 rng(g_test_seed_8);
  std::uniform_real_distribution<float> dist(0.0f,1.0f);
  for(auto &v:input_data)
  {
    v=dist(rng);
  }

  CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                        {g_batch,g_image_height,g_image_width,g_channels},
                                                        stream);

  // Forward with training=true
  CAIF_DeviceTensor output=vit.Forward(input,true);

  // Create grad_output
  std::vector<float> grad_out_data(g_batch*g_num_classes,1.0f);
  CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(grad_out_data.data(),
                                                              {g_batch,g_num_classes},
                                                              stream);

  // Backward
  vit.ZeroGradients();
  CAIF_DeviceTensor grad_input=vit.Backward(grad_output);

  // Check grad_input shape
  const auto &shape=grad_input.Shape();
  assert(shape.size()==4);
  assert(shape[0]==g_batch);
  assert(shape[1]==g_image_height);
  assert(shape[2]==g_image_width);
  assert(shape[3]==g_channels);

  // Verify gradients are valid
  std::vector<float> grad_host=ToVector(grad_input.ToHost());
  for(const auto &v:grad_host)
  {
    assert(std::isfinite(v)==true);
  }

  std::cout<<"PASSED"<<std::endl;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main()
{
  std::cout<<"=== Multi-Modal Embedding Tests ==="<<std::endl;

  try
  {
    TestTabularEmbeddingForward();
    TestTabularEmbedding3D();
    TestTabularEmbeddingGradient();
    TestSpectrogramEmbeddingForward();
    TestSpectrogramEmbeddingWithCLS();
    TestSpectrogramEmbeddingGradient();
    TestViTModelForward();
    TestViTModelBackward();

    std::cout<<"\n=== All Multi-Modal Embedding Tests PASSED ==="<<std::endl;
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cerr<<"Test failed with exception: "<<e.what()<<std::endl;
    return 1;
  }
}
