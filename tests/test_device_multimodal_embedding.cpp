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
#include "caif_run_context.h"
#include "caif_exception.h"
#include "caif_test_harness.h"
#include "ise_lib/ise_out.h"
#include <cmath>
#include <random>
#include <vector>

namespace instance
{

constexpr float g_caif_mmembed_test_tolerance=1e-3f;
constexpr float g_caif_mmembed_test_grad_tolerance=5e-2f;
constexpr uint32_t g_caif_mmembed_test_seed_1=42;
constexpr uint32_t g_caif_mmembed_test_seed_2=123;
constexpr uint32_t g_caif_mmembed_test_seed_3=456;
constexpr uint32_t g_caif_mmembed_test_seed_4=789;
constexpr uint32_t g_caif_mmembed_test_seed_5=101;
constexpr uint32_t g_caif_mmembed_test_seed_6=202;
constexpr uint32_t g_caif_mmembed_test_seed_7=303;
constexpr uint32_t g_caif_mmembed_test_seed_8=404;
constexpr float g_caif_mmembed_test_finite_diff_eps=1e-3f;
constexpr float g_caif_mmembed_test_img_max=1.0f;
constexpr float g_caif_mmembed_test_img_min=0.0f;
constexpr float g_caif_mmembed_test_rand_min=-1.0f;
constexpr float g_caif_mmembed_test_rand_max=1.0f;
constexpr float g_caif_mmembed_test_rope_base=10000.0f;
constexpr float g_caif_mmembed_test_dropout=0.0f;

//------------------------------------------------------------------------------
// Multi-modal embedding tests: tabular, spectrogram, ViT.
//------------------------------------------------------------------------------
class CAIF_MultiModalEmbeddingTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> ToVector(const CAIF_HostTensor &host_tensor);
    static std::vector<float> CpuLinearProject(const std::vector<float> &input,
                                               const std::vector<float> &weight,
                                               const std::vector<float> &bias,
                                               uint32_t batch,
                                               uint32_t seq_len,
                                               uint32_t in_dim,
                                               uint32_t out_dim);

    static void TestTabularEmbeddingForward();
    static void TestTabularEmbedding3D();
    static void TestTabularEmbeddingGradient();
    static void TestSpectrogramEmbeddingForward();
    static void TestSpectrogramEmbeddingWithCLS();
    static void TestSpectrogramEmbeddingGradient();
    static void TestViTModelForward();
    static void TestViTModelBackward();
};

//------------------------------------------------------------------------------
// Helper: Convert CAIF_HostTensor to std::vector<float>
//------------------------------------------------------------------------------
std::vector<float> CAIF_MultiModalEmbeddingTests::ToVector(const CAIF_HostTensor &host_tensor)
{
  const size_t n=host_tensor.TotalElements();
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
std::vector<float> CAIF_MultiModalEmbeddingTests::CpuLinearProject(
  const std::vector<float> &input,
  const std::vector<float> &weight,
  const std::vector<float> &bias,
  const uint32_t batch,
  const uint32_t seq_len,
  const uint32_t in_dim,
  const uint32_t out_dim)
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
void CAIF_MultiModalEmbeddingTests::TestTabularEmbeddingForward()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    constexpr uint32_t num_features=8;
    constexpr uint32_t dim=16;
    constexpr uint32_t batch=2;

    CAIF_DeviceTabularEmbeddingConfig config(num_features,dim);

    CAIF_DeviceTabularEmbedding<float,float> embed(config,stream);

    // Create input [batch, num_features]
    std::vector<float> input_data(batch*num_features);
    std::mt19937 rng(g_caif_mmembed_test_seed_1);
    std::uniform_real_distribution<float> dist(g_caif_mmembed_test_rand_min,g_caif_mmembed_test_rand_max);
    for(auto &v:input_data)
    {
      v=dist(rng);
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                             {batch,num_features},
                                                             stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=embed.Forward(input,ctx);

    // Check output shape: [batch, 1, dim]
    const auto &shape=output.Shape();
    bool passed=true;
    if(shape.size()!=3||shape[0]!=batch||shape[1]!=1||shape[2]!=dim)
    {
      passed=false;
    }

    if(passed==true)
    {
      // Verify with CPU reference
      std::vector<float> w_host=ToVector(embed.ParameterTensor(0).ToHost());
      std::vector<float> b_host=ToVector(embed.ParameterTensor(1).ToHost());
      std::vector<float> expected=CpuLinearProject(input_data,w_host,b_host,batch,1,num_features,dim);
      std::vector<float> output_host=ToVector(output.ToHost());
      for(size_t i=0;i<expected.size();++i)
      {
        const float diff=std::fabs(output_host[i]-expected[i]);
        if(diff>g_caif_mmembed_test_tolerance)
        {
          ISE_Out::ErrLog()<<"TabularFwd mismatch at "
                           <<i
                           <<": "
                           <<output_host[i]
                           <<" vs "
                           <<expected[i]
                           <<"\n";
          passed=false;
          break;
        }
      }
    }

    CAIF_TestHarness::Report("MultiModalEmbed::TabularForward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MultiModalEmbed::TabularForward")
}

//------------------------------------------------------------------------------
// Test: TabularEmbedding 3D input
//------------------------------------------------------------------------------
void CAIF_MultiModalEmbeddingTests::TestTabularEmbedding3D()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    constexpr uint32_t num_features=4;
    constexpr uint32_t dim=8;
    constexpr uint32_t batch=2;
    constexpr uint32_t seq_len=3;

    CAIF_DeviceTabularEmbeddingConfig config(num_features,dim);

    CAIF_DeviceTabularEmbedding<float,float> embed(config,stream);

    // Create input [batch, seq_len, num_features]
    std::vector<float> input_data(batch*seq_len*num_features);
    std::mt19937 rng(g_caif_mmembed_test_seed_2);
    std::uniform_real_distribution<float> dist(g_caif_mmembed_test_rand_min,g_caif_mmembed_test_rand_max);
    for(auto &v:input_data)
    {
      v=dist(rng);
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                             {batch,seq_len,num_features},
                                                             stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=embed.Forward(input,ctx);

    // Check output shape: [batch, seq_len, dim]
    const auto &shape=output.Shape();
    const bool passed=(shape.size()==3&&shape[0]==batch&&shape[1]==seq_len&&shape[2]==dim);
    CAIF_TestHarness::Report("MultiModalEmbed::Tabular3D",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MultiModalEmbed::Tabular3D")
}

//------------------------------------------------------------------------------
// Test: TabularEmbedding gradient check
//------------------------------------------------------------------------------
void CAIF_MultiModalEmbeddingTests::TestTabularEmbeddingGradient()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    constexpr uint32_t num_features=4;
    constexpr uint32_t dim=6;
    constexpr uint32_t batch=2;

    CAIF_DeviceTabularEmbeddingConfig config(num_features,dim);

    CAIF_DeviceTabularEmbedding<float,float> embed(config,stream);

    std::vector<float> input_data(batch*num_features);
    std::mt19937 rng(g_caif_mmembed_test_seed_3);
    std::uniform_real_distribution<float> dist(g_caif_mmembed_test_rand_min,g_caif_mmembed_test_rand_max);
    for(auto &v:input_data)
    {
      v=dist(rng);
    }

    // Forward and backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                             {batch,num_features},
                                                             stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=embed.Forward(input,ctx);

    // Create grad_output (ones)
    std::vector<float> grad_out_data(batch*1*dim,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(grad_out_data.data(),
                                                                   {batch,1,dim},
                                                                   stream);

    embed.ZeroGradients();
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=embed.Backward(grad_output,ctx);

    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    // Finite difference check for input
    std::vector<float> grad_input_host=ToVector(grad_input.ToHost());
    bool passed=true;
    for(uint32_t i=0;i<batch*num_features&&passed==true;++i)
    {
      std::vector<float> perturbed=input_data;
      perturbed[i]+=g_caif_mmembed_test_finite_diff_eps;
      CAIF_DeviceTensor input_plus=CAIF_DeviceTensor::FromHostData(perturbed.data(),
                                                                     {batch,num_features},
                                                                     stream);
      CAIF_DeviceTensor out_plus=embed.Forward(input_plus,ctx);
      std::vector<float> out_plus_host=ToVector(out_plus.ToHost());

      perturbed[i]=input_data[i]-g_caif_mmembed_test_finite_diff_eps;
      CAIF_DeviceTensor input_minus=CAIF_DeviceTensor::FromHostData(perturbed.data(),
                                                                      {batch,num_features},
                                                                      stream);
      CAIF_DeviceTensor out_minus=embed.Forward(input_minus,ctx);
      std::vector<float> out_minus_host=ToVector(out_minus.ToHost());

      float numerical_grad=0.0f;
      for(size_t j=0;j<out_plus_host.size();++j)
      {
        numerical_grad+=(out_plus_host[j]-out_minus_host[j])/(2.0f*g_caif_mmembed_test_finite_diff_eps);
      }

      const float diff=std::fabs(grad_input_host[i]-numerical_grad);
      if(diff>g_caif_mmembed_test_grad_tolerance)
      {
        ISE_Out::ErrLog()<<"TabularGrad mismatch at input["
                         <<i
                         <<"]: "
                         <<grad_input_host[i]
                         <<" vs "
                         <<numerical_grad
                         <<"\n";
        passed=false;
      }
    }

    CAIF_TestHarness::Report("MultiModalEmbed::TabularGradient",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MultiModalEmbed::TabularGradient")
}

//------------------------------------------------------------------------------
// Test: SpectrogramEmbedding forward
//------------------------------------------------------------------------------
void CAIF_MultiModalEmbeddingTests::TestSpectrogramEmbeddingForward()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    constexpr uint32_t freq_bins=80;
    constexpr uint32_t dim=64;
    constexpr uint32_t batch=2;
    constexpr uint32_t time_frames=10;

    CAIF_DeviceSpectrogramEmbeddingConfig config(freq_bins,dim,false);

    CAIF_DeviceSpectrogramEmbedding<float,float> embed(config,stream);

    // Create input [batch, time_frames, freq_bins]
    std::vector<float> input_data(batch*time_frames*freq_bins);
    std::mt19937 rng(g_caif_mmembed_test_seed_4);
    std::uniform_real_distribution<float> dist(g_caif_mmembed_test_rand_min,g_caif_mmembed_test_rand_max);
    for(auto &v:input_data)
    {
      v=dist(rng);
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                             {batch,time_frames,freq_bins},
                                                             stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=embed.Forward(input,ctx);

    // Check output shape: [batch, time_frames, dim]
    const auto &shape=output.Shape();
    const bool passed=(shape.size()==3&&shape[0]==batch&&shape[1]==time_frames&&shape[2]==dim);
    CAIF_TestHarness::Report("MultiModalEmbed::SpectrogramForward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MultiModalEmbed::SpectrogramForward")
}

//------------------------------------------------------------------------------
// Test: SpectrogramEmbedding with CLS token
//------------------------------------------------------------------------------
void CAIF_MultiModalEmbeddingTests::TestSpectrogramEmbeddingWithCLS()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    constexpr uint32_t freq_bins=40;
    constexpr uint32_t dim=32;
    constexpr uint32_t batch=2;
    constexpr uint32_t time_frames=5;

    CAIF_DeviceSpectrogramEmbeddingConfig config(freq_bins,dim,true);

    CAIF_DeviceSpectrogramEmbedding<float,float> embed(config,stream);

    // Create input [batch, time_frames, freq_bins]
    std::vector<float> input_data(batch*time_frames*freq_bins);
    std::mt19937 rng(g_caif_mmembed_test_seed_5);
    std::uniform_real_distribution<float> dist(g_caif_mmembed_test_rand_min,g_caif_mmembed_test_rand_max);
    for(auto &v:input_data)
    {
      v=dist(rng);
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                             {batch,time_frames,freq_bins},
                                                             stream);
    ctx.SetTraining(false);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=embed.Forward(input,ctx);

    // Check output shape: [batch, time_frames+1, dim] (CLS token prepended)
    const auto &shape=output.Shape();
    bool passed=(shape.size()==3&&shape[0]==batch&&shape[1]==time_frames+1&&shape[2]==dim);

    if(passed==true)
    {
      // Verify CLS token is the same across batches
      std::vector<float> output_host=ToVector(output.ToHost());
      std::vector<float> cls_token=ToVector(embed.ParameterTensor(2).ToHost());
      for(uint32_t b=0;b<batch&&passed==true;++b)
      {
        for(uint32_t d=0;d<dim;++d)
        {
          // Position 0 of each batch
          const float out_val=output_host[b*(time_frames+1)*dim+d];
          const float diff=std::fabs(out_val-cls_token[d]);
          if(diff>g_caif_mmembed_test_tolerance)
          {
            ISE_Out::ErrLog()<<"CLS token mismatch at batch "
                             <<b
                             <<" dim "
                             <<d
                             <<"\n";
            passed=false;
            break;
          }
        }
      }
    }

    CAIF_TestHarness::Report("MultiModalEmbed::SpectrogramWithCLS",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MultiModalEmbed::SpectrogramWithCLS")
}

//------------------------------------------------------------------------------
// Test: SpectrogramEmbedding gradient
//------------------------------------------------------------------------------
void CAIF_MultiModalEmbeddingTests::TestSpectrogramEmbeddingGradient()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    constexpr uint32_t freq_bins=8;
    constexpr uint32_t dim=6;
    constexpr uint32_t batch=2;
    constexpr uint32_t time_frames=3;

    CAIF_DeviceSpectrogramEmbeddingConfig config(freq_bins,dim,false);

    CAIF_DeviceSpectrogramEmbedding<float,float> embed(config,stream);

    std::vector<float> input_data(batch*time_frames*freq_bins);
    std::mt19937 rng(g_caif_mmembed_test_seed_6);
    std::uniform_real_distribution<float> dist(g_caif_mmembed_test_rand_min,g_caif_mmembed_test_rand_max);
    for(auto &v:input_data)
    {
      v=dist(rng);
    }

    // Forward and backward
    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                             {batch,time_frames,freq_bins},
                                                             stream);
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=embed.Forward(input,ctx);

    // Create grad_output (ones)
    std::vector<float> grad_out_data(batch*time_frames*dim,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(grad_out_data.data(),
                                                                   {batch,time_frames,dim},
                                                                   stream);

    embed.ZeroGradients();
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=embed.Backward(grad_output,ctx);

    // Just check shapes
    const auto &grad_shape=grad_input.Shape();
    const bool passed=(grad_shape.size()==3&&grad_shape[0]==batch&&
                       grad_shape[1]==time_frames&&grad_shape[2]==freq_bins);
    CAIF_TestHarness::Report("MultiModalEmbed::SpectrogramGradient",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MultiModalEmbed::SpectrogramGradient")
}

//------------------------------------------------------------------------------
// Test: ViT model construction and forward
//------------------------------------------------------------------------------
void CAIF_MultiModalEmbeddingTests::TestViTModelForward()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    constexpr uint32_t image_height=32;
    constexpr uint32_t image_width=32;
    constexpr uint32_t channels=3;
    constexpr uint32_t patch_size=8;
    constexpr uint32_t dim=64;
    constexpr uint32_t num_layers=2;
    constexpr uint32_t num_heads=4;
    constexpr uint32_t ffn_dim=128;
    constexpr uint32_t num_classes=10;
    constexpr uint32_t batch=2;
    constexpr float dropout_rate=g_caif_mmembed_test_dropout;

    CAIF_DeviceViTModelConfig config(image_height,
                                     image_width,
                                     channels,
                                     patch_size,
                                     dim,
                                     num_layers,
                                     num_heads,
                                     ffn_dim,
                                     dropout_rate,
                                     num_classes,
                                     false,
                                     g_caif_mmembed_test_rope_base);
    CAIF_DeviceViTModel<float,float> vit(config,stream);

    // Check computed properties
    constexpr uint32_t expected_num_patches=(image_height/patch_size)*(image_width/patch_size);
    bool passed=(vit.NumPatches()==expected_num_patches&&
                 vit.SequenceLength()==expected_num_patches+1);

    if(passed==true)
    {
      // Create input [batch, H, W, C]
      std::vector<float> input_data(batch*image_height*image_width*channels);
      std::mt19937 rng(g_caif_mmembed_test_seed_7);
      std::uniform_real_distribution<float> dist(g_caif_mmembed_test_img_min,g_caif_mmembed_test_img_max);
      for(auto &v:input_data)
      {
        v=dist(rng);
      }

      CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                               {batch,image_height,image_width,channels},
                                                               stream);
      ctx.SetTraining(false);
      ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
      CAIF_DeviceTensor output=vit.Forward(input,ctx);

      // Check output shape: [batch, num_classes]
      const auto &shape=output.Shape();
      if(shape.size()!=2||shape[0]!=batch||shape[1]!=num_classes)
      {
        passed=false;
      }

      if(passed==true)
      {
        // Verify output contains valid values (not NaN/Inf)
        std::vector<float> output_host=ToVector(output.ToHost());
        for(const auto &v:output_host)
        {
          if(std::isfinite(v)==false)
          {
            passed=false;
            break;
          }
        }
      }
    }

    CAIF_TestHarness::Report("MultiModalEmbed::ViTForward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MultiModalEmbed::ViTForward")
}

//------------------------------------------------------------------------------
// Test: ViT model backward
//------------------------------------------------------------------------------
void CAIF_MultiModalEmbeddingTests::TestViTModelBackward()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    constexpr uint32_t image_height=16;
    constexpr uint32_t image_width=16;
    constexpr uint32_t channels=3;
    constexpr uint32_t patch_size=8;
    constexpr uint32_t dim=32;
    constexpr uint32_t num_layers=1;
    constexpr uint32_t num_heads=2;
    constexpr uint32_t ffn_dim=64;
    constexpr uint32_t num_classes=5;
    constexpr uint32_t batch=2;
    constexpr float dropout_rate=g_caif_mmembed_test_dropout;

    CAIF_DeviceViTModelConfig config(image_height,
                                     image_width,
                                     channels,
                                     patch_size,
                                     dim,
                                     num_layers,
                                     num_heads,
                                     ffn_dim,
                                     dropout_rate,
                                     num_classes,
                                     false,
                                     g_caif_mmembed_test_rope_base);
    CAIF_DeviceViTModel<float,float> vit(config,stream);

    // Create input
    std::vector<float> input_data(batch*image_height*image_width*channels);
    std::mt19937 rng(g_caif_mmembed_test_seed_8);
    std::uniform_real_distribution<float> dist(g_caif_mmembed_test_img_min,g_caif_mmembed_test_img_max);
    for(auto &v:input_data)
    {
      v=dist(rng);
    }

    CAIF_DeviceTensor input=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                             {batch,image_height,image_width,channels},
                                                             stream);

    // Forward with training=true
    ctx.SetTraining(true);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    CAIF_DeviceTensor output=vit.Forward(input,ctx);

    // Create grad_output
    std::vector<float> grad_out_data(batch*num_classes,1.0f);
    CAIF_DeviceTensor grad_output=CAIF_DeviceTensor::FromHostData(grad_out_data.data(),
                                                                   {batch,num_classes},
                                                                   stream);

    // Backward
    vit.ZeroGradients();
    ctx.SetPass(CAIF_RunContext::Pass_e::Backward_e);
    CAIF_DeviceTensor grad_input=vit.Backward(grad_output,ctx);

    // Check grad_input shape
    const auto &shape=grad_input.Shape();
    bool passed=(shape.size()==4&&shape[0]==batch&&shape[1]==image_height&&
                 shape[2]==image_width&&shape[3]==channels);

    if(passed==true)
    {
      // Verify gradients are valid
      std::vector<float> grad_host=ToVector(grad_input.ToHost());
      for(const auto &v:grad_host)
      {
        if(std::isfinite(v)==false)
        {
          passed=false;
          break;
        }
      }
    }

    CAIF_TestHarness::Report("MultiModalEmbed::ViTBackward",passed);
  }
  CAIF_TEST_CATCH_BLOCK("MultiModalEmbed::ViTBackward")
}

void CAIF_MultiModalEmbeddingTests::RunAll()
{
  ISE_Out::Out()<<"=== Multi-Modal Embedding Tests ==="
                <<"\n\n";
  TestTabularEmbeddingForward();
  TestTabularEmbedding3D();
  TestTabularEmbeddingGradient();
  TestSpectrogramEmbeddingForward();
  TestSpectrogramEmbeddingWithCLS();
  TestSpectrogramEmbeddingGradient();
  TestViTModelForward();
  TestViTModelBackward();
  ISE_Out::Out()<<"\n=== All Multi-Modal Embedding Tests done ==="
                <<"\n";
}

}//end instance namespace

int main()
{
  instance::CAIF_MultiModalEmbeddingTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
