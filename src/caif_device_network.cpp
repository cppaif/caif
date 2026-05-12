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
// Device-resident neural network implementation
//------------------------------------------------------------------------------

#include "caif_device_network.h"
#include "caif_ops.h"
#include "caif_run_context.h"
#include "caif_run_context_scope.h"
#include "caif_safetensors_format.h"
#include "caif_cuda_kernels.h"
#include "caif_host_tensor.h"
#include "caif_exception.h"
// `total_sq` is the fp32 gradient-L2 reduction accumulator. Per-site
// `total_sq.DevicePtr<float>()` reads name this contract inline, per
// the type-dispatch full plan (Phase 2).

#include "caif_adam_optimizer.h"
#include "caif_offloaded_adam.h"
#include "caif_sgd_optimizer.h"
#include "caif_momentum_optimizer.h"
#include "caif_rmsprop_optimizer.h"
#include "caif_adagrad_optimizer.h"
#include <fstream>
#include <sstream>
#include <cmath>

namespace instance
{

CAIF_DeviceNetwork::CAIF_DeviceNetwork(CAIF_CudaStream &stream):CAIF_DeviceContainer(stream),
                                                                _input_size(0),
                                                                _output_size(0),
                                                                _optimizer()
{
}

CAIF_DeviceNetwork::~CAIF_DeviceNetwork()=default;

void CAIF_DeviceNetwork::ClearOptimizer()
{
  _optimizer.reset();
}

void CAIF_DeviceNetwork::SetOptimizer(std::unique_ptr<CAIF_Optimizer> optimizer)
{
  _optimizer=std::move(optimizer);
}

CAIF_DeviceNetwork::CAIF_DeviceNetwork(CAIF_DeviceNetwork &&other):CAIF_DeviceContainer(std::move(other)),
                                                                   _input_size(other._input_size),
                                                                   _output_size(other._output_size),
                                                                   _optimizer(std::move(other._optimizer)),
                                                                   _pending_prefix_lengths(other._pending_prefix_lengths),
                                                                   _pending_encoder_context(other._pending_encoder_context),
                                                                   _pending_grad_encoder_context(other._pending_grad_encoder_context),
                                                                   _pending_position_bias(other._pending_position_bias),
                                                                   _pending_grad_position_bias(other._pending_grad_position_bias)
{
  other._input_size=0;
  other._output_size=0;
  other._pending_prefix_lengths=nullptr;
  other._pending_encoder_context=nullptr;
  other._pending_grad_encoder_context=nullptr;
  other._pending_position_bias=nullptr;
  other._pending_grad_position_bias=nullptr;
}

CAIF_DeviceNetwork &CAIF_DeviceNetwork::operator=(CAIF_DeviceNetwork &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceContainer::operator=(std::move(other));
      _input_size=other._input_size;
      _output_size=other._output_size;
      SetOptimizer(std::move(other._optimizer));
      _pending_prefix_lengths=other._pending_prefix_lengths;
      _pending_encoder_context=other._pending_encoder_context;
      _pending_grad_encoder_context=other._pending_grad_encoder_context;
      _pending_position_bias=other._pending_position_bias;
      _pending_grad_position_bias=other._pending_grad_position_bias;
      other._input_size=0;
      other._output_size=0;
      other._pending_prefix_lengths=nullptr;
      other._pending_encoder_context=nullptr;
      other._pending_grad_encoder_context=nullptr;
      other._pending_position_bias=nullptr;
      other._pending_grad_position_bias=nullptr;
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::AddDenseLayer(uint32_t input_size,
                                       uint32_t output_size,
                                       CAIF_DeviceActivation_e activation,
                                       bool use_bias)
{
  try
  {
    if(input_size==0)
    {
      if(LayerCount()==0)
      {
        THROW_CAIFE("DeviceNetwork: first layer must specify input_size");
      }
      input_size=_output_size;
    }

    if(LayerCount()!=0)
    {
      if(input_size!=_output_size)
      {
        THROW_CAIFE("DeviceNetwork: layer input_size must match previous output_size");
      }
    }

    std::unique_ptr<CAIF_DeviceDenseLayer<float,float>> layer=
      std::make_unique<CAIF_DeviceDenseLayer<float,float>>(input_size,
                                              output_size,
                                              activation,
                                              Stream(),
                                              use_bias);
    CAIF_DeviceContainer::AddLayer(std::move(layer));

    if(LayerCount()==1)
    {
      _input_size=input_size;
    }
    _output_size=output_size;

    ClearOptimizer();
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::AddLayer(std::unique_ptr<CAIF_DeviceLayer> layer)
{
  try
  {
    CAIF_DeviceContainer::AddLayer(std::move(layer));
    ClearOptimizer();
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::SetLayerTrainable(size_t index,bool trainable)
{
  try
  {
    CAIF_DeviceContainer::SetLayerTrainable(index,trainable);
    ClearOptimizer();
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::StashSidebandIntoContext(CAIF_RunContext &ctx)const
{
  try
  {
    if(_pending_prefix_lengths!=nullptr)
    {
      ctx.SetPrefixLengths(*_pending_prefix_lengths);
    }
    if(_pending_encoder_context!=nullptr)
    {
      ctx.SetEncoderContext(*_pending_encoder_context);
    }
    if(_pending_grad_encoder_context!=nullptr)
    {
      ctx.SetGradEncoderContext(*_pending_grad_encoder_context);
    }
    if(_pending_position_bias!=nullptr)
    {
      ctx.SetPositionBias(*_pending_position_bias);
    }
    if(_pending_grad_position_bias!=nullptr)
    {
      ctx.SetGradPositionBias(*_pending_grad_position_bias);
    }
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceNetwork::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(LayerCount()==0)
    {
      THROW_CAIFE("DeviceNetwork: no layers added");
    }

    CAIF_RunContext ctx;
    ctx.SetStream(Stream());
    ctx.SetTraining(training);
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);
    StashSidebandIntoContext(ctx);

    return CAIF_DeviceLayer::Forward(input,ctx);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(LayerCount()==0)
    {
      THROW_CAIFE("DeviceNetwork: no layers added");
    }

    CAIF_RunContext ctx;
    ctx.SetStream(Stream());
    ctx.SetTraining(true);
    CAIF_RunContextPassScope pass_scope(ctx,CAIF_RunContext::Pass_e::Backward_e);
    StashSidebandIntoContext(ctx);

    CAIF_DeviceLayer::Backward(grad_output,ctx);
  }
  CAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceNetwork::Description()const
{
  try
  {
    return "Network("+std::to_string(LayerCount())+" layers)";
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::InitializeAdam(float lr,
                                        float beta1,
                                        float beta2,
                                        float epsilon,
                                        float weight_decay)
{
  try
  {
    SetOptimizer(std::make_unique<CAIF_AdamOptimizer>(lr,beta1,beta2,epsilon,
                                                      weight_decay,Stream()));
    Optimizer().Initialize(*this);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::InitializeOffloadedAdam(float lr,
                                                  float beta1,
                                                  float beta2,
                                                  float epsilon,
                                                  float weight_decay)
{
  try
  {
    SetOptimizer(std::make_unique<CAIF_OffloadedAdam>(lr,beta1,beta2,epsilon,
                                                       weight_decay,Stream()));
    Optimizer().Initialize(*this);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::InitializeSgd(float lr,float weight_decay)
{
  try
  {
    SetOptimizer(std::make_unique<CAIF_SgdOptimizer>(lr,weight_decay,Stream()));
    Optimizer().Initialize(*this);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::InitializeMomentum(float lr,
                                            float momentum,
                                            float weight_decay)
{
  try
  {
    SetOptimizer(std::make_unique<CAIF_MomentumOptimizer>(lr,momentum,
                                                          weight_decay,Stream()));
    Optimizer().Initialize(*this);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::InitializeRmsprop(float lr,
                                           float alpha,
                                           float epsilon,
                                           float weight_decay)
{
  try
  {
    SetOptimizer(std::make_unique<CAIF_RmspropOptimizer>(lr,alpha,epsilon,
                                                         weight_decay,Stream()));
    Optimizer().Initialize(*this);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::InitializeAdaGrad(float lr,
                                           float epsilon,
                                           float weight_decay)
{
  try
  {
    SetOptimizer(std::make_unique<CAIF_AdaGradOptimizer>(lr,epsilon,
                                                         weight_decay,Stream()));
    Optimizer().Initialize(*this);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::OptimizerStep()
{
  try
  {
    if(HasOptimizer()==false)
    {
      THROW_CAIFE("DeviceNetwork: must call Initialize{Adam,Sgd,Momentum,"
                  "Rmsprop,AdaGrad} before OptimizerStep");
    }
    Optimizer().Step(*this);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::SetLearningRate(float lr)
{
  try
  {
    if(HasOptimizer()==false)
    {
      THROW_CAIFE("DeviceNetwork: must call Initialize{Adam,Sgd,Momentum,"
                  "Rmsprop,AdaGrad} before SetLearningRate");
    }
    Optimizer().SetLearningRate(lr);
  }
  CAIF_CATCH_BLOCK()
}

float CAIF_DeviceNetwork::GradientNormSquared()
{
  try
  {
#ifdef USE_CAIF_CUDA
    CAIF_DeviceTensor total_sq=CAIF_DeviceTensor::Zeros({1},Stream());

    for(size_t i=0;i<LayerCount();++i)
    {
      if(IsLayerTrainable(i)==false)
      {
        continue;
      }
      CAIF_DeviceLayer &layer=Layer(i);
      for(size_t p=0;p<layer.ParameterTensorCount();++p)
      {
        const CAIF_DeviceTensor &grad=layer.GradientTensor(p);
        const int n=static_cast<int>(grad.TotalElements());
        if(n>0)
        {
          // Per-layer gradient dtype follows that layer's template
          // instantiation; the network is dtype-erased so we dispatch
          // on the runtime dtype here. total_sq is fp32 and accumulates
          // across all layers.
          switch(grad.Dtype())
          {
            case CAIF_DataType::CAIF_DataType_e::Float32:
              launch_sum_of_squares<float>(grad.DevicePtr<float>(),
                                            total_sq.DevicePtr<float>(),  // fp32: gradient-L2 accumulator
                                            n,
                                            Stream().Handle());
              break;
            case CAIF_DataType::CAIF_DataType_e::Float16:
              launch_sum_of_squares<__half>(grad.DevicePtr<__half>(),
                                             total_sq.DevicePtr<float>(),  // fp32: gradient-L2 accumulator
                                             n,
                                             Stream().Handle());
              break;
            case CAIF_DataType::CAIF_DataType_e::BFloat16:
              launch_sum_of_squares<__nv_bfloat16>(
                                grad.DevicePtr<__nv_bfloat16>(),
                                total_sq.DevicePtr<float>(),  // fp32: gradient-L2 accumulator
                                n,
                                Stream().Handle());
              break;
            default:
              THROW_CAIFE("CAIF_DeviceNetwork::GradientNormSquared: "
                          "unsupported gradient dtype");
          }
        }
      }
    }

    CAIF_HostTensor host_sq=total_sq.ToHost();
    return host_sq.At({0});
#else
    return 0.0f;
#endif
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::ScaleGradients(float coef)
{
  try
  {
#ifdef USE_CAIF_CUDA
    if(coef==1.0f)
    {
      return;
    }
    for(size_t i=0;i<LayerCount();++i)
    {
      if(IsLayerTrainable(i)==false)
      {
        continue;
      }
      CAIF_DeviceLayer &layer=Layer(i);
      for(size_t p=0;p<layer.ParameterTensorCount();++p)
      {
        CAIF_DeviceTensor &grad=layer.GradientTensor(p);
        const int n=static_cast<int>(grad.TotalElements());
        if(n>0)
        {
          CAIF_Ops::Scale(grad,coef);
        }
      }
    }
#else
    (void)coef;
#endif
  }
  CAIF_CATCH_BLOCK()
}

float CAIF_DeviceNetwork::ClipGradientNorm(float max_norm)
{
  try
  {
#ifdef USE_CAIF_CUDA
    const float total_norm=std::sqrt(GradientNormSquared());
    if(total_norm>max_norm)
    {
      ScaleGradients(max_norm/total_norm);
    }
    return total_norm;
#else
    (void)max_norm;
    return 0.0f;
#endif
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceDenseLayer<float,float> &CAIF_DeviceNetwork::DenseLayer(size_t index)
{
  try
  {
    CAIF_DeviceDenseLayer<float,float> *dense=dynamic_cast<CAIF_DeviceDenseLayer<float,float> *>(&Layer(index));
    if(dense==nullptr)
    {
      THROW_CAIFE("DeviceNetwork::DenseLayer: layer at index is not a dense layer");
    }
    return *dense;
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceDenseLayer<float,float> &CAIF_DeviceNetwork::DenseLayer(size_t index)const
{
  try
  {
    const CAIF_DeviceDenseLayer<float,float> *dense=
      dynamic_cast<const CAIF_DeviceDenseLayer<float,float> *>(&Layer(index));
    if(dense==nullptr)
    {
      THROW_CAIFE("DeviceNetwork::DenseLayer: layer at index is not a dense layer");
    }
    return *dense;
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// SafeTensors Format Support
//------------------------------------------------------------------------------

void CAIF_DeviceNetwork::SaveSafeTensors(const std::string &path)const
{
  try
  {
    CAIF_SafeTensorsFormat format;
    Save(path,format);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::LoadSafeTensors(const std::string &path)
{
  try
  {
    CAIF_SafeTensorsFormat format;
    Load(path,format);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::Save(const std::string &path,const CAIF_ModelFormat &format)const
{
  try
  {
    if(HasStream()==false)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }
    if(LayerCount()==0)
    {
      THROW_CAIFE("DeviceNetwork: no layers to save");
    }

    std::vector<std::pair<std::string, const CAIF_DeviceTensor*>> tensors;

    for(size_t layer_idx=0;layer_idx<LayerCount();++layer_idx)
    {
      const CAIF_DeviceLayer &layer=Layer(layer_idx);

      std::string prefix="layers."+std::to_string(layer_idx)+".";
      std::vector<std::string> param_names=layer.ParameterNames(prefix);

      if(param_names.size()!=layer.ParameterTensorCount())
      {
        THROW_CAIFE("DeviceNetwork: ParameterNames count mismatch with ParameterTensorCount");
      }

      for(size_t p=0;p<layer.ParameterTensorCount();++p)
      {
        tensors.push_back({param_names[p],&layer.ParameterTensor(p)});
      }
    }

    std::map<std::string, std::string> metadata;
    metadata["format"]="aif_device_network";
    metadata["layer_count"]=std::to_string(LayerCount());

    std::ostringstream layer_desc;
    for(size_t i=0;i<LayerCount();++i)
    {
      if(i>0)
      {
        layer_desc<<";";
      }
      layer_desc<<Layer(i).Description();
    }
    metadata["layer_descriptions"]=layer_desc.str();

    format.Save(path,tensors,metadata);
  }
  CAIF_CATCH_BLOCK()
}

static bool IsFloatDtype(CAIF_DataType::CAIF_DataType_e dt)
{
  return dt==CAIF_DataType::CAIF_DataType_e::Float32
       ||dt==CAIF_DataType::CAIF_DataType_e::Float16
       ||dt==CAIF_DataType::CAIF_DataType_e::BFloat16;
}

void CAIF_DeviceNetwork::Load(const std::string &path,const CAIF_ModelFormat &format)
{
  try
  {
    if(LayerCount()==0)
    {
      THROW_CAIFE("DeviceNetwork: must add layers before loading weights");
    }

    std::map<std::string, CAIF_DeviceTensor> loaded_tensors=format.Load(path,Stream());

    CAIF_RunContext ctx;
    ctx.SetStream(Stream());

    for(size_t layer_idx=0;layer_idx<LayerCount();++layer_idx)
    {
      CAIF_DeviceLayer &layer=Layer(layer_idx);

      std::string prefix="layers."+std::to_string(layer_idx)+".";
      std::vector<std::string> param_names=layer.ParameterNames(prefix);

      for(size_t p=0;p<layer.ParameterTensorCount();++p)
      {
        const std::string &name=param_names[p];

        auto it=loaded_tensors.find(name);
        if(it==loaded_tensors.end())
        {
          THROW_CAIFE(("DeviceNetwork: missing tensor in file: "+name).c_str());
        }

        CAIF_DeviceTensor &param=layer.ParameterTensor(p);
        const CAIF_DeviceTensor &loaded=it->second;

        if(param.Shape()!=loaded.Shape())
        {
          std::ostringstream msg;
          msg<<"DeviceNetwork: shape mismatch for "<<name<<": expected [";
          for(size_t s=0;s<param.Shape().size();++s)
          {
            if(s>0)
            {
              msg<<",";
            }
            msg<<param.Shape()[s];
          }
          msg<<"] but got [";
          for(size_t s=0;s<loaded.Shape().size();++s)
          {
            if(s>0)
            {
              msg<<",";
            }
            msg<<loaded.Shape()[s];
          }
          msg<<"]";
          THROW_CAIFE(msg.str().c_str());
        }

        if(param.Dtype()==loaded.Dtype())
        {
          // Same dtype — raw-byte copy works for any dtype (including
          // quantized/packed int8/int4 weights).
          std::vector<char> bytes(loaded.SizeBytes());
          loaded.CopyToHostRaw(bytes.data());
          param.CopyFromHostRaw(bytes.data(),bytes.size());
        }
        else
        {
          // Cross-dtype: cast on load.  Supported for float dtypes only
          // (fp32 / fp16 / bf16).  Quantized dtypes would need dequant
          // (int→float) or requant (float→int) kernels, which aren't wired
          // into the on-load path yet; reject loudly.
          if(IsFloatDtype(param.Dtype())==false||IsFloatDtype(loaded.Dtype())==false)
          {
            std::ostringstream msg;
            msg<<"DeviceNetwork: on-load dtype conversion not supported for "
               <<name
               <<": loaded="
               <<CAIF_DataType(loaded.Dtype()).Name()
               <<" target="
               <<CAIF_DataType(param.Dtype()).Name()
               <<" (only fp32/fp16/bf16 cross-dtype loads are supported)";
            THROW_CAIFE(msg.str().c_str());
          }
          CAIF_Ops::Cast(loaded,param,ctx);
        }
      }
    }

    ClearOptimizer();
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Legacy Binary Format Support (Dense Layers Only)
//------------------------------------------------------------------------------

static constexpr uint32_t g_caif_device_network_magic=0x4149464E;  // "AIFN"
static constexpr uint32_t g_caif_device_network_version=1;

void CAIF_DeviceNetwork::SaveModel(const std::string &filepath,bool save_optimizer_state)const
{
  try
  {
    if(HasStream()==false)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }

    std::ofstream out(filepath,std::ios::binary);
    if(out.is_open()==false)
    {
      THROW_CAIFE("DeviceNetwork: cannot open file for writing: "+filepath);
    }

    out.write(reinterpret_cast<const char *>(&g_caif_device_network_magic),sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&g_caif_device_network_version),sizeof(uint32_t));

    const uint32_t layer_count=static_cast<uint32_t>(LayerCount());
    out.write(reinterpret_cast<const char *>(&layer_count),sizeof(uint32_t));

    for(size_t idx=0;idx<LayerCount();++idx)
    {
      const CAIF_DeviceDenseLayer<float,float> *dense=
        dynamic_cast<const CAIF_DeviceDenseLayer<float,float> *>(&Layer(idx));
      if(dense==nullptr)
      {
        THROW_CAIFE("DeviceNetwork::SaveModel: only dense layers are currently serializable");
      }

      const uint32_t input_size=dense->InputSize();
      const uint32_t output_size=dense->OutputSize();
      const int32_t activation=static_cast<int32_t>(dense->Activation());
      uint8_t use_bias=0;
      if(dense->UseBias()==true)
      {
        use_bias=1;
      }

      out.write(reinterpret_cast<const char *>(&input_size),sizeof(uint32_t));
      out.write(reinterpret_cast<const char *>(&output_size),sizeof(uint32_t));
      out.write(reinterpret_cast<const char *>(&activation),sizeof(int32_t));
      out.write(reinterpret_cast<const char *>(&use_bias),sizeof(uint8_t));

      const CAIF_DeviceTensor &weights=dense->Weights();
      const size_t weight_count=weights.TotalElements();
      std::vector<float> weight_data(weight_count);
      weights.CopyToHost(weight_data.data());
      out.write(reinterpret_cast<const char *>(weight_data.data()),
                static_cast<std::streamsize>(weight_count*sizeof(float)));

      if(dense->UseBias()==true)
      {
        const CAIF_DeviceTensor &bias=dense->Bias();
        const size_t bias_count=bias.TotalElements();
        std::vector<float> bias_data(bias_count);
        bias.CopyToHost(bias_data.data());
        out.write(reinterpret_cast<const char *>(bias_data.data()),
                  static_cast<std::streamsize>(bias_count*sizeof(float)));
      }
    }

    // Legacy binary format (deprecated in favor of SafeTensors) no
    // longer ships optimizer state — the field is kept for backward
    // file-format compatibility but always serialized as 0. Callers
    // that want optimizer-state checkpointing should switch to the
    // SafeTensors path.
    (void)save_optimizer_state;
    const uint8_t has_optimizer=0;
    out.write(reinterpret_cast<const char *>(&has_optimizer),sizeof(uint8_t));

    out.close();
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::LoadModel(const std::string &filepath,bool load_optimizer_state)
{
  try
  {
    if(LayerCount()!=0)
    {
      THROW_CAIFE("DeviceNetwork: cannot load into network with existing layers");
    }

    std::ifstream in(filepath,std::ios::binary);
    if(in.is_open()==false)
    {
      THROW_CAIFE("DeviceNetwork: cannot open file for reading: "+filepath);
    }

    uint32_t magic=0;
    uint32_t version=0;
    in.read(reinterpret_cast<char *>(&magic),sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&version),sizeof(uint32_t));

    if(magic!=g_caif_device_network_magic)
    {
      THROW_CAIFE("DeviceNetwork: invalid file format (bad magic number)");
    }
    if(version!=g_caif_device_network_version)
    {
      THROW_CAIFE("DeviceNetwork: unsupported file version");
    }

    uint32_t layer_count=0;
    in.read(reinterpret_cast<char *>(&layer_count),sizeof(uint32_t));

    for(uint32_t i=0;i<layer_count;++i)
    {
      uint32_t input_size=0;
      uint32_t output_size=0;
      int32_t activation_int=0;
      uint8_t use_bias=0;

      in.read(reinterpret_cast<char *>(&input_size),sizeof(uint32_t));
      in.read(reinterpret_cast<char *>(&output_size),sizeof(uint32_t));
      in.read(reinterpret_cast<char *>(&activation_int),sizeof(int32_t));
      in.read(reinterpret_cast<char *>(&use_bias),sizeof(uint8_t));

      const CAIF_DeviceActivation_e activation=
        static_cast<CAIF_DeviceActivation_e>(activation_int);
      const bool has_bias=(use_bias!=0);

      std::unique_ptr<CAIF_DeviceDenseLayer<float,float>> layer=
        std::make_unique<CAIF_DeviceDenseLayer<float,float>>(input_size,
                                                output_size,
                                                activation,
                                                Stream(),
                                                has_bias);

      const size_t weight_count=
        static_cast<size_t>(input_size)*static_cast<size_t>(output_size);
      std::vector<float> weight_data(weight_count);
      in.read(reinterpret_cast<char *>(weight_data.data()),
              static_cast<std::streamsize>(weight_count*sizeof(float)));
      layer->Weights().CopyFromHost(weight_data.data(),weight_count);

      if(has_bias==true)
      {
        const size_t bias_count=static_cast<size_t>(output_size);
        std::vector<float> bias_data(bias_count);
        in.read(reinterpret_cast<char *>(bias_data.data()),
                static_cast<std::streamsize>(bias_count*sizeof(float)));
        layer->Bias().CopyFromHost(bias_data.data(),bias_count);
      }

      if(i==0)
      {
        _input_size=input_size;
      }
      _output_size=output_size;

      CAIF_DeviceContainer::AddLayer(std::move(layer));
    }

    // Legacy binary format no longer round-trips optimizer state. If
    // an old file claims to have a trailing Adam state block (the
    // optimizer-state serialization predates the optimizer-abstraction
    // refactor), we skip past it without reconstructing anything.
    // SafeTensors is the supported path for stateful checkpointing.
    (void)load_optimizer_state;
    uint8_t has_optimizer=0;
    in.read(reinterpret_cast<char *>(&has_optimizer),sizeof(uint8_t));

    if(has_optimizer==1)
    {
      // Skip the legacy Adam-state trailer: 5 floats (lr, beta1, beta2,
      // epsilon, weight_decay) + 1 int (t) + uint32 moment_count + per
      // moment (size_t m_count + 2*m_count floats for m and v).
      in.seekg(static_cast<std::streamoff>(5*sizeof(float)+sizeof(int)),std::ios::cur);
      uint32_t moment_count=0;
      in.read(reinterpret_cast<char *>(&moment_count),sizeof(uint32_t));
      for(uint32_t i=0;i<moment_count;++i)
      {
        size_t m_count=0;
        in.read(reinterpret_cast<char *>(&m_count),sizeof(size_t));
        in.seekg(static_cast<std::streamoff>(m_count*sizeof(float)*2),std::ios::cur);
      }
    }
    ClearOptimizer();

    in.close();
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
