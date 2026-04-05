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

#include "caif_device_network.h"
#include "caif_device_ops.h"
#include "caif_safetensors_format.h"
#include "caif_cuda_kernels.h"
#include "caif_host_tensor.h"
#include "caif_exception.h"
#include <fstream>
#include <sstream>
#include <cmath>

namespace instance
{

CAIF_DeviceNetwork::CAIF_DeviceNetwork(CAIF_CudaStream &stream):_stream(&stream),
                                                             _layers(),
                                                             _trainable(),
                                                             _input_size(0),
                                                             _output_size(0),
                                                             _adam_initialized(false),
                                                             _adam_lr(0.001f),
                                                             _adam_beta1(0.9f),
                                                             _adam_beta2(0.999f),
                                                             _adam_epsilon(1e-8f),
                                                             _adam_weight_decay(0.0f),
                                                             _adam_t(0),
                                                             _adam_m(),
                                                             _adam_v()
{
}

CAIF_DeviceNetwork::CAIF_DeviceNetwork(CAIF_DeviceNetwork &&other):_stream(other._stream),
                                                                _layers(std::move(other._layers)),
                                                                _trainable(std::move(other._trainable)),
                                                                _input_size(other._input_size),
                                                                _output_size(other._output_size),
                                                                _adam_initialized(other._adam_initialized),
                                                                _adam_lr(other._adam_lr),
                                                                _adam_beta1(other._adam_beta1),
                                                                _adam_beta2(other._adam_beta2),
                                                                _adam_epsilon(other._adam_epsilon),
                                                                _adam_weight_decay(other._adam_weight_decay),
                                                                _adam_t(other._adam_t),
                                                                _adam_m(std::move(other._adam_m)),
                                                                _adam_v(std::move(other._adam_v))
{
  other._stream=nullptr;
  other._input_size=0;
  other._output_size=0;
  other._adam_initialized=false;
}

CAIF_DeviceNetwork &CAIF_DeviceNetwork::operator=(CAIF_DeviceNetwork &&other)
{
  try
  {
    if(this!=&other)
    {
      _stream=other._stream;
      _layers=std::move(other._layers);
      _trainable=std::move(other._trainable);
      _input_size=other._input_size;
      _output_size=other._output_size;
      _adam_initialized=other._adam_initialized;
      _adam_lr=other._adam_lr;
      _adam_beta1=other._adam_beta1;
      _adam_beta2=other._adam_beta2;
      _adam_epsilon=other._adam_epsilon;
      _adam_weight_decay=other._adam_weight_decay;
      _adam_t=other._adam_t;
      _adam_m=std::move(other._adam_m);
      _adam_v=std::move(other._adam_v);
      other._stream=nullptr;
      other._input_size=0;
      other._output_size=0;
      other._adam_initialized=false;
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
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }

    // Auto-infer input size from previous layer if 0
    if(input_size==0)
    {
      if(_layers.empty()==true)
      {
        THROW_CAIFE("DeviceNetwork: first layer must specify input_size");
      }
      input_size=_output_size;
    }

    // Validate layer connectivity
    if(_layers.empty()==false)
    {
      if(input_size!=_output_size)
      {
        THROW_CAIFE("DeviceNetwork: layer input_size must match previous output_size");
      }
    }

    auto layer=std::make_unique<CAIF_DeviceDenseLayer>(input_size,
                                                       output_size,
                                                       activation,
                                                       *_stream,
                                                       use_bias);
    _layers.push_back(std::move(layer));

    // Track network IO sizes
    if(_layers.size()==1)
    {
      _input_size=input_size;
    }
    _output_size=output_size;

    // Invalidate Adam state since architecture changed
    _adam_initialized=false;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::AddLayer(std::unique_ptr<CAIF_DeviceLayer> layer)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }

    if(layer==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: cannot add null layer");
    }

    _layers.push_back(std::move(layer));
    _trainable.push_back(true);

    // Invalidate Adam state since architecture changed
    _adam_initialized=false;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::SetLayerTrainable(size_t index,bool trainable)
{
  try
  {
    if(index>=_layers.size())
    {
      THROW_CAIFE("DeviceNetwork::SetLayerTrainable: index out of range");
    }
    _trainable[index]=trainable;

    // Invalidate Adam state since trainability changed
    _adam_initialized=false;
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_DeviceNetwork::IsLayerTrainable(size_t index)const
{
  try
  {
    if(index>=_layers.size())
    {
      THROW_CAIFE("DeviceNetwork::IsLayerTrainable: index out of range");
    }
    return _trainable[index];
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceNetwork::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }
    if(_layers.empty()==true)
    {
      THROW_CAIFE("DeviceNetwork: no layers added");
    }

    CAIF_DeviceTensor current=input.Clone();

    for(size_t li=0;li<_layers.size();++li)
    {
      current=_layers[li]->Forward(current,training);
    }

    return current;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }
    if(_layers.empty()==true)
    {
      THROW_CAIFE("DeviceNetwork: no layers added");
    }

    CAIF_DeviceTensor grad=grad_output.Clone();

    // Backward through layers in reverse order
    for(auto it=_layers.rbegin();it!=_layers.rend();++it)
    {
      grad=(*it)->Backward(grad);
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::ZeroGradients()
{
  try
  {
    for(auto &layer:_layers)
    {
      layer->ZeroGradients();
    }
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
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }

    _adam_lr=lr;
    _adam_beta1=beta1;
    _adam_beta2=beta2;
    _adam_epsilon=epsilon;
    _adam_weight_decay=weight_decay;
    _adam_t=0;

    // Clear existing state
    _adam_m.clear();
    _adam_v.clear();

    // Create moment tensors only for trainable layers
    for(size_t i=0;i<_layers.size();++i)
    {
      if(_trainable[i]==false)
      {
        continue;
      }
      for(size_t p=0;p<_layers[i]->ParameterTensorCount();++p)
      {
        const auto &shape=_layers[i]->ParameterTensor(p).Shape();
        _adam_m.push_back(
          CAIF_DeviceTensor::Zeros(std::vector<uint32_t>(shape.begin(),shape.end()),*_stream));
        _adam_v.push_back(
          CAIF_DeviceTensor::Zeros(std::vector<uint32_t>(shape.begin(),shape.end()),*_stream));
      }
    }

    _adam_initialized=true;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::AdamStep()
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }
    if(_adam_initialized==false)
    {
      THROW_CAIFE("DeviceNetwork: must call InitializeAdam before AdamStep");
    }

    _adam_t++;

    size_t moment_idx=0;
    for(size_t i=0;i<_layers.size();++i)
    {
      if(_trainable[i]==false)
      {
        continue;
      }
      for(size_t p=0;p<_layers[i]->ParameterTensorCount();++p)
      {
        CAIF_DeviceOps::AdamUpdate(_layers[i]->ParameterTensor(p),
                                  _layers[i]->GradientTensor(p),
                                  _adam_m[moment_idx],
                                  _adam_v[moment_idx],
                                  _adam_lr,
                                  _adam_beta1,
                                  _adam_beta2,
                                  _adam_epsilon,
                                  _adam_weight_decay,
                                  _adam_t);
        moment_idx++;
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

float CAIF_DeviceNetwork::ClipGradientNorm(float max_norm)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }

#ifdef USE_CAIF_CUDA
    // Accumulate sum of squares across all trainable gradient tensors
    CAIF_DeviceTensor total_sq=CAIF_DeviceTensor::Zeros({1},*_stream);

    for(size_t i=0;i<_layers.size();++i)
    {
      if(_trainable[i]==false)
      {
        continue;
      }
      for(size_t p=0;p<_layers[i]->ParameterTensorCount();++p)
      {
        const CAIF_DeviceTensor &grad=_layers[i]->GradientTensor(p);
        const int n=static_cast<int>(grad.TotalElements());
        if(n>0)
        {
          launch_sum_of_squares(grad.DevicePtr(),
                                total_sq.DevicePtr(),
                                n,
                                _stream->Handle());
        }
      }
    }

    // Read total norm to host
    CAIF_HostTensor host_sq=total_sq.ToHost();
    const float total_norm=std::sqrt(host_sq.At({0}));

    // Scale gradients if norm exceeds max_norm
    if(total_norm>max_norm)
    {
      const float clip_coeff=max_norm/total_norm;
      for(size_t i=0;i<_layers.size();++i)
      {
        if(_trainable[i]==false)
        {
          continue;
        }
        for(size_t p=0;p<_layers[i]->ParameterTensorCount();++p)
        {
          CAIF_DeviceTensor &grad=_layers[i]->GradientTensor(p);
          const int n=static_cast<int>(grad.TotalElements());
          if(n>0)
          {
            launch_elementwise_mul_scalar(grad.DevicePtr(),
                                         clip_coeff,
                                         grad.DevicePtr(),
                                         n,
                                         _stream->Handle());
          }
        }
      }
    }

    return total_norm;
#else
    (void)max_norm;
    return 0.0f;
#endif
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceDenseLayer &CAIF_DeviceNetwork::DenseLayer(size_t index)
{
  try
  {
    CAIF_DeviceDenseLayer *dense=dynamic_cast<CAIF_DeviceDenseLayer *>(_layers[index].get());
    if(dense==nullptr)
    {
      THROW_CAIFE("DeviceNetwork::DenseLayer: layer at index is not a dense layer");
    }
    return *dense;
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceDenseLayer &CAIF_DeviceNetwork::DenseLayer(size_t index)const
{
  try
  {
    const CAIF_DeviceDenseLayer *dense=
      dynamic_cast<const CAIF_DeviceDenseLayer *>(_layers[index].get());
    if(dense==nullptr)
    {
      THROW_CAIFE("DeviceNetwork::DenseLayer: layer at index is not a dense layer");
    }
    return *dense;
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceNetwork::TotalParameterCount()const
{
  try
  {
    size_t count=0;
    for(const auto &layer:_layers)
    {
      count+=layer->TotalParameterCount();
    }
    return count;
  }
  CAIF_CATCH_BLOCK()
}

uint32_t CAIF_DeviceNetwork::InputSize()const
{
  try
  {
    return _input_size;
  }
  CAIF_CATCH_BLOCK()
}

uint32_t CAIF_DeviceNetwork::OutputSize()const
{
  try
  {
    return _output_size;
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
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }
    if(_layers.empty()==true)
    {
      THROW_CAIFE("DeviceNetwork: no layers to save");
    }

    // Collect all parameter tensors with their names
    std::vector<std::pair<std::string, const CAIF_DeviceTensor*>> tensors;

    for(size_t layer_idx=0;layer_idx<_layers.size();++layer_idx)
    {
      const auto &layer=_layers[layer_idx];

      // Build layer prefix: "layers.0.", "layers.1.", etc.
      std::string prefix="layers."+std::to_string(layer_idx)+".";

      // Get parameter names for this layer
      std::vector<std::string> param_names=layer->ParameterNames(prefix);

      // Sanity check
      if(param_names.size()!=layer->ParameterTensorCount())
      {
        THROW_CAIFE("DeviceNetwork: ParameterNames count mismatch with ParameterTensorCount");
      }

      // Add each parameter tensor
      for(size_t p=0;p<layer->ParameterTensorCount();++p)
      {
        tensors.push_back({param_names[p],&layer->ParameterTensor(p)});
      }
    }

    // Build metadata
    std::map<std::string, std::string> metadata;
    metadata["format"]="aif_device_network";
    metadata["layer_count"]=std::to_string(_layers.size());

    // Store layer descriptions for reconstruction hints
    std::ostringstream layer_desc;
    for(size_t i=0;i<_layers.size();++i)
    {
      if(i>0)
      {
        layer_desc<<";";
      }
      layer_desc<<_layers[i]->Description();
    }
    metadata["layer_descriptions"]=layer_desc.str();

    // Save using the format
    format.Save(path,tensors,metadata);
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::Load(const std::string &path,const CAIF_ModelFormat &format)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }
    if(_layers.empty()==true)
    {
      THROW_CAIFE("DeviceNetwork: must add layers before loading weights");
    }

    // Load tensors from file
    std::map<std::string, CAIF_DeviceTensor> loaded_tensors=format.Load(path,*_stream);

    // Map loaded tensors to layer parameters
    for(size_t layer_idx=0;layer_idx<_layers.size();++layer_idx)
    {
      auto &layer=_layers[layer_idx];

      // Build layer prefix
      std::string prefix="layers."+std::to_string(layer_idx)+".";

      // Get expected parameter names
      std::vector<std::string> param_names=layer->ParameterNames(prefix);

      // Load each parameter
      for(size_t p=0;p<layer->ParameterTensorCount();++p)
      {
        const std::string &name=param_names[p];

        auto it=loaded_tensors.find(name);
        if(it==loaded_tensors.end())
        {
          THROW_CAIFE(("DeviceNetwork: missing tensor in file: "+name).c_str());
        }

        CAIF_DeviceTensor &param=layer->ParameterTensor(p);
        const CAIF_DeviceTensor &loaded=it->second;

        // Verify shape match
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

        // Copy data from loaded tensor to parameter
        std::vector<float> data(loaded.TotalElements());
        loaded.CopyToHost(data.data());
        param.CopyFromHost(data.data(),data.size());
      }
    }

    // Reset Adam state since weights changed
    _adam_initialized=false;
    _adam_m.clear();
    _adam_v.clear();
  }
  CAIF_CATCH_BLOCK()
}

//------------------------------------------------------------------------------
// Legacy Binary Format Support (Dense Layers Only)
//------------------------------------------------------------------------------

// File format constants
static constexpr uint32_t g_caif_device_network_magic=0x4149464E;  // "AIFN"
static constexpr uint32_t g_caif_device_network_version=1;

void CAIF_DeviceNetwork::SaveModel(const std::string &filepath,bool save_optimizer_state)const
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }

    std::ofstream out(filepath,std::ios::binary);
    if(out.is_open()==false)
    {
      THROW_CAIFE("DeviceNetwork: cannot open file for writing: "+filepath);
    }

    // Write header
    out.write(reinterpret_cast<const char *>(&g_caif_device_network_magic),sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&g_caif_device_network_version),sizeof(uint32_t));

    // Write layer count
    const uint32_t layer_count=static_cast<uint32_t>(_layers.size());
    out.write(reinterpret_cast<const char *>(&layer_count),sizeof(uint32_t));

    // Write each layer's architecture and parameters
    for(const auto &layer:_layers)
    {
      const CAIF_DeviceDenseLayer *dense=
        dynamic_cast<const CAIF_DeviceDenseLayer *>(layer.get());
      if(dense==nullptr)
      {
        THROW_CAIFE("DeviceNetwork::SaveModel: only dense layers are currently serializable");
      }

      // Architecture
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

      // Weights [input_size x output_size]
      const CAIF_DeviceTensor &weights=dense->Weights();
      const size_t weight_count=weights.TotalElements();
      std::vector<float> weight_data(weight_count);
      weights.CopyToHost(weight_data.data());
      out.write(reinterpret_cast<const char *>(weight_data.data()),
                static_cast<std::streamsize>(weight_count*sizeof(float)));

      // Bias [output_size] (if used)
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

    // Write optimizer state flag
    uint8_t has_optimizer=0;
    if(_adam_initialized==true&&save_optimizer_state==true)
    {
      has_optimizer=1;
    }
    out.write(reinterpret_cast<const char *>(&has_optimizer),sizeof(uint8_t));

    if(has_optimizer==1)
    {
      // Write Adam hyperparameters
      out.write(reinterpret_cast<const char *>(&_adam_lr),sizeof(float));
      out.write(reinterpret_cast<const char *>(&_adam_beta1),sizeof(float));
      out.write(reinterpret_cast<const char *>(&_adam_beta2),sizeof(float));
      out.write(reinterpret_cast<const char *>(&_adam_epsilon),sizeof(float));
      out.write(reinterpret_cast<const char *>(&_adam_weight_decay),sizeof(float));
      out.write(reinterpret_cast<const char *>(&_adam_t),sizeof(int));

      // Write moment tensors
      const uint32_t moment_count=static_cast<uint32_t>(_adam_m.size());
      out.write(reinterpret_cast<const char *>(&moment_count),sizeof(uint32_t));

      for(size_t i=0;i<_adam_m.size();++i)
      {
        // M tensor
        const size_t m_count=_adam_m[i].TotalElements();
        std::vector<float> m_data(m_count);
        _adam_m[i].CopyToHost(m_data.data());
        out.write(reinterpret_cast<const char *>(&m_count),sizeof(size_t));
        out.write(reinterpret_cast<const char *>(m_data.data()),
                  static_cast<std::streamsize>(m_count*sizeof(float)));

        // V tensor
        const size_t v_count=_adam_v[i].TotalElements();
        std::vector<float> v_data(v_count);
        _adam_v[i].CopyToHost(v_data.data());
        out.write(reinterpret_cast<const char *>(v_data.data()),
                  static_cast<std::streamsize>(v_count*sizeof(float)));
      }
    }

    out.close();
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceNetwork::LoadModel(const std::string &filepath,bool load_optimizer_state)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceNetwork: network has been moved from");
    }
    if(_layers.empty()==false)
    {
      THROW_CAIFE("DeviceNetwork: cannot load into network with existing layers");
    }

    std::ifstream in(filepath,std::ios::binary);
    if(in.is_open()==false)
    {
      THROW_CAIFE("DeviceNetwork: cannot open file for reading: "+filepath);
    }

    // Read and verify header
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

    // Read layer count
    uint32_t layer_count=0;
    in.read(reinterpret_cast<char *>(&layer_count),sizeof(uint32_t));

    // Read and create each layer
    for(uint32_t i=0;i<layer_count;++i)
    {
      // Read architecture
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

      // Create layer
      auto layer=std::make_unique<CAIF_DeviceDenseLayer>(input_size,
                                                         output_size,
                                                         activation,
                                                         *_stream,
                                                         has_bias);

      // Read weights
      const size_t weight_count=
        static_cast<size_t>(input_size)*static_cast<size_t>(output_size);
      std::vector<float> weight_data(weight_count);
      in.read(reinterpret_cast<char *>(weight_data.data()),
              static_cast<std::streamsize>(weight_count*sizeof(float)));
      layer->Weights().CopyFromHost(weight_data.data(),weight_count);

      // Read bias if present
      if(has_bias==true)
      {
        const size_t bias_count=static_cast<size_t>(output_size);
        std::vector<float> bias_data(bias_count);
        in.read(reinterpret_cast<char *>(bias_data.data()),
                static_cast<std::streamsize>(bias_count*sizeof(float)));
        layer->Bias().CopyFromHost(bias_data.data(),bias_count);
      }

      // Track network IO sizes
      if(i==0)
      {
        _input_size=input_size;
      }
      _output_size=output_size;

      _layers.push_back(std::move(layer));
    }

    // Read optimizer state
    uint8_t has_optimizer=0;
    in.read(reinterpret_cast<char *>(&has_optimizer),sizeof(uint8_t));

    if(has_optimizer==1&&load_optimizer_state==true)
    {
      // Read Adam hyperparameters
      in.read(reinterpret_cast<char *>(&_adam_lr),sizeof(float));
      in.read(reinterpret_cast<char *>(&_adam_beta1),sizeof(float));
      in.read(reinterpret_cast<char *>(&_adam_beta2),sizeof(float));
      in.read(reinterpret_cast<char *>(&_adam_epsilon),sizeof(float));
      in.read(reinterpret_cast<char *>(&_adam_weight_decay),sizeof(float));
      in.read(reinterpret_cast<char *>(&_adam_t),sizeof(int));

      // Read moment tensor count
      uint32_t moment_count=0;
      in.read(reinterpret_cast<char *>(&moment_count),sizeof(uint32_t));

      _adam_m.clear();
      _adam_v.clear();

      for(uint32_t i=0;i<moment_count;++i)
      {
        // Read M tensor
        size_t m_count=0;
        in.read(reinterpret_cast<char *>(&m_count),sizeof(size_t));
        std::vector<float> m_data(m_count);
        in.read(reinterpret_cast<char *>(m_data.data()),
                static_cast<std::streamsize>(m_count*sizeof(float)));

        // Store as 1D tensor since optimizer doesn't care about shape
        CAIF_DeviceTensor m_tensor=
          CAIF_DeviceTensor::Uninitialized({static_cast<uint32_t>(m_count)},*_stream);
        m_tensor.CopyFromHost(m_data.data(),m_count);
        _adam_m.push_back(std::move(m_tensor));

        // Read V tensor (same size as M)
        std::vector<float> v_data(m_count);
        in.read(reinterpret_cast<char *>(v_data.data()),
                static_cast<std::streamsize>(m_count*sizeof(float)));
        CAIF_DeviceTensor v_tensor=
          CAIF_DeviceTensor::Uninitialized({static_cast<uint32_t>(m_count)},*_stream);
        v_tensor.CopyFromHost(v_data.data(),m_count);
        _adam_v.push_back(std::move(v_tensor));
      }

      _adam_initialized=true;
    }
    else if(has_optimizer==1&&load_optimizer_state==false)
    {
      // Skip optimizer state
      float dummy_f;
      int dummy_i;
      in.read(reinterpret_cast<char *>(&dummy_f),sizeof(float));  // lr
      in.read(reinterpret_cast<char *>(&dummy_f),sizeof(float));  // beta1
      in.read(reinterpret_cast<char *>(&dummy_f),sizeof(float));  // beta2
      in.read(reinterpret_cast<char *>(&dummy_f),sizeof(float));  // epsilon
      in.read(reinterpret_cast<char *>(&dummy_f),sizeof(float));  // weight_decay
      in.read(reinterpret_cast<char *>(&dummy_i),sizeof(int));    // t

      uint32_t moment_count=0;
      in.read(reinterpret_cast<char *>(&moment_count),sizeof(uint32_t));

      for(uint32_t i=0;i<moment_count;++i)
      {
        size_t m_count=0;
        in.read(reinterpret_cast<char *>(&m_count),sizeof(size_t));
        // Skip M and V data
        in.seekg(static_cast<std::streamoff>(m_count*sizeof(float)*2),std::ios::cur);
      }

      _adam_initialized=false;
    }

    in.close();
  }
  CAIF_CATCH_BLOCK()
}

}//end instance namespace
