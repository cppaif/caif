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

#include "caif_device_linear_head.h"
#include "caif_constants.h"
#include "caif_ops.h"
#include "caif_exception.h"
#include <cmath>
#include <vector>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceLinearHead<ComputeT,StorageT>::CAIF_DeviceLinearHead(const Config_t &config,
                                                                CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _weight_tied(false),
                                          _frozen(false),
                                          _weight(),
                                          _weight_grad(),
                                          _tied_weight(nullptr),
                                          _tied_weight_grad(nullptr),
                                          _bias(),
                                          _bias_grad(),
                                          _cached_input(),
                                          _cached_shape()
{
  try
  {
    if(config.input_dim==0)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: input_dim must be > 0");
    }
    if(config.output_dim==0)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: output_dim must be > 0");
    }

    const float limit=std::sqrt(g_caif_xavier_uniform_scale/
                                 static_cast<float>(config.input_dim+config.output_dim));
    const size_t weight_size=static_cast<size_t>(config.input_dim)*config.output_dim;
    std::vector<float> w_init(weight_size);
    for(size_t i=0;i<weight_size;++i)
    {
      const float t=static_cast<float>(i)*g_caif_golden_ratio_frac;
      w_init[i]=(t-std::floor(t))*2.0f*limit-limit;
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    SetWeight(CAIF_DeviceTensor::Uninitialized({config.input_dim,config.output_dim},stream,sdt));
    WeightMut().CopyFromHostFp32(w_init.data(),weight_size);
    _weight_grad=CAIF_DeviceTensor::Zeros({config.input_dim,config.output_dim},stream,sdt);

    if(config.use_bias==true)
    {
      _bias=CAIF_DeviceTensor::Zeros({config.output_dim},stream,sdt);
      _bias_grad=CAIF_DeviceTensor::Zeros({config.output_dim},stream,sdt);
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLinearHead<ComputeT,StorageT>::CAIF_DeviceLinearHead(const Config_t &config,
                                                                CAIF_DeviceTensor &tied_weight,
                                                                CAIF_DeviceTensor &tied_weight_grad,
                                                                CAIF_CudaStream &stream):
                                          CAIF_DeviceLayerTyped<ComputeT,StorageT>(stream),
                                          _config(config),
                                          _weight_tied(true),
                                          _frozen(false),
                                          _weight(),
                                          _weight_grad(),
                                          _tied_weight(&tied_weight),
                                          _tied_weight_grad(&tied_weight_grad),
                                          _bias(),
                                          _bias_grad(),
                                          _cached_input(),
                                          _cached_shape()
{
  try
  {
    if(config.input_dim==0)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: input_dim must be > 0");
    }
    if(config.output_dim==0)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: output_dim must be > 0");
    }

    const std::vector<uint32_t> &tied_shape=tied_weight.Shape();
    if(tied_shape.size()!=2)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: tied weight must be 2D");
    }
    if(tied_shape[0]!=config.output_dim||tied_shape[1]!=config.input_dim)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead: tied weight shape must be [output_dim, input_dim]");
    }

    const CAIF_DataType::CAIF_DataType_e sdt=StorageDtype();
    if(config.use_bias==true)
    {
      _bias=CAIF_DeviceTensor::Zeros({config.output_dim},stream,sdt);
      _bias_grad=CAIF_DeviceTensor::Zeros({config.output_dim},stream,sdt);
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLinearHead<ComputeT,StorageT>::CAIF_DeviceLinearHead(CAIF_DeviceLinearHead &&other):
                              CAIF_DeviceLayerTyped<ComputeT,StorageT>(std::move(other)),
                              _config(other._config),
                              _weight_tied(other._weight_tied),
                              _frozen(other._frozen),
                              _weight(std::move(other._weight)),
                              _weight_grad(std::move(other._weight_grad)),
                              _tied_weight(other._tied_weight),
                              _tied_weight_grad(other._tied_weight_grad),
                              _bias(std::move(other._bias)),
                              _bias_grad(std::move(other._bias_grad)),
                              _cached_input(std::move(other._cached_input)),
                              _cached_shape(std::move(other._cached_shape))
{
  other._tied_weight=nullptr;
  other._tied_weight_grad=nullptr;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceLinearHead<ComputeT,StorageT> &
CAIF_DeviceLinearHead<ComputeT,StorageT>::operator=(CAIF_DeviceLinearHead &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayerTyped<ComputeT,StorageT>::operator=(std::move(other));
    _config=other._config;
    _weight_tied=other._weight_tied;
    _frozen=other._frozen;
    _weight=std::move(other._weight);
    _weight_grad=std::move(other._weight_grad);
    _tied_weight=other._tied_weight;
    _tied_weight_grad=other._tied_weight_grad;
    _bias=std::move(other._bias);
    _bias_grad=std::move(other._bias_grad);
    _cached_input=std::move(other._cached_input);
    _cached_shape=std::move(other._cached_shape);
    other._tied_weight=nullptr;
    other._tied_weight_grad=nullptr;
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceLinearHead<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                       CAIF_RunContext &ctx)
{
  try
  {
    AssertInputDtype(input);
    const std::vector<uint32_t> &shape=input.Shape();
    if(shape.empty()==true)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::Forward: input shape is empty");
    }
    if(shape.back()!=_config.input_dim)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::Forward: last dim must match input_dim");
    }

    uint32_t n=1;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      n*=shape[i];
    }

    CAIF_DeviceTensor flat_input=input.Clone();
    flat_input.Reshape({n,_config.input_dim});

    CAIF_DeviceTensor output=AllocateOutput({n,_config.output_dim},ctx);

    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();
    if(_weight_tied==true)
    {
      CAIF_Ops::MatMulTransposeB(flat_input,*_tied_weight,output,ctx,cdt);
    }
    else
    {
      CAIF_Ops::MatMul(flat_input,_weight,output,ctx,cdt);
    }

    if(_config.use_bias==true)
    {
      CAIF_Ops::BiasAdd(output,_bias,output);
    }

    std::vector<uint32_t> out_shape;
    for(size_t i=0;i<shape.size()-1;++i)
    {
      out_shape.push_back(shape[i]);
    }
    out_shape.push_back(_config.output_dim);
    output.Reshape(out_shape);

    if(ctx.Training()==true)
    {
      _cached_input=input.Clone();
      _cached_shape=std::vector<uint32_t>(shape.begin(),shape.end());
    }

    return output;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DeviceLinearHead<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                        CAIF_RunContext &ctx)
{
  try
  {
    if(_cached_shape.empty()==true)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::Backward: must call Forward with training=true first");
    }

    const std::vector<uint32_t> &grad_shape=grad_output.Shape();
    uint32_t n=1;
    for(size_t i=0;i<grad_shape.size()-1;++i)
    {
      n*=grad_shape[i];
    }

    CAIF_DeviceTensor flat_grad=grad_output.Clone();
    flat_grad.Reshape({n,_config.output_dim});

    CAIF_DeviceTensor flat_input=_cached_input.Clone();
    flat_input.Reshape({n,_config.input_dim});

    CAIF_DeviceTensor grad_input=AllocateOutput({n,_config.input_dim},ctx);

    const CAIF_DataType::CAIF_DataType_e cdt=ComputeDtype();
    if(_frozen==true)
    {
      if(_weight_tied==true)
      {
        CAIF_Ops::MatMul(flat_grad,*_tied_weight,grad_input,ctx,cdt);
      }
      else
      {
        CAIF_Ops::MatMulTransposeB(flat_grad,_weight,grad_input,ctx,cdt);
      }
    }
    else
    {
      if(_config.use_bias==true)
      {
        CAIF_Ops::BiasGradient(flat_grad,_bias_grad);
      }

      if(_weight_tied==true)
      {
        const CAIF_DataType::CAIF_DataType_e wdt=_tied_weight_grad->Dtype();
        CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized(
                                          {_config.output_dim,_config.input_dim},
                                          ctx.Stream(),wdt);
        CAIF_Ops::MatMulTransposeA(flat_grad,flat_input,grad_w_delta,ctx,cdt);
        CAIF_Ops::Add(*_tied_weight_grad,grad_w_delta,*_tied_weight_grad);

        CAIF_Ops::MatMul(flat_grad,*_tied_weight,grad_input,ctx,cdt);
      }
      else
      {
        const CAIF_DataType::CAIF_DataType_e wdt=_weight_grad.Dtype();
        CAIF_DeviceTensor grad_w_delta=CAIF_DeviceTensor::Uninitialized(
                                          {_config.input_dim,_config.output_dim},
                                          ctx.Stream(),wdt);
        CAIF_Ops::MatMulTransposeA(flat_input,flat_grad,grad_w_delta,ctx,cdt);
        CAIF_Ops::Add(_weight_grad,grad_w_delta,_weight_grad);

        CAIF_Ops::MatMulTransposeB(flat_grad,_weight,grad_input,ctx,cdt);
      }
    }

    grad_input.Reshape(_cached_shape);
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLinearHead<ComputeT,StorageT>::ZeroGradients()
{
  try
  {
    if(_weight_tied==false)
    {
      _weight_grad.FillZero();
    }
    if(_config.use_bias==true)
    {
      _bias_grad.FillZero();
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLinearHead<ComputeT,StorageT>::ParameterTensorCount()const
{
  try
  {
    size_t count=0;
    if(_weight_tied==false)
    {
      count+=1;
    }
    if(_config.use_bias==true)
    {
      count+=1;
    }
    return count;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceLinearHead<ComputeT,StorageT>::ParameterTensor(size_t index)
{
  try
  {
    if(_weight_tied==false)
    {
      if(index==0)
      {
        return _weight;
      }
      if(index==1&&_config.use_bias==true)
      {
        return _bias;
      }
    }
    else
    {
      if(index==0&&_config.use_bias==true)
      {
        return _bias;
      }
    }
    THROW_CAIFE("CAIF_DeviceLinearHead::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceLinearHead<ComputeT,StorageT>::ParameterTensor(size_t index)const
{
  try
  {
    if(_weight_tied==false)
    {
      if(index==0)
      {
        return _weight;
      }
      if(index==1&&_config.use_bias==true)
      {
        return _bias;
      }
    }
    else
    {
      if(index==0&&_config.use_bias==true)
      {
        return _bias;
      }
    }
    THROW_CAIFE("CAIF_DeviceLinearHead::ParameterTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor &
CAIF_DeviceLinearHead<ComputeT,StorageT>::GradientTensor(size_t index)
{
  try
  {
    if(_weight_tied==false)
    {
      if(index==0)
      {
        return _weight_grad;
      }
      if(index==1&&_config.use_bias==true)
      {
        return _bias_grad;
      }
    }
    else
    {
      if(index==0&&_config.use_bias==true)
      {
        return _bias_grad;
      }
    }
    THROW_CAIFE("CAIF_DeviceLinearHead::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
const CAIF_DeviceTensor &
CAIF_DeviceLinearHead<ComputeT,StorageT>::GradientTensor(size_t index)const
{
  try
  {
    if(_weight_tied==false)
    {
      if(index==0)
      {
        return _weight_grad;
      }
      if(index==1&&_config.use_bias==true)
      {
        return _bias_grad;
      }
    }
    else
    {
      if(index==0&&_config.use_bias==true)
      {
        return _bias_grad;
      }
    }
    THROW_CAIFE("CAIF_DeviceLinearHead::GradientTensor: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DeviceLinearHead<ComputeT,StorageT>::TotalParameterCount()const
{
  try
  {
    size_t total=0;
    if(_weight_tied==false)
    {
      total+=static_cast<size_t>(_config.input_dim)*_config.output_dim;
    }
    if(_config.use_bias==true)
    {
      total+=_config.output_dim;
    }
    return total;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceLinearHead<ComputeT,StorageT>::Description()const
{
  try
  {
    std::string desc="LinearHead(in="+std::to_string(_config.input_dim)+
                     ",out="+std::to_string(_config.output_dim)+")";
    if(_weight_tied==true)
    {
      desc="LinearHead(in="+std::to_string(_config.input_dim)+
           ",out="+std::to_string(_config.output_dim)+
           ",tied=true)";
    }
    return desc;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceLinearHead<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    if(_weight_tied==false)
    {
      names.push_back(prefix+"weight");
    }
    if(_config.use_bias==true)
    {
      names.push_back(prefix+"bias");
    }
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLinearHead<ComputeT,StorageT>::LoadWeight(CAIF_DeviceTensor &&weight)
{
  try
  {
    if(_weight_tied==true)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::LoadWeight: cannot load weight into a weight-tied head");
    }
    const std::vector<uint32_t> &shape=weight.Shape();
    if(shape.size()!=2 ||
       shape[0]!=_config.input_dim ||
       shape[1]!=_config.output_dim)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::LoadWeight: shape mismatch, expected [input_dim, output_dim]");
    }
    _weight=std::move(weight);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceLinearHead<ComputeT,StorageT>::LoadBias(CAIF_DeviceTensor &&bias)
{
  try
  {
    if(_config.use_bias==false)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::LoadBias: use_bias is false");
    }
    const std::vector<uint32_t> &shape=bias.Shape();
    if(shape.size()!=1 || shape[0]!=_config.output_dim)
    {
      THROW_CAIFE("CAIF_DeviceLinearHead::LoadBias: shape mismatch, expected [output_dim]");
    }
    _bias=std::move(bias);
  }
  CAIF_CATCH_BLOCK()
}

// Explicit instantiations — full 3x3 (ComputeT, StorageT) grid.
template class CAIF_DeviceLinearHead<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceLinearHead<float,__half>;
template class CAIF_DeviceLinearHead<float,__nv_bfloat16>;
template class CAIF_DeviceLinearHead<__half,float>;
template class CAIF_DeviceLinearHead<__half,__half>;
template class CAIF_DeviceLinearHead<__half,__nv_bfloat16>;
template class CAIF_DeviceLinearHead<__nv_bfloat16,float>;
template class CAIF_DeviceLinearHead<__nv_bfloat16,__half>;
template class CAIF_DeviceLinearHead<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
