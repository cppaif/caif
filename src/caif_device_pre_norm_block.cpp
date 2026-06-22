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

#include "caif_device_pre_norm_block.h"
#include "caif_ops.h"
#include "caif_constants.h"
#include "caif_serialization_constants.h"
#include "caif_exception.h"

#include <sstream>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DevicePreNormBlock<ComputeT,StorageT>::CAIF_DevicePreNormBlock(SubLayerVec_t sub_layers,
                                                                    CAIF_CudaStream &stream):
                                                                    CAIF_DeviceContainer(stream),
                                                                    _norm_prefixes(),
                                                                    _layer_prefixes(),
                                                                    _norms_trainable(true),
                                                                    _checkpointed(false),
                                                                    _saved_input(),
                                                                    _offload_scheduler()
{
  try
  {
    if(sub_layers.empty()==true)
    {
      THROW_CAIFE("PreNormBlock: must have at least one sub-layer");
    }

    NormPrefixes().reserve(sub_layers.size());
    LayerPrefixes().reserve(sub_layers.size());

    for(size_t i=0;i<sub_layers.size();++i)
    {
      if(sub_layers[i].norm==nullptr)
      {
        THROW_CAIFE("PreNormBlock: sub-layer "+std::to_string(i)+" has null norm");
      }
      if(sub_layers[i].layer==nullptr)
      {
        THROW_CAIFE("PreNormBlock: sub-layer "+std::to_string(i)+" has null layer");
      }

      NormPrefixes().push_back(std::move(sub_layers[i].norm_prefix));
      LayerPrefixes().push_back(std::move(sub_layers[i].layer_prefix));
      AddLayer(std::move(sub_layers[i].norm));
      AddLayer(std::move(sub_layers[i].layer));
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePreNormBlock<ComputeT,StorageT>::CAIF_DevicePreNormBlock(
                                                  CAIF_DevicePreNormBlock &&other):
                                          CAIF_DeviceContainer(std::move(other)),
                                          _norm_prefixes(std::move(other._norm_prefixes)),
                                          _layer_prefixes(std::move(other._layer_prefixes)),
                                          _norms_trainable(other._norms_trainable),
                                          _checkpointed(other._checkpointed),
                                          _saved_input(std::move(other._saved_input)),
                                          _offload_scheduler(std::move(other._offload_scheduler))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DevicePreNormBlock<ComputeT,StorageT> &
CAIF_DevicePreNormBlock<ComputeT,StorageT>::operator=(CAIF_DevicePreNormBlock &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceContainer::operator=(std::move(other));
    SetNormPrefixes(std::move(other.NormPrefixes()));
    SetLayerPrefixes(std::move(other.LayerPrefixes()));
    SetNormsTrainable(other.NormsTrainable());
    SetCheckpointed(other.Checkpointed());
    SetSavedInput(std::move(other.SavedInput()));
    _offload_scheduler=std::move(other._offload_scheduler);
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DevicePreNormBlock<ComputeT,StorageT>::ForwardLoop(const CAIF_DeviceTensor &input,
                                                        CAIF_RunContext &ctx)
{
  try
  {
    CAIF_DeviceTensor x=input.Clone();

    const size_t stage_count=SubLayerCount();
    for(size_t i=0;i<stage_count;++i)
    {
      CAIF_DeviceLayer &norm=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_norm_offset);
      CAIF_DeviceLayer &sub=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_layer_offset);

      if(HasOffloadScheduler()==true)
      {
        OffloadSchedulerMut().OnEnterForwardStage(i,ctx.Stream());
      }

      CAIF_DeviceTensor normed=norm.Forward(x,ctx);
      CAIF_DeviceTensor sub_out=sub.Forward(normed,ctx);

      CAIF_DeviceTensor residual=CAIF_DeviceTensor::Uninitialized(x.Shape(),
                                                                   Stream(),
                                                                   x.Dtype());
      CAIF_Ops::Add(x,sub_out,residual);
      x=std::move(residual);

      if(HasOffloadScheduler()==true)
      {
        OffloadSchedulerMut().OnExitForwardStage(i,ctx.Stream());
      }
    }

    return x;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DevicePreNormBlock<ComputeT,StorageT>::BackwardLoop(const CAIF_DeviceTensor &grad_output,
                                                         CAIF_RunContext &ctx)
{
  try
  {
    CAIF_DeviceTensor grad=grad_output.Clone();

    const size_t stage_count=SubLayerCount();
    for(size_t ri=0;ri<stage_count;++ri)
    {
      const size_t i=stage_count-1-ri;
      CAIF_DeviceLayer &norm=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_norm_offset);
      CAIF_DeviceLayer &sub=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_layer_offset);

      if(HasOffloadScheduler()==true)
      {
        OffloadSchedulerMut().OnEnterBackwardStage(i,ctx.Stream());
      }

      CAIF_DeviceTensor d_sub=sub.Backward(grad,ctx);
      CAIF_DeviceTensor d_normed=norm.Backward(d_sub,ctx);

      CAIF_DeviceTensor d_residual=CAIF_DeviceTensor::Uninitialized(grad.Shape(),
                                                                     Stream(),
                                                                     grad.Dtype());
      CAIF_Ops::Add(grad,d_normed,d_residual);
      grad=std::move(d_residual);

      if(HasOffloadScheduler()==true)
      {
        OffloadSchedulerMut().OnExitBackwardStage(i,ctx.Stream());
      }
    }

    return grad;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DevicePreNormBlock<ComputeT,StorageT>::ForwardImpl(const CAIF_DeviceTensor &input,
                                                        CAIF_RunContext &ctx)
{
  try
  {
    if(Checkpointed()==true && ctx.Training()==true)
    {
      // Save the block input for the recompute in BackwardImpl, then run
      // the forward sweep with caching off — every saving sublayer
      // (MHA / MLA / FFN / MoE / norms) gates its `_cached_*` writes on
      // ctx.Training(), so flipping the flag here disables their caching
      // for free. Caches will be repopulated on the recompute pass at
      // the top of BackwardImpl.
      SetSavedInput(input.Clone());
      ctx.SetTraining(false);
      try
      {
        CAIF_DeviceTensor output=ForwardLoop(input,ctx);
        ctx.SetTraining(true);
        return output;
      }
      catch(...)
      {
        ctx.SetTraining(true);
        throw;
      }
    }
    return ForwardLoop(input,ctx);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DevicePreNormBlock<ComputeT,StorageT>::BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                                         CAIF_RunContext &ctx)
{
  try
  {
    if(Checkpointed()==true && HasSavedInput()==true)
    {
      // Recompute the forward sweep so each saving sublayer repopulates
      // its `_cached_*` members against the same inputs as the original
      // (skipped-cache) forward sweep. ctx.Training() is true here —
      // we're inside a training step's backward pass — so sublayers
      // will cache normally. The recomputed output is discarded; only
      // the side-effect populated caches matter for BackwardLoop.
      ForwardLoop(SavedInput(),ctx);
      ClearSavedInput();
    }
    return BackwardLoop(grad_output,ctx);
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DevicePreNormBlock<ComputeT,StorageT>::SetNormsTrainable(bool trainable)
{
  try
  {
    _norms_trainable=trainable;
    const size_t stage_count=SubLayerCount();
    for(size_t i=0;i<stage_count;++i)
    {
      SetLayerTrainable(i*g_caif_prenorm_stage_stride+g_caif_prenorm_norm_offset,trainable);
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DevicePreNormBlock<ComputeT,StorageT>::Description()const
{
  std::ostringstream ss;
  ss<<g_serial_tag_pre_norm_block
    <<g_serial_open_paren
    <<g_serial_kv_stages
    <<SubLayerCount()
    <<g_serial_comma
    <<g_serial_kv_params
    <<TotalParameterCount()
    <<g_serial_close_paren;
  return ss.str();
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DevicePreNormBlock<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    const size_t stage_count=SubLayerCount();
    for(size_t i=0;i<stage_count;++i)
    {
      const CAIF_DeviceLayer &norm=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_norm_offset);
      const CAIF_DeviceLayer &sub=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_layer_offset);

      auto norm_names=norm.ParameterNames(prefix+NormPrefixes()[i]);
      names.insert(names.end(),norm_names.begin(),norm_names.end());

      auto layer_names=sub.ParameterNames(prefix+LayerPrefixes()[i]);
      names.insert(names.end(),layer_names.begin(),layer_names.end());
    }
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
size_t CAIF_DevicePreNormBlock<ComputeT,StorageT>::FrozenTensorCount()const
{
  try
  {
    size_t total=0;
    const size_t stage_count=SubLayerCount();
    for(size_t i=0;i<stage_count;++i)
    {
      const CAIF_DeviceLayer &norm=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_norm_offset);
      const CAIF_DeviceLayer &sub=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_layer_offset);
      total+=norm.FrozenTensorCount();
      total+=sub.FrozenTensorCount();
    }
    return total;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTensor
CAIF_DevicePreNormBlock<ComputeT,StorageT>::FrozenTensorFP32(size_t index)const
{
  try
  {
    size_t offset=0;
    const size_t stage_count=SubLayerCount();
    for(size_t i=0;i<stage_count;++i)
    {
      const CAIF_DeviceLayer &norm=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_norm_offset);
      const size_t nc=norm.FrozenTensorCount();
      if(index<offset+nc)
      {
        return norm.FrozenTensorFP32(index-offset);
      }
      offset+=nc;
      const CAIF_DeviceLayer &sub=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_layer_offset);
      const size_t lc=sub.FrozenTensorCount();
      if(index<offset+lc)
      {
        return sub.FrozenTensorFP32(index-offset);
      }
      offset+=lc;
    }
    THROW_CAIFE("DevicePreNormBlock::FrozenTensorFP32: index out of range");
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DevicePreNormBlock<ComputeT,StorageT>::FrozenTensorNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    const size_t stage_count=SubLayerCount();
    for(size_t i=0;i<stage_count;++i)
    {
      const CAIF_DeviceLayer &norm=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_norm_offset);
      const CAIF_DeviceLayer &sub=Layer(i*g_caif_prenorm_stage_stride+g_caif_prenorm_layer_offset);

      auto norm_names=norm.FrozenTensorNames(prefix+NormPrefixes()[i]);
      names.insert(names.end(),norm_names.begin(),norm_names.end());

      auto layer_names=sub.FrozenTensorNames(prefix+LayerPrefixes()[i]);
      names.insert(names.end(),layer_names.begin(),layer_names.end());
    }
    return names;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_BlockOffloadScheduler &
CAIF_DevicePreNormBlock<ComputeT,StorageT>::OffloadSchedulerMut()
{
  try
  {
    if(_offload_scheduler==nullptr)
    {
      _offload_scheduler.reset(new CAIF_BlockOffloadScheduler());
    }
    return *_offload_scheduler;
  }
  CAIF_CATCH_BLOCK()
}

template class CAIF_DevicePreNormBlock<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DevicePreNormBlock<float,__half>;
template class CAIF_DevicePreNormBlock<float,__nv_bfloat16>;
template class CAIF_DevicePreNormBlock<__half,float>;
template class CAIF_DevicePreNormBlock<__half,__half>;
template class CAIF_DevicePreNormBlock<__half,__nv_bfloat16>;
template class CAIF_DevicePreNormBlock<__nv_bfloat16,float>;
template class CAIF_DevicePreNormBlock<__nv_bfloat16,__half>;
template class CAIF_DevicePreNormBlock<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
