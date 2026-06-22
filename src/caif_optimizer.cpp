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

#include "caif_optimizer.h"
#include "caif_device_network.h"
#include "caif_device_layer.h"
#include "caif_ops.h"
#include "caif_settings.h"
#include "caif_run_context.h"
#include "caif_exception.h"

#ifdef USE_CAIF_CUDA
#include <cuda_runtime.h>
#endif

namespace instance
{

CAIF_Optimizer::CAIF_Optimizer(const float lr,
                               const float weight_decay,
                               CAIF_CudaStream &stream):_lr(lr),
                                                        _weight_decay(weight_decay),
                                                        _t(0),
                                                        _stream(&stream),
                                                        _master(),
                                                        _grad_fp32(),
                                                        _trainable_refs(),
                                                        _batch_num_tensors(0),
                                                        _batch_total_elements(0)
{
}

void CAIF_Optimizer::Initialize(CAIF_DeviceNetwork &network)
{
  try
  {
    MastersMut().clear();
    GradFp32sMut().clear();
    TrainableRefsMut().clear();
    ResetStepCount();

    const CAIF_DataType::CAIF_DataType_e fp32=CAIF_DataType::CAIF_DataType_e::Float32;

    for(size_t i=0;i<network.LayerCount();++i)
    {
      if(network.IsLayerTrainable(i)==false)
      {
        continue;
      }

      CAIF_DeviceLayer &layer=network.Layer(i);
      const size_t pc=layer.ParameterTensorCount();
      for(size_t p=0;p<pc;++p)
      {
        if(layer.IsParameterTrainable(p)==false)
        {
          continue;
        }
        const CAIF_DeviceTensor &param=layer.ParameterTensor(p);
        if(param.Dtype()==fp32)
        {
          MastersMut().emplace_back();
          GradFp32sMut().emplace_back();
        }
        else
        {
          MastersMut().push_back(param.To(fp32));
          GradFp32sMut().push_back(CAIF_DeviceTensor::Zeros(param.Shape(),Stream(),fp32));
        }
        TrainableRefsMut().push_back(CAIF_Optimizer::ParamRef_t(i,p));
        AllocateState(param);
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_Optimizer::Step(CAIF_DeviceNetwork &network)
{
  try
  {
    IncrementStep();

    // Multi-tensor ("foreach") fast path: one kernel launch for every parameter
    // when the subclass supports it and every param is fp32. Falls through to
    // the per-parameter loop below otherwise (AMP, or optimizers without a
    // batched override). StepCount() was already advanced, so Adam's bias
    // correction is identical on both paths.
    if(BatchedStep(network)==true)
    {
      return;
    }

    CAIF_RunContext ctx;
    ctx.SetStream(Stream());
    ctx.SetPass(CAIF_RunContext::Pass_e::Forward_e);

    const CAIF_Optimizer::ParamRefVec_t &refs=TrainableRefs();
    for(size_t idx=0;idx<refs.size();++idx)
    {
      CAIF_DeviceLayer &layer=network.Layer(refs[idx].LayerIndex());
      CAIF_DeviceTensor &param=layer.ParameterTensor(refs[idx].ParamIndex());
      const CAIF_DeviceTensor &grad=layer.GradientTensor(refs[idx].ParamIndex());
      CAIF_DeviceTensor &master=MastersMut()[idx];
      if(master.TotalElements()==0)
      {
        UpdateOne(param,grad,idx);
      }
      else
      {
        CAIF_DeviceTensor &grad_fp32=GradFp32sMut()[idx];
        CAIF_Ops::Cast(grad,grad_fp32,ctx);
        UpdateOne(master,grad_fp32,idx);
        CAIF_Ops::Cast(master,param,ctx);
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_Optimizer::UnscaleGradsCheckInf(CAIF_DeviceNetwork &network,
                                          float inv_scale,
                                          CAIF_DeviceTensor &found_inf)
{
  try
  {
    const CAIF_Optimizer::ParamRefVec_t &refs=TrainableRefs();
    for(size_t idx=0;idx<refs.size();++idx)
    {
      CAIF_DeviceLayer &layer=network.Layer(refs[idx].LayerIndex());
      CAIF_DeviceTensor &grad=layer.GradientTensor(refs[idx].ParamIndex());
      CAIF_Ops::UnscaleCheckInf(grad,inv_scale,found_inf);
    }
  }
  CAIF_CATCH_BLOCK()
}

bool CAIF_Optimizer::BuildSharedBatch(CAIF_DeviceNetwork &network)
{
#ifdef USE_CAIF_CUDA
  if(CAIF_Settings::MultiTensorOptimizer()==false)
  {
    return false;
  }
  const CAIF_Optimizer::ParamRefVec_t &refs=TrainableRefs();
  const size_t n=refs.size();
  if(n==0)
  {
    return false;
  }
  // Batched path is fp32-only: any AMP master present -> per-param fallback.
  for(size_t idx=0;idx<n;++idx)
  {
    if(Masters()[idx].TotalElements()!=0)
    {
      return false;
    }
  }

  HostTargetsMut().resize(n);
  HostGradsMut().resize(n);
  HostOffsetsMut().resize(n+1);
  int64_t total=0;
  for(size_t idx=0;idx<n;++idx)
  {
    CAIF_DeviceLayer &layer=network.Layer(refs[idx].LayerIndex());
    CAIF_DeviceTensor &param=layer.ParameterTensor(refs[idx].ParamIndex());
    CAIF_DeviceTensor &grad=layer.GradientTensor(refs[idx].ParamIndex());
    HostOffsetsMut()[idx]=total;
    HostTargetsMut()[idx]=param.DevicePtr<float>();
    HostGradsMut()[idx]=grad.DevicePtr<float>();
    total+=static_cast<int64_t>(param.TotalElements());
  }
  HostOffsetsMut()[n]=total;
  SetBatchNumTensors(static_cast<int>(n));
  SetBatchTotalElements(total);

  UploadScratch(DeviceOffsetsMut(),HostOffsets().data(),(n+1)*sizeof(int64_t),Stream());
  UploadScratch(DeviceTargetsMut(),HostTargetsMut().data(),n*sizeof(float *),Stream());
  UploadScratch(DeviceGradsMut(),HostGradsMut().data(),n*sizeof(const float *),Stream());
  return true;
#else
  (void)network;
  return false;
#endif
}

const int64_t *CAIF_Optimizer::BatchOffsetsDevice()
{
#ifdef USE_CAIF_CUDA
  return reinterpret_cast<const int64_t *>(DeviceOffsetsMut().DeviceDataRaw());
#else
  return nullptr;
#endif
}

float *const *CAIF_Optimizer::BatchTargetsDevice()
{
#ifdef USE_CAIF_CUDA
  return reinterpret_cast<float *const *>(DeviceTargetsMut().DeviceDataRaw());
#else
  return nullptr;
#endif
}

const float *const *CAIF_Optimizer::BatchGradsDevice()
{
#ifdef USE_CAIF_CUDA
  return reinterpret_cast<const float *const *>(DeviceGradsMut().DeviceDataRaw());
#else
  return nullptr;
#endif
}

void CAIF_Optimizer::UploadScratch(CAIF_DeviceTensor &scratch,
                                   const void *host,
                                   size_t bytes,
                                   CAIF_CudaStream &stream)
{
#ifdef USE_CAIF_CUDA
  if(bytes==0)
  {
    return;
  }
  if(scratch.TotalElements()<bytes)
  {
    scratch=CAIF_DeviceTensor::Uninitialized({static_cast<uint32_t>(bytes)},
                                             stream,
                                             CAIF_DataType::CAIF_DataType_e::Int8);
  }
  cudaMemcpyAsync(scratch.DeviceDataRaw(),
                  host,
                  bytes,
                  cudaMemcpyHostToDevice,
                  stream.Handle());
#else
  (void)scratch;
  (void)host;
  (void)bytes;
  (void)stream;
#endif
}

}//end instance namespace
