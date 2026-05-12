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
#include "caif_run_context.h"
#include "caif_exception.h"

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
                                                        _trainable_refs()
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

}//end instance namespace
