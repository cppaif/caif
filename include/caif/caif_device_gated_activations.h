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
// AIF - AI Framework
// Gated activation leaf classes (header-only)
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_GATED_ACTIVATIONS_H
#define CAIF_DEVICE_GATED_ACTIVATIONS_H

#include "caif_device_activation.h"
#include "caif_cuda_kernels.h"

namespace instance
{

//------------------------------------------------------------------------------
// SwiGLU: output = swish(gate) * up
//------------------------------------------------------------------------------
class CAIF_DeviceSwiGLUActivation:public CAIF_DeviceGatedActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &gate_input,
                 const CAIF_DeviceTensor &up_input,
                 CAIF_DeviceTensor &output)const override
    {
      const int n=static_cast<int>(gate_input.TotalElements());
      launch_gated_activation_forward(gate_input.DevicePtr(),
                                      up_input.DevicePtr(),
                                      output.DevicePtr(),
                                      CAIF_GATED_OP_SWISH,
                                      n,
                                      output.Stream().Handle());
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &cached_gate_input,
                  const CAIF_DeviceTensor &cached_up_input,
                  CAIF_DeviceTensor &grad_gate,
                  CAIF_DeviceTensor &grad_up)const override
    {
      const int n=static_cast<int>(grad_output.TotalElements());
      launch_gated_activation_backward(grad_output.DevicePtr(),
                                       cached_gate_input.DevicePtr(),
                                       cached_up_input.DevicePtr(),
                                       grad_gate.DevicePtr(),
                                       grad_up.DevicePtr(),
                                       CAIF_GATED_OP_SWISH,
                                       n,
                                       grad_gate.Stream().Handle());
    }

    std::string Description()const override{return "SwiGLU";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceSwiGLUActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// GeGLU: output = gelu(gate) * up
//------------------------------------------------------------------------------
class CAIF_DeviceGeGLUActivation:public CAIF_DeviceGatedActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &gate_input,
                 const CAIF_DeviceTensor &up_input,
                 CAIF_DeviceTensor &output)const override
    {
      const int n=static_cast<int>(gate_input.TotalElements());
      launch_gated_activation_forward(gate_input.DevicePtr(),
                                      up_input.DevicePtr(),
                                      output.DevicePtr(),
                                      CAIF_GATED_OP_GELU,
                                      n,
                                      output.Stream().Handle());
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &cached_gate_input,
                  const CAIF_DeviceTensor &cached_up_input,
                  CAIF_DeviceTensor &grad_gate,
                  CAIF_DeviceTensor &grad_up)const override
    {
      const int n=static_cast<int>(grad_output.TotalElements());
      launch_gated_activation_backward(grad_output.DevicePtr(),
                                       cached_gate_input.DevicePtr(),
                                       cached_up_input.DevicePtr(),
                                       grad_gate.DevicePtr(),
                                       grad_up.DevicePtr(),
                                       CAIF_GATED_OP_GELU,
                                       n,
                                       grad_gate.Stream().Handle());
    }

    std::string Description()const override{return "GeGLU";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceGeGLUActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// ReGLU: output = relu(gate) * up
//------------------------------------------------------------------------------
class CAIF_DeviceReGLUActivation:public CAIF_DeviceGatedActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &gate_input,
                 const CAIF_DeviceTensor &up_input,
                 CAIF_DeviceTensor &output)const override
    {
      const int n=static_cast<int>(gate_input.TotalElements());
      launch_gated_activation_forward(gate_input.DevicePtr(),
                                      up_input.DevicePtr(),
                                      output.DevicePtr(),
                                      CAIF_GATED_OP_RELU,
                                      n,
                                      output.Stream().Handle());
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &cached_gate_input,
                  const CAIF_DeviceTensor &cached_up_input,
                  CAIF_DeviceTensor &grad_gate,
                  CAIF_DeviceTensor &grad_up)const override
    {
      const int n=static_cast<int>(grad_output.TotalElements());
      launch_gated_activation_backward(grad_output.DevicePtr(),
                                       cached_gate_input.DevicePtr(),
                                       cached_up_input.DevicePtr(),
                                       grad_gate.DevicePtr(),
                                       grad_up.DevicePtr(),
                                       CAIF_GATED_OP_RELU,
                                       n,
                                       grad_gate.Stream().Handle());
    }

    std::string Description()const override{return "ReGLU";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceReGLUActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// GLU: output = sigmoid(gate) * up
//------------------------------------------------------------------------------
class CAIF_DeviceGLUActivation:public CAIF_DeviceGatedActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &gate_input,
                 const CAIF_DeviceTensor &up_input,
                 CAIF_DeviceTensor &output)const override
    {
      const int n=static_cast<int>(gate_input.TotalElements());
      launch_gated_activation_forward(gate_input.DevicePtr(),
                                      up_input.DevicePtr(),
                                      output.DevicePtr(),
                                      CAIF_GATED_OP_SIGMOID,
                                      n,
                                      output.Stream().Handle());
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &cached_gate_input,
                  const CAIF_DeviceTensor &cached_up_input,
                  CAIF_DeviceTensor &grad_gate,
                  CAIF_DeviceTensor &grad_up)const override
    {
      const int n=static_cast<int>(grad_output.TotalElements());
      launch_gated_activation_backward(grad_output.DevicePtr(),
                                       cached_gate_input.DevicePtr(),
                                       cached_up_input.DevicePtr(),
                                       grad_gate.DevicePtr(),
                                       grad_up.DevicePtr(),
                                       CAIF_GATED_OP_SIGMOID,
                                       n,
                                       grad_gate.Stream().Handle());
    }

    std::string Description()const override{return "GLU";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceGLUActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// Bilinear: output = gate * up (identity gate activation)
//------------------------------------------------------------------------------
class CAIF_DeviceBilinearActivation:public CAIF_DeviceGatedActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &gate_input,
                 const CAIF_DeviceTensor &up_input,
                 CAIF_DeviceTensor &output)const override
    {
      const int n=static_cast<int>(gate_input.TotalElements());
      launch_gated_activation_forward(gate_input.DevicePtr(),
                                      up_input.DevicePtr(),
                                      output.DevicePtr(),
                                      CAIF_GATED_OP_LINEAR,
                                      n,
                                      output.Stream().Handle());
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &cached_gate_input,
                  const CAIF_DeviceTensor &cached_up_input,
                  CAIF_DeviceTensor &grad_gate,
                  CAIF_DeviceTensor &grad_up)const override
    {
      const int n=static_cast<int>(grad_output.TotalElements());
      launch_gated_activation_backward(grad_output.DevicePtr(),
                                       cached_gate_input.DevicePtr(),
                                       cached_up_input.DevicePtr(),
                                       grad_gate.DevicePtr(),
                                       grad_up.DevicePtr(),
                                       CAIF_GATED_OP_LINEAR,
                                       n,
                                       grad_gate.Stream().Handle());
    }

    std::string Description()const override{return "Bilinear";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceBilinearActivation>();
    }

  protected:

  private:
};

}//end instance namespace

#endif  // CAIF_DEVICE_GATED_ACTIVATIONS_H
