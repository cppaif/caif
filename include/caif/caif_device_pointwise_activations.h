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
// Pointwise activation leaf classes (header-only)
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_POINTWISE_ACTIVATIONS_H
#define CAIF_DEVICE_POINTWISE_ACTIVATIONS_H

#include "caif_device_activation.h"
#include "caif_device_ops.h"

namespace instance
{

//------------------------------------------------------------------------------
// ReLU: f(x) = max(0, x)
//------------------------------------------------------------------------------
class CAIF_DeviceReLUActivation:public CAIF_DevicePointwiseActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override
    {
      CAIF_DeviceOps::ReLU(input,output);
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override
    {
      (void)post_activation;
      CAIF_DeviceOps::ReLUBackward(grad_output,pre_activation,grad_input);
    }

    std::string Description()const override{return "ReLU";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceReLUActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// GELU: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//------------------------------------------------------------------------------
class CAIF_DeviceGELUActivation:public CAIF_DevicePointwiseActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override
    {
      CAIF_DeviceOps::GELU(input,output);
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override
    {
      (void)post_activation;
      CAIF_DeviceOps::GELUBackward(grad_output,pre_activation,grad_input);
    }

    std::string Description()const override{return "GELU";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceGELUActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// Sigmoid: f(x) = 1 / (1 + exp(-x))
//------------------------------------------------------------------------------
class CAIF_DeviceSigmoidActivation:public CAIF_DevicePointwiseActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override
    {
      CAIF_DeviceOps::Sigmoid(input,output);
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override
    {
      (void)pre_activation;
      CAIF_DeviceOps::SigmoidBackward(grad_output,post_activation,grad_input);
    }

    bool NeedsPostActivation()const override{return true;}

    std::string Description()const override{return "Sigmoid";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceSigmoidActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// Tanh: f(x) = tanh(x)
//------------------------------------------------------------------------------
class CAIF_DeviceTanhActivation:public CAIF_DevicePointwiseActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override
    {
      CAIF_DeviceOps::Tanh(input,output);
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override
    {
      (void)pre_activation;
      CAIF_DeviceOps::TanhBackward(grad_output,post_activation,grad_input);
    }

    bool NeedsPostActivation()const override{return true;}

    std::string Description()const override{return "Tanh";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceTanhActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// Swish: f(x) = x * sigmoid(x)
//------------------------------------------------------------------------------
class CAIF_DeviceSwishActivation:public CAIF_DevicePointwiseActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override
    {
      CAIF_DeviceOps::Swish(input,output);
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override
    {
      CAIF_DeviceOps::SwishBackward(grad_output,pre_activation,
                                    post_activation,grad_input);
    }

    bool NeedsPostActivation()const override{return true;}

    std::string Description()const override{return "Swish";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceSwishActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// LeakyReLU: f(x) = max(alpha * x, x)
//------------------------------------------------------------------------------
class CAIF_DeviceLeakyReLUActivation:public CAIF_DevicePointwiseActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override
    {
      CAIF_DeviceOps::LeakyReLU(input,output);
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override
    {
      (void)post_activation;
      CAIF_DeviceOps::LeakyReLUBackward(grad_output,pre_activation,grad_input);
    }

    std::string Description()const override{return "LeakyReLU";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceLeakyReLUActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// ELU: f(x) = x if x > 0, else alpha * (exp(x) - 1)
//------------------------------------------------------------------------------
class CAIF_DeviceELUActivation:public CAIF_DevicePointwiseActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override
    {
      CAIF_DeviceOps::ELU(input,output);
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override
    {
      CAIF_DeviceOps::ELUBackward(grad_output,pre_activation,
                                  post_activation,grad_input);
    }

    bool NeedsPostActivation()const override{return true;}

    std::string Description()const override{return "ELU";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceELUActivation>();
    }

  protected:

  private:
};

//------------------------------------------------------------------------------
// Linear (identity): f(x) = x
//------------------------------------------------------------------------------
class CAIF_DeviceLinearActivation:public CAIF_DevicePointwiseActivation
{
  public:
    void Forward(const CAIF_DeviceTensor &input,
                 CAIF_DeviceTensor &output)const override
    {
      output=input.Clone();
    }

    void Backward(const CAIF_DeviceTensor &grad_output,
                  const CAIF_DeviceTensor &pre_activation,
                  const CAIF_DeviceTensor &post_activation,
                  CAIF_DeviceTensor &grad_input)const override
    {
      (void)pre_activation;
      (void)post_activation;
      grad_input=grad_output.Clone();
    }

    std::string Description()const override{return "Linear";}

    std::unique_ptr<CAIF_DeviceActivation> Clone()const override
    {
      return std::make_unique<CAIF_DeviceLinearActivation>();
    }

  protected:

  private:
};

}//end instance namespace

#endif  // CAIF_DEVICE_POINTWISE_ACTIVATIONS_H
