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
// Test: Stage 5b device-layer ports — Flatten, Reshape, Dropout, BatchNorm,
// Max/Avg-Pool2D, Conv2D. Uses the shared harness + gradcheck.
//------------------------------------------------------------------------------
#include "caif_device_batch_norm.h"
#include "caif_device_conv2d.h"
#include "caif_device_dropout.h"
#include "caif_device_flatten.h"
#include "caif_device_pooling2d.h"
#include "caif_device_max_pooling2d.h"
#include "caif_device_average_pooling2d.h"
#include "caif_device_reshape.h"
#include "caif_cuda_stream.h"
#include "caif_device_tensor.h"
#include "caif_exception.h"
#include "caif_gradcheck.h"
#include "caif_run_context.h"
#include "caif_test_harness.h"
#include "caif_tolerances.h"
#include "ise_lib/ise_out.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace instance
{

constexpr float g_caif_visionports_test_rand_lo=-0.5f;
constexpr float g_caif_visionports_test_rand_hi=0.5f;
constexpr float g_caif_visionports_test_bn_mean_tol=5.0e-3f;
constexpr float g_caif_visionports_test_bn_var_tol=5.0e-3f;
constexpr float g_caif_visionports_test_reldiff_floor=1e-6f;
constexpr float g_caif_visionports_test_dropout_rate_a=0.3f;
constexpr float g_caif_visionports_test_dropout_rate_b=0.5f;
constexpr float g_caif_visionports_test_dropout_rate_c=0.25f;
constexpr uint64_t g_caif_visionports_test_seed_a=7777ULL;
constexpr uint64_t g_caif_visionports_test_seed_b=8888ULL;
constexpr float g_caif_visionports_test_tol_fp32=1e-5f;
constexpr float g_caif_visionports_test_tol_fp16_pool=5e-3f;
constexpr float g_caif_visionports_test_tol_bf16_pool=2e-2f;
constexpr float g_caif_visionports_test_tol_fp32_conv=1e-3f;
constexpr float g_caif_visionports_test_tol_fp16_conv=5e-2f;
constexpr float g_caif_visionports_test_tol_bf16_conv=5e-2f;
constexpr float g_caif_visionports_test_tol_fp32_bn=1e-3f;
constexpr float g_caif_visionports_test_tol_fp16_bn=5e-2f;
constexpr float g_caif_visionports_test_tol_bf16_bn=5e-2f;
constexpr float g_caif_visionports_test_tol_fp32_bn_bwd=1e-3f;
constexpr float g_caif_visionports_test_tol_fp16_bn_bwd=8e-2f;
constexpr float g_caif_visionports_test_tol_bf16_bn_bwd=1e-1f;

//------------------------------------------------------------------------------
// Layer-specific gradcheck functors that host each layer and drive
// CAIF_GradCheck. Each implements only ForwardOnly; the analytical gradient
// is produced by calling Forward+Backward once in RunFullPass prior to the
// CAIF_GradCheck::Check invocation.
//------------------------------------------------------------------------------

class CAIF_FlattenFunctor:public CAIF_GradCheckTargetFunctor
{
  public:
    explicit CAIF_FlattenFunctor(CAIF_CudaStream &stream):_layer(stream){}

    CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                  CAIF_RunContext &ctx)override
    {
      return _layer.Forward(perturbed,ctx);
    }

    CAIF_DeviceFlatten<float,float> &Layer(){return _layer;}

  protected:

  private:
    CAIF_DeviceFlatten<float,float> _layer;
};

class CAIF_ReshapeFunctor:public CAIF_GradCheckTargetFunctor
{
  public:
    CAIF_ReshapeFunctor(const std::vector<uint32_t> &target,
                         CAIF_CudaStream &stream):_layer(target,stream){}

    CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                  CAIF_RunContext &ctx)override
    {
      return _layer.Forward(perturbed,ctx);
    }

    CAIF_DeviceReshape<float,float> &Layer(){return _layer;}

  protected:

  private:
    CAIF_DeviceReshape<float,float> _layer;
};

class CAIF_BatchNormFunctor:public CAIF_GradCheckTargetFunctor
{
  public:
    CAIF_BatchNormFunctor(const uint32_t features,
                           CAIF_CudaStream &stream):_layer(features,stream){}

    CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                  CAIF_RunContext &ctx)override
    {
      return _layer.Forward(perturbed,ctx);
    }

    CAIF_DeviceBatchNorm<float,float> &Layer(){return _layer;}

  protected:

  private:
    CAIF_DeviceBatchNorm<float,float> _layer;
};

class CAIF_MaxPoolFunctor:public CAIF_GradCheckTargetFunctor
{
  public:
    CAIF_MaxPoolFunctor(const CAIF_DevicePooling2DConfig &cfg,
                         CAIF_CudaStream &stream):_layer(cfg,stream){}

    CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                  CAIF_RunContext &ctx)override
    {
      return _layer.Forward(perturbed,ctx);
    }

    CAIF_DeviceMaxPooling2D<float,float> &Layer(){return _layer;}

  protected:

  private:
    CAIF_DeviceMaxPooling2D<float,float> _layer;
};

class CAIF_AvgPoolFunctor:public CAIF_GradCheckTargetFunctor
{
  public:
    CAIF_AvgPoolFunctor(const CAIF_DevicePooling2DConfig &cfg,
                         CAIF_CudaStream &stream):_layer(cfg,stream){}

    CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                  CAIF_RunContext &ctx)override
    {
      return _layer.Forward(perturbed,ctx);
    }

    CAIF_DeviceAveragePooling2D<float,float> &Layer(){return _layer;}

  protected:

  private:
    CAIF_DeviceAveragePooling2D<float,float> _layer;
};

class CAIF_Conv2DFunctor:public CAIF_GradCheckTargetFunctor
{
  public:
    CAIF_Conv2DFunctor(const CAIF_DeviceConv2DConfig &cfg,
                        CAIF_CudaStream &stream):_layer(cfg,stream){}

    CAIF_DeviceTensor ForwardOnly(const CAIF_DeviceTensor &perturbed,
                                  CAIF_RunContext &ctx)override
    {
      return _layer.Forward(perturbed,ctx);
    }

    CAIF_DeviceConv2D<float,float> &Layer(){return _layer;}

  protected:

  private:
    CAIF_DeviceConv2D<float,float> _layer;
};

//------------------------------------------------------------------------------
// Vision-port device layer tests.
//------------------------------------------------------------------------------
class CAIF_VisionPortsTests
{
  public:
    static void RunAll();

  protected:

  private:
    static std::vector<float> MakeRandomVec(const size_t n,const uint32_t seed);
    static void WriteHost(CAIF_DeviceTensor &t,const std::vector<float> &data);
    static std::vector<float> ReadHost(const CAIF_DeviceTensor &t);
    static CAIF_DeviceTensor MakeHostFromVec(const std::vector<uint32_t> &shape,
                                              const std::vector<float> &data);
    static float MaxAbs(const std::vector<float> &v);
    static size_t Product(const std::vector<uint32_t> &shape);
    static float MaxRelDiff(const std::vector<float> &got,
                             const std::vector<float> &expected);
    static bool RunGradCheckOnFunctor(CAIF_GradCheckTargetFunctor &functor,
                                       CAIF_DeviceLayer &layer,
                                       const std::vector<uint32_t> &input_shape,
                                       const uint32_t seed,
                                       CAIF_RunContext &ctx);

    static bool TestFlattenForwardShape();
    static bool TestFlattenGradCheck();
    static bool TestReshapeForwardShape();
    static bool TestReshapeGradCheck();
    static bool TestDropoutInferenceIsIdentity();
    static bool TestDropoutTrainingDeterministic();
    static bool TestDropoutBackwardMatchesMask();
    static bool TestBatchNormForwardStats();
    static bool TestBatchNormGradCheck();
    static bool TestMaxPoolForwardShape();
    static bool TestMaxPoolGradCheck();
    static bool TestAvgPoolForwardMean();
    static bool TestAvgPoolGradCheck();
    static bool TestConv2DForwardShape();
    static bool TestConv2DGradCheck();

    template<typename StorageT>
    static bool TestMaxPool2DDevice(const CAIF_DataType::CAIF_DataType_e storage_dt,
                                     const float tol);
    template<typename StorageT>
    static bool TestAvgPool2DDevice(const CAIF_DataType::CAIF_DataType_e storage_dt,
                                     const float tol);
    template<typename StorageT>
    static bool TestConv2DDevice(const CAIF_DataType::CAIF_DataType_e storage_dt,
                                  const float tol);
    template<typename StorageT>
    static bool TestBatchNormDevice(const CAIF_DataType::CAIF_DataType_e storage_dt,
                                     const float tol);
    // Device-path BN backward parity vs host fp32 backward. Verifies cuDNN
    // `cudnnBatchNormalizationBackward` produces grad_input matching the
    // host loop's analytic gradient up to dtype-appropriate tolerance.
    template<typename StorageT>
    static bool TestBatchNormBackwardDevice(const CAIF_DataType::CAIF_DataType_e storage_dt,
                                             const float tol);
};

std::vector<float> CAIF_VisionPortsTests::MakeRandomVec(const size_t n,const uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(g_caif_visionports_test_rand_lo,
                                             g_caif_visionports_test_rand_hi);
  std::vector<float> v(n);
  for(size_t i=0;i<n;++i)
  {
    v[i]=dist(gen);
  }
  return v;
}

void CAIF_VisionPortsTests::WriteHost(CAIF_DeviceTensor &t,const std::vector<float> &data)
{
  std::memcpy(t.DeviceDataRaw(),data.data(),data.size()*sizeof(float));
}

std::vector<float> CAIF_VisionPortsTests::ReadHost(const CAIF_DeviceTensor &t)
{
  std::vector<float> out(t.TotalElements());
  std::memcpy(out.data(),t.DeviceDataRaw(),out.size()*sizeof(float));
  return out;
}

CAIF_DeviceTensor CAIF_VisionPortsTests::MakeHostFromVec(const std::vector<uint32_t> &shape,
                                                           const std::vector<float> &data)
{
  CAIF_DeviceTensor t=CAIF_DeviceTensor::ZerosHost(shape);
  WriteHost(t,data);
  return t;
}

float CAIF_VisionPortsTests::MaxAbs(const std::vector<float> &v)
{
  float m=0.0f;
  for(const float x:v)
  {
    const float a=std::fabs(x);
    if(a>m)
    {
      m=a;
    }
  }
  return m;
}

size_t CAIF_VisionPortsTests::Product(const std::vector<uint32_t> &shape)
{
  size_t p=1u;
  for(const uint32_t d:shape)
  {
    p*=static_cast<size_t>(d);
  }
  return p;
}

float CAIF_VisionPortsTests::MaxRelDiff(const std::vector<float> &got,
                                         const std::vector<float> &expected)
{
  // Global-denominator relative diff (matches test_device_matmul_dtype's
  // RelClose pattern): max_abs_err / max_abs_ref. This avoids amplifying
  // near-zero output elements into unbounded relative error.
  float max_abs_ref=0.0f;
  for(float r:expected)
  {
    if(std::fabs(r)>max_abs_ref)
    {
      max_abs_ref=std::fabs(r);
    }
  }
  const float denom=std::max(max_abs_ref,g_caif_visionports_test_reldiff_floor);
  float worst=0.0f;
  for(size_t i=0;i<got.size();++i)
  {
    const float rel=std::fabs(got[i]-expected[i])/denom;
    if(rel>worst)
    {
      worst=rel;
    }
  }
  return worst;
}

bool CAIF_VisionPortsTests::RunGradCheckOnFunctor(CAIF_GradCheckTargetFunctor &functor,
                                                    CAIF_DeviceLayer &layer,
                                                    const std::vector<uint32_t> &input_shape,
                                                    const uint32_t seed,
                                                    CAIF_RunContext &ctx)
{
  try
  {
    const size_t n=Product(input_shape);
    const std::vector<float> x_host=MakeRandomVec(n,seed);
    CAIF_DeviceTensor x=MakeHostFromVec(input_shape,x_host);

    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    const std::vector<uint32_t> y_shape=y.Shape();
    const size_t ny=y.TotalElements();
    const std::vector<float> g_host=MakeRandomVec(ny,seed+13);
    CAIF_DeviceTensor g=MakeHostFromVec(y_shape,g_host);

    CAIF_DeviceTensor dx=layer.Backward(g,ctx);
    const std::vector<float> analytical=ReadHost(dx);

    return CAIF_GradCheck::Check(functor,x_host,input_shape,g_host,analytical,ctx,
                                 CAIF_Tolerances::GradcheckRel(),
                                 CAIF_DeviceTensor::Location_e::Host_e);
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestFlattenForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const std::vector<uint32_t> shape={2u,3u,4u,5u};
    const std::vector<float> data=MakeRandomVec(Product(shape),101);
    CAIF_DeviceTensor x=MakeHostFromVec(shape,data);

    CAIF_DeviceFlatten<float,float> layer(stream);
    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    if(y.Shape().size()!=2u)
    {
      return false;
    }
    if(y.Shape()[0]!=2u || y.Shape()[1]!=3u*4u*5u)
    {
      return false;
    }
    const std::vector<float> y_vec=ReadHost(y);
    for(size_t i=0;i<data.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(data[i],y_vec[i],
                                      CAIF_Tolerances::ShapeIdentity())==false)
      {
        return false;
      }
    }
    return true;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestFlattenGradCheck()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_FlattenFunctor functor(stream);
    return RunGradCheckOnFunctor(functor,functor.Layer(),{2u,3u,4u},103,ctx);
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestReshapeForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const std::vector<uint32_t> in_shape={2u,6u,4u};
    const std::vector<uint32_t> target_shape={2u,4u,6u};
    const std::vector<float> data=MakeRandomVec(Product(in_shape),201);
    CAIF_DeviceTensor x=MakeHostFromVec(in_shape,data);

    CAIF_DeviceReshape<float,float> layer(target_shape,stream);
    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    if(y.Shape()!=target_shape)
    {
      return false;
    }
    const std::vector<float> y_vec=ReadHost(y);
    for(size_t i=0;i<data.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(data[i],y_vec[i],
                                      CAIF_Tolerances::ShapeIdentity())==false)
      {
        return false;
      }
    }
    return true;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestReshapeGradCheck()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_ReshapeFunctor functor({2u,4u,6u},stream);
    return RunGradCheckOnFunctor(functor,functor.Layer(),{2u,6u,4u},203,ctx);
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestDropoutInferenceIsIdentity()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(false);

    const std::vector<uint32_t> shape={4u,8u};
    const std::vector<float> data=MakeRandomVec(Product(shape),301);
    CAIF_DeviceTensor x=MakeHostFromVec(shape,data);

    CAIF_DeviceDropout<float,float> layer(g_caif_visionports_test_dropout_rate_a,stream);
    CAIF_DeviceTensor y=layer.Forward(x,ctx);
    const std::vector<float> y_vec=ReadHost(y);
    for(size_t i=0;i<data.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(data[i],y_vec[i],
                                      CAIF_Tolerances::ShapeIdentity())==false)
      {
        return false;
      }
    }
    return true;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestDropoutTrainingDeterministic()
{
  try
  {
    CAIF_CudaStream stream;
    const std::vector<uint32_t> shape={4u,8u};
    const std::vector<float> data=MakeRandomVec(Product(shape),311);

    CAIF_RunContext ctx_a;
    ctx_a.SetStream(stream);
    ctx_a.SetTraining(true);
    ctx_a.SetRandomSeed(g_caif_visionports_test_seed_a);
    CAIF_DeviceTensor x_a=MakeHostFromVec(shape,data);
    CAIF_DeviceDropout<float,float> layer_a(g_caif_visionports_test_dropout_rate_b,stream);
    const CAIF_DeviceTensor y_a=layer_a.Forward(x_a,ctx_a);
    const std::vector<float> y_a_vec=ReadHost(y_a);

    CAIF_RunContext ctx_b;
    ctx_b.SetStream(stream);
    ctx_b.SetTraining(true);
    ctx_b.SetRandomSeed(g_caif_visionports_test_seed_a);
    CAIF_DeviceTensor x_b=MakeHostFromVec(shape,data);
    CAIF_DeviceDropout<float,float> layer_b(g_caif_visionports_test_dropout_rate_b,stream);
    const CAIF_DeviceTensor y_b=layer_b.Forward(x_b,ctx_b);
    const std::vector<float> y_b_vec=ReadHost(y_b);

    if(y_a_vec.size()!=y_b_vec.size())
    {
      return false;
    }
    for(size_t i=0;i<y_a_vec.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(y_a_vec[i],y_b_vec[i],
                                      CAIF_Tolerances::ShapeIdentity())==false)
      {
        return false;
      }
    }
    return true;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestDropoutBackwardMatchesMask()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    ctx.SetRandomSeed(g_caif_visionports_test_seed_b);

    const std::vector<uint32_t> shape={4u,8u};
    const std::vector<float> x_data(Product(shape),1.0f);
    const std::vector<float> g_data(Product(shape),1.0f);
    CAIF_DeviceTensor x=MakeHostFromVec(shape,x_data);
    CAIF_DeviceTensor g=MakeHostFromVec(shape,g_data);

    CAIF_DeviceDropout<float,float> layer(g_caif_visionports_test_dropout_rate_c,stream);
    const CAIF_DeviceTensor y=layer.Forward(x,ctx);
    const CAIF_DeviceTensor dx=layer.Backward(g,ctx);
    const std::vector<float> y_vec=ReadHost(y);
    const std::vector<float> dx_vec=ReadHost(dx);

    for(size_t i=0;i<y_vec.size();++i)
    {
      if(CAIF_TestHarness::FloatEqual(y_vec[i],dx_vec[i],
                                      CAIF_Tolerances::ShapeIdentity())==false)
      {
        return false;
      }
    }
    return true;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestBatchNormForwardStats()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    const uint32_t batch=16u;
    const uint32_t features=4u;
    const std::vector<uint32_t> shape={batch,features};
    const std::vector<float> data=MakeRandomVec(Product(shape),401);
    CAIF_DeviceTensor x=MakeHostFromVec(shape,data);

    CAIF_DeviceBatchNorm<float,float> layer(features,stream);
    const CAIF_DeviceTensor y=layer.Forward(x,ctx);
    const std::vector<float> y_vec=ReadHost(y);

    for(uint32_t f=0;f<features;++f)
    {
      float sum=0.0f;
      float sum_sq=0.0f;
      for(uint32_t b=0;b<batch;++b)
      {
        const float v=y_vec[b*features+f];
        sum+=v;
        sum_sq+=v*v;
      }
      const float mean=sum/static_cast<float>(batch);
      const float var=sum_sq/static_cast<float>(batch)-mean*mean;
      if(std::fabs(mean)>g_caif_visionports_test_bn_mean_tol)
      {
        return false;
      }
      if(std::fabs(var-1.0f)>g_caif_visionports_test_bn_var_tol)
      {
        return false;
      }
    }
    return true;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestBatchNormGradCheck()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);
    CAIF_BatchNormFunctor functor(4u,stream);
    return RunGradCheckOnFunctor(functor,functor.Layer(),{8u,4u},403,ctx);
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestMaxPoolForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const uint32_t n=2u;
    const uint32_t h=4u;
    const uint32_t w=4u;
    const uint32_t c=3u;
    const std::vector<uint32_t> shape={n,h,w,c};
    const std::vector<float> data=MakeRandomVec(Product(shape),501);
    CAIF_DeviceTensor x=MakeHostFromVec(shape,data);

    CAIF_DevicePooling2DConfig cfg={2u,2u,2u,2u};
    CAIF_DeviceMaxPooling2D<float,float> layer(cfg,stream);
    const CAIF_DeviceTensor y=layer.Forward(x,ctx);
    if(y.Shape().size()!=4u)
    {
      return false;
    }
    if(y.Shape()[0]!=n || y.Shape()[1]!=h/2u)
    {
      return false;
    }
    if(y.Shape()[2]!=w/2u || y.Shape()[3]!=c)
    {
      return false;
    }
    return true;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestMaxPoolGradCheck()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DevicePooling2DConfig cfg={2u,2u,2u,2u};
    CAIF_MaxPoolFunctor functor(cfg,stream);
    return RunGradCheckOnFunctor(functor,functor.Layer(),{1u,4u,4u,2u},503,ctx);
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestAvgPoolForwardMean()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const std::vector<uint32_t> shape={1u,2u,2u,1u};
    const std::vector<float> data={1.0f,2.0f,3.0f,4.0f};
    CAIF_DeviceTensor x=MakeHostFromVec(shape,data);

    CAIF_DevicePooling2DConfig cfg={2u,2u,2u,2u};
    CAIF_DeviceAveragePooling2D<float,float> layer(cfg,stream);
    const CAIF_DeviceTensor y=layer.Forward(x,ctx);
    const std::vector<float> y_vec=ReadHost(y);
    if(y_vec.size()!=1u)
    {
      return false;
    }
    const float expected=(1.0f+2.0f+3.0f+4.0f)/4.0f;
    return CAIF_TestHarness::FloatEqual(y_vec[0],expected,
                                        CAIF_Tolerances::ShapeIdentity());
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestAvgPoolGradCheck()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    CAIF_DevicePooling2DConfig cfg={2u,2u,2u,2u};
    CAIF_AvgPoolFunctor functor(cfg,stream);
    return RunGradCheckOnFunctor(functor,functor.Layer(),{1u,4u,4u,2u},603,ctx);
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestConv2DForwardShape()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const uint32_t n=1u;
    const uint32_t h=5u;
    const uint32_t w=5u;
    const uint32_t cin=3u;
    const uint32_t cout=4u;
    const uint32_t kh=3u;
    const uint32_t kw=3u;
    const std::vector<uint32_t> shape={n,h,w,cin};
    const std::vector<float> data=MakeRandomVec(Product(shape),701);
    CAIF_DeviceTensor x=MakeHostFromVec(shape,data);

    CAIF_DeviceConv2DConfig cfg={cin,cout,kh,kw,1u,1u};
    CAIF_DeviceConv2D<float,float> layer(cfg,stream);
    const CAIF_DeviceTensor y=layer.Forward(x,ctx);
    if(y.Shape().size()!=4u || y.Shape()[0]!=n)
    {
      return false;
    }
    if(y.Shape()[1]!=(h-kh)+1u || y.Shape()[2]!=(w-kw)+1u)
    {
      return false;
    }
    if(y.Shape()[3]!=cout)
    {
      return false;
    }
    const std::vector<float> y_vec=ReadHost(y);
    return MaxAbs(y_vec)>0.0f;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

bool CAIF_VisionPortsTests::TestConv2DGradCheck()
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const uint32_t cin=2u;
    const uint32_t cout=3u;
    CAIF_DeviceConv2DConfig cfg={cin,cout,3u,3u,1u,1u};
    CAIF_Conv2DFunctor functor(cfg,stream);
    return RunGradCheckOnFunctor(functor,functor.Layer(),{1u,5u,5u,cin},703,ctx);
  }
  CAIF_CATCH_BLOCK();
  return false;
}

template<typename StorageT>
bool CAIF_VisionPortsTests::TestMaxPool2DDevice(
  const CAIF_DataType::CAIF_DataType_e storage_dt,
  const float tol)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const uint32_t n=2u;
    const uint32_t h=4u;
    const uint32_t w=4u;
    const uint32_t c=3u;
    const std::vector<uint32_t> shape={n,h,w,c};
    const std::vector<float> data=MakeRandomVec(Product(shape),801);

    CAIF_DevicePooling2DConfig cfg_host={2u,2u,2u,2u};
    CAIF_DeviceMaxPooling2D<float,float> host_layer(cfg_host,stream);
    CAIF_DeviceTensor x_host=MakeHostFromVec(shape,data);
    CAIF_DeviceTensor y_host=host_layer.Forward(x_host,ctx);
    const std::vector<float> ref=ReadHost(y_host);

    CAIF_DeviceTensor x_fp32_dev=CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
    CAIF_DeviceTensor x_dev=x_fp32_dev.To(storage_dt);
    CAIF_DevicePooling2DConfig cfg_dev={2u,2u,2u,2u};
    CAIF_DeviceMaxPooling2D<float,StorageT> dev_layer(cfg_dev,stream);
    CAIF_DeviceTensor y_dev=dev_layer.Forward(x_dev,ctx);
    CAIF_DeviceTensor y_dev_fp32=y_dev.To(CAIF_DataType::CAIF_DataType_e::Float32);
    std::vector<float> got(y_dev_fp32.TotalElements());
    y_dev_fp32.CopyToHost(got.data());
    stream.Synchronize();

    if(got.size()!=ref.size())
    {
      return false;
    }
    return MaxRelDiff(got,ref)<=tol;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

template<typename StorageT>
bool CAIF_VisionPortsTests::TestAvgPool2DDevice(
  const CAIF_DataType::CAIF_DataType_e storage_dt,
  const float tol)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const uint32_t n=2u;
    const uint32_t h=4u;
    const uint32_t w=4u;
    const uint32_t c=3u;
    const std::vector<uint32_t> shape={n,h,w,c};
    const std::vector<float> data=MakeRandomVec(Product(shape),811);

    CAIF_DevicePooling2DConfig cfg_host={2u,2u,2u,2u};
    CAIF_DeviceAveragePooling2D<float,float> host_layer(cfg_host,stream);
    CAIF_DeviceTensor x_host=MakeHostFromVec(shape,data);
    CAIF_DeviceTensor y_host=host_layer.Forward(x_host,ctx);
    const std::vector<float> ref=ReadHost(y_host);

    CAIF_DeviceTensor x_fp32_dev=CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
    CAIF_DeviceTensor x_dev=x_fp32_dev.To(storage_dt);
    CAIF_DevicePooling2DConfig cfg_dev={2u,2u,2u,2u};
    CAIF_DeviceAveragePooling2D<float,StorageT> dev_layer(cfg_dev,stream);
    CAIF_DeviceTensor y_dev=dev_layer.Forward(x_dev,ctx);
    CAIF_DeviceTensor y_dev_fp32=y_dev.To(CAIF_DataType::CAIF_DataType_e::Float32);
    std::vector<float> got(y_dev_fp32.TotalElements());
    y_dev_fp32.CopyToHost(got.data());
    stream.Synchronize();

    if(got.size()!=ref.size())
    {
      return false;
    }
    return MaxRelDiff(got,ref)<=tol;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

template<typename StorageT>
bool CAIF_VisionPortsTests::TestConv2DDevice(
  const CAIF_DataType::CAIF_DataType_e storage_dt,
  const float tol)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);

    const uint32_t n=1u;
    const uint32_t h=5u;
    const uint32_t w=5u;
    const uint32_t cin=3u;
    const uint32_t cout=4u;
    const uint32_t kh=3u;
    const uint32_t kw=3u;
    const std::vector<uint32_t> shape={n,h,w,cin};
    const std::vector<float> data=MakeRandomVec(Product(shape),821);

    CAIF_DeviceConv2DConfig cfg_host={cin,cout,kh,kw,1u,1u};
    CAIF_DeviceConv2D<float,float> host_layer(cfg_host,stream);
    CAIF_DeviceTensor x_host=MakeHostFromVec(shape,data);
    CAIF_DeviceTensor y_host=host_layer.Forward(x_host,ctx);
    const std::vector<float> ref=ReadHost(y_host);

    CAIF_DeviceTensor x_fp32_dev=CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
    CAIF_DeviceTensor x_dev=x_fp32_dev.To(storage_dt);
    CAIF_DeviceConv2DConfig cfg_dev={cin,cout,kh,kw,1u,1u};
    CAIF_DeviceConv2D<float,StorageT> dev_layer(cfg_dev,stream);
    CAIF_DeviceTensor y_dev=dev_layer.Forward(x_dev,ctx);
    CAIF_DeviceTensor y_dev_fp32=y_dev.To(CAIF_DataType::CAIF_DataType_e::Float32);
    std::vector<float> got(y_dev_fp32.TotalElements());
    y_dev_fp32.CopyToHost(got.data());
    stream.Synchronize();

    if(got.size()!=ref.size())
    {
      return false;
    }
    const float diff=MaxRelDiff(got,ref);
    if(diff>tol)
    {
      ISE_Out::Out()<<"  conv2d device dtype: max_rel_diff="
                    <<diff
                    <<" tol="
                    <<tol
                    <<"\n";
    }
    return diff<=tol;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

template<typename StorageT>
bool CAIF_VisionPortsTests::TestBatchNormDevice(
  const CAIF_DataType::CAIF_DataType_e storage_dt,
  const float tol)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    const uint32_t batch=16u;
    const uint32_t features=8u;
    const std::vector<uint32_t> shape={batch,features};
    const std::vector<float> data=MakeRandomVec(Product(shape),831);

    // Reference: host fp32 path.
    CAIF_DeviceBatchNorm<float,float> host_layer(features,stream);
    CAIF_DeviceTensor x_host=MakeHostFromVec(shape,data);
    CAIF_DeviceTensor y_host=host_layer.Forward(x_host,ctx);
    const std::vector<float> ref=ReadHost(y_host);

    // Device path at the templated cell.
    CAIF_DeviceTensor x_fp32_dev=CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
    CAIF_DeviceTensor x_dev=x_fp32_dev.To(storage_dt);
    CAIF_DeviceBatchNorm<float,StorageT> dev_layer(features,stream);
    CAIF_DeviceTensor y_dev=dev_layer.Forward(x_dev,ctx);
    CAIF_DeviceTensor y_dev_fp32=y_dev.To(CAIF_DataType::CAIF_DataType_e::Float32);
    std::vector<float> got(y_dev_fp32.TotalElements());
    y_dev_fp32.CopyToHost(got.data());
    stream.Synchronize();

    if(got.size()!=ref.size())
    {
      return false;
    }
    const float diff=MaxRelDiff(got,ref);
    if(diff>tol)
    {
      ISE_Out::Out()<<"  batch_norm device dtype: max_rel_diff="
                    <<diff
                    <<" tol="
                    <<tol
                    <<"\n";
    }
    return diff<=tol;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

template<typename StorageT>
bool CAIF_VisionPortsTests::TestBatchNormBackwardDevice(
  const CAIF_DataType::CAIF_DataType_e storage_dt,
  const float tol)
{
  try
  {
    CAIF_CudaStream stream;
    CAIF_RunContext ctx;
    ctx.SetStream(stream);
    ctx.SetTraining(true);

    const uint32_t batch=16u;
    const uint32_t features=8u;
    const std::vector<uint32_t> shape={batch,features};
    const std::vector<float> data=MakeRandomVec(Product(shape),841);
    const std::vector<float> grad_data=MakeRandomVec(Product(shape),842);

    // Reference: host fp32 forward + backward.
    CAIF_DeviceBatchNorm<float,float> host_layer(features,stream);
    CAIF_DeviceTensor x_host=MakeHostFromVec(shape,data);
    CAIF_DeviceTensor y_host=host_layer.Forward(x_host,ctx);
    CAIF_DeviceTensor go_host=MakeHostFromVec(shape,grad_data);
    CAIF_DeviceTensor gi_host=host_layer.Backward(go_host,ctx);
    const std::vector<float> gi_ref=ReadHost(gi_host);

    // Device path on the templated cell.
    CAIF_DeviceTensor x_fp32_dev=CAIF_DeviceTensor::FromHostData(data.data(),shape,stream);
    CAIF_DeviceTensor x_dev=x_fp32_dev.To(storage_dt);
    CAIF_DeviceBatchNorm<float,StorageT> dev_layer(features,stream);
    CAIF_DeviceTensor y_dev=dev_layer.Forward(x_dev,ctx);
    CAIF_DeviceTensor go_fp32_dev=CAIF_DeviceTensor::FromHostData(grad_data.data(),
                                                                    shape,stream);
    CAIF_DeviceTensor go_dev=go_fp32_dev.To(storage_dt);
    CAIF_DeviceTensor gi_dev=dev_layer.Backward(go_dev,ctx);
    CAIF_DeviceTensor gi_dev_fp32=gi_dev.To(CAIF_DataType::CAIF_DataType_e::Float32);
    std::vector<float> gi_got(gi_dev_fp32.TotalElements());
    gi_dev_fp32.CopyToHost(gi_got.data());
    stream.Synchronize();

    if(gi_got.size()!=gi_ref.size())
    {
      return false;
    }
    const float diff=MaxRelDiff(gi_got,gi_ref);
    if(diff>tol)
    {
      ISE_Out::Out()<<"  batch_norm device backward: max_rel_diff="
                    <<diff
                    <<" tol="
                    <<tol
                    <<"\n";
    }
    return diff<=tol;
  }
  CAIF_CATCH_BLOCK();
  return false;
}

void CAIF_VisionPortsTests::RunAll()
{
  ISE_Out::Out()<<"Stage 5b Device-Layer Port Tests\n";
  ISE_Out::Out()<<"=================================\n";
  CAIF_TestHarness::Reset();

  CAIF_TestHarness::Report("Flatten forward shape",TestFlattenForwardShape());
  CAIF_TestHarness::Report("Flatten gradcheck",TestFlattenGradCheck());

  CAIF_TestHarness::Report("Reshape forward shape",TestReshapeForwardShape());
  CAIF_TestHarness::Report("Reshape gradcheck",TestReshapeGradCheck());

  CAIF_TestHarness::Report("Dropout inference identity",TestDropoutInferenceIsIdentity());
  CAIF_TestHarness::Report("Dropout training deterministic",TestDropoutTrainingDeterministic());
  CAIF_TestHarness::Report("Dropout backward matches mask",TestDropoutBackwardMatchesMask());

  CAIF_TestHarness::Report("BatchNorm forward zero-mean unit-var",TestBatchNormForwardStats());
  CAIF_TestHarness::Report("BatchNorm gradcheck",TestBatchNormGradCheck());

  CAIF_TestHarness::Report("MaxPool2D forward shape",TestMaxPoolForwardShape());
  CAIF_TestHarness::Report("MaxPool2D gradcheck",TestMaxPoolGradCheck());

  CAIF_TestHarness::Report("AvgPool2D forward mean",TestAvgPoolForwardMean());
  CAIF_TestHarness::Report("AvgPool2D gradcheck",TestAvgPoolGradCheck());

  CAIF_TestHarness::Report("Conv2D forward shape",TestConv2DForwardShape());
  CAIF_TestHarness::Report("Conv2D gradcheck",TestConv2DGradCheck());

  // cuDNN device-path tests (G1). Tolerance reflects each dtype's
  // representable precision; pooling preserves exact values for max/avg
  // so fp32 matches host exactly, while fp16/bf16 lose precision.
  typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
  CAIF_TestHarness::Report("MaxPool2D device fp32",
                           TestMaxPool2DDevice<float>(Dtype_e::Float32,
                                                      g_caif_visionports_test_tol_fp32));
  CAIF_TestHarness::Report("MaxPool2D device fp16",
                           TestMaxPool2DDevice<__half>(Dtype_e::Float16,
                                                       g_caif_visionports_test_tol_fp16_pool));
  CAIF_TestHarness::Report("MaxPool2D device bf16",
                           TestMaxPool2DDevice<__nv_bfloat16>(
                             Dtype_e::BFloat16,
                             g_caif_visionports_test_tol_bf16_pool));
  CAIF_TestHarness::Report("AvgPool2D device fp32",
                           TestAvgPool2DDevice<float>(Dtype_e::Float32,
                                                      g_caif_visionports_test_tol_fp32));
  CAIF_TestHarness::Report("AvgPool2D device fp16",
                           TestAvgPool2DDevice<__half>(Dtype_e::Float16,
                                                       g_caif_visionports_test_tol_fp16_pool));
  CAIF_TestHarness::Report("AvgPool2D device bf16",
                           TestAvgPool2DDevice<__nv_bfloat16>(
                             Dtype_e::BFloat16,
                             g_caif_visionports_test_tol_bf16_pool));
  CAIF_TestHarness::Report("Conv2D device fp32",
                           TestConv2DDevice<float>(Dtype_e::Float32,
                                                   g_caif_visionports_test_tol_fp32_conv));
  CAIF_TestHarness::Report("Conv2D device fp16",
                           TestConv2DDevice<__half>(Dtype_e::Float16,
                                                    g_caif_visionports_test_tol_fp16_conv));
  CAIF_TestHarness::Report("Conv2D device bf16",
                           TestConv2DDevice<__nv_bfloat16>(Dtype_e::BFloat16,
                                                            g_caif_visionports_test_tol_bf16_conv));

  CAIF_TestHarness::Report("BatchNorm device fp32",
                           TestBatchNormDevice<float>(Dtype_e::Float32,
                                                      g_caif_visionports_test_tol_fp32_bn));
  CAIF_TestHarness::Report("BatchNorm device fp16",
                           TestBatchNormDevice<__half>(Dtype_e::Float16,
                                                       g_caif_visionports_test_tol_fp16_bn));
  CAIF_TestHarness::Report("BatchNorm device bf16",
                           TestBatchNormDevice<__nv_bfloat16>(
                             Dtype_e::BFloat16,
                             g_caif_visionports_test_tol_bf16_bn));
  CAIF_TestHarness::Report("BatchNorm device backward fp32",
                           TestBatchNormBackwardDevice<float>(
                             Dtype_e::Float32,
                             g_caif_visionports_test_tol_fp32_bn_bwd));
  CAIF_TestHarness::Report("BatchNorm device backward fp16",
                           TestBatchNormBackwardDevice<__half>(
                             Dtype_e::Float16,
                             g_caif_visionports_test_tol_fp16_bn_bwd));
  CAIF_TestHarness::Report("BatchNorm device backward bf16",
                           TestBatchNormBackwardDevice<__nv_bfloat16>(
                             Dtype_e::BFloat16,
                             g_caif_visionports_test_tol_bf16_bn_bwd));
}

}//end instance namespace

int main()
{
  instance::CAIF_VisionPortsTests::RunAll();
  return instance::CAIF_TestHarness::FinalExitCode();
}
