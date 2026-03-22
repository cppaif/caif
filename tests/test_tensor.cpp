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

#include "caif_tensor.h"
#include "caif_framework.h"
#include <iostream>
#include <vector>
#include <random>

using namespace instance;
using CAIF_DataType_e = CAIF_DataType::CAIF_DataType_e;

// Global framework for tests
static CAIF_Framework g_test_framework;

static int ExpectEqual(const std::vector<uint32_t> &a,const std::vector<uint32_t> &b,const char *msg)
{
  if(a.size()!=b.size())
  {
    std::cout<<"FAIL: "<<msg<<" size mismatch\n";
    return 1;
  }
  for(size_t i=0;i<a.size();++i)
  {
    if(a[i]!=b[i])
    {
      std::cout<<"FAIL: "
               <<msg
               <<" at dim "
               <<i
               <<" got="
               <<a[i]
               <<" exp="
               <<b[i]
               <<"\n";
      return 1;
    }
  }
  return 0;
}

static int TestTransposeBasics()
{
  try
  {
    CAIF_Tensor t(g_test_framework,{2,3,4},CAIF_DataType::CAIF_DataType_e::Float32);
    float *p=t.MutableData<float>();
    if(p==nullptr)
    {
      std::cout<<"transpose: alloc fail\n";
      return 1;
    }
    float *d=p;
    for(size_t i=0;i<t.NumElements();++i)
    {
      d[i]=static_cast<float>(i);
    }

    // Permute [0,1,2] -> [1,0,2]
    auto u=t.Transpose({1,0,2});
    if(ExpectEqual(u.Shape(),std::vector<uint32_t>{3,2,4},"transpose shape")!=0)
    {
      return 1;
    }

    // Check a few mapped elements
    const auto s=CAIF_Tensor::CalculateStrides(t.Shape());
    const auto su=CAIF_Tensor::CalculateStrides(u.Shape());
    const float *ud=static_cast<const float*>(u.Data());
    const float *td=static_cast<const float*>(t.Data());
    // Sample couple of indices
    {
      // t[1,2,3] maps to u[2,1,3]
      size_t tidx=1*s[0]+2*s[1]+3*s[2];
      size_t uidx=2*su[0]+1*su[1]+3*su[2];
      if(ud[uidx]!=td[tidx])
      {
        std::cout<<"transpose: value mismatch\n";
        return 1;
      }
    }
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"transpose ex: "<<e.what()<<"\n";
    return 1;
  }
}

static int TestSliceBasics()
{
  try
  {
    CAIF_Tensor t(g_test_framework,{4,5,6},CAIF_DataType::CAIF_DataType_e::Float32);
    float *p=t.MutableData<float>();
    if(p==nullptr)
    {
      std::cout<<"slice: alloc fail\n";
      return 1;
    }
    float *d=p;
    for(size_t i=0;i<t.NumElements();++i)
    {
      d[i]=static_cast<float>(i);
    }

    std::vector<std::pair<uint32_t,uint32_t>> ranges;
    ranges.push_back({1,3}); // batch 2 elements
    ranges.push_back({0,5});
    ranges.push_back({2,6}); // last 4 columns
    auto s=t.Slice(ranges);
    if(ExpectEqual(s.Shape(),std::vector<uint32_t>{2,5,4},"slice shape")!=0)
    {
      return 1;
    }

    // Verify a few mapped values
    const auto ts=CAIF_Tensor::CalculateStrides(t.Shape());
    const auto ss=CAIF_Tensor::CalculateStrides(s.Shape());
    const float *sd=static_cast<const float*>(s.Data());
    const float *td=static_cast<const float*>(t.Data());
    // pick s[1,4,3] -> t[1+1, 4+0, 3+2] = t[2,4,5]
    size_t sidx=1*ss[0]+4*ss[1]+3*ss[2];
    size_t tidx=(1+ranges[0].first)*ts[0]+(4+ranges[1].first)*ts[1]+(3+ranges[2].first)*ts[2];
    if(sd[sidx]!=td[tidx])
    {
      std::cout<<"slice: value mismatch\n";
      return 1;
    }
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"slice ex: "<<e.what()<<"\n";
    return 1;
  }
}

static int TestReduceAlongAxis()
{
  try
  {
    CAIF_Tensor t(g_test_framework,{2,3,4},CAIF_DataType::CAIF_DataType_e::Float32);
    float *p=t.MutableData<float>();
    if(p==nullptr)
    {
      std::cout<<"reduce: alloc fail\n";
      return 1;
    }
    float *d=p;
    for(size_t i=0;i<t.NumElements();++i)
    {
      d[i]=1.0f;
    }

    auto sum0=t.Sum(0); // expect shape {3,4} with all 2's
    if(ExpectEqual(sum0.Shape(),std::vector<uint32_t>{3,4},"sum shape")!=0)
    {
      return 1;
    }
    const float *sd=static_cast<const float*>(sum0.Data());
    for(size_t i=0;i<sum0.NumElements();++i)
    {
      if(sd[i]!=2.0f)
      {
        std::cout<<"sum0 val mismatch\n";
        return 1;
      }
    }

    auto sum1=t.Sum(1); // expect {2,4} with all 3's
    if(ExpectEqual(sum1.Shape(),std::vector<uint32_t>{2,4},"sum1 shape")!=0)
    {
      return 1;
    }
    const float *s1=static_cast<const float*>(sum1.Data());
    for(size_t i=0;i<sum1.NumElements();++i)
    {
      if(s1[i]!=3.0f)
      {
        std::cout<<"sum1 val mismatch\n";
        return 1;
      }
    }
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"reduce ex: "<<e.what()<<"\n";
    return 1;
  }
}

static std::pair<uint32_t,uint32_t> PickRange(std::mt19937 &rng,uint32_t n)
{
  uint32_t a=static_cast<uint32_t>(rng()%n);
  uint32_t b=a+static_cast<uint32_t>(rng()%(n-a));
  if(b==a)
  {
    if(a<n)
    {
      b=a+1;
    }
    else
    {
      b=n;
    }
  }
  if(b>n)
  {
    b=n;
  }
  return {a,b};
}

static int TestFuzzSlices()
{
  try
  {
    std::mt19937 rng(123);
    for(int it=0; it<100; ++it)
    {
      const uint32_t B=static_cast<uint32_t>(1+rng()%4);
      const uint32_t H=static_cast<uint32_t>(2+rng()%6);
      const uint32_t W=static_cast<uint32_t>(2+rng()%6);
      const uint32_t C=static_cast<uint32_t>(1+rng()%4);
      CAIF_Tensor t(g_test_framework,{B,H,W,C},CAIF_DataType::CAIF_DataType_e::Float32);
      float *p=t.MutableData<float>();
      if(p==nullptr)
      {
        return 1;
      }
      float *d=p;
      for(size_t i=0;i<t.NumElements();++i)
      {
        d[i]=static_cast<float>(i%97);
      }
      std::vector<std::pair<uint32_t,uint32_t>> r;
      r.push_back(PickRange(rng,B));
      r.push_back(PickRange(rng,H));
      r.push_back(PickRange(rng,W));
      r.push_back(PickRange(rng,C));
      auto s=t.Slice(r);
      // Validate sizes
      std::vector<uint32_t> exp={r[0].second-r[0].first,
                                r[1].second-r[1].first,
                                r[2].second-r[2].first,
                                r[3].second-r[3].first};
      if(ExpectEqual(s.Shape(),exp,"fuzz slice shape")!=0)
      {
        return 1;
      }
    }
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"fuzz slice ex: "<<e.what()<<"\n";
    return 1;
  }
}

int TestSlices()
{
  try
  {
    std::vector<uint32_t> shape={12,360,640,3};
    CAIF_Tensor t(g_test_framework,shape,CAIF_DataType::CAIF_DataType_e::Float32);
    // Fill with dummy data
    float *md=t.MutableData<float>();
    if(md!=nullptr)
    {
      float *ptr=md;
      const size_t n=t.NumElements();
      for(size_t i=0;i<n;++i)
      {
        ptr[i]=static_cast<float>(i%255)/255.0f;
      }
    }
    // View first batch range [0,12)
    CAIF_Tensor view=t.SliceViewBatch({0,12});
    // Copy a sub-slice [0,12) x [0,360) x [0,640) x [0,3)
    std::vector<std::pair<uint32_t,uint32_t>> ranges;
    ranges.push_back({0,12});
    ranges.push_back({0,360});
    ranges.push_back({0,640});
    ranges.push_back({0,3});
    CAIF_Tensor copy=view.Slice(ranges);
    std::cout<<"Slice test ok: "<<copy.ToString()<<"\n";
    return 0;
  }
  catch(const std::exception &e)
  {
    std::cout<<"Slice test failed: "<<e.what()<<"\n";
    return 1;
  }
}

int main()
{
  int rc=0;
  rc|=TestTransposeBasics();
  rc|=TestSliceBasics();
  rc|=TestReduceAlongAxis();
  rc|=TestFuzzSlices();
  if(rc!=0)
  {
    return rc;
  }
  return TestSlices();
} 