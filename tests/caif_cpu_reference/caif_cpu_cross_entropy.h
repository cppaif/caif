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
// Shared CPU-reference cross-entropy loss + gradient.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_CROSS_ENTROPY_H
#define CAIF_CPU_REFERENCE_CROSS_ENTROPY_H

#include <cmath>
#include <cstdint>
#include <vector>

namespace instance
{

class CAIF_CpuCrossEntropy
{
  public:
    static float Loss(const std::vector<float> &logits,
                      const std::vector<int> &targets,
                      const int32_t n,const int32_t vocab_size,
                      const int32_t ignore_index)
    {
      float total_loss=0.0f;
      int32_t count=0;
      for(int32_t i=0;i<n;++i)
      {
        if(targets[i]!=ignore_index)
        {
          const float max_logit=RowMax(logits,i,vocab_size);
          const float sum_exp=RowSumExp(logits,i,vocab_size,max_logit);
          const float log_sum_exp=max_logit+std::log(sum_exp);
          const float target_logit=logits[i*vocab_size+targets[i]];
          total_loss+=log_sum_exp-target_logit;
          ++count;
        }
      }
      if(count==0)
      {
        return 0.0f;
      }
      return total_loss/static_cast<float>(count);
    }

    static void Gradient(const std::vector<float> &logits,
                         const std::vector<int> &targets,
                         std::vector<float> &grad,
                         const int32_t n,const int32_t vocab_size,
                         const int32_t ignore_index)
    {
      grad.resize(static_cast<size_t>(n)*static_cast<size_t>(vocab_size));
      for(int32_t i=0;i<n;++i)
      {
        if(targets[i]==ignore_index)
        {
          for(int32_t v=0;v<vocab_size;++v)
          {
            grad[i*vocab_size+v]=0.0f;
          }
        }
        else
        {
          const float max_logit=RowMax(logits,i,vocab_size);
          const float sum_exp=RowSumExp(logits,i,vocab_size,max_logit);
          for(int32_t v=0;v<vocab_size;++v)
          {
            const float softmax_val=std::exp(logits[i*vocab_size+v]-max_logit)/
                                      sum_exp;
            float g=softmax_val;
            if(v==targets[i])
            {
              g-=1.0f;
            }
            grad[i*vocab_size+v]=g/static_cast<float>(n);
          }
        }
      }
    }

  protected:
  private:
    static float RowMax(const std::vector<float> &logits,const int32_t i,
                        const int32_t vocab_size)
    {
      float max_logit=-1e30f;
      for(int32_t v=0;v<vocab_size;++v)
      {
        const float val=logits[i*vocab_size+v];
        if(val>max_logit)
        {
          max_logit=val;
        }
      }
      return max_logit;
    }

    static float RowSumExp(const std::vector<float> &logits,const int32_t i,
                           const int32_t vocab_size,const float max_logit)
    {
      float sum_exp=0.0f;
      for(int32_t v=0;v<vocab_size;++v)
      {
        sum_exp+=std::exp(logits[i*vocab_size+v]-max_logit);
      }
      return sum_exp;
    }
};

}//end instance namespace

#endif
