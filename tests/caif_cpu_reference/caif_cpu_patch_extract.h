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
// Shared CPU-reference image-to-patch extraction primitive.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_PATCH_EXTRACT_H
#define CAIF_CPU_REFERENCE_PATCH_EXTRACT_H

#include <cstdint>

namespace instance
{

class CAIF_CpuPatchExtract
{
  public:
    static void Apply(const float *input,float *output,const int32_t batch,
                      const int32_t height,const int32_t width,
                      const int32_t channels,const int32_t patch_size,
                      const int32_t num_patches_h,const int32_t num_patches_w)
    {
      const int32_t patch_flat=patch_size*patch_size*channels;
      const int32_t num_patches=num_patches_h*num_patches_w;
      for(int32_t b=0;b<batch;++b)
      {
        for(int32_t ph=0;ph<num_patches_h;++ph)
        {
          for(int32_t pw=0;pw<num_patches_w;++pw)
          {
            const int32_t patch_idx=ph*num_patches_w+pw;
            const int32_t out_row=b*num_patches+patch_idx;
            for(int32_t lh=0;lh<patch_size;++lh)
            {
              for(int32_t lw=0;lw<patch_size;++lw)
              {
                const int32_t gh=ph*patch_size+lh;
                const int32_t gw=pw*patch_size+lw;
                for(int32_t c=0;c<channels;++c)
                {
                  const int32_t flat_pos=(lh*patch_size+lw)*channels+c;
                  const int32_t in_idx=((b*height+gh)*width+gw)*channels+c;
                  output[out_row*patch_flat+flat_pos]=input[in_idx];
                }
              }
            }
          }
        }
      }
    }

  protected:
  private:
};

}//end instance namespace

#endif
