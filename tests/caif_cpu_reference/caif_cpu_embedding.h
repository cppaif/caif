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
// Shared CPU-reference embedding-lookup primitive.
//------------------------------------------------------------------------------
#ifndef CAIF_CPU_REFERENCE_EMBEDDING_H
#define CAIF_CPU_REFERENCE_EMBEDDING_H

#include <cstdint>

namespace instance
{

class CAIF_CpuEmbedding
{
  public:
    static void Lookup(const float *table,const uint32_t *ids,float *output,
                       const int32_t num_tokens,const int32_t dim)
    {
      for(int32_t t=0;t<num_tokens;++t)
      {
        const uint32_t id=ids[t];
        for(int32_t d=0;d<dim;++d)
        {
          output[t*dim+d]=table[id*dim+d];
        }
      }
    }

  protected:
  private:
};

}//end instance namespace

#endif
