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

#include "retrainer/rtnr_token_loader.h"

#include <fstream>
#include <sstream>
#include <algorithm>

#include "ise_lib/ise_out.h"

using namespace instance;

namespace instance
{

RTNR_TokenLoader::RTNR_TokenLoader():_vocab_size(0),
                                      _max_len(0),
                                      _loaded(false)
{
}

void RTNR_TokenLoader::Load(const std::string &path,int32_t separator_token)
{
  try
  {
    std::ifstream file(path);
    if(file.is_open()==false)
    {
      THROW_RTNRE("Failed to open token file: "+path);
    }

    _sequences.clear();
    _vocab_size=0;
    _max_len=0;

    std::vector<uint32_t> current_seq;
    std::string line;

    while(std::getline(file,line))
    {
      // Trim whitespace
      size_t start=line.find_first_not_of(" \t\r\n");
      if(start==std::string::npos)
      {
        // Empty line - sequence boundary
        if(current_seq.empty()==false)
        {
          if(current_seq.size()>_max_len)
          {
            _max_len=static_cast<uint32_t>(current_seq.size());
          }
          _sequences.push_back(std::move(current_seq));
          current_seq.clear();
        }
        continue;
      }

      size_t end=line.find_last_not_of(" \t\r\n");
      line=line.substr(start,end-start+1);

      // Parse token ID
      int32_t token_id=0;
      try
      {
        token_id=std::stoi(line);
      }
      catch(const std::exception &)
      {
        // Skip invalid lines
        continue;
      }

      if(token_id<0)
      {
        // Negative token - treat as separator
        if(current_seq.empty()==false)
        {
          if(current_seq.size()>_max_len)
          {
            _max_len=static_cast<uint32_t>(current_seq.size());
          }
          _sequences.push_back(std::move(current_seq));
          current_seq.clear();
        }
        continue;
      }

      // Check for separator token
      if(separator_token>=0&&token_id==separator_token)
      {
        if(current_seq.empty()==false)
        {
          if(current_seq.size()>_max_len)
          {
            _max_len=static_cast<uint32_t>(current_seq.size());
          }
          _sequences.push_back(std::move(current_seq));
          current_seq.clear();
        }
        continue;
      }

      const uint32_t token=static_cast<uint32_t>(token_id);
      current_seq.push_back(token);

      if(token+1>_vocab_size)
      {
        _vocab_size=token+1;
      }
    }

    // Don't forget the last sequence
    if(current_seq.empty()==false)
    {
      if(current_seq.size()>_max_len)
      {
        _max_len=static_cast<uint32_t>(current_seq.size());
      }
      _sequences.push_back(std::move(current_seq));
    }

    _loaded=true;

    ISE_Out::Out()<<"[RTNR] Loaded "<<_sequences.size()<<" sequences, vocab_size="
                  <<_vocab_size<<", max_len="<<_max_len<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_TokenLoader::Load")
}

void RTNR_TokenLoader::LoadBinary(const std::string &path)
{
  try
  {
    std::ifstream file(path,std::ios::binary);
    if(file.is_open()==false)
    {
      THROW_RTNRE("Failed to open binary token file: "+path);
    }

    // Read header
    uint32_t magic=0;
    uint32_t version=0;
    uint32_t num_sequences=0;
    uint32_t max_seq_len=0;

    file.read(reinterpret_cast<char *>(&magic),sizeof(magic));
    file.read(reinterpret_cast<char *>(&version),sizeof(version));
    file.read(reinterpret_cast<char *>(&num_sequences),sizeof(num_sequences));
    file.read(reinterpret_cast<char *>(&max_seq_len),sizeof(max_seq_len));

    if(magic!=g_rtnr_token_magic)
    {
      THROW_RTNRE("Invalid token file magic number");
    }

    if(version!=g_rtnr_token_version)
    {
      THROW_RTNRE("Unsupported token file version");
    }

    // Read sequence table
    struct SequenceInfo_t
    {
      uint32_t offset;
      uint32_t length;
    };

    std::vector<SequenceInfo_t> seq_table(num_sequences);
    for(uint32_t i=0;i<num_sequences;++i)
    {
      file.read(reinterpret_cast<char *>(&seq_table[i].offset),sizeof(uint32_t));
      file.read(reinterpret_cast<char *>(&seq_table[i].length),sizeof(uint32_t));
    }

    // Calculate data section start
    const size_t data_start=16+num_sequences*8;

    // Read sequences
    _sequences.clear();
    _sequences.reserve(num_sequences);
    _vocab_size=0;
    _max_len=0;

    for(uint32_t i=0;i<num_sequences;++i)
    {
      const uint32_t seq_len=seq_table[i].length;
      std::vector<uint32_t> tokens(seq_len);

      file.seekg(static_cast<std::streamoff>(data_start+seq_table[i].offset));
      file.read(reinterpret_cast<char *>(tokens.data()),
                static_cast<std::streamsize>(seq_len*sizeof(uint32_t)));

      for(uint32_t j=0;j<seq_len;++j)
      {
        if(tokens[j]+1>_vocab_size)
        {
          _vocab_size=tokens[j]+1;
        }
      }

      if(seq_len>_max_len)
      {
        _max_len=seq_len;
      }

      _sequences.push_back(std::move(tokens));
    }

    _loaded=true;

    ISE_Out::Out()<<"[RTNR] Loaded binary: "<<_sequences.size()<<" sequences, vocab_size="
                  <<_vocab_size<<", max_len="<<_max_len<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_TokenLoader::LoadBinary")
}

void RTNR_TokenLoader::SaveBinary(const std::string &path)const
{
  try
  {
    if(_loaded==false)
    {
      THROW_RTNRE("No data loaded to save");
    }

    std::ofstream file(path,std::ios::binary);
    if(file.is_open()==false)
    {
      THROW_RTNRE("Failed to create output file: "+path);
    }

    const uint32_t magic=g_rtnr_token_magic;
    const uint32_t version=g_rtnr_token_version;
    const uint32_t num_sequences=static_cast<uint32_t>(_sequences.size());
    const uint32_t max_seq_len=_max_len;

    file.write(reinterpret_cast<const char *>(&magic),sizeof(magic));
    file.write(reinterpret_cast<const char *>(&version),sizeof(version));
    file.write(reinterpret_cast<const char *>(&num_sequences),sizeof(num_sequences));
    file.write(reinterpret_cast<const char *>(&max_seq_len),sizeof(max_seq_len));

    // Calculate offsets and write sequence table
    uint32_t current_offset=0;
    for(uint32_t i=0;i<num_sequences;++i)
    {
      const uint32_t length=static_cast<uint32_t>(_sequences[i].size());
      file.write(reinterpret_cast<const char *>(&current_offset),sizeof(current_offset));
      file.write(reinterpret_cast<const char *>(&length),sizeof(length));
      current_offset+=length*sizeof(uint32_t);
    }

    // Write token data
    for(uint32_t i=0;i<num_sequences;++i)
    {
      const std::vector<uint32_t> &seq=_sequences[i];
      file.write(reinterpret_cast<const char *>(seq.data()),
                 static_cast<std::streamsize>(seq.size()*sizeof(uint32_t)));
    }

    ISE_Out::Out()<<"[RTNR] Saved binary: "<<path<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_TokenLoader::SaveBinary")
}

std::pair<CAIF_DeviceTensor,CAIF_DeviceTensor>
RTNR_TokenLoader::Batch(const std::vector<size_t> &indices,
                         uint32_t max_len,
                         CAIF_CudaStream &stream)const
{
  try
  {
    if(_loaded==false)
    {
      THROW_RTNRE("Token data not loaded");
    }

    const uint32_t batch_size=static_cast<uint32_t>(indices.size());
    if(batch_size==0)
    {
      THROW_RTNRE("Empty batch indices");
    }

    // Allocate host buffers
    // For causal LM: target[i] = input[i+1]
    std::vector<float> input_data(static_cast<size_t>(batch_size)*max_len,0.0f);
    std::vector<float> target_data(static_cast<size_t>(batch_size)*max_len,0.0f);

    for(uint32_t b=0;b<batch_size;++b)
    {
      const size_t seq_idx=indices[b];
      if(seq_idx>=_sequences.size())
      {
        THROW_RTNRE("Sequence index out of range");
      }

      const std::vector<uint32_t> &seq=_sequences[seq_idx];
      const uint32_t seq_len=static_cast<uint32_t>(seq.size());

      // Need at least 2 tokens for input->target pairs
      uint32_t copy_len=0;
      if(seq_len>0)
      {
        copy_len=seq_len-1;
      }
      if(copy_len>max_len)
      {
        copy_len=max_len;
      }

      for(uint32_t t=0;t<copy_len;++t)
      {
        input_data[static_cast<size_t>(b)*max_len+t]=static_cast<float>(seq[t]);
        target_data[static_cast<size_t>(b)*max_len+t]=static_cast<float>(seq[t+1]);
      }
    }

    CAIF_DeviceTensor input_tensor=CAIF_DeviceTensor::FromHostData(input_data.data(),
                                                                  {batch_size,max_len},
                                                                  stream);
    CAIF_DeviceTensor target_tensor=CAIF_DeviceTensor::FromHostData(target_data.data(),
                                                                   {batch_size,max_len},
                                                                   stream);

    return std::make_pair(std::move(input_tensor),std::move(target_tensor));
  }
  RTNR_CATCH_BLOCK("RTNR_TokenLoader::Batch")
}

std::vector<uint32_t> RTNR_TokenLoader::Sequence(size_t index)const
{
  try
  {
    if(_loaded==false)
    {
      THROW_RTNRE("Token data not loaded");
    }

    if(index>=_sequences.size())
    {
      THROW_RTNRE("Sequence index out of range");
    }

    return _sequences[index];
  }
  RTNR_CATCH_BLOCK("RTNR_TokenLoader::Sequence")
}

size_t RTNR_TokenLoader::TotalTokens()const
{
  size_t total=0;
  for(const auto &seq:_sequences)
  {
    total+=seq.size();
  }
  return total;
}

}
