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

/**
 * @file rtnr_token_loader.h
 * @brief Token data loader for LLM retraining
 *
 * Loads tokenized text from simple text files where each line contains
 * one token ID. Sequences are separated by a special separator token.
 */

#pragma once

#include "rtnr_exception.h"
#include "rtnr_constants.h"

#include <string>
#include <vector>
#include <cstdint>
#include <utility>

#include "caif/caif_device_tensor.h"
#include "caif/caif_cuda_stream.h"

namespace instance
{

/**
 * @brief Loads tokenized text for language model training
 *
 * Text format (.tokens):
 *   One integer token ID per line
 *   Empty line or special separator token marks sequence boundary
 *
 * Example file:
 *   1234
 *   5678
 *   9012
 *
 *   3456  <- new sequence starts here
 *   7890
 */
class RTNR_TokenLoader
{
  public:
    RTNR_TokenLoader();

    /**
     * @brief Load token data from text file (one token per line)
     * @param path Path to .tokens file
     * @param separator_token Token ID that marks sequence boundaries (-1 for empty line)
     */
    void Load(const std::string &path,int32_t separator_token=-1);

    /**
     * @brief Load token data from binary .tok file
     * @param path Path to .tok file
     */
    void LoadBinary(const std::string &path);

    /**
     * @brief Save loaded data to binary .tok file for faster loading
     * @param path Output path
     */
    void SaveBinary(const std::string &path)const;

    /**
     * @brief Get a batch of sequences for causal language modeling
     * @param indices Sequence indices to include in batch
     * @param max_len Maximum sequence length (truncate/pad to this)
     * @param stream CUDA stream
     * @return Pair of (input_ids, target_ids) where target[i] = input[i+1]
     */
    std::pair<CAIF_DeviceTensor,CAIF_DeviceTensor> Batch(const std::vector<size_t> &indices,
                                                        uint32_t max_len,
                                                        CAIF_CudaStream &stream)const;

    /**
     * @brief Get a single sequence as host vector
     * @param index Sequence index
     * @return Token IDs for the sequence
     */
    std::vector<uint32_t> Sequence(size_t index)const;

    /**
     * @brief Get number of loaded sequences
     */
    size_t NumSequences()const{return _sequences.size();}

    /**
     * @brief Get detected vocabulary size (max token ID + 1)
     */
    uint32_t VocabSize()const{return _vocab_size;}

    /**
     * @brief Get maximum sequence length in loaded data
     */
    uint32_t MaxLength()const{return _max_len;}

    /**
     * @brief Check if data is loaded
     */
    bool IsLoaded()const{return _loaded;}

    /**
     * @brief Get total number of tokens across all sequences
     */
    size_t TotalTokens()const;

  protected:

  private:
    std::vector<std::vector<uint32_t>> _sequences;
    uint32_t _vocab_size;
    uint32_t _max_len;
    bool _loaded;
};

}
