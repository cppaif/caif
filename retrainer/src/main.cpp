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

#include <iostream>
#include <string>
#include <sstream>

#include "retrainer/rtnr_trainer.h"
#include "ise_lib/ise_out.h"

using namespace instance;

static std::vector<std::string> ParseCommaList(const std::string &s)
{
  std::vector<std::string> result;
  std::istringstream stream(s);
  std::string token;
  while(std::getline(stream,token,','))
  {
    if(token.empty()==false)
    {
      result.push_back(token);
    }
  }
  return result;
}

static void PrintUsage()
{
  ISE_Out::Out()<<"Usage:"<<std::endl;
  ISE_Out::Out()<<"  retrainer -m train [options]"<<std::endl;
  ISE_Out::Out()<<"  retrainer -m finetune-glm [options]"<<std::endl;
  ISE_Out::Out()<<"  retrainer -m finetune-qwen [options]"<<std::endl;
  ISE_Out::Out()<<std::endl;
  ISE_Out::Out()<<"Modes:"<<std::endl;
  ISE_Out::Out()<<"  train        Train a model from scratch"<<std::endl;
  ISE_Out::Out()<<"  finetune     Fine-tune a pre-trained safetensors model"<<std::endl;
  ISE_Out::Out()<<"  finetune-glm Fine-tune a GLM model from HuggingFace directory"<<std::endl;
  ISE_Out::Out()<<"  finetune-qwen Fine-tune a Qwen model from HuggingFace directory"<<std::endl;
  ISE_Out::Out()<<std::endl;
  ISE_Out::Out()<<"Required flags:"<<std::endl;
  ISE_Out::Out()<<"  -m, --mode <mode>         Mode: train, finetune, or finetune-glm"<<std::endl;
  ISE_Out::Out()<<"  -t, --train-data <file>   Training data (.tokens or .tok)"<<std::endl;
  ISE_Out::Out()<<"  -o, --output <file>       Output model path (.safetensors)"<<std::endl;
  ISE_Out::Out()<<std::endl;
  ISE_Out::Out()<<"Architecture flags (for train mode):"<<std::endl;
  ISE_Out::Out()<<"  --vocab-size <n>          Vocabulary size"<<std::endl;
  ISE_Out::Out()<<"  --dim <n>                 Model dimension"<<std::endl;
  ISE_Out::Out()<<"  --num-heads <n>           Number of attention heads"<<std::endl;
  ISE_Out::Out()<<"  --num-kv-heads <n>        Number of KV heads (for GQA, default=num-heads)"<<std::endl;
  ISE_Out::Out()<<"  --num-layers <n>          Number of transformer layers"<<std::endl;
  ISE_Out::Out()<<"  --ffn-dim <n>             FFN hidden dim (default: auto)"<<std::endl;
  ISE_Out::Out()<<"  --max-seq-len <n>         Maximum sequence length (default: 512)"<<std::endl;
  ISE_Out::Out()<<std::endl;
  ISE_Out::Out()<<"Fine-tune flags:"<<std::endl;
  ISE_Out::Out()<<"  --base-model <file>       Base model to fine-tune (.safetensors)"<<std::endl;
  ISE_Out::Out()<<std::endl;
  ISE_Out::Out()<<"GLM fine-tune flags (for finetune-glm mode):"<<std::endl;
  ISE_Out::Out()<<"  --model-dir <path>        HuggingFace model directory (required)"<<std::endl;
  ISE_Out::Out()<<"  --storage-dtype <dtype>   Weight storage: fp32, fp16, bf16, int8, int4 (default: bf16)"<<std::endl;
  ISE_Out::Out()<<"  --lora-rank <n>           LoRA rank, 0 disables LoRA (default: 16)"<<std::endl;
  ISE_Out::Out()<<"  --lora-alpha <f>          LoRA alpha scaling factor (default: 32.0)"<<std::endl;
  ISE_Out::Out()<<"  --lora-targets <list>     Comma-separated targets: q,kv,o,gate,up,down (default: q,kv,o)"<<std::endl;
  ISE_Out::Out()<<"  --save-lora-only          Save only LoRA weights, not base model"<<std::endl;
  ISE_Out::Out()<<"  --resume-lora <file>      Load LoRA weights to resume training"<<std::endl;
  ISE_Out::Out()<<"  --resume-step <n>        Resume training from step N (skip prior steps)"<<std::endl;
  ISE_Out::Out()<<std::endl;
  ISE_Out::Out()<<"Training flags:"<<std::endl;
  ISE_Out::Out()<<"  -e, --epochs <n>          Number of epochs (default: 3)"<<std::endl;
  ISE_Out::Out()<<"  -b, --batch-size <n>      Batch size (default: 4)"<<std::endl;
  ISE_Out::Out()<<"  -l, --learning-rate <r>   Learning rate (default: 2e-5)"<<std::endl;
  ISE_Out::Out()<<"  --grad-accum <n>          Gradient accumulation steps (default: 4)"<<std::endl;
  ISE_Out::Out()<<"  --warmup-ratio <r>        Warmup ratio (default: 0.1)"<<std::endl;
  ISE_Out::Out()<<"  --val-data <file>         Validation data (optional)"<<std::endl;
  ISE_Out::Out()<<"  --checkpoint-interval <n> Save LoRA checkpoint every N steps (default: 1000, 0=off)"<<std::endl;
  ISE_Out::Out()<<"  --checkpoint-path <file>  Checkpoint file path (default: <output>.checkpoint)"<<std::endl;
  ISE_Out::Out()<<"  --min-loss <f>            Stop when epoch avg loss < this (default: 0=off)"<<std::endl;
  ISE_Out::Out()<<"  --min-epochs <n>          Min epochs before min-loss can trigger (default: 0)"<<std::endl;
  ISE_Out::Out()<<"  --log-interval <n>        Log every N steps (default: 10)"<<std::endl;
  ISE_Out::Out()<<"  -v, --verbose             Verbose output"<<std::endl;
}

int main(int argc,char **argv)
{
  try
  {
    std::string mode;
    std::string train_data_path;
    std::string val_data_path;
    std::string base_model_path;
    std::string output_path;

    RTNR_TrainConfig config;

    // Parse arguments
    for(int i=1;i<argc;++i)
    {
      const std::string arg=argv[i];

      if(arg=="-m"||arg=="--mode")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        mode=argv[i];
        continue;
      }

      if(arg=="-t"||arg=="--train-data")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        train_data_path=argv[i];
        continue;
      }

      if(arg=="--val-data")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        val_data_path=argv[i];
        continue;
      }

      if(arg=="--base-model")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        base_model_path=argv[i];
        continue;
      }

      if(arg=="-o"||arg=="--output")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        output_path=argv[i];
        continue;
      }

      if(arg=="--vocab-size")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.vocab_size=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--dim")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.dim=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--num-heads")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.num_heads=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--num-kv-heads")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.num_kv_heads=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--num-layers")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.num_layers=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--ffn-dim")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.ffn_dim=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--max-seq-len")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.max_seq_len=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="-e"||arg=="--epochs")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.epochs=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="-b"||arg=="--batch-size")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.batch_size=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="-l"||arg=="--learning-rate")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.learning_rate=std::stof(argv[i]);
        continue;
      }

      if(arg=="--grad-accum")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.grad_accum_steps=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--warmup-ratio")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.warmup_ratio=std::stof(argv[i]);
        continue;
      }

      if(arg=="--log-interval")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.log_interval=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--checkpoint-interval")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.checkpoint_interval=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--checkpoint-path")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.checkpoint_path=argv[i];
        continue;
      }

      if(arg=="--model-dir")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.model_dir=argv[i];
        continue;
      }

      if(arg=="--storage-dtype")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.storage_dtype=argv[i];
        continue;
      }

      if(arg=="--lora-rank")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.lora_r=static_cast<uint32_t>(std::stoul(argv[i]));
        config.use_lora=(config.lora_r>0);
        continue;
      }

      if(arg=="--lora-alpha")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.lora_alpha=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--lora-targets")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.lora_targets=ParseCommaList(argv[i]);
        continue;
      }

      if(arg=="--save-lora-only")
      {
        config.save_lora_only=true;
        continue;
      }

      if(arg=="--resume-lora")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.resume_lora_path=argv[i];
        continue;
      }

      if(arg=="--resume-step")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.resume_step=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="--min-loss")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.min_loss=std::stof(argv[i]);
        continue;
      }

      if(arg=="--min-epochs")
      {
        if(i+1>=argc)
        {
          ISE_Out::ErrLog()<<"Error: "<<arg<<" requires a value"<<std::endl;
          return 1;
        }
        ++i;
        config.min_epochs=static_cast<uint32_t>(std::stoul(argv[i]));
        continue;
      }

      if(arg=="-v"||arg=="--verbose")
      {
        config.verbose=true;
        continue;
      }

      if(arg=="-h"||arg=="--help")
      {
        PrintUsage();
        return 0;
      }

      ISE_Out::ErrLog()<<"Unknown argument: "<<arg<<std::endl;
      return 1;
    }

    // Validate mode
    if(mode.empty()==true)
    {
      PrintUsage();
      return 1;
    }

    if(mode!="train" && mode!="finetune" && mode!="finetune-glm" &&
       mode!="finetune-qwen")
    {
      ISE_Out::ErrLog()<<"Unknown mode: "<<mode<<std::endl;
      ISE_Out::ErrLog()<<"Valid modes: train, finetune, finetune-glm, finetune-qwen"<<std::endl;
      return 1;
    }

    // Validate required arguments
    if(train_data_path.empty()==true)
    {
      ISE_Out::ErrLog()<<"Error: --train-data is required"<<std::endl;
      return 1;
    }

    if(output_path.empty()==true)
    {
      ISE_Out::ErrLog()<<"Error: --output is required"<<std::endl;
      return 1;
    }

    // Set model type for GLM/Qwen mode
    if(mode=="finetune-glm")
    {
      config.model_type="glm";
    }
    else if(mode=="finetune-qwen")
    {
      config.model_type="qwen";
    }

    // Set default LoRA targets if not specified and LoRA is enabled
    if(config.use_lora==true&&config.lora_targets.empty()==true)
    {
      config.lora_targets={"q","kv","o"};
    }

    // Default checkpoint path from output path
    // (must be set before Initialize copies config into trainer)
    if(config.checkpoint_path.empty()==true &&
       config.checkpoint_interval>0 &&
       config.save_lora_only==true)
    {
      // e.g. "tmp/foo.safetensors" -> "tmp/foo.checkpoint.safetensors"
      std::string base=output_path;
      const size_t dot_pos=base.rfind('.');
      if(dot_pos!=std::string::npos)
      {
        config.checkpoint_path=
          base.substr(0,dot_pos)+".checkpoint"+base.substr(dot_pos);
      }
      else
      {
        config.checkpoint_path=base+".checkpoint";
      }
      ISE_Out::Out()<<"[RTNR] Checkpoint path: "
                    <<config.checkpoint_path<<std::endl;
    }

    // Create trainer
    RTNR_Trainer trainer;
    trainer.Initialize(config);

    // Load data first (except for GLM/Qwen mode which uses vocab from config)
    if(mode!="finetune-glm" && mode!="finetune-qwen")
    {
      trainer.LoadData(train_data_path,val_data_path);
    }

    if(mode=="train")
    {
      // Validate architecture flags
      if(config.dim==0)
      {
        ISE_Out::ErrLog()<<"Error: --dim is required for train mode"<<std::endl;
        return 1;
      }
      if(config.num_heads==0)
      {
        ISE_Out::ErrLog()<<"Error: --num-heads is required for train mode"<<std::endl;
        return 1;
      }
      if(config.num_layers==0)
      {
        ISE_Out::ErrLog()<<"Error: --num-layers is required for train mode"<<std::endl;
        return 1;
      }

      trainer.BuildModel();
    }
    else if(mode=="finetune")
    {
      if(base_model_path.empty()==true)
      {
        ISE_Out::ErrLog()<<"Error: --base-model is required for finetune mode"<<std::endl;
        return 1;
      }

      trainer.LoadModel(base_model_path);
    }
    else if(mode=="finetune-glm")
    {
      if(config.model_dir.empty()==true)
      {
        ISE_Out::ErrLog()<<"Error: --model-dir is required for finetune-glm mode"<<std::endl;
        return 1;
      }

      // Build and load GLM model, then load data
      trainer.BuildGLMModel();
      trainer.LoadData(train_data_path,val_data_path);
    }
    else if(mode=="finetune-qwen")
    {
      if(config.model_dir.empty()==true)
      {
        ISE_Out::ErrLog()<<"Error: --model-dir is required for finetune-qwen mode"
                         <<std::endl;
        return 1;
      }

      // Build and load Qwen model, then load data
      trainer.BuildQwenModel();
      trainer.LoadData(train_data_path,val_data_path);
    }

    // Train
    const float final_loss=trainer.Train();
    ISE_Out::Out()<<"[RTNR] Final loss: "<<final_loss<<std::endl;

    // Save
    if(config.save_lora_only==true)
    {
      trainer.SaveLoRAWeights(output_path);
    }
    else
    {
      trainer.SaveModel(output_path);
    }

    ISE_Out::Out()<<"[RTNR] Done"<<std::endl;
    return 0;
  }
  RTNR_CATCH_BLOCK("main")
}
