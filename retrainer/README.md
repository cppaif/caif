# Retrainer

Fine-tune open-weight LLMs using the CAIF (C++ AI Framework) library with CUDA
acceleration. The full pipeline runs from HuggingFace model download through
LoRA training to Ollama deployment — no PyTorch training loop required.

## Table of Contents

- [Overview](#overview)
- [Supported Models](#supported-models)
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Full Pipeline: HuggingFace to Ollama](#full-pipeline-huggingface-to-ollama)
  - [Step 1: Download a Model](#step-1-download-a-model)
  - [Step 2: Prepare Training Data](#step-2-prepare-training-data)
  - [Step 3: Tokenize](#step-3-tokenize)
  - [Step 4: Fine-Tune with LoRA](#step-4-fine-tune-with-lora)
  - [Step 5: Merge LoRA into Base Model](#step-5-merge-lora-into-base-model)
  - [Step 6: Deploy to Ollama](#step-6-deploy-to-ollama)
- [Qwen Example Pipeline](#qwen-example-pipeline)
- [Alternative: Direct GGUF Export](#alternative-direct-gguf-export)
- [Script Reference](#script-reference)
- [Retrainer CLI Reference](#retrainer-cli-reference)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

## Overview

The retrainer is a C++ application that fine-tunes transformer language models
using LoRA (Low-Rank Adaptation) adapters. Training runs entirely on the GPU
via CAIF's CUDA kernels — the only Python involved is for data preparation
and post-training model conversion.

**External services used by the pipeline scripts:**

- **[HuggingFace Hub](https://huggingface.co/)** — source for downloading
  pre-trained model weights, tokenizers, and config files. You need a
  (free) HuggingFace account for gated models. The `download_model.py`,
  `convert_hf_model.py`, `merge_lora.py`, `tokenize_chat.py`,
  `rtnr_tokenize.py`, and `export_gguf*.py` scripts all use the HuggingFace
  `transformers` and/or `huggingface_hub` Python packages.

- **[Ollama](https://ollama.com/)** — local model runtime for deploying and
  serving your fine-tuned model. Ollama must be installed on the machine
  where you run `ollama create` and `ollama run`. The retrainer itself does
  not call Ollama — you use it after training to serve the merged model.

The pipeline:

```
HuggingFace Model (safetensors)
        |
        v
  [download_model.py]    — download model weights + tokenizer
        |
        v
  [rtnr_tokenize.py]     — tokenize plain text
  [tokenize_chat.py]     — tokenize chat/instruction JSONL
        |
        v
  [retrainer]             — GPU fine-tuning with LoRA (C++/CUDA)
        |
        v
  LoRA weights (.safetensors)
        |
        +---> [merge_lora.py]      — merge into HF model dir
        |           |
        |           v
        |     Merged safetensors --> ollama create
        |
        +---> [export_gguf.py]     — direct GGUF export (GLM/DeepSeek2)
        +---> [export_gguf_qwen.py] — direct GGUF export (Qwen/standard MHA)
                    |
                    v
              model.gguf --> ollama create
```

## Supported Models

| Model | Architecture | Mode | GGUF Export Script |
|-------|-------------|------|--------------------|
| GLM-4.7-Flash (30B-A3B MoE) | DeepSeek2 (MLA + MoE) | `finetune-glm` | `export_gguf.py` |
| Qwen2.5-Coder series | Standard MHA (GQA) | `finetune-qwen` | `export_gguf_qwen.py` |

The model builders (`rtnr_glm_model_builder`, `rtnr_qwen_model_builder`)
assemble CAIF layers from HuggingFace config.json. Adding support for a new
architecture means writing a new model builder that maps HuggingFace weight
names to CAIF layers.

## Prerequisites

### System Requirements

- **Linux** (tested on openSUSE, Ubuntu, Fedora)
- **NVIDIA GPU** with compute capability 7.0+ (Volta or newer)
- **CUDA Toolkit 12.0+** with cuBLAS and cuDNN
- **CMake 3.18+**
- **GCC 13+** (C++23 support required)

### External Tools

- **[Ollama](https://ollama.com/)** — install from https://ollama.com/download
  for model deployment (Step 6). Not needed for training itself.
- **[HuggingFace account](https://huggingface.co/join)** — free account
  needed for gated models. Log in with `huggingface-cli login` before
  downloading gated model weights.

### Python Requirements (for pipeline scripts only)

The C++ retrainer binary has no Python dependency. Python 3.8+ is only needed
for the data preparation and model conversion scripts.

```bash
pip install torch safetensors transformers huggingface_hub numpy gguf
```

Package purposes:

| Package | Used By | Purpose |
|---------|---------|---------|
| `torch` | merge_lora, export_gguf* | Tensor operations for LoRA merge |
| `safetensors` | merge_lora, export_gguf*, inspect_safetensors | Read/write safetensors files |
| `transformers` | tokenize_chat, rtnr_tokenize, export_gguf* | HuggingFace tokenizers |
| `huggingface_hub` | download_model, convert_hf_model | Download from HuggingFace Hub |
| `numpy` | export_gguf* | Tensor conversion for GGUF |
| `gguf` | export_gguf* | Write GGUF format files |

### Disk Space

Budget generously for model weights:

| Model | Download Size | INT4 Training VRAM | Merged Output |
|-------|--------------|-------------------|---------------|
| GLM-4.7-Flash (30B) | ~60 GB | ~18 GB | ~60 GB |
| Qwen2.5-Coder-1.5B | ~3 GB | ~2 GB | ~3 GB |

## Building

The retrainer is an optional component of the CAIF build. Enable it with
`-DCAIF_BUILD_RETRAINER=ON`:

```bash
cd /path/to/caif
mkdir build && cd build

# Build CAIF + retrainer
cmake .. -DCAIF_BUILD_RETRAINER=ON
make -j$(nproc)
```

The retrainer binary will be at `build/retrainer/retrainer`.

### Build Options

```
cmake .. \
  -DCAIF_BUILD_RETRAINER=ON \
  -DCAIF_BUILD_CUDA=ON \
  -DOPENBLAS_ROOT=/opt/OpenBLAS \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
```

CUDA must be enabled (`CAIF_BUILD_CUDA=ON`, which is the default) — the
retrainer will not build without it.

## Full Pipeline: HuggingFace to Ollama

This section walks through the complete GLM-4.7-Flash pipeline. For a Qwen
example, see [Qwen Example Pipeline](#qwen-example-pipeline) below.

All commands assume you are in the CAIF root directory.

### Step 1: Download a Model

**Requires:** `huggingface_hub` Python package, internet connection.

Download model weights and tokenizer from HuggingFace:

```bash
python3 retrainer/scripts/download_model.py \
    --model zai-org/GLM-4.7-Flash \
    --output-dir ./models/GLM-4.7-Flash
```

This downloads all safetensors shards, config.json, and tokenizer files. For
large models (GLM-4.7-Flash is ~60 GB) this may take a while.

To see model info without downloading:

```bash
python3 retrainer/scripts/download_model.py \
    --model zai-org/GLM-4.7-Flash \
    --info-only
```

### Step 2: Prepare Training Data

The retrainer expects training data as a token file — one token ID per line,
with blank lines separating sequences. You need to prepare your text data
before tokenizing.

**For plain text** (e.g., a corpus of documents):

Create one or more `.txt` files with your training text. The tokenizer will
split on double newlines to create separate training sequences.

**For chat/instruction data**:

Create a JSONL file where each line contains a chat conversation:

```json
{"messages":[{"role":"user","content":"What is X?"},{"role":"assistant","content":"X is..."}]}
{"messages":[{"role":"user","content":"How do I Y?"},{"role":"assistant","content":"To Y, you..."}]}
```

### Step 3: Tokenize

**Requires:** `transformers` Python package (for HuggingFace tokenizers).

**For plain text files:**

```bash
python3 retrainer/scripts/rtnr_tokenize.py \
    -i your_training_text.txt \
    -o train.tokens \
    --tokenizer ./models/GLM-4.7-Flash
```

**For a directory of text files:**

```bash
python3 retrainer/scripts/rtnr_tokenize.py \
    -i ./data/ \
    -o train.tokens \
    --tokenizer ./models/GLM-4.7-Flash \
    --glob "*.txt"
```

**For chat/instruction JSONL:**

```bash
python3 retrainer/scripts/tokenize_chat.py \
    -i train.jsonl \
    -o train.tokens \
    --tokenizer ./models/GLM-4.7-Flash
```

The chat tokenizer uses `AutoTokenizer.apply_chat_template()` which
automatically applies the correct chat format for any HuggingFace model.

Useful tokenizer options:

```
--add-bos         Add BOS token at start of each sequence
--add-eos         Add EOS token at end of each sequence
--max-length 2048 Truncate sequences longer than N tokens
--binary          Output binary format (.tok) for faster loading
--dry-run         Print statistics without writing output
```

### Step 4: Fine-Tune with LoRA

Run the retrainer to fine-tune with LoRA adapters:

```bash
./build/retrainer/retrainer \
    -m finetune-glm \
    --model-dir ./models/GLM-4.7-Flash \
    --train-data train.tokens \
    --output lora_weights.safetensors \
    --storage-dtype int4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-targets q,kv,o \
    --save-lora-only \
    --epochs 3 \
    --batch-size 1 \
    --learning-rate 2e-5 \
    --grad-accum 4
```

**Key parameters:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `-m` | Mode | `finetune-glm` or `finetune-qwen` |
| `--model-dir` | Downloaded HuggingFace model | `./models/GLM-4.7-Flash` |
| `--train-data` | Tokenized training data | `train.tokens` |
| `--output` | Output LoRA weights | `lora_weights.safetensors` |
| `--storage-dtype` | Weight storage precision | `int4` (saves VRAM), `fp16`, `bf16` |
| `--lora-rank` | LoRA adapter rank | 8, 16, or 32 |
| `--lora-alpha` | LoRA scaling factor | Usually 2x rank |
| `--lora-targets` | Which projections to adapt | `q,kv,o` (GLM) or `q,k,v,o` (Qwen) |
| `--save-lora-only` | Only save LoRA delta weights | Recommended |
| `--epochs` | Passes over training data | 1-5 |
| `--batch-size` | Samples per batch | 1 for INT4 (memory limited) |
| `--learning-rate` | Optimizer learning rate | 1e-5 to 5e-5 |
| `--grad-accum` | Gradient accumulation steps | 4-8 |

**Additional options:**

```
--log-interval 10         Print loss every N steps
--checkpoint-interval 500 Save checkpoint every N steps
--resume-lora FILE        Resume training from a LoRA checkpoint
--resume-step N           Skip first N steps when resuming
-v                        Verbose output
```

**Memory guidance:**

- INT4 storage (`--storage-dtype int4`) dramatically reduces VRAM usage. A 30B
  model that needs ~60 GB in FP16 fits in ~18 GB with INT4.
- Batch size of 1 is typical for large models on consumer GPUs. Use
  `--grad-accum` to simulate larger effective batch sizes.
- LoRA rank 16 is a good starting point. Higher rank increases capacity but
  uses more memory.

### Step 5: Merge LoRA into Base Model

**Requires:** `torch`, `safetensors` Python packages.

After training, merge the LoRA weights back into the base model:

```bash
python3 retrainer/scripts/merge_lora.py \
    --model-dir ./models/GLM-4.7-Flash \
    --lora lora_weights.safetensors \
    --output ./models/GLM-4.7-Flash-finetuned \
    --lora-alpha 32 \
    --lora-rank 16
```

The script auto-detects the model architecture (GLM, Qwen, Llama, etc.) from
`config.json` and uses the correct projection names for LoRA merging. It
creates a complete model directory with all original files (config.json,
tokenizer, etc.) plus the modified weight shards.

**Important:** The `--lora-alpha` and `--lora-rank` values must match what you
used during training. These control the scaling factor `alpha/rank` applied to
the LoRA delta.

### Step 6: Deploy to Ollama

**Requires:** [Ollama](https://ollama.com/) installed locally.

Create an Ollama `Modelfile` that points to your merged model.

**GLM-4.7-Flash Modelfile:**

```
FROM ./models/GLM-4.7-Flash-finetuned/

TEMPLATE """[gMASK]<sop>{{- range .Messages }}<|{{ .Role }}|>
{{ .Content }}{{- end }}<|assistant|>"""

PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|observation|>"
```

Save this as `Modelfile`, then import and quantize:

```bash
# Import with quantization (Q4_K_M is a good balance of quality and size)
ollama create my-finetuned-glm -f Modelfile --quantize Q4_K_M

# Test it
ollama run my-finetuned-glm "Hello, how are you?"
```

**Notes on the Modelfile:**

- The `TEMPLATE` must match your model's chat format. See model-specific
  examples below.
- The `PARAMETER stop` tokens must also match the model.
- Ollama handles quantization during `ollama create`. Use `Q4_K_M` for a good
  size/quality trade-off, or `Q8_0` for higher quality.

## Qwen Example Pipeline

This section shows the complete pipeline for Qwen2.5-Coder-1.5B-Instruct, a
much smaller model suitable for consumer GPUs (needs ~2 GB VRAM with INT4).

```bash
# Step 1: Download
python3 retrainer/scripts/download_model.py \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output-dir ./models/Qwen2.5-Coder-1.5B-Instruct

# Step 2+3: Tokenize (chat format)
python3 retrainer/scripts/tokenize_chat.py \
    -i train.jsonl \
    -o train.tokens \
    --tokenizer ./models/Qwen2.5-Coder-1.5B-Instruct

# Step 4: Fine-tune (note: finetune-qwen mode, q,k,v,o targets)
./build/retrainer/retrainer \
    -m finetune-qwen \
    --model-dir ./models/Qwen2.5-Coder-1.5B-Instruct \
    --train-data train.tokens \
    --output lora_weights.safetensors \
    --storage-dtype int4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-targets q,k,v,o \
    --save-lora-only \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --grad-accum 2

# Step 5: Merge LoRA (auto-detects Qwen architecture)
python3 retrainer/scripts/merge_lora.py \
    --model-dir ./models/Qwen2.5-Coder-1.5B-Instruct \
    --lora lora_weights.safetensors \
    --output ./models/Qwen2.5-Coder-finetuned \
    --lora-alpha 32 \
    --lora-rank 16

# Step 6: Deploy to Ollama
cat > Modelfile << 'EOF'
FROM ./models/Qwen2.5-Coder-finetuned/

TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|endoftext|>"
EOF

ollama create my-finetuned-qwen -f Modelfile --quantize Q4_K_M
ollama run my-finetuned-qwen "Write a hello world in Python"
```

**Key differences from the GLM pipeline:**

| | GLM-4.7-Flash | Qwen2.5-Coder-1.5B |
|-|--------------|---------------------|
| Mode | `finetune-glm` | `finetune-qwen` |
| LoRA targets | `q,kv,o` (MLA projections) | `q,k,v,o` (standard MHA) |
| Batch size | 1 (30B model, memory-limited) | 4+ (1.5B model, fits easily) |
| GGUF export | `export_gguf.py` (DeepSeek2 arch) | `export_gguf_qwen.py` (Qwen2 arch) |
| Modelfile TEMPLATE | `[gMASK]<sop>...` | `<\|im_start\|>...` |
| Stop tokens | `<\|endoftext\|>`, `<\|user\|>` | `<\|im_end\|>`, `<\|im_start\|>` |

## Alternative: Direct GGUF Export

Instead of merging into safetensors and letting Ollama quantize, you can export
directly to GGUF format with built-in quantization.

**Requires:** `torch`, `safetensors`, `numpy`, `gguf` Python packages.

**For GLM-4.7-Flash:**

```bash
python3 retrainer/scripts/export_gguf.py \
    --model-dir ./models/GLM-4.7-Flash \
    --lora lora_weights.safetensors \
    --output model-finetuned.gguf \
    --quant q8_0 \
    --lora-alpha 32 \
    --lora-rank 16
```

**For Qwen2.5-Coder:**

```bash
python3 retrainer/scripts/export_gguf_qwen.py \
    --model-dir ./models/Qwen2.5-Coder-1.5B-Instruct \
    --lora lora_weights.safetensors \
    --output model-finetuned.gguf \
    --quant q8_0 \
    --lora-alpha 32 \
    --lora-rank 16
```

The Qwen exporter also works without LoRA (omit `--lora`) to convert an
unmodified model to GGUF.

Available quantization types: `f32`, `f16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`,
`q8_0`.

Then import the GGUF into Ollama with a matching Modelfile:

```bash
# For GLM
cat > Modelfile << 'EOF'
FROM ./model-finetuned.gguf

TEMPLATE """[gMASK]<sop>{{- range .Messages }}<|{{ .Role }}|>
{{ .Content }}{{- end }}<|assistant|>"""

PARAMETER stop "<|endoftext|>"
EOF

ollama create my-finetuned-model -f Modelfile
```

**When to use direct GGUF export vs. merge + Ollama quantize:**

- **Merge + Ollama** (recommended): Simpler, Ollama handles quantization with
  more format options (Q4_K_M, Q5_K_M, etc.), and you keep a full-precision
  merged checkpoint you can re-quantize later.
- **Direct GGUF**: Useful if you need a standalone GGUF file for llama.cpp or
  other runtimes, or if you want fine control over the quantization process.

## Script Reference

All scripts are in `retrainer/scripts/` and take `--help` for full usage.

| Script | Purpose | Requires |
|--------|---------|----------|
| `download_model.py` | Download model weights and tokenizer from HuggingFace | `huggingface_hub` |
| `convert_hf_model.py` | Merge sharded safetensors into a single file | `safetensors`, `huggingface_hub` |
| `rtnr_tokenize.py` | Tokenize plain text files into token ID sequences | `transformers` |
| `tokenize_chat.py` | Tokenize chat JSONL using model's chat template | `transformers` |
| `merge_lora.py` | Merge LoRA weights into base model (auto-detects arch) | `torch`, `safetensors` |
| `export_gguf.py` | Export GLM/DeepSeek2 + LoRA to quantized GGUF | `torch`, `safetensors`, `numpy`, `gguf`, `transformers` |
| `export_gguf_qwen.py` | Export Qwen/MHA + LoRA to quantized GGUF | `torch`, `safetensors`, `numpy`, `gguf`, `transformers` |
| `inspect_safetensors.py` | List tensors, shapes, and dtypes in a safetensors file | `safetensors` |
| `inspect_training_data.py` | Browse and search JSONL training data | (stdlib only) |

## Retrainer CLI Reference

```
Usage: retrainer -m <mode> [options]

Modes:
  finetune-glm     Fine-tune GLM architecture (MLA + MoE)
  finetune-qwen    Fine-tune Qwen architecture (standard MHA)

Required:
  --model-dir DIR   HuggingFace model directory (config.json + safetensors)
  --train-data FILE Tokenized training data (.tokens or .tok)
  --output FILE     Output LoRA weights file (.safetensors)

Training:
  --epochs N            Number of training epochs (default: 1)
  --batch-size N        Batch size (default: 1)
  --learning-rate F     Learning rate (default: 2e-5)
  --grad-accum N        Gradient accumulation steps (default: 1)
  --log-interval N      Log loss every N steps (default: 10)
  --checkpoint-interval N  Save checkpoint every N steps (default: 0 = off)

LoRA:
  --lora-rank N         LoRA rank (default: 16)
  --lora-alpha F        LoRA alpha scaling (default: 32.0)
  --lora-targets STR    Comma-separated projection targets
                        GLM: q,kv,o   Qwen: q,k,v,o
  --save-lora-only      Only save LoRA adapter weights (not full model)

Memory:
  --storage-dtype TYPE  Weight storage: fp32, fp16, bf16, int8, int4

Resume:
  --resume-lora FILE    Load LoRA weights and continue training
  --resume-step N       Skip first N steps (use with --resume-lora)

Other:
  -v                    Verbose output
```

## Architecture

```
retrainer/
├── include/retrainer/
│   ├── rtnr_constants.h           Training constants and defaults
│   ├── rtnr_exception.h           Exception class and RTNR_THROW macro
│   ├── rtnr_glm_model_builder.h   GLM-4.7-Flash model assembly from CAIF layers
│   ├── rtnr_qwen_model_builder.h  Qwen model assembly from CAIF layers
│   ├── rtnr_token_loader.h        Token file I/O and batching
│   └── rtnr_trainer.h             Training loop, optimizer, checkpointing
├── src/
│   ├── main.cpp                   CLI entry point and argument parsing
│   ├── rtnr_glm_model_builder.cpp GLM config parsing + CAIF layer construction
│   ├── rtnr_qwen_model_builder.cpp Qwen config parsing + CAIF layer construction
│   ├── rtnr_token_loader.cpp      Loads .tokens files into GPU tensors
│   └── rtnr_trainer.cpp           Forward/backward, loss, LoRA save/load
├── scripts/                       Python pipeline utilities (see Script Reference)
├── CMakeLists.txt
└── README.md
```

The model builders translate a HuggingFace `config.json` into a CAIF
`CAIF_DeviceNetwork`:

1. Parse architecture parameters (hidden_size, num_layers, etc.)
2. Create CAIF layers (embedding, attention, FFN, normalization)
3. Load safetensors weights into the layers
4. Attach LoRA adapters to specified projection layers
5. Freeze base weights; only LoRA parameters are trainable

The trainer then runs a standard training loop: forward pass, cross-entropy
loss, backward pass, optimizer step. LoRA weights are saved periodically and
at the end of training.

## Troubleshooting

### Out of memory (OOM)

- Use `--storage-dtype int4` to reduce VRAM usage dramatically
- Set `--batch-size 1` and increase `--grad-accum` instead
- Reduce `--lora-rank` (8 instead of 16)
- Target fewer projections (`--lora-targets q,o` instead of `q,kv,o`)

### Training loss not decreasing

- Check your training data quality and quantity
- Try a higher learning rate (e.g., 5e-5)
- Ensure the tokenizer matches the model (use the same model directory)
- Verify token file format: one integer per line, blank lines between sequences

### Ollama model produces garbage

- Verify the `TEMPLATE` in your Modelfile matches the model's chat format
- Ensure `--lora-alpha` and `--lora-rank` in merge_lora.py match training
- Try without quantization first (`ollama create` without `--quantize`) to
  rule out quantization issues
- Check that all safetensors shards were downloaded (compare file count with
  `model.safetensors.index.json`)

### merge_lora.py reports "unknown LoRA key"

The script auto-detects architecture from `config.json`. If your model is not
recognized, the detection defaults to standard MHA projections (q_proj,
k_proj, v_proj, o_proj). To add a new architecture, update the
`ARCH_PROJECTIONS` and `MODEL_TYPE_MAP` dictionaries at the top of the script.

### export_gguf.py errors

- Ensure the `gguf` Python package is installed: `pip install gguf`
- Use `export_gguf.py` for GLM/DeepSeek2 architecture models
- Use `export_gguf_qwen.py` for Qwen/Llama/Mistral/standard MHA models
- The GGUF weight name mappings are architecture-specific — using the wrong
  export script will produce incorrect weight names

### HuggingFace download issues

- Log in with `huggingface-cli login` for gated models
- Use `--info-only` to verify the model exists before downloading
- Set `HF_HOME` environment variable to change the cache location
- For slow connections, the download resumes automatically if interrupted
