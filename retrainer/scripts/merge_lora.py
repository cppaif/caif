#!/usr/bin/env python3
# Copyright 2026 Eric Malloy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Merge LoRA weights into base HuggingFace model, saving as safetensors.

The merged model directory can be loaded directly by Ollama:
    ollama create mymodel -f Modelfile
    # where Modelfile contains: FROM /path/to/merged/

Supports:
  - GLM-4.7-Flash (MLA projections: q_a_proj, q_b_proj, kv_a_proj_with_mqa,
    kv_b_proj, o_proj)
  - Qwen2.5 / standard MHA models (q_proj, k_proj, v_proj, o_proj)

Architecture is auto-detected from config.json model_type. To add support
for a new architecture, add its projection list to ARCH_PROJECTIONS below.

Requires:
    pip install torch safetensors

    Ollama (https://ollama.com/) for deployment after merging.

Usage:
    python merge_lora.py \
        --model-dir /path/to/model-directory/ \
        --lora /path/to/lora_weights.safetensors \
        --output /path/to/merged/
"""

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Projection names per architecture.
# The retrainer saves LoRA keys as: layers.{N}.self_attn.{proj}.lora_{a|b}.weight
# HF base weight names are:         model.layers.{N}.self_attn.{proj}.weight
ARCH_PROJECTIONS={
    "glm": [
        "q_a_proj", "q_b_proj",
        "kv_a_proj_with_mqa", "kv_b_proj",
        "o_proj",
    ],
    "qwen2": [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
    # Standard MHA (Llama, Mistral, etc.) — same as Qwen
    "llama": [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
}

# Map from HuggingFace config.json model_type to our architecture key
MODEL_TYPE_MAP={
    "glm":      "glm",
    "chatglm":  "glm",
    "qwen2":    "qwen2",
    "qwen":     "qwen2",
    "llama":    "llama",
    "mistral":  "llama",
}


def detect_architecture(config):
    """Detect architecture from HuggingFace config.json."""
    model_type=config.get("model_type", "").lower()

    # Try direct match
    if model_type in MODEL_TYPE_MAP:
        return MODEL_TYPE_MAP[model_type]

    # Check for MLA indicators (GLM/DeepSeek)
    if "q_lora_rank" in config or "kv_lora_rank" in config:
        return "glm"

    # Check for num_key_value_heads (GQA/MHA — Qwen, Llama)
    if "num_key_value_heads" in config:
        return "qwen2"

    print(f"Warning: unknown model_type '{model_type}', defaulting to standard MHA",
          file=sys.stderr)
    return "qwen2"


def build_lora_to_hf_map(num_layers, projections):
    """Build mapping from retrainer LoRA names to HF weight names."""
    m={}
    for i in range(num_layers):
        rtnr_idx=i+1
        for proj in projections:
            lora_a=f"layers.{rtnr_idx}.self_attn.{proj}.lora_a.weight"
            lora_b=f"layers.{rtnr_idx}.self_attn.{proj}.lora_b.weight"
            hf_name=f"model.layers.{i}.self_attn.{proj}.weight"
            m[lora_a]={"hf_name": hf_name, "type": "a"}
            m[lora_b]={"hf_name": hf_name, "type": "b"}
    return m


def load_lora_deltas(lora_path, num_layers, lora_alpha, lora_rank, projections):
    """Load LoRA weights and compute delta: (alpha/rank) * B @ A for each projection."""
    lora_map=build_lora_to_hf_map(num_layers, projections)
    deltas={}

    f=safe_open(lora_path, framework="pt")

    pairs={}
    for key in f.keys():
        if key not in lora_map:
            print(f"  Warning: unknown LoRA key: {key}", file=sys.stderr)
            continue
        info=lora_map[key]
        hf_name=info["hf_name"]
        if hf_name not in pairs:
            pairs[hf_name]={}
        pairs[hf_name][info["type"]]=f.get_tensor(key)

    scale=lora_alpha/lora_rank

    for hf_name, ab in pairs.items():
        if "a" not in ab or "b" not in ab:
            print(f"  Warning: incomplete LoRA pair for {hf_name}", file=sys.stderr)
            continue
        a=ab["a"].float()
        b=ab["b"].float()
        delta=scale*(b @ a)
        deltas[hf_name]=delta

    return deltas


def main():
    parser=argparse.ArgumentParser(description="Merge LoRA into HF model for Ollama")
    parser.add_argument("--model-dir", required=True, help="HuggingFace model directory")
    parser.add_argument("--lora", required=True, help="LoRA safetensors from retrainer")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    args=parser.parse_args()

    # Load config
    config_path=os.path.join(args.model_dir, "config.json")
    with open(config_path) as f:
        config=json.load(f)
    num_layers=config["num_hidden_layers"]

    # Detect architecture and get projection names
    arch=detect_architecture(config)
    projections=ARCH_PROJECTIONS[arch]
    print(f"Architecture: {arch} ({len(projections)} projections per layer)")

    # Load LoRA deltas
    print(f"Loading LoRA from: {args.lora}")
    deltas=load_lora_deltas(args.lora, num_layers, args.lora_alpha, args.lora_rank,
                            projections)
    print(f"  {len(deltas)} LoRA deltas computed")

    # Load shard index
    index_path=os.path.join(args.model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index=json.load(f)
    weight_map=index["weight_map"]

    # Create output dir
    os.makedirs(args.output, exist_ok=True)

    # Copy non-weight files (config, tokenizer, etc.)
    for fname in os.listdir(args.model_dir):
        if fname.endswith(".safetensors"):
            continue
        if fname=="model.safetensors.index.json":
            continue
        src=os.path.join(args.model_dir, fname)
        dst=os.path.join(args.output, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

    # Group tensors by shard
    shard_tensors={}
    for tensor_name, shard_name in weight_map.items():
        if shard_name not in shard_tensors:
            shard_tensors[shard_name]=[]
        shard_tensors[shard_name].append(tensor_name)

    # Process each shard: merge LoRA deltas, save
    merge_count=0
    for shard_name in sorted(shard_tensors.keys()):
        tensor_names=shard_tensors[shard_name]
        shard_path=os.path.join(args.model_dir, shard_name)
        print(f"  Processing {shard_name} ({len(tensor_names)} tensors)")

        shard_file=safe_open(shard_path, framework="pt")
        tensors={}

        for name in tensor_names:
            tensor=shard_file.get_tensor(name)
            if name in deltas:
                tensor=tensor.float()+deltas[name]
                tensor=tensor.to(torch.bfloat16)
                merge_count+=1
                print(f"    Merged LoRA: {name}")
            tensors[name]=tensor

        out_path=os.path.join(args.output, shard_name)
        save_file(tensors, out_path)
        del tensors, shard_file

    # Copy the index file (shard names are the same)
    shutil.copy2(index_path, os.path.join(args.output, "model.safetensors.index.json"))

    print(f"\nMerged {merge_count} tensors into {args.output}")
    print("Done!")


if __name__=="__main__":
    main()
