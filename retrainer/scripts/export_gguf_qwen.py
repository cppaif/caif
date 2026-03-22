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
Export a Qwen2.5 model with merged LoRA weights to GGUF format for Ollama/llama.cpp.

Reads base weights from HuggingFace sharded safetensors, merges LoRA adapters
from the retrainer's safetensors output, and writes a quantized GGUF file.

Supports standard MHA transformer models (Qwen2.5, Llama, Mistral, etc.).
For GLM-4.7-Flash (MLA + MoE), use export_gguf.py instead.

Requires:
    pip install torch safetensors numpy gguf transformers

Usage:
    python export_gguf_qwen.py \
        --model-dir /path/to/Qwen2.5-Coder-1.5B-Instruct/ \
        --lora /path/to/lora_weights.safetensors \
        --output model.gguf \
        --quant q8_0
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

import gguf
import torch
from safetensors import safe_open


def build_hf_to_gguf_map(num_layers):
    """Build mapping from HF weight names to GGUF tensor names for standard MHA."""
    m={}

    # Embedding and output
    m["model.embed_tokens.weight"]="token_embd.weight"
    m["model.norm.weight"]="output_norm.weight"
    m["lm_head.weight"]="output.weight"

    for i in range(num_layers):
        pfx=f"model.layers.{i}"
        blk=f"blk.{i}"

        # Standard MHA attention
        m[f"{pfx}.self_attn.q_proj.weight"]=f"{blk}.attn_q.weight"
        m[f"{pfx}.self_attn.k_proj.weight"]=f"{blk}.attn_k.weight"
        m[f"{pfx}.self_attn.v_proj.weight"]=f"{blk}.attn_v.weight"
        m[f"{pfx}.self_attn.o_proj.weight"]=f"{blk}.attn_output.weight"

        # QKV biases (Qwen2.5 uses biases on Q/K/V)
        m[f"{pfx}.self_attn.q_proj.bias"]=f"{blk}.attn_q.bias"
        m[f"{pfx}.self_attn.k_proj.bias"]=f"{blk}.attn_k.bias"
        m[f"{pfx}.self_attn.v_proj.bias"]=f"{blk}.attn_v.bias"

        # Norms
        m[f"{pfx}.input_layernorm.weight"]=f"{blk}.attn_norm.weight"
        m[f"{pfx}.post_attention_layernorm.weight"]=f"{blk}.ffn_norm.weight"

        # SwiGLU FFN
        m[f"{pfx}.mlp.gate_proj.weight"]=f"{blk}.ffn_gate.weight"
        m[f"{pfx}.mlp.up_proj.weight"]=f"{blk}.ffn_up.weight"
        m[f"{pfx}.mlp.down_proj.weight"]=f"{blk}.ffn_down.weight"

    return m


def build_lora_to_hf_map(num_layers):
    """Build mapping from retrainer LoRA names to HF weight names for standard MHA.

    Retrainer saves: layers.{i}.self_attn.{proj}.lora_{a|b}.weight
    HF base names:   model.layers.{i}.self_attn.{proj}.weight
    """
    m={}
    projections=["q_proj", "k_proj", "v_proj", "o_proj"]
    # Retrainer uses 1-based layer indices (CAIF network: 0=embed, 1-N=blocks)
    # HF uses 0-based layer indices
    for i in range(num_layers):
        rtnr_idx=i+1
        for proj in projections:
            lora_a=f"layers.{rtnr_idx}.self_attn.{proj}.lora_a.weight"
            lora_b=f"layers.{rtnr_idx}.self_attn.{proj}.lora_b.weight"
            hf_name=f"model.layers.{i}.self_attn.{proj}.weight"
            m[lora_a]={"hf_name": hf_name, "type": "a"}
            m[lora_b]={"hf_name": hf_name, "type": "b"}
    return m


def load_lora_deltas(lora_path, num_layers, lora_alpha, lora_rank):
    """Load LoRA weights and compute delta: (alpha/rank) * B @ A for each projection."""
    lora_map=build_lora_to_hf_map(num_layers)
    deltas={}

    f=safe_open(lora_path, framework="pt")
    keys=f.keys()

    pairs={}
    for key in keys:
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


def load_safetensors_index(model_dir):
    """Load the safetensors index to find which shard has each tensor."""
    index_path=os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index=json.load(f)
        return index["weight_map"]

    # Single-file model (no index)
    single=os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        f=safe_open(single, framework="pt")
        return {key: "model.safetensors" for key in f.keys()}

    raise FileNotFoundError(f"No safetensors files found in {model_dir}")


def get_quant_type(quant_name):
    """Map quantization name to gguf type."""
    quant_map={
        "f32": gguf.GGMLQuantizationType.F32,
        "f16": gguf.GGMLQuantizationType.F16,
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
        "q4_1": gguf.GGMLQuantizationType.Q4_1,
        "q5_0": gguf.GGMLQuantizationType.Q5_0,
        "q5_1": gguf.GGMLQuantizationType.Q5_1,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
    }
    if quant_name not in quant_map:
        print(f"Unknown quantization: {quant_name}", file=sys.stderr)
        print(f"Available: {', '.join(quant_map.keys())}", file=sys.stderr)
        sys.exit(1)
    return quant_map[quant_name]


def should_quantize(tensor_name, tensor_shape):
    """Determine if a tensor should be quantized (only large weight matrices)."""
    skip_patterns=["_norm", "token_embd", "output_norm", ".bias"]
    for p in skip_patterns:
        if p in tensor_name:
            return False
    if ".weight" not in tensor_name:
        return False
    if len(tensor_shape)<2:
        return False
    if tensor_shape[-1]%32!=0:
        return False
    return True


def convert_tensor(data, gguf_name, quant_type):
    """Convert tensor data to the target quantization format."""
    if not should_quantize(gguf_name, data.shape):
        return data.astype(np.float32), gguf.GGMLQuantizationType.F32

    if quant_type==gguf.GGMLQuantizationType.F32:
        return data.astype(np.float32), gguf.GGMLQuantizationType.F32

    if quant_type==gguf.GGMLQuantizationType.F16:
        return data.astype(np.float16), gguf.GGMLQuantizationType.F16

    f32_data=data.astype(np.float32)
    quantized=gguf.quants.quantize(f32_data, quant_type)
    return quantized, quant_type


def main():
    parser=argparse.ArgumentParser(
        description="Export Qwen/MHA model + LoRA to GGUF for Ollama/llama.cpp"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="HuggingFace model directory (contains config.json + safetensors)"
    )
    parser.add_argument(
        "--lora", default=None,
        help="LoRA weights safetensors file from retrainer (optional)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--quant", default="f16",
        help="Quantization type: f32, f16, q4_0, q5_0, q8_0 (default: f16)"
    )
    parser.add_argument(
        "--lora-alpha", type=float, default=32.0,
        help="LoRA alpha (must match training, default: 32.0)"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank (must match training, default: 16)"
    )
    args=parser.parse_args()

    # Load config
    config_path=os.path.join(args.model_dir, "config.json")
    with open(config_path, "r") as f:
        config=json.load(f)

    num_layers=config["num_hidden_layers"]
    dim=config["hidden_size"]
    num_heads=config["num_attention_heads"]
    num_kv_heads=config.get("num_key_value_heads", num_heads)
    vocab_size=config["vocab_size"]
    rope_theta=config.get("rope_theta", 10000.0)
    rms_norm_eps=config.get("rms_norm_eps", 1e-6)
    ffn_dim=config.get("intermediate_size", dim*4)
    head_dim=dim//num_heads
    tie_word_embeddings=config.get("tie_word_embeddings", False)

    print(f"Model: {num_layers} layers, dim={dim}, heads={num_heads}, kv_heads={num_kv_heads}")
    print(f"FFN dim: {ffn_dim}, vocab: {vocab_size}, head_dim: {head_dim}")

    # Build name mappings
    hf_to_gguf=build_hf_to_gguf_map(num_layers)

    # Load LoRA deltas (optional)
    lora_deltas={}
    if args.lora:
        print(f"Loading LoRA from: {args.lora}")
        lora_deltas=load_lora_deltas(
            args.lora, num_layers, args.lora_alpha, args.lora_rank
        )
        print(f"  {len(lora_deltas)} LoRA deltas computed")

    # Load safetensors index
    weight_map=load_safetensors_index(args.model_dir)

    quant_type=get_quant_type(args.quant)
    print(f"Quantization: {args.quant}")

    # Determine GGUF architecture name
    model_type=config.get("model_type", "qwen2").lower()
    if model_type in ("qwen2", "qwen"):
        gguf_arch="qwen2"
    else:
        gguf_arch="llama"

    # Create GGUF writer
    writer=gguf.GGUFWriter(args.output, gguf_arch)

    # Write metadata
    writer.add_block_count(num_layers)
    writer.add_embedding_length(dim)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_vocab_size(vocab_size)
    writer.add_context_length(config.get("max_position_embeddings", 32768))
    writer.add_rope_freq_base(rope_theta)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_feed_forward_length(ffn_dim)
    writer.add_rope_dimension_count(head_dim)

    # Write tokenizer from the model directory
    tokenizer_path=os.path.join(args.model_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

        writer.add_tokenizer_model("gpt2")

        tokens=[]
        scores=[]
        token_types=[]
        for i in range(vocab_size):
            try:
                token=tokenizer.convert_ids_to_tokens(i)
                if token is None:
                    token=""
            except Exception:
                token=""
            tokens.append(token.encode("utf-8", errors="replace"))
            scores.append(0.0)
            token_types.append(gguf.TokenType.NORMAL)

        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(token_types)

        # Add BPE merges
        tok_json_path=os.path.join(args.model_dir, "tokenizer.json")
        with open(tok_json_path, "r") as tf:
            tok_json=json.load(tf)
        if "model" in tok_json and "merges" in tok_json["model"]:
            raw_merges=tok_json["model"]["merges"]
            merges=[" ".join(m) if isinstance(m, list) else m for m in raw_merges]
            writer.add_token_merges(merges)
            print(f"  Tokenizer: {len(merges)} BPE merges")

        if tokenizer.bos_token_id is not None:
            writer.add_bos_token_id(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                writer.add_eos_token_id(tokenizer.eos_token_id[0])
            else:
                writer.add_eos_token_id(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            writer.add_pad_token_id(tokenizer.pad_token_id)

    # Group HF names by shard for efficient loading
    shard_groups={}
    for hf_name in hf_to_gguf.keys():
        if hf_name not in weight_map:
            continue
        shard=weight_map[hf_name]
        if shard not in shard_groups:
            shard_groups[shard]=[]
        shard_groups[shard].append(hf_name)

    tensor_count=0
    print(f"Processing tensors across {len(shard_groups)} shards...")

    for shard_name, hf_names in sorted(shard_groups.items()):
        shard_path=os.path.join(args.model_dir, shard_name)
        print(f"  Loading shard: {shard_name} ({len(hf_names)} tensors)")

        shard_file=safe_open(shard_path, framework="pt")

        for hf_name in hf_names:
            tensor=shard_file.get_tensor(hf_name)

            # Merge LoRA delta if available
            if hf_name in lora_deltas:
                delta=lora_deltas[hf_name]
                tensor=tensor.float()+delta
                print(f"    Merged LoRA: {hf_name}")

            data=tensor.float().numpy()
            gguf_name=hf_to_gguf[hf_name]
            converted, dtype=convert_tensor(data, gguf_name, quant_type)
            writer.add_tensor(gguf_name, converted, raw_dtype=dtype)
            tensor_count+=1

        del shard_file

    # Handle tied embeddings (lm_head shares weights with embed_tokens)
    if tie_word_embeddings and "lm_head.weight" not in weight_map:
        if "model.embed_tokens.weight" in weight_map:
            shard=weight_map["model.embed_tokens.weight"]
            shard_path=os.path.join(args.model_dir, shard)
            f=safe_open(shard_path, framework="pt")
            tensor=f.get_tensor("model.embed_tokens.weight")
            data=tensor.float().numpy()
            converted, dtype=convert_tensor(data, "output.weight", quant_type)
            writer.add_tensor("output.weight", converted, raw_dtype=dtype)
            tensor_count+=1
            print(f"  Added tied lm_head from embedding")

    print(f"Wrote {tensor_count} tensors")

    # Finalize
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    output_size=os.path.getsize(args.output)
    print(f"Output: {args.output} ({output_size/(1024*1024*1024):.1f} GB)")
    print("Done!")


if __name__=="__main__":
    main()
