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
Export a model with merged LoRA weights to GGUF format for Ollama/llama.cpp.

Reads base weights from HuggingFace sharded safetensors, merges LoRA adapters
from the retrainer's safetensors output, and writes a quantized GGUF file.

Currently supports GLM-4.7-Flash (DeepSeek2 architecture with MLA + MoE).
For Qwen/Llama/standard MHA models, use export_gguf_qwen.py instead.

Requires:
    pip install torch safetensors numpy gguf transformers

Usage:
    python export_gguf.py \
        --model-dir /path/to/model-directory/ \
        --lora /path/to/lora_weights.safetensors \
        --output model.gguf \
        --quant q8_0
"""

import argparse
import json
import os
import sys
import struct
import numpy as np
from pathlib import Path

import gguf
import torch
from safetensors import safe_open


# HuggingFace name -> GGUF name mapping for GLM-4.7-Flash (MLA + MoE)
# Expert weights are NOT mapped here; they are stacked into combined tensors separately.
def build_hf_to_gguf_map(num_layers):
    """Build mapping from HF weight names to GGUF tensor names (non-expert tensors only)."""
    m={}

    # Embedding and output
    m["model.embed_tokens.weight"]="token_embd.weight"
    m["model.norm.weight"]="output_norm.weight"
    m["lm_head.weight"]="output.weight"

    for i in range(num_layers):
        pfx=f"model.layers.{i}"
        blk=f"blk.{i}"

        # MLA attention
        m[f"{pfx}.self_attn.q_a_proj.weight"]=f"{blk}.attn_q_a.weight"
        m[f"{pfx}.self_attn.q_a_layernorm.weight"]=f"{blk}.attn_q_a_norm.weight"
        m[f"{pfx}.self_attn.q_b_proj.weight"]=f"{blk}.attn_q_b.weight"
        m[f"{pfx}.self_attn.kv_a_proj_with_mqa.weight"]=f"{blk}.attn_kv_a_mqa.weight"
        m[f"{pfx}.self_attn.kv_a_layernorm.weight"]=f"{blk}.attn_kv_a_norm.weight"
        m[f"{pfx}.self_attn.kv_b_proj.weight"]=f"{blk}.attn_kv_b.weight"
        m[f"{pfx}.self_attn.o_proj.weight"]=f"{blk}.attn_output.weight"

        # Norms
        m[f"{pfx}.input_layernorm.weight"]=f"{blk}.attn_norm.weight"
        m[f"{pfx}.post_attention_layernorm.weight"]=f"{blk}.ffn_norm.weight"

        # Dense FFN (layer 0)
        m[f"{pfx}.mlp.gate_proj.weight"]=f"{blk}.ffn_gate.weight"
        m[f"{pfx}.mlp.up_proj.weight"]=f"{blk}.ffn_up.weight"
        m[f"{pfx}.mlp.down_proj.weight"]=f"{blk}.ffn_down.weight"

        # MoE router
        m[f"{pfx}.mlp.gate.weight"]=f"{blk}.ffn_gate_inp.weight"

        # Shared expert
        m[f"{pfx}.mlp.shared_experts.gate_proj.weight"]=f"{blk}.ffn_gate_shexp.weight"
        m[f"{pfx}.mlp.shared_experts.up_proj.weight"]=f"{blk}.ffn_up_shexp.weight"
        m[f"{pfx}.mlp.shared_experts.down_proj.weight"]=f"{blk}.ffn_down_shexp.weight"

    return m


def build_expert_groups(num_layers, num_experts):
    """Build mapping of expert HF names grouped by layer and type for stacking.

    Returns dict: (layer_idx, expert_type) -> list of (expert_idx, hf_name)
    where expert_type is 'gate', 'up', or 'down'.
    """
    groups={}
    for i in range(num_layers):
        pfx=f"model.layers.{i}"
        for etype, proj in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
            key=(i, etype)
            groups[key]=[]
            for j in range(num_experts):
                hf_name=f"{pfx}.mlp.experts.{j}.{proj}.weight"
                groups[key].append((j, hf_name))
    return groups


def build_lora_to_hf_map(num_layers):
    """Build mapping from retrainer LoRA names to HF weight names.

    Retrainer saves: layers.{i}.self_attn.{proj}.lora_{a|b}.weight
    HF base names:   model.layers.{i}.self_attn.{proj}.weight
    """
    m={}
    projections=[
        "q_a_proj", "q_b_proj",
        "kv_a_proj_with_mqa", "kv_b_proj",
        "o_proj",
    ]
    # Retrainer uses 1-based layer indices (CAIF network: 0=embed, 1-N=blocks)
    # HF uses 0-based layer indices (model.layers.0 through model.layers.46)
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

    # Group by HF name
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
        # LoRA forward: output += (input @ A^T @ B^T) * scale
        # Weight delta: W_new = W_old + scale * (B @ A)
        delta=scale*(b @ a)
        deltas[hf_name]=delta

    return deltas


def load_safetensors_index(model_dir):
    """Load the safetensors index to find which shard has each tensor."""
    index_path=os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index=json.load(f)
    return index["weight_map"]


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
    # Don't quantize norms, biases, embeddings, or small tensors
    skip_patterns=["_norm", "token_embd", "output_norm"]
    for p in skip_patterns:
        if p in tensor_name:
            return False
    # Only quantize weight matrices (2D or 3D stacked experts)
    if ".weight" not in tensor_name:
        return False
    if len(tensor_shape)<2:
        return False
    # Last dimension must be divisible by 32 for block quantization
    if tensor_shape[-1]%32!=0:
        return False
    return True


def convert_tensor(data, gguf_name, quant_type):
    """Convert tensor data to the target quantization format.

    Returns (converted_data, actual_dtype) where actual_dtype is the GGMLQuantizationType.
    """
    if not should_quantize(gguf_name, data.shape):
        # Norms, embeddings: keep as f32
        return data.astype(np.float32), gguf.GGMLQuantizationType.F32

    if quant_type==gguf.GGMLQuantizationType.F32:
        return data.astype(np.float32), gguf.GGMLQuantizationType.F32

    if quant_type==gguf.GGMLQuantizationType.F16:
        return data.astype(np.float16), gguf.GGMLQuantizationType.F16

    # Block quantization (Q4_0, Q8_0, etc.)
    f32_data=data.astype(np.float32)
    quantized=gguf.quants.quantize(f32_data, quant_type)
    return quantized, quant_type


def main():
    parser=argparse.ArgumentParser(
        description="Export model + LoRA to GGUF for Ollama/llama.cpp"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="HuggingFace model directory (contains config.json + shards)"
    )
    parser.add_argument(
        "--lora", required=True,
        help="LoRA weights safetensors file from retrainer"
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
    vocab_size=config["vocab_size"]
    rope_theta=config.get("rope_theta", 1000000.0)
    rms_norm_eps=config.get("rms_norm_eps", 1e-5)
    q_lora_rank=config.get("q_lora_rank", 768)
    kv_lora_rank=config.get("kv_lora_rank", 512)
    qk_rope_head_dim=config.get("qk_rope_head_dim", 64)
    qk_nope_head_dim=config.get("qk_nope_head_dim", 192)
    v_head_dim=config.get("v_head_dim", 256)
    ffn_dim=config.get("intermediate_size", dim*4)
    num_experts=config.get("n_routed_experts", 64)
    num_experts_per_tok=config.get("num_experts_per_tok", 4)
    n_shared_experts=config.get("n_shared_experts", 0)
    moe_hidden_dim=config.get("moe_intermediate_size", 1536)
    first_k_dense=config.get("first_k_dense_replace", 1)

    print(f"Model: {num_layers} layers, dim={dim}, heads={num_heads}, vocab={vocab_size}")
    print(f"MLA: q_lora_rank={q_lora_rank}, kv_lora_rank={kv_lora_rank}")
    print(f"MoE: {num_experts} experts, top_k={num_experts_per_tok}, shared={n_shared_experts}")

    # Build name mappings
    hf_to_gguf=build_hf_to_gguf_map(num_layers)
    expert_groups=build_expert_groups(num_layers, num_experts)

    # Load LoRA deltas
    print(f"Loading LoRA from: {args.lora}")
    lora_deltas=load_lora_deltas(
        args.lora, num_layers, args.lora_alpha, args.lora_rank
    )
    print(f"  {len(lora_deltas)} LoRA deltas computed")

    # Load safetensors index
    weight_map=load_safetensors_index(args.model_dir)

    quant_type=get_quant_type(args.quant)
    print(f"Quantization: {args.quant}")

    # Create GGUF writer
    writer=gguf.GGUFWriter(args.output, "deepseek2")

    # Write metadata
    writer.add_block_count(num_layers)
    writer.add_embedding_length(dim)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_heads)
    writer.add_vocab_size(vocab_size)
    writer.add_context_length(config.get("max_position_embeddings", 131072))
    writer.add_rope_freq_base(rope_theta)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_feed_forward_length(ffn_dim)
    writer.add_expert_count(num_experts)
    writer.add_expert_used_count(num_experts_per_tok)
    writer.add_expert_shared_count(n_shared_experts)
    writer.add_expert_feed_forward_length(moe_hidden_dim)
    writer.add_leading_dense_block_count(first_k_dense)
    writer.add_key_length(qk_nope_head_dim+qk_rope_head_dim)
    writer.add_value_length(v_head_dim)
    writer.add_q_lora_rank(q_lora_rank)
    writer.add_kv_lora_rank(kv_lora_rank)
    writer.add_rope_dimension_count(qk_rope_head_dim)
    writer.add_expert_weights_scale(config.get("routed_scaling_factor", 1.0))
    writer.add_expert_weights_norm(config.get("norm_topk_prob", False))

    # Write tokenizer from the model directory
    tokenizer_path=os.path.join(args.model_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

        # Add tokenizer model type
        writer.add_tokenizer_model("gpt2")

        # Add token list
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

        # Add BPE merges (required for gpt2 tokenizer type)
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

    # Collect all HF tensor names we need (non-expert + expert)
    all_hf_names=set(hf_to_gguf.keys())
    for key, pairs in expert_groups.items():
        for j, hf_name in pairs:
            all_hf_names.add(hf_name)

    # Group ALL HF names by shard for efficient loading
    shard_groups={}
    for hf_name in all_hf_names:
        if hf_name not in weight_map:
            continue
        shard=weight_map[hf_name]
        if shard not in shard_groups:
            shard_groups[shard]=[]
        shard_groups[shard].append(hf_name)

    # Build reverse index: hf_name -> (layer_idx, etype, expert_idx)
    expert_reverse={}
    for (layer_idx, etype), pairs in expert_groups.items():
        for j, hf_name in pairs:
            expert_reverse[hf_name]=(layer_idx, etype, j)

    etype_to_gguf={"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}

    tensor_count=0
    print(f"Processing tensors across {len(shard_groups)} shards...")

    for shard_name, hf_names in sorted(shard_groups.items()):
        shard_path=os.path.join(args.model_dir, shard_name)
        print(f"  Loading shard: {shard_name} ({len(hf_names)} tensors)")

        shard_file=safe_open(shard_path, framework="pt")

        # Per-shard expert accumulator: (layer_idx, etype) -> {expert_idx: data}
        shard_experts={}

        for hf_name in hf_names:
            tensor=shard_file.get_tensor(hf_name)

            # Merge LoRA delta if available
            if hf_name in lora_deltas:
                delta=lora_deltas[hf_name]
                tensor=tensor.float()+delta
                print(f"    Merged LoRA: {hf_name}")

            data=tensor.float().numpy()

            if hf_name in hf_to_gguf:
                # Non-expert tensor: write directly
                gguf_name=hf_to_gguf[hf_name]
                converted, dtype=convert_tensor(data, gguf_name, quant_type)
                writer.add_tensor(gguf_name, converted, raw_dtype=dtype)
                tensor_count+=1
            elif hf_name in expert_reverse:
                # Expert tensor: collect per-shard for stacking
                layer_idx, etype, j=expert_reverse[hf_name]
                key=(layer_idx, etype)
                if key not in shard_experts:
                    shard_experts[key]={}
                shard_experts[key][j]=data

        del shard_file

        # Stack and write any complete expert groups from this shard
        for (layer_idx, etype), parts in shard_experts.items():
            if len(parts)==num_experts:
                gguf_name=f"blk.{layer_idx}.{etype_to_gguf[etype]}.weight"
                stacked=np.stack([parts[j] for j in range(num_experts)], axis=0)
                converted, dtype=convert_tensor(stacked, gguf_name, quant_type)
                writer.add_tensor(gguf_name, converted, raw_dtype=dtype)
                tensor_count+=1
                del stacked, converted
            else:
                print(f"    Warning: incomplete expert group blk.{layer_idx}.{etype} "
                      f"({len(parts)}/{num_experts})", file=sys.stderr)

        del shard_experts

    # Handle tensors not in any shard (like lm_head if tied)
    if "lm_head.weight" not in weight_map:
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
