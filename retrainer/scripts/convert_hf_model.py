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
Download and convert HuggingFace models to single SafeTensors file for CAIF.

Usage:
    python convert_hf_model.py --model THUDM/glm-4-9b-chat --output model.safetensors

This script:
1. Downloads model from HuggingFace Hub (or uses cached)
2. Loads all weight shards
3. Merges into single SafeTensors file
4. Optionally prints model architecture info

Requires:
    pip install safetensors huggingface_hub torch
"""

import argparse
import os
import sys
import json
from pathlib import Path


def download_model(model_id, cache_dir=None):
    """Download model files from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id}...")
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    print(f"Model cached at: {path}")
    return path


def load_sharded_safetensors(model_path):
    """Load all safetensors shards and merge into single dict."""
    from safetensors import safe_open

    model_path = Path(model_path)

    # Check for index file
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = sorted(set(index["weight_map"].values()))
        print(f"Found {len(shard_files)} shards")
    else:
        # Find all safetensors files
        shard_files = sorted([f.name for f in model_path.glob("*.safetensors")])
        if not shard_files:
            raise FileNotFoundError("No safetensors files found")
        print(f"Found {len(shard_files)} safetensors files")

    # Load all tensors
    all_tensors = {}
    total_params = 0

    for shard_name in shard_files:
        shard_path = model_path / shard_name
        print(f"  Loading {shard_name}...")

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                all_tensors[key] = tensor
                total_params += tensor.numel()

    print(f"Loaded {len(all_tensors)} tensors, {total_params:,} parameters")
    return all_tensors


def save_safetensors(tensors, output_path, metadata=None):
    """Save tensors to single SafeTensors file."""
    from safetensors.torch import save_file

    if metadata is None:
        metadata = {}

    print(f"Saving to {output_path}...")
    save_file(tensors, output_path, metadata=metadata)

    # Print file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {size_mb:.1f} MB")


def print_model_info(model_path):
    """Print model architecture from config.json."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        print("\nModel architecture:")
        for key in ['hidden_size', 'num_attention_heads', 'num_key_value_heads',
                    'num_hidden_layers', 'intermediate_size', 'vocab_size',
                    'max_position_embeddings', 'model_type']:
            if key in config:
                print(f"  {key}: {config[key]}")
        return config
    return None


def print_weight_names(tensors, limit=50):
    """Print weight tensor names and shapes."""
    print(f"\nWeight tensors ({len(tensors)} total):")
    for i, (name, tensor) in enumerate(sorted(tensors.items())):
        if i >= limit:
            print(f"  ... and {len(tensors) - limit} more")
            break
        print(f"  {name}: {list(tensor.shape)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to single SafeTensors file"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., THUDM/glm-4-9b-chat)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output SafeTensors file path"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only print model info, don't convert"
    )
    parser.add_argument(
        "--show-weights",
        action="store_true",
        help="Print weight tensor names"
    )
    parser.add_argument(
        "--local-path",
        default=None,
        help="Use local model path instead of downloading"
    )

    args = parser.parse_args()

    # Get model path
    if args.local_path:
        model_path = args.local_path
    else:
        model_path = download_model(args.model, args.cache_dir)

    # Print model info
    config = print_model_info(model_path)

    if args.info_only:
        return

    # Load tensors
    tensors = load_sharded_safetensors(model_path)

    if args.show_weights:
        print_weight_names(tensors)

    # Build metadata
    metadata = {"source_model": args.model}
    if config:
        for key in ['hidden_size', 'num_attention_heads', 'num_hidden_layers', 'vocab_size']:
            if key in config:
                metadata[key] = str(config[key])

    # Save merged file
    save_safetensors(tensors, args.output, metadata)

    print("\nDone!")
    print(f"Use with retrainer:")
    print(f"  ./retrainer -m finetune --base-model {args.output} ...")


if __name__ == "__main__":
    main()
