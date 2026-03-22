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
Download HuggingFace model safetensors to a local directory.

Requires:
    pip install huggingface_hub

    For gated models, log in first:
        huggingface-cli login

Usage:
    python download_model.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --output-dir ./models/qwen
    python download_model.py --model zai-org/GLM-4.7-Flash --output-dir ./models/glm4 --info-only

This downloads safetensors files with their original names (not as blobs).
"""

import argparse
import os
import sys
import json
from pathlib import Path


def list_model_files(model_id):
    """List all files in a HuggingFace model repo."""
    from huggingface_hub import list_repo_files

    return list(list_repo_files(model_id))


def download_model(model_id, output_dir, patterns=None):
    """
    Download model files from HuggingFace Hub to output_dir with original names.

    Args:
        model_id: HuggingFace model ID (e.g., 'zai-org/GLM-4.7-Flash')
        output_dir: Local directory to save files
        patterns: List of file patterns to download (default: safetensors + config)
    """
    from huggingface_hub import hf_hub_download

    if patterns is None:
        patterns = ['*.safetensors', 'config.json', 'tokenizer*', '*.json']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all files in repo
    all_files = list_model_files(model_id)

    # Filter by patterns
    import fnmatch
    files_to_download = []
    for f in all_files:
        for pattern in patterns:
            if fnmatch.fnmatch(f, pattern):
                files_to_download.append(f)
                break

    print(f"Downloading {len(files_to_download)} files from {model_id}...")

    downloaded = []
    for filename in sorted(files_to_download):
        print(f"  {filename}...", end=" ", flush=True)

        # Download to HF cache, then copy/link to output_dir
        cached_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False  # Copy files, don't symlink
        )

        # hf_hub_download with local_dir puts file at output_dir/filename
        output_path = output_dir / filename
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"{size_mb:.1f} MB")
            downloaded.append(str(output_path))
        else:
            print("done")
            downloaded.append(cached_path)

    return downloaded


def print_model_info(model_id):
    """Print model architecture and file list."""
    from huggingface_hub import hf_hub_download

    print(f"Model: {model_id}")
    print()

    # List all files
    files = list_model_files(model_id)
    safetensor_files = [f for f in files if f.endswith('.safetensors')]

    print(f"Files ({len(files)} total, {len(safetensor_files)} safetensors):")
    for f in sorted(files):
        print(f"  {f}")
    print()

    # Try to get config
    try:
        config_path = hf_hub_download(model_id, 'config.json')
        with open(config_path) as f:
            config = json.load(f)

        print("Architecture:")
        keys = ['hidden_size', 'num_attention_heads', 'num_key_value_heads',
                'num_hidden_layers', 'intermediate_size', 'vocab_size',
                'max_position_embeddings', 'model_type', 'torch_dtype']
        for key in keys:
            if key in config:
                print(f"  {key}: {config[key]}")

        # Estimate size
        if 'hidden_size' in config and 'num_hidden_layers' in config:
            d = config['hidden_size']
            n = config['num_hidden_layers']
            v = config.get('vocab_size', 100000)
            # Rough estimate: embedding + layers
            params = v * d + n * (4 * d * d + 3 * d * config.get('intermediate_size', 4*d))
            print(f"  estimated_params: ~{params/1e9:.1f}B")

    except Exception as e:
        print(f"Could not load config: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model safetensors files"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model ID (e.g., zai-org/GLM-4.7-Flash)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: ./models/<model-name>)"
    )
    parser.add_argument(
        "--info-only", "-i",
        action="store_true",
        help="Only print model info, don't download"
    )
    parser.add_argument(
        "--safetensors-only",
        action="store_true",
        help="Only download .safetensors files (skip tokenizer, config)"
    )

    args = parser.parse_args()

    # Print model info
    print_model_info(args.model)

    if args.info_only:
        return

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: ./models/<model-name>
        model_name = args.model.split('/')[-1]
        output_dir = f"./models/{model_name}"

    print()
    print(f"Output directory: {output_dir}")
    print()

    # Set patterns
    if args.safetensors_only:
        patterns = ['*.safetensors', 'model.safetensors.index.json']
    else:
        patterns = ['*.safetensors', '*.json', 'tokenizer.model']

    # Download
    downloaded = download_model(args.model, output_dir, patterns)

    print()
    print(f"Downloaded {len(downloaded)} files to {output_dir}")


if __name__ == "__main__":
    main()
