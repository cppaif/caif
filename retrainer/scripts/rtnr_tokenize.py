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
Tokenize text files into simple format: one token ID per line.
Empty lines separate sequences.

Requires:
    pip install transformers

Usage:
    python tokenize.py --input <text_file> --output <tokens_file> --tokenizer <name_or_path>

Examples:
    # Using a HuggingFace tokenizer
    python tokenize.py -i train.txt -o train.tokens --tokenizer Qwen/Qwen2.5-Coder-1.5B-Instruct

    # Using a local tokenizer.json
    python tokenize.py -i train.txt -o train.tokens --tokenizer ./tokenizer.json

    # Process a directory of files
    python tokenize.py -i ./data/ -o train.tokens --tokenizer gpt2 --glob "*.txt"
"""

import argparse
import os
import sys
from pathlib import Path


def load_tokenizer(tokenizer_path):
    """Load tokenizer from HuggingFace or local path."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=False
        )
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)


def tokenize_text(text, tokenizer):
    """Tokenize text and return list of token IDs."""
    # Use encode without special tokens for raw tokenization
    # Add special tokens based on the model's expected format
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens


def process_file(input_path, tokenizer):
    """Process a single text file and yield sequences of tokens."""
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Split by double newlines (paragraphs) for natural sequence boundaries
    # Or treat entire file as one sequence
    paragraphs = content.split('\n\n')

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        tokens = tokenize_text(para, tokenizer)
        if tokens:
            yield tokens


def process_directory(input_dir, tokenizer, glob_pattern="*.txt"):
    """Process all matching files in a directory."""
    input_path = Path(input_dir)
    for file_path in sorted(input_path.glob(glob_pattern)):
        if file_path.is_file():
            print(f"Processing: {file_path}", file=sys.stderr)
            for tokens in process_file(str(file_path), tokenizer):
                yield tokens


def write_tokens(output_path, token_sequences, binary=False):
    """Write token sequences to output file."""
    if binary:
        write_tokens_binary(output_path, token_sequences)
    else:
        write_tokens_text(output_path, token_sequences)


def write_tokens_text(output_path, token_sequences):
    """Write tokens as text: one token per line, empty line between sequences."""
    with open(output_path, 'w', encoding='utf-8') as f:
        first_seq = True
        for tokens in token_sequences:
            if not first_seq:
                f.write('\n')  # Empty line between sequences
            first_seq = False

            for token_id in tokens:
                f.write(f"{token_id}\n")


def write_tokens_binary(output_path, token_sequences):
    """Write tokens in binary format for faster loading."""
    import struct

    sequences = list(token_sequences)
    num_sequences = len(sequences)
    max_len = max(len(seq) for seq in sequences) if sequences else 0

    # Magic: "RTNT" = 0x52544E54
    magic = 0x52544E54
    version = 1

    with open(output_path, 'wb') as f:
        # Header: magic, version, num_sequences, max_len
        f.write(struct.pack('<IIII', magic, version, num_sequences, max_len))

        # Build sequence table
        current_offset = 0
        for seq in sequences:
            length = len(seq)
            f.write(struct.pack('<II', current_offset, length))
            current_offset += length * 4  # 4 bytes per uint32

        # Write token data
        for seq in sequences:
            for token_id in seq:
                f.write(struct.pack('<I', token_id))

    print(f"Wrote binary: {num_sequences} sequences, max_len={max_len}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text files for LLM training"
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help="Input text file or directory"
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help="Output tokens file (.tokens for text, .tok for binary)"
    )
    parser.add_argument(
        '--tokenizer',
        required=True,
        help="HuggingFace tokenizer name or local path"
    )
    parser.add_argument(
        '--glob',
        default="*.txt",
        help="Glob pattern for directory input (default: *.txt)"
    )
    parser.add_argument(
        '--binary',
        action='store_true',
        help="Output binary format (.tok) instead of text"
    )
    parser.add_argument(
        '--add-bos',
        action='store_true',
        help="Add BOS token at start of each sequence"
    )
    parser.add_argument(
        '--add-eos',
        action='store_true',
        help="Add EOS token at end of each sequence"
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=0,
        help="Maximum sequence length (0 = no limit)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Print stats without writing output"
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}", file=sys.stderr)

    # Get special token IDs
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    # Process input
    input_path = Path(args.input)

    def generate_sequences():
        if input_path.is_dir():
            raw_seqs = process_directory(str(input_path), tokenizer, args.glob)
        else:
            raw_seqs = process_file(str(input_path), tokenizer)

        for tokens in raw_seqs:
            # Add special tokens if requested
            if args.add_bos and bos_id is not None:
                tokens = [bos_id] + tokens
            if args.add_eos and eos_id is not None:
                tokens = tokens + [eos_id]

            # Truncate if needed
            if args.max_length > 0 and len(tokens) > args.max_length:
                tokens = tokens[:args.max_length]

            yield tokens

    if args.dry_run:
        # Just count
        total_seqs = 0
        total_tokens = 0
        max_len = 0

        for tokens in generate_sequences():
            total_seqs += 1
            total_tokens += len(tokens)
            if len(tokens) > max_len:
                max_len = len(tokens)

        print(f"Sequences: {total_seqs}")
        print(f"Total tokens: {total_tokens}")
        print(f"Max length: {max_len}")
        print(f"Avg length: {total_tokens / total_seqs if total_seqs > 0 else 0:.1f}")
    else:
        # Determine output format
        binary = args.binary or args.output.endswith('.tok')

        # Collect sequences (need to iterate twice for binary format)
        sequences = list(generate_sequences())

        print(f"Writing {len(sequences)} sequences to {args.output}", file=sys.stderr)
        write_tokens(args.output, sequences, binary=binary)

        total_tokens = sum(len(seq) for seq in sequences)
        print(f"Done: {len(sequences)} sequences, {total_tokens} tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
