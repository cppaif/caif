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
Tokenize JSONL chat messages into token sequences using the model's chat template.

Works with any HuggingFace model that supports apply_chat_template() —
the tokenizer handles the model-specific formatting automatically.

Requires:
    pip install transformers

Input:  JSONL with {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
Output: Token file (.tokens) with one token ID per line, empty lines between sequences.

Usage:
    python tokenize_chat.py \
        -i train.jsonl \
        -o train.tokens \
        --tokenizer ./models/GLM-4.7-Flash

    python tokenize_chat.py \
        -i train.jsonl \
        -o train.tokens \
        --tokenizer ./models/Qwen2.5-Coder-1.5B-Instruct

    python tokenize_chat.py --dry-run -i train.jsonl \
        --tokenizer ./models/GLM-4.7-Flash
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path


def load_tokenizer(tokenizer_path):
    """Load tokenizer from HuggingFace model directory or name."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)


def tokenize_chat(messages, tokenizer):
    """Tokenize a chat message list using the model's chat template.

    Returns list of token IDs.
    """
    # Get the templated text first, then encode — some tokenizers
    # return Encoding objects instead of flat int lists from apply_chat_template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    token_ids = tokenizer.encode(text)
    return token_ids


def read_jsonl(input_path):
    """Read JSONL file, yield parsed message dicts."""
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "messages" in data:
                    yield data["messages"]
                else:
                    print(f"Warning: line {line_num} missing 'messages' key",
                          file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"Warning: line {line_num} JSON error: {e}",
                      file=sys.stderr)


def write_tokens_text(output_path, token_sequences):
    """Write token sequences as text: one token per line, empty line between sequences."""
    with open(output_path, 'w', encoding='utf-8') as f:
        first = True
        for tokens in token_sequences:
            if not first:
                f.write('\n')
            first = False
            for tid in tokens:
                f.write(f"{tid}\n")


def write_tokens_binary(output_path, sequences):
    """Write token sequences in binary format for faster loading."""
    num_sequences = len(sequences)
    max_len = max(len(s) for s in sequences) if sequences else 0

    magic = 0x52544E54  # "RTNT"
    version = 1

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<IIII', magic, version, num_sequences, max_len))

        current_offset = 0
        for seq in sequences:
            length = len(seq)
            f.write(struct.pack('<II', current_offset, length))
            current_offset += length * 4

        for seq in sequences:
            for tid in seq:
                f.write(struct.pack('<I', tid))

    print(f"Wrote binary: {num_sequences} sequences, max_len={max_len}",
          file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize JSONL chat messages for LLM training"
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help="Input JSONL file with chat messages"
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help="Output token file (.tokens or .tok)"
    )
    parser.add_argument(
        '--tokenizer',
        required=True,
        help="HuggingFace tokenizer path or model directory"
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)"
    )
    parser.add_argument(
        '--binary',
        action='store_true',
        help="Output binary format (.tok)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Print statistics without writing output"
    )

    args = parser.parse_args()

    if not args.output and not args.dry_run:
        parser.error("Either --output or --dry-run is required")

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr)
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}", file=sys.stderr)

    # Process messages
    print(f"Reading: {args.input}", file=sys.stderr)
    total_messages = 0
    total_tokens = 0
    skipped = 0
    max_len = 0
    min_len = float('inf')
    len_histogram = {}  # bucket -> count

    sequences = []

    for messages in read_jsonl(args.input):
        total_messages += 1
        tokens = tokenize_chat(messages, tokenizer)

        if len(tokens) > args.max_length:
            skipped += 1
            continue

        seq_len = len(tokens)
        total_tokens += seq_len

        if seq_len > max_len:
            max_len = seq_len
        if seq_len < min_len:
            min_len = seq_len

        # Histogram in 128-token buckets
        bucket = (seq_len // 128) * 128
        len_histogram[bucket] = len_histogram.get(bucket, 0) + 1

        sequences.append(tokens)

    kept = len(sequences)
    avg_len = total_tokens / kept if kept > 0 else 0

    print(f"\nStatistics:", file=sys.stderr)
    print(f"  Total messages:  {total_messages}", file=sys.stderr)
    print(f"  Kept:            {kept}", file=sys.stderr)
    print(f"  Skipped (>{args.max_length}): {skipped}", file=sys.stderr)
    print(f"  Total tokens:    {total_tokens}", file=sys.stderr)
    print(f"  Min length:      {min_len}", file=sys.stderr)
    print(f"  Max length:      {max_len}", file=sys.stderr)
    print(f"  Avg length:      {avg_len:.1f}", file=sys.stderr)
    print(f"\n  Length distribution:", file=sys.stderr)

    for bucket in sorted(len_histogram.keys()):
        count = len_histogram[bucket]
        bar = '#' * min(50, count // max(1, kept // 50))
        print(f"    {bucket:>5}-{bucket+127:>5}: {count:>6} {bar}", file=sys.stderr)

    if args.dry_run:
        return

    # Write output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    binary = args.binary or args.output.endswith('.tok')

    if binary:
        write_tokens_binary(args.output, sequences)
    else:
        write_tokens_text(args.output, sequences)

    print(f"\nWrote {kept} sequences ({total_tokens} tokens) to {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
