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
Inspect JSONL training data (chat messages format).

Usage:
    # Show 5 random examples
    python inspect_training_data.py train.jsonl

    # Show examples 10-14
    python inspect_training_data.py train.jsonl --start 10 --count 5

    # Search for examples containing a keyword
    python inspect_training_data.py train.jsonl --search "keyword"

    # Show file size and line count
    python inspect_training_data.py train.jsonl --summary
"""

import argparse
import json
import os
import random
import sys


def load_examples(path):
    """Load all examples from JSONL."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def print_example(idx, example):
    """Pretty-print a single example."""
    msgs = example["messages"]
    user_msg = msgs[0]["content"] if len(msgs) > 0 else ""
    asst_msg = msgs[1]["content"] if len(msgs) > 1 else ""

    print(f"\n{'='*80}")
    print(f"  Example #{idx}")
    print(f"{'='*80}")
    print(f"\n--- USER ---")
    print(user_msg)
    print(f"\n--- ASSISTANT ---")
    print(asst_msg)
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect training data")
    parser.add_argument("input", help="JSONL file to inspect")
    parser.add_argument("--start", type=int, default=None,
                        help="Start index")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of examples to show (default: 5)")
    parser.add_argument("--search", type=str, default=None,
                        help="Search for examples containing this text")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary only")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for random sampling")

    args = parser.parse_args()

    file_size = os.path.getsize(args.input)
    examples = load_examples(args.input)

    if args.summary:
        # Count user/assistant token lengths roughly
        user_lens = []
        asst_lens = []
        for ex in examples:
            msgs = ex["messages"]
            user_lens.append(len(msgs[0]["content"]))
            asst_lens.append(len(msgs[1]["content"]))

        print(f"File:             {args.input}")
        print(f"Size:             {file_size / 1024 / 1024:.1f} MB")
        print(f"Examples:         {len(examples)}")
        print(f"Avg user chars:   {sum(user_lens)/len(user_lens):.0f}")
        print(f"Avg asst chars:   {sum(asst_lens)/len(asst_lens):.0f}")
        print(f"Min asst chars:   {min(asst_lens)}")
        print(f"Max asst chars:   {max(asst_lens)}")
        return

    if args.search:
        matches = []
        for i, ex in enumerate(examples):
            text = json.dumps(ex)
            if args.search.lower() in text.lower():
                matches.append((i, ex))

        print(f"Found {len(matches)} examples matching '{args.search}'")
        for i, (idx, ex) in enumerate(matches[:args.count]):
            print_example(idx, ex)
        return

    if args.start is not None:
        for i in range(args.start, min(args.start + args.count, len(examples))):
            print_example(i, examples[i])
    else:
        # Random sample
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(examples)), min(args.count, len(examples)))
        for idx in sorted(indices):
            print_example(idx, examples[idx])


if __name__ == "__main__":
    main()
