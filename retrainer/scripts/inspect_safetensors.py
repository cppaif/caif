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

"""Inspect a safetensors file: list tensor names, shapes, and dtypes.

Requires:
    pip install safetensors torch
"""

import argparse
import sys
from safetensors import safe_open


def main():
    parser=argparse.ArgumentParser(description="Inspect safetensors file")
    parser.add_argument("file", help="Path to .safetensors file")
    args=parser.parse_args()

    f=safe_open(args.file, framework="pt")
    keys=sorted(f.keys())
    print(f"Tensors: {len(keys)}")
    for k in keys:
        t=f.get_tensor(k)
        print(f"  {k}: {list(t.shape)} {t.dtype}")


if __name__=="__main__":
    main()
