#!/usr/bin/env python3
"""
PyTorch benchmark for T5 relative position bias and GeGLU vs SwiGLU.
Run alongside CAIF C++ benchmarks for direct comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# T5 relative position bias (from HuggingFace T5 implementation)
# ---------------------------------------------------------------------------
def t5_relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets = num_buckets // 2
        ret = (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.clamp(n, min=0)

    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) /
        math.log(max_distance / max_exact) *
        (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)
    ret = ret + torch.where(is_small, n, val_if_large)
    return ret


def compute_t5_bias(num_heads, q_len, k_len, num_buckets, max_distance,
                    bidirectional, embedding, device):
    q_pos = torch.arange(q_len, dtype=torch.long, device=device).unsqueeze(1)
    k_pos = torch.arange(k_len, dtype=torch.long, device=device).unsqueeze(0)
    relative_position = k_pos - q_pos
    buckets = t5_relative_position_bucket(relative_position, bidirectional,
                                          num_buckets, max_distance)
    # embedding: [num_heads, num_buckets]
    # buckets: [q_len, k_len]
    bias = embedding[:, buckets.view(-1)].view(num_heads, q_len, k_len)
    return bias


# ---------------------------------------------------------------------------
# Benchmark: RPB forward
# ---------------------------------------------------------------------------
def bench_rpb_forward():
    num_heads = 8
    num_buckets = 32
    max_distance = 128
    seq_len = 512
    warmup = 10
    iters = 100
    device = "cuda"

    embedding = torch.randn(num_heads, num_buckets, device=device)

    for _ in range(warmup):
        compute_t5_bias(num_heads, seq_len, seq_len, num_buckets,
                        max_distance, True, embedding, device)
    sync()

    start = time.perf_counter()
    for _ in range(iters):
        compute_t5_bias(num_heads, seq_len, seq_len, num_buckets,
                        max_distance, True, embedding, device)
    sync()
    elapsed = time.perf_counter() - start

    per_iter_us = elapsed * 1e6 / iters
    print(f"[BENCH] PyTorch RPB forward (heads={num_heads},"
          f"seq={seq_len},"
          f"buckets={num_buckets}): {per_iter_us:.1f} us/iter"
          f" ({iters} iters)")


# ---------------------------------------------------------------------------
# Benchmark: T5 attention forward vs base MHA forward
# ---------------------------------------------------------------------------
def bench_t5_vs_base_mha():
    batch = 4
    seq_len = 128
    dim = 256
    num_heads = 8
    head_dim = dim // num_heads
    warmup = 5
    iters = 50
    device = "cuda"

    # Base MHA (no position bias, uses flash attention via F.scaled_dot_product_attention)
    q_proj = nn.Linear(dim, dim, bias=False).to(device)
    k_proj = nn.Linear(dim, dim, bias=False).to(device)
    v_proj = nn.Linear(dim, dim, bias=False).to(device)
    o_proj = nn.Linear(dim, dim, bias=False).to(device)

    x = torch.randn(batch, seq_len, dim, device=device)

    def base_mha_forward(x):
        B, S, D = x.shape
        q = q_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return o_proj(attn_out)

    # T5 attention (explicit scores + position bias, no flash)
    rpb_embedding = torch.randn(num_heads, 32, device=device)

    def t5_mha_forward(x):
        B, S, D = x.shape
        q = q_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        bias = compute_t5_bias(num_heads, S, S, 32, 128, True,
                               rpb_embedding, device)
        scores = scores + bias.unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return o_proj(attn_out)

    with torch.no_grad():
        # Warmup base
        for _ in range(warmup):
            base_mha_forward(x)
        sync()

        start = time.perf_counter()
        for _ in range(iters):
            base_mha_forward(x)
        sync()
        base_ms = (time.perf_counter() - start) * 1000 / iters

        # Warmup T5
        for _ in range(warmup):
            t5_mha_forward(x)
        sync()

        start = time.perf_counter()
        for _ in range(iters):
            t5_mha_forward(x)
        sync()
        t5_ms = (time.perf_counter() - start) * 1000 / iters

    overhead = ((t5_ms - base_ms) / base_ms) * 100
    print(f"[BENCH] PyTorch base MHA forward (batch={batch},"
          f"seq={seq_len},"
          f"dim={dim},"
          f"heads={num_heads}): {base_ms:.4f} ms/iter")
    print(f"[BENCH] PyTorch T5 attention forward (same config + RPB): "
          f"{t5_ms:.4f} ms/iter")
    print(f"[BENCH] PyTorch T5 overhead vs base MHA: {overhead:.1f}%")


# ---------------------------------------------------------------------------
# Benchmark: T5 attention forward+backward
# ---------------------------------------------------------------------------
def bench_t5_fwd_bwd():
    batch = 4
    seq_len = 128
    dim = 256
    num_heads = 8
    head_dim = dim // num_heads
    warmup = 5
    iters = 50
    device = "cuda"

    q_proj = nn.Linear(dim, dim, bias=False).to(device)
    k_proj = nn.Linear(dim, dim, bias=False).to(device)
    v_proj = nn.Linear(dim, dim, bias=False).to(device)
    o_proj = nn.Linear(dim, dim, bias=False).to(device)
    rpb_embedding = nn.Parameter(torch.randn(num_heads, 32, device=device))

    x = torch.randn(batch, seq_len, dim, device=device, requires_grad=False)

    def t5_mha_fwd_bwd(x):
        x = x.detach().requires_grad_(True)
        B, S, D = x.shape
        q = q_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        bias = compute_t5_bias(num_heads, S, S, 32, 128, True,
                               rpb_embedding, device)
        scores = scores + bias.unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        out = o_proj(attn_out)
        loss = out.sum()
        loss.backward()

    for _ in range(warmup):
        q_proj.zero_grad()
        k_proj.zero_grad()
        v_proj.zero_grad()
        o_proj.zero_grad()
        rpb_embedding.grad = None
        t5_mha_fwd_bwd(x)
    sync()

    start = time.perf_counter()
    for _ in range(iters):
        q_proj.zero_grad()
        k_proj.zero_grad()
        v_proj.zero_grad()
        o_proj.zero_grad()
        rpb_embedding.grad = None
        t5_mha_fwd_bwd(x)
    sync()
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters

    print(f"[BENCH] PyTorch T5 attention fwd+bwd (batch={batch},"
          f"seq={seq_len},"
          f"dim={dim},"
          f"heads={num_heads}): {elapsed_ms:.4f} ms/iter")


# ---------------------------------------------------------------------------
# Benchmark: GeGLU vs SwiGLU FFN forward
# ---------------------------------------------------------------------------
def bench_geglu_vs_swiglu_forward():
    bs = 2048
    dim = 512
    hidden_dim = 1024
    warmup = 10
    iters = 100
    device = "cuda"

    w_gate = nn.Linear(dim, hidden_dim, bias=False).to(device)
    w_up = nn.Linear(dim, hidden_dim, bias=False).to(device)
    w_down = nn.Linear(hidden_dim, dim, bias=False).to(device)

    x = torch.randn(bs, dim, device=device)

    def swiglu_forward(x):
        gate = F.silu(w_gate(x))
        up = w_up(x)
        return w_down(gate * up)

    def geglu_forward(x):
        gate = F.gelu(w_gate(x))
        up = w_up(x)
        return w_down(gate * up)

    with torch.no_grad():
        # SwiGLU
        for _ in range(warmup):
            swiglu_forward(x)
        sync()

        start = time.perf_counter()
        for _ in range(iters):
            swiglu_forward(x)
        sync()
        swiglu_ms = (time.perf_counter() - start) * 1000 / iters

        # GeGLU
        for _ in range(warmup):
            geglu_forward(x)
        sync()

        start = time.perf_counter()
        for _ in range(iters):
            geglu_forward(x)
        sync()
        geglu_ms = (time.perf_counter() - start) * 1000 / iters

    diff = ((geglu_ms - swiglu_ms) / swiglu_ms) * 100
    print(f"[BENCH] PyTorch SwiGLU FFN forward (bs={bs},"
          f"dim={dim},"
          f"hidden={hidden_dim}): {swiglu_ms:.4f} ms/iter")
    print(f"[BENCH] PyTorch GeGLU FFN forward  (same config): "
          f"{geglu_ms:.4f} ms/iter")
    print(f"[BENCH] PyTorch GeGLU vs SwiGLU diff: {diff:.2f}%")


# ---------------------------------------------------------------------------
# Benchmark: GeGLU vs SwiGLU FFN forward+backward
# ---------------------------------------------------------------------------
def bench_geglu_vs_swiglu_fwd_bwd():
    bs = 2048
    dim = 512
    hidden_dim = 1024
    warmup = 5
    iters = 50
    device = "cuda"

    def make_ffn():
        return (nn.Linear(dim, hidden_dim, bias=False).to(device),
                nn.Linear(dim, hidden_dim, bias=False).to(device),
                nn.Linear(hidden_dim, dim, bias=False).to(device))

    sw_gate, sw_up, sw_down = make_ffn()
    ge_gate, ge_up, ge_down = make_ffn()

    x = torch.randn(bs, dim, device=device, requires_grad=False)

    def swiglu_fwd_bwd(x):
        x = x.detach().requires_grad_(True)
        gate = F.silu(sw_gate(x))
        up = sw_up(x)
        out = sw_down(gate * up)
        out.sum().backward()

    def geglu_fwd_bwd(x):
        x = x.detach().requires_grad_(True)
        gate = F.gelu(ge_gate(x))
        up = ge_up(x)
        out = ge_down(gate * up)
        out.sum().backward()

    # SwiGLU
    for _ in range(warmup):
        sw_gate.zero_grad()
        sw_up.zero_grad()
        sw_down.zero_grad()
        swiglu_fwd_bwd(x)
    sync()

    start = time.perf_counter()
    for _ in range(iters):
        sw_gate.zero_grad()
        sw_up.zero_grad()
        sw_down.zero_grad()
        swiglu_fwd_bwd(x)
    sync()
    swiglu_ms = (time.perf_counter() - start) * 1000 / iters

    # GeGLU
    for _ in range(warmup):
        ge_gate.zero_grad()
        ge_up.zero_grad()
        ge_down.zero_grad()
        geglu_fwd_bwd(x)
    sync()

    start = time.perf_counter()
    for _ in range(iters):
        ge_gate.zero_grad()
        ge_up.zero_grad()
        ge_down.zero_grad()
        geglu_fwd_bwd(x)
    sync()
    geglu_ms = (time.perf_counter() - start) * 1000 / iters

    diff = ((geglu_ms - swiglu_ms) / swiglu_ms) * 100
    print(f"[BENCH] PyTorch SwiGLU FFN fwd+bwd (bs={bs},"
          f"dim={dim},"
          f"hidden={hidden_dim}): {swiglu_ms:.4f} ms/iter")
    print(f"[BENCH] PyTorch GeGLU FFN fwd+bwd  (same config): "
          f"{geglu_ms:.4f} ms/iter")
    print(f"[BENCH] PyTorch GeGLU vs SwiGLU fwd+bwd diff: {diff:.2f}%")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    print("--- RPB & T5 Attention ---")
    bench_rpb_forward()
    bench_t5_vs_base_mha()
    bench_t5_fwd_bwd()

    print("\n--- GeGLU vs SwiGLU ---")
    bench_geglu_vs_swiglu_forward()
    bench_geglu_vs_swiglu_fwd_bwd()
