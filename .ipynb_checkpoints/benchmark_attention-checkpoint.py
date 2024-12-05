import time
import torch
import numpy as np
from attention_mechanisms import DotProductAttention, MultiHeadAttention, SparseAttention, create_attention_mask

def generate_random_data(batch_size, seq_length, embed_dim):
    query = torch.randn(batch_size, seq_length, embed_dim)
    key = torch.randn(batch_size, seq_length, embed_dim)
    value = torch.randn(batch_size, seq_length, embed_dim)
    return query, key, value

def benchmark_attention(attention_mechanism, query, key, value, mask=None, num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = attention_mechanism(query, key, value, mask)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def main():
    batch_size = 32
    seq_length = 100
    embed_dim = 512
    num_heads = 8
    block_size = 16

    query, key, value = generate_random_data(batch_size, seq_length, embed_dim)
    mask = create_attention_mask(seq_length).to(query.device)

    attention_mechanisms = {
        "Dot Product Attention": DotProductAttention(embed_dim),
        "Multi-Head Attention": MultiHeadAttention(embed_dim, num_heads),
        "Sparse Attention": SparseAttention(embed_dim, block_size)
    }

    results = {}

    for name, mechanism in attention_mechanisms.items():
        avg_time = benchmark_attention(mechanism, query, key, value, mask)
        results[name] = avg_time
        print(f"{name} - Average time per run: {avg_time:.6f} seconds")

    # Find the fastest mechanism
    fastest = min(results, key=results.get)
    print(f"\nFastest mechanism: {fastest}")

    # Calculate relative speed-up
    baseline = results["Dot Product Attention"]
    for name, time in results.items():
        speedup = baseline / time
        print(f"{name} speedup relative to Dot Product Attention: {speedup:.2f}x")

if __name__ == "__main__":
    main()