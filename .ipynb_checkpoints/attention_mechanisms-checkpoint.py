import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, query, key, value, mask=None):
        raise NotImplementedError

class DotProductAttention(BaseAttention):
    def __init__(self, embed_dim):
        super().__init__(embed_dim)

    def forward(self, query, key, value, mask=None):
        # query, key, value shape: (batch_size, seq_len, embed_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights

class MultiHeadAttention(BaseAttention):
    def __init__(self, embed_dim, num_heads):
        super().__init__(embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(attn_output)

        return output, attn_weights

class SparseAttention(BaseAttention):
    def __init__(self, embed_dim, block_size):
        super().__init__(embed_dim)
        self.block_size = block_size

    def forward(self, query, key, value, mask=None):
        # Simplified sparse attention: only attend to nearby blocks
        batch_size, seq_len, _ = query.size()
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        
        # Create a block diagonal mask
        block_mask = torch.ones_like(scores).tril(self.block_size - 1)
        for i in range(1, seq_len // self.block_size):
            block_mask[:, i*self.block_size:(i+1)*self.block_size, (i-1)*self.block_size:i*self.block_size] = 1

        if mask is not None:
            block_mask = block_mask * mask

        scores = scores.masked_fill(block_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights

# Helper function to create attention mask
def create_attention_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0)  # Add batch dimension