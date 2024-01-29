import torch
import torch.nn as nn
from torch.nn import Functional as F
import math

vocab_size = 10000
max_seq_len = 256
n_head = 6
n_embd = 384
head_size = n_embd // n_head
dropout = 0.2

class FeedForwardNN(nn.Moduel):
    def __init__(self):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self):
        super.__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        score = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        score = score.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        score = nn.Softmax(score, dim=-1)
        self.dropout(score)

        v = self.value(v)
        out = score @ v

        return out

class MultiHeadAttention(nn.Moduel):
    def __init__(self):
        super.__init__()
        self.heads = nn.ModuleLists([Head() for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out        

class GPTBlock(nn.Module):
    def __init__(self):
        super.__init__()
        self.mha = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffn = FeedForwardNN()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        out = x + (self.mha(self.ln1(x)))
        out = out + (self.ffn(self.ln2(x)))

        return self.dropout(out)
