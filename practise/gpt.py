import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
trainset_ratio = 0.9
batch_size = 64
block_size = 256 # max sequence length
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 384 # embedding dimension
n_layers = 6
n_head = 6
learning_rate = 3e-4
dropout = 0.2
max_iters = 5000

class Head(nn.Module):
    "one head of self-attention"

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size: (batch, time_step, n_embd)
        # output size: (batch, time_step, head_size)
        B, T, C = x.shape
        k = self.key(x)  # (batch, time_step, head_size)
        q = self.query(x)
        # compute attention scores: (batch, time_step, time_step)
        score = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])  # k矩阵后面两个维度交换顺序
        # masked_fill: fiills elements of with value where mask is True
        score = score.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        score= F.softmax(score, dim=-1)  # softmax over row (last dimension)
        score = self.dropout(score)
        v = self.value(x)
        out = score @ v  # (B, T, T) * (B, T, head_size) = (B, T, head_size)

        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(n_head))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def foward(self, x):
        # input: (batch, time_seq, n_embd)
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(x))

        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # input: (batch, time_seq, n_embd)
        # output: (batch, time_seq, n_embd)
        x = x + self.sa(self.ln1(x))  # add & pre-norm
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__() # 调用父类的构造函数
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[GPTBlock(n_embd, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if(isinstance(module, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if(isinstance(module, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # batch_size, time_step
        device = idx.device
        tok_emb = self.token_embedding_table(idx)  # (batch, time_step, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (time_step, n_embd)
        x = tok_emb + pos_emb # broadcast, (batch, time_step, n_embd)
        x = self.blocks(x)    # (batch, time_step, n_embd)
        x = self.ln_f(x)      # pre-norm, (batch, time_step, n_embd)
        logits = self.lm_head(x) # (batch, time_step, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # reshape: (batch, time_step, vocab_size) -> (batch*time_step, vocab_size)
            targets = targets.view(B*T)  # (batch, time_step, 1)? -> (batch*time_step)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        pass

if __name__ == '__main__':
    # 1. prepare data
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # get vocab size 
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # encode, decode
    # just char <-> index
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for i,c in enumerate(chars)}     
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # get train/eval data    
    data = torch.tensor(encode(text), dtype=torch.int) # int(32 bit) is enough，don't need long
    split_at = int(len(text) * trainset_ratio)
    train_data = data[:split_at]
    val_data = data[split_at:]

    def get_batch(split):

        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)

        return x, y

    # 2. create model
    model = GPTLanguageModel(vocab_size)
    model = model.to(device)
    # number of parameters
    print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")

    # 3. train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        
        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True) # 将优化器的梯度清零
        loss.backward()
        optimizer.step()