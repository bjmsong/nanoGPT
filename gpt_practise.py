import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
trainset_ratio = 0.9
batch_size = 64
block_size = 256 # max sequence length
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 384
n_layers = 6
n_head = 6
learning_rate = 

class GPTBlock(nn.Module):
    # n_embd: embedding dimension, n_head: the number of heads we'd like
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # Key, query, value projections for all heads in batch
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

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
        B, T = idx.shape  # batch_size, seq_len
        device = idx.device
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb # broadcast
        x = self.blocks(x)    # (B, )
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # reshape
            targets = targets.view(B*T)
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

    # 3. train model

