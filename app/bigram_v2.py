import torch
import torch.nn as nn
from torch.nn import ReLU, functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

# used for reproducibility
torch.manual_seed(1337)

# read the dataset and inspect it
with open('../data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# extract all characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a character level tokenizer (characters to integers)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s] # take a string and output list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # take a list of integers and output a string

# split the data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% will be train data
train_data = data[:n]
val_data = data[n:]

# load the data
def get_batch(split):
    # generate small batch of data of inputs X and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# one head of self-attention
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) ----> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) ----> (B, T, C)
        return out

# multiple heads of self-attention in parallel
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

# simple Linear layer followed by non-linearity
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

# Transformer block: communication followed by computation
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimmension, n_head: number of heads we want
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, index_X, targets=None):
        B, T = index_X.shape

        # index and targets are both (B, T) tensor of integers
        token_embd = self.token_embedding_table(index_X) # (B, T, C) = (batch, time, channel)
        positional_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embd + positional_embd # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # reshape logits to two-dimensional array and preserve C on the second dimension -> PyTorch Cross Entropy docs
            targets = targets.view(B * T) # reshape targets to one-dimensional array -> PyTorch Cross Entropy docs

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index_X, max_new_tokens):
        # index_X is the (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop index_X to the last block_size tokens
            index_X_cond = index_X[:, -block_size:]
            # get the predictions
            logits, loss = self(index_X_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_X_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index_X = torch.cat((index_X, index_X_next), dim=1) # (B, T+1)
        return index_X

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    X_batch, Y_batch = get_batch('train')
    # evaluate the loss
    logits, loss = model(X_batch, Y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))