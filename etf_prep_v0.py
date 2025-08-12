# prepare a ETF for GPT-2 small embedding
# working in FP32

import torch
from torch import nn
import torch.nn.functional as F

# hyperparameters
num_iterations = 2000
cooldown_frac = 0.4 # fraction of iterations to cooldown

# functions
def norm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

class ETFpreparer(nn.Module):
    def __init__(self, vocab_size: int = 50257, model_dim: int = 768):
        super(ETFpreparer, self).__init__()
        self.vocab_size = vocab_size
        self.W = nn.Parameter(torch.empty(vocab_size, model_dim))
        self.W.data.normal_(mean=0.0, std=1.0)

    def forward(self, batch_size: int = 8192):
        overlap_sum = 0.0

        for i in range(0, self.vocab_size, batch_size):
            end_i = min(i + batch_size, self.vocab_size)
            W_i = norm(self.W[i:end_i])

            for j in range(i, self.vocab_size, batch_size):
                end_j = min(j + batch_size, self.vocab_size)
                W_j = norm(self.W[j:end_j])

                # Compute batched dot products
                dot_products = torch.matmul(W_i, W_j.T)
                
                # For the diagonal block, exclude lower triangle and diagonal
                if i == j:
                    dot_products = dot_products.triu(diagonal=1)

                # Accumulate the sum of squared dot products
                overlap_sum += (dot_products ** 2).sum()

        return overlap_sum / (self.vocab_size * (self.vocab_size - 1) / 2)
    
model = ETFpreparer()
model = model.cuda() # to V100
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# training loop
for step in range(num_iterations):
    # update learning rate
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * get_lr(step)

    # forward pass
    loss = model()

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # update weights
    optimizer.step()

    # print progress
    if step % 100 == 0:
        print(f"Step {step}, Loss {loss.item():.4f}, ETF {1 / 768:.4f}")

# save the W matrix
torch.save(norm(model.W.cpu()), "etf_weight.pt")