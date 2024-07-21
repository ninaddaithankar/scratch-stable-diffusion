import math
import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):

	def __init__(self, n_heads, d_embed, W_qkv_bias=True, W_o_bias=True):
		super().__init__()
		
		self.heads = n_heads
		self.d_head = d_embed//n_heads

		self.W_qkv = nn.Linear(d_embed, d_embed * 3, bias = W_qkv_bias)
		self.W_o   = nn.Linear(d_embed, d_embed, bias = W_o_bias)

	def forward(self, x, causal_mask=True):

		b, pixels, d_embed = x.shape

		q, k, v = self.W_qkv(x).chunk(3, dim= -1)

		q.view((b, pixels, self.heads, self.d_head)).transpose(1, 2)
		k.view((b, pixels, self.heads, self.d_head)).transpose(1, 2)
		v.view((b, pixels, self.heads, self.d_head)).transpose(1, 2)

		weights = q @ k.transpose(-1, -2)

		if causal_mask:
			mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
			weights.masked_fill_(mask, -torch.inf)
		
		weights /= math.sqrt(self.d_head)

		weights = F.softmax(weights, dim= -1)

		outputs = weights @ v

		outputs = outputs.transpose(1, 2).reshape((b, pixels, d_embed))
		outputs = self.W_o(outputs)

		return outputs
