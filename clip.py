import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):

	def __init__(self, vocab_size, embedding_size, n_tokens):
		super().__init__()

		self.token_embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_size)
		self.positional_encodings = nn.Parameter(torch.zeros((n_tokens, embedding_size)))

	def forward(self, tokens):

		x = self.token_embeddings(tokens)
		x += self.positional_encodings

		return x


class CLIPLayer(nn.Module):

	def __init__(self, n_heads, embedding_size):
		super().__init__()

		self.layernorm1 = nn.LayerNorm(embedding_size)
		self.layernorm2 = nn.LayerNorm(embedding_size)

		self.self_attention = SelfAttention(n_heads, embedding_size)

		self.linear1 = nn.Linear(embedding_size, embedding_size * 4)
		self.linear2 = nn.Linear(embedding_size * 4, embedding_size)

	def forward(self, x):

		residue = x

		# SELF ATTENTION
		x = self.layernorm1(x)
		x = self.self_attention(x, causal_mask = True)
		x += residue

		residue = x

		# FEED FORWARD
		x = self.layernorm2(x)
		x = self.linear1(x)
		x = x * torch.sigmoid(1.702*x)		# this is GeLU

		x = self.linear2(x)
		x += residue

		return x
	

class CLIPEncoder(nn.Module):

	def __init__(self):
		super().__init__()

		# EMBEDDINGS TABLE
		self.embeddings = CLIPEmbedding(49408, 768, 77)

		# 12 TRANSFORMER ENCODER LAYERS
		self.cliplayers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])

		self.layernorm = nn.LayerNorm(768)

	def forward(self, tokens):
		tokens = tokens.type(torch.long)

		state = self.embeddings(tokens)
		for layer in self.cliplayers:
			state = layer(state)		
		output = self.layernorm(state)

		return output
	


		