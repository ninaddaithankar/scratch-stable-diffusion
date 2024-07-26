import torch
from torch import nn
from torch.nn import functional as F
from unet import UNet, UNetOutputLayer

class TimeEmbedding(nn.Module):

	def __init__(self, embedding_size):
		super().__init__()

		self.linear1 = nn.Linear(embedding_size, embedding_size * 4)
		self.linear2 = nn.Linear(embedding_size * 4, embedding_size * 4)

	def forward(self, time):

		time = self.linear1(time)
		time = F.silu(time)
		time = self.linear2(time)

		return time
	

class Diffusion(nn.Module):

	def __init__(self):
		super().__init__()

		self.time_embedding = TimeEmbedding(320)
		self.unet = UNet()
		self.final_layer = UNetOutputLayer(320, 4)

	def forward(self, latent, context, time):

		time = self.time_embedding(time)
		output = self.unet(latent, context, time)
		output = self.final_layer(output)
