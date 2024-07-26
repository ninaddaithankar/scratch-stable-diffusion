import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class ResidualBlock(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.groupnorm_in = nn.GroupNorm(32, in_channels)
		self.groupnorm_out = nn.GroupNorm(32, out_channels)

		self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

		if in_channels == out_channels:
			self.residual_layer = nn.Identity()
		else:
			self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

	def forward(self, x):
		residue = x

		x = self.groupnorm_in(x)
		x = F.silu(x)

		x = self.conv_in(x)

		x = self.groupnorm_out(x)
		x = F.silu(x)

		x = self.conv_out(x)

		x = x + self.residual_layer(residue)

		return x


class SelfAttentionBlock(nn.Module):

	def __init__(self, in_channels):
		super().__init__()
		
		self.groupnorm = nn.GroupNorm(32, in_channels)
		self.attention = SelfAttention(1, in_channels)

	def forward(self, x):
		residue = x

		x = self.groupnorm(x)

		b, c, h, w = x.shape

		x = x.view((b, c, h*w))
		x = x.transpose(-1, -2)

		x = self.attention(x)

		x = x.transpose(-1, -2)
		x = x.view((b, c, h, w))

		x += residue

		return x

