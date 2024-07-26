import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class UNetResidualBlock(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.groupnorm1 = nn.GroupNorm(32, in_channels)
		self.groupnorm2 = nn.GroupNorm(32, out_channels)

		self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

		self.linear_time = nn.Linear(1280, out_channels)

		if in_channels == out_channels:
			self.residual_layer = nn.Identity()
		else:
			self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

	def forward(self, x, time):

		residue = x

		x = self.groupnorm1(x)
		x = F.silu(x)
		x = self.conv_feature(x)

		time = self.linear_time(time)
		time = F.silu(time)

		# (b, out_channels, h, w) + (1, out_channels, 1, 1)
		x_merged = x + time.unsqueeze(-1).unsqueeze(-1)

		x_merged = self.groupnorm2(x_merged)
		x_merged = F.silu(x_merged)
		x_merged = self.conv_merged(x_merged)

		x_merged += self.residual_layer(residue)

		return x_merged
	


class UNetAttentionBlock(nn.Module):

	def __init__(self, n_heads, d_head, d_context=768):
		super().__init__()

		channels = n_heads * d_head

		self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)

		self.layernorm1 = nn.LayerNorm(channels)
		self.layernorm2 = nn.LayerNorm(channels)
		self.layernorm3 = nn.LayerNorm(channels)

		self.conv_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
		self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

		self.self_attention = SelfAttention(n_heads, channels, W_qkv_bias=False)
		self.cross_attention = CrossAttention(n_heads, channels, d_context, W_qkv_bias=False)

		self.linear_geglu1 = nn.Linear(channels, 4 * channels * 2)
		self.linear_geglu2 = nn.Linear(4 * channels, channels)

	def forward(self, x, context):

		residue_long = x

		x = self.groupnorm(x)
		x = self.conv_in(x)

		n, c, h, w = x.shape

		x = x.view((n, c, h * w))
		x = x.transpose(-1, -2)

		residue_short = x

		x = self.layernorm1(x)
		x = self.self_attention(x)
		x += residue_short
		
		residue_short = x

		x = self.layernorm2(x)
		x = self.cross_attention(x, context)
		x += residue_short

		residue_short = x

		x = self.layernorm3(x)
		x, gate = self.linear_geglu1(x).chunk(2, dim = -1)
		x = x * F.gelu(gate)
		x = self.linear_geglu2(x)
		x += residue_short

		x = x.transpose(-1, -2)
		x = x.view((n, c, h, w))

		x = self.conv_out(x)
		x += residue_long

		return x
	


class Upsample(nn.Module):

	def __init__(self, channels):
		super().__init__()

		self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

	def forward(self, x):

		x = F.interpolate(x, scale_factor=2, mode="nearest")
		return self.conv(x)
	

class SwitchSequential(nn.Sequential):
	
	def forward(self, x, context, time):
		for layer in self.layers:
			if isinstance(layer, UNetAttentionBlock):
				x = layer(x, context)
			elif isinstance(layer, UNetResidualBlock):
				x = layer(x, time)
			else:
				x = layer(x)
		return x
	

class UNet(nn.Module):

	def __init__(self):
		self.encoder = nn.ModuleList([
			# b, c, h/8, w/8
			SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

			SwitchSequential(UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)),
			SwitchSequential(UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)),

			# DOWNSAMPLE: b, c, h/8, w/8 -> b, c, h/16, w/16
			SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

			SwitchSequential(UNetResidualBlock(320, 640), UNetAttentionBlock(8, 80)),
			SwitchSequential(UNetResidualBlock(640, 640), UNetAttentionBlock(8, 80)),

			# DOWNSAMPLE: b, c, h/16, w/16 -> b, c, h/32, w/32
			SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

			SwitchSequential(UNetResidualBlock(640, 1280), UNetAttentionBlock(8, 160)),
			SwitchSequential(UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)),

			# DOWNSAMPLE: b, c, h/32, w/32 -> b, c, h/64, w/64
			SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

			SwitchSequential(UNetResidualBlock(1280, 1280)),
			SwitchSequential(UNetResidualBlock(1280, 1280)),
		])

		self.bottleneck = SwitchSequential(
			UNetResidualBlock(1280, 1280),
			UNetAttentionBlock(8, 160),
			UNetResidualBlock(1280, 1280),
		)

		self.decoder = nn.ModuleList([
			SwitchSequential(UNetResidualBlock(2560, 1280)),
			SwitchSequential(UNetResidualBlock(2560, 1280)),

			# UPSAMPLE
			SwitchSequential(UNetResidualBlock(2560, 1280), Upsample(1280)),

			SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
			SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),

			# UPSAMPLE
			SwitchSequential(UNetResidualBlock(1920, 1280), UNetAttentionBlock(8, 160), Upsample(1280)),

			SwitchSequential(UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)),
			SwitchSequential(UNetResidualBlock(1280, 640), UNetAttentionBlock(8, 80)),

			# UPSAMPLE
			SwitchSequential(UNetResidualBlock(960, 640), UNetAttentionBlock(8, 80), Upsample(640)),

			SwitchSequential(UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)),
			SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)),
			SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)),
		])

	def forward(self, x, context, time):
		
		skip_connections = []
		for layer in self.encoder:
			x = layer(x, context, time)
			skip_connections.append(x)

		x = self.bottleneck(x, context, time)

		for layer in self.decoder:
			x = torch.cat((x, skip_connections.pop()), dim=1)
			x = layer(x, context, time)

		return x
	

class UNetOutputLayer(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.groupnorm = nn.GroupNorm(32, in_channels)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

	def forward(self, x):
		
		x = self.groupnorm(x)
		x = F.silu(x)
		x = self.conv(x)

		return x





		


