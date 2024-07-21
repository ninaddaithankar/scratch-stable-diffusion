import torch
from torch import nn
from torch.nn import functional as F
from autoencoder.decoder import ResidualBlock, AttentionBlock

class Encoder(nn.Sequential):
	def __init__(self):
		super().__init__(
			# the comment below each layer shows the shape after applying the operation

			nn.Conv2d(3, 128, kernel_size=3, padding=1),
			# Out: 128, h, w
			ResidualBlock(128, 128),										# elongate
			# Out: 128, h, w
			ResidualBlock(128, 128),
			# Out: 128, h, w

			
			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),		# downsample
			# Out: 128, h/2, w/2
			ResidualBlock(128, 256),										# elongate
			# Out: 256, h/2, w/2
			ResidualBlock(256, 256),
			# Out: 256, h/2, w/2


			nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 		# downsample
			# Out: 256, h/4, w/4
			ResidualBlock(256, 512),										# elongate
			# Out: 512, h/4, w/4
			ResidualBlock(512, 512),
			# Out: 512, h/4, w/4
			ResidualBlock(512, 512),
			# Out: 512, h/4, w/4


			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),		# downsample
			# Out: 512, h/8, w/8
			ResidualBlock(512, 512),										# elongate
			# Out: 512, h/8, w/8
			ResidualBlock(512, 512),
			# Out: 512, h/8, w/8
			ResidualBlock(512, 512),
			# Out: 512, h/8, w/8


			AttentionBlock(512),											# attention
			# Out: 512, h/8, w/8
			ResidualBlock(512, 512),
			# Out: 512, h/8, w/8


			nn.GroupNorm(num_groups=32, num_channels=512),					# normalize
			# Out: 512, h/8, w/8
			nn.SiLU(),														# non-linearity
			# Out: 512, h/8, w/8


			nn.Conv2d(512, 8, kernel_size=3, padding=1),					# shorten
			# Out: 8, h/8, w/8


			nn.Conv2d(8, 8, kernel_size=1, padding=0),
			# Out: 8, h/8, w/8
		)

	def forward(self, x, noise):

		for module in self:
			# if current module is downsampling (has stride 2), apply asymmetric padding
			if getattr(module, 'stride', None) == (2, 2):
				x = F.pad(x, (0, 1, 0, 1))

			x = module(x)

		# split into two tensors for mean and log variance
		mean, log_variance = torch.chunk(x, chunks=2, dim=1)

		# clamp log variance values between -30, 20
		log_variance = torch.clamp(log_variance, min=-30, max=20)
		
		# exponentiate log variance to get variance
		variance = log_variance.exp()

		# square root variance to get standard deviation
		stddev = variance.sqrt()

		# get x as a normal distribution using N(mean, stdev) = mean + stddev * N(0, 1) 
		x = mean + stddev * noise

		# scale by constant 0.18215 coz for some abstract reason pros at LMU did it
		x *= 0.18215

		return x