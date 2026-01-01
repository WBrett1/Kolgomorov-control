import torch
from torch import nn
from torch.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np


class DiscreteSymmetries(nn.Module):
    def __init__(self):
        super(DiscreteSymmetries, self).__init__()

    def reflect_shift_y_operator(self, x, a : int):

        xf = fftshift(fft2(x, dim=(-2, -1)), dim=(-2, -1))
        kf = 4

        N = x.shape[-1]
        k = torch.arange(-N//2, N//2)
        kx, ky = torch.meshgrid(k, k, indexing='ij')

        s = a * np.pi / kf

        shift = torch.exp(1j * ky.to(x.device) * s)

        xf_shift = shift[None, None, :, :].to(xf.device) * xf

        x_shift = ifft2(ifftshift(xf_shift, dim=(-2, -1)), dim=(-2, -1)).real

        if a % 2 == 1:
            ones = torch.ones_like(x_shift)
            ones[..., 0, :, :] = -1
            x_shift = ones* torch.flip(x_shift, (-2,)) # flip along x axis

        return x_shift

    def flip_operator(self, x):
        return -1* torch.flip(x, (-2, -1))

    def forward(self, x):

        splits = []
        # double reflect
        for b in range(2):
        # shift reflect
            for a in range(8):
                if b == 0:
                    splits.append(self.reflect_shift_y_operator(x, a))
                else:
                    splits.append(self.flip_operator(self.reflect_shift_y_operator(x, a)))

        return torch.stack(splits, dim=1) # B x 16 symmetries x C x N x N

    def inverse(self, x, j):
       
        if j >=8:
            x = self.flip_operator(x)
            x = self.reflect_shift_y_operator(x, -1*(j % 8))
        else:
            x = self.reflect_shift_y_operator(x, -1*(j % 8))
        return x

class STN(nn.Module):
    """spatial transform network"""

    def __init__(self, shape):
        super(STN, self).__init__()

        ## trainable linear layer to estimate the phaseshift of the first fourier mode uh_01
        self.linear = nn.Linear(torch.prod(torch.tensor(shape)), 2, bias=False)

    def shift_x_operator(self, x, s):
        # dim(x) = B x C x N x N
        # dim(s) = B

        N = x.shape[-1]
        k = torch.arange(-N//2, N//2)
        kx, ky = torch.meshgrid(k, k, indexing='ij') # N x N

        xf = fftshift(fft2(x, dim=(-2, -1)), dim=(-2, -1))

        shift = torch.exp(1j *torch.einsum('ij, lh -> ijlh', s.unsqueeze(-1), kx.to(s.device)) ) # B x C x N x N
        xf_shift = shift.to(xf.device) * xf
        x_shift = ifft2(ifftshift(xf_shift, dim=(-2, -1)), dim=(-2, -1)).real
        return x_shift

    def forward(self, x):

        cos_sin = self.linear(x.reshape(*x.shape[:-3], -1))
        cos = cos_sin[:, 0]
        sin = cos_sin[:, 1]
        sx = torch.angle(cos + 1j* sin) # estimate the phase of the first fourier mode uh_01

        return self.shift_x_operator(x, sx), sx


class SymAutoencoder(nn.Module):
    """Symmetry-aware autoencoder"""

    def __init__(self, autoencoder, stn, discrete_symmetries):
        super(SymAutoencoder, self).__init__()

        self.autoencoder = autoencoder
        self.latent_dim = autoencoder.latent_dim

        self.stn = stn
        self.discrete_symmetries = discrete_symmetries

    def forward(self, x):
        B = x.shape[0] ## batch size
        input_dsym = self.discrete_symmetries(x) ## B x 16 x C x N x N with all 16 discrete symmetries
        input_sym, sx = self.stn(input_dsym.reshape(-1, 2, 32, 32)) ## B * 16 x C x N x N  phase shifted to the fourier slice

        # Forward pass
        output_sym = self.autoencoder(input_sym).reshape(-1, 16, 2, 32, 32) ## B x 16 x C x N x N

        return input_sym.reshape(-1, 16, 2, 32, 32), output_sym, sx.reshape(B, 16)

    def name(self):
        return 'S' + self.autoencoder.name()

    def encoder(self, x):
        # normal operation
        B = x.shape[0]
        input_dsym = self.discrete_symmetries(x)
        input_sym, sx = self.stn(input_dsym.reshape(-1, 2, 32, 32))

        encoded = self.autoencoder.encoder(input_sym)
        decoded = self.autoencoder.decoder(encoded).reshape(-1, 16, 2, 32, 32)
        
        argmin_red = torch.argmin(
            torch.mean((decoded - input_sym.reshape(-1, 16, 2, 32, 32))**2, dim=(-3,-2,-1)), 
            dim=1).reshape(-1,1) 

        encoded = encoded.reshape(-1, 16, self.autoencoder.latent_dim)
        sx = sx.reshape(-1, 16)

        # select the encoded vector and sx corresponding to the symmetry with the smallest reconstruction error
        encoded_l = []
        sx_l = []
        for i in range(B):
            encoded_l.append(encoded[i, argmin_red[i]])
            sx_l.append(sx[i, argmin_red[i]])

        encoded = torch.stack(encoded_l)
        sx = torch.stack(sx_l)

        reconstruct_errors = []
        for i in range(B):
            d = self.stn.shift_x_operator(decoded[i, argmin_red[i]], -1*sx[i])
            d = self.discrete_symmetries.inverse(d, argmin_red[i])
            reconstruct_errors.append(torch.mean((d - x[i])**2))


        return encoded, sx, argmin_red, torch.stack(reconstruct_errors)
    
    def decoder(self, encoded, sx, argmin_red, *args):

        decoded = self.autoencoder.decoder(encoded)

        x = self.stn.shift_x_operator(decoded, -1*sx)
        x_l = []
        for i,j in enumerate(argmin_red):
            x_l.append(self.discrete_symmetries.inverse(x[i], j))
        x = torch.vstack(x_l)

        return x
        
    

class TransAutoencoder(nn.Module):
    def __init__(self, autoencoder, stn, ):
        super(TransAutoencoder, self).__init__()

        self.autoencoder = autoencoder
        self.latent_dim = autoencoder.latent_dim
        self.stn = stn

    def forward(self, x):
        input_sym, sx = self.stn(x) # returns the fourier sliced batch

        # Forward pass
        output_sym = self.autoencoder(input_sym) # autoencoder cycle

        return input_sym, output_sym, sx

    def name(self):
        return 'T' + self.autoencoder.name()

    def encoder(self, x):

        input_sym, sx = self.stn(x)


        encoded = self.autoencoder.encoder(input_sym)

        decoded = self.autoencoder.decoder(encoded)
        reconstruct_errors = []
        xhat = self.stn.shift_x_operator(decoded, -1*sx)
        reconstruct_errors = torch.mean((xhat - x)**2, dim=(-3,-2,-1))


        return encoded, sx, reconstruct_errors
    
    def decoder(self, encoded, sx):
        decoded = self.autoencoder.decoder(encoded)

        xhat = self.stn.shift_x_operator(decoded, -1*sx)
        return xhat