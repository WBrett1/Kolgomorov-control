import torch
import torch.nn as nn
import numpy as np

 
class Encoder(nn.Module):
    def __init__(self, image_size, channels, embedding_dim, activation=nn.Tanh(), base=32):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(channels, base, kernel_size=3, stride=2, padding=1, padding_mode='circular',)
        self.conv2 = nn.Conv2d(base, base*2, kernel_size=3, stride=2, padding=1, padding_mode='circular',)
        self.conv3 = nn.Conv2d(base*2, base*4, kernel_size=3, stride=2, padding=1, padding_mode='circular',)
        self.activation = activation
        self.shape_before_flattening = None
        flattened_size = (image_size // 8) * (image_size // 8) * 4*base
        self.fc = nn.Linear(flattened_size, embedding_dim, bias=False)
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        self.shape_before_flattening = x.shape[1:]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening, channels, activation=nn.Tanh(), base=32):
        super(Decoder, self).__init__()
        self.activation = activation
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        self.reshape_dim = shape_before_flattening
        self.circularpad = nn.CircularPad2d((1,0,1,0))

        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(base*2, base, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(base, base, kernel_size=3, stride=2, padding=1, output_padding=0)

        self.conv1 = nn.Conv2d(base, channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='circular')

    def forward(self, x):
        x = self.activation(self.fc(x))
        x = x.view(x.size(0), *self.reshape_dim)
        x = self.activation(self.circularpad(self.deconv1(x)))
        x = self.activation(self.circularpad(self.deconv2(x)))
        x = self.activation(self.circularpad(self.deconv3(x)))
        x = self.conv1(x)

        return x

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64, image_size=32, channels=2, activation=nn.Tanh(), base=32):
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(image_size, channels, latent_dim, activation=activation, base=base)

        dummy = torch.randn(1,2,32,32)
        self.encoder(dummy)

        self.decoder = Decoder(latent_dim, self.encoder.shape_before_flattening, channels, activation=activation, base=base)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def name(self):
        return f'AutoEncoder_Nd{self.latent_dim}'
