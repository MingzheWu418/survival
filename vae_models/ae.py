import torch.nn as nn
from .types_ import *


class Autoencoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 *kwargs):

        super(Autoencoder, self).__init__()

        encoding_layers = []
        if hidden_dims:
            encoding_layers.append(
                nn.Sequential(
                    nn.Linear(input_size, hidden_dims[0]),
                    nn.ReLU()
                )
            )

            for i in range(len(hidden_dims)-1):
                encoding_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                        nn.ReLU()
                    )
                )

            encoding_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[-1], latent_dim)
                )
            )
        else:
            encoding_layers.append(
                nn.Sequential(
                    nn.Linear(input_size, latent_dim)
                )
            )

        decoding_layers = []
        if hidden_dims:
            decoding_layers.append(
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dims[-1]),
                    nn.ReLU()
                )
            )

            for i in range(len(hidden_dims)-1):
                decoding_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[-i-1], hidden_dims[-i]),
                        nn.ReLU()
                    )
                )

            decoding_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[0], input_size),
                )
            )
        else:
            decoding_layers.append(
                nn.Sequential(
                    nn.Linear(latent_dim, input_size),
                )
            )

        self.encoder = nn.Sequential(*encoding_layers)
        self.decoder = nn.Sequential(*decoding_layers)

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_size,512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, latent_dim)
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, input_size)
        # )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, rep):
        return self.decoder(rep)