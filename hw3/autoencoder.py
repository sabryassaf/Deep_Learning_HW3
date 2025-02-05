import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        # Define a series of convolutional layers with varying kernel sizes and additional Dropout for regularization
        layers = []
        filters = [in_channels, 64, 128, 256, out_channels]
        kernel_sizes = 5
        stride = 2 
        padding = 2
        for i in range(len(filters) - 1):
            layers.append(nn.Conv2d(in_channels=filters[i],
                                    out_channels=filters[i + 1],
                                    kernel_size=kernel_sizes,
                                    stride=stride,
                                    padding=padding))
            layers.append(nn.BatchNorm2d(filters[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.25))  
        self.encoder_dropout = nn.Dropout(0.25) 
        modules.extend(layers)
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        # Define a series of transposed convolutional layers with varying kernel sizes and additional Dropout
        trans_layers = []
        filters = [in_channels, 512, 256, 128, out_channels]
        kernel_sizes = 4
        stride = 2
        padding = 1
        for i in range(len(filters) - 1):
            trans_layers.append(nn.ConvTranspose2d(in_channels=filters[i],
                                                  out_channels=filters[i + 1],
                                                  kernel_size=kernel_sizes,
                                                  stride=stride,
                                                  padding=padding))
            if i < len(filters) - 2:
                trans_layers.append(nn.BatchNorm2d(filters[i + 1]))
                trans_layers.append(nn.LeakyReLU(0.2))
                trans_layers.append(nn.Dropout(0.25))  # Added Dropout for regularization
        self.decoder_dropout = nn.Dropout(0.25)  # Additional Dropout layer
        trans_layers.append(nn.Tanh())  
        modules.extend(trans_layers)
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        # Define separate linear layers for mean and log variance with different initialization
        self.fc_mu = nn.Linear(n_features, z_dim)
        self.fc_logvar = nn.Linear(n_features, z_dim)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.constant_(self.fc_mu.bias, 0)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, 0)

        # Define a linear layer to transform latent space back to feature space
        self.fc_z = nn.Linear(z_dim, n_features)
        nn.init.xavier_uniform_(self.fc_z.weight)
        nn.init.constant_(self.fc_z.bias, 0)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        # Pass input through encoder and flatten
        encoded = self.features_encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)

        # Compute mean and log variance
        mu = self.fc_mu(encoded_flat)
        log_sigma2 = self.fc_logvar(encoded_flat)

        # Reparameterization trick
        std = torch.exp(0.5 * log_sigma2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer
        #  2. Apply features decoder
        # ====== YOUR CODE: ======
        # Transform latent vector back to feature space
        transformed_z = self.fc_z(z)
        reshaped_z = transformed_z.view(-1, *self.features_shape)

        # Decode to reconstruct the image
        x_rec = self.features_decoder(reshaped_z)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            samples = torch.randn(n, self.z_dim).to(device)
            samples = self.decode(samples)
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    # Get input dimensions
    dx = torch.prod(torch.tensor(x.shape[1:]))

    # Data reconstruction loss using norm
    reconstruction_error = torch.norm((x - xr).reshape(x.shape[0], -1), dim=1)
    data_loss = torch.mean(reconstruction_error ** 2) / (dx * x_sigma2)

    # KL divergence loss using norm for mu term
    kldiv_loss = torch.mean(
        torch.exp(z_log_sigma2).sum(1) + 
        torch.norm(z_mu, dim=1) ** 2 - 
        z_mu.shape[1] - 
        z_log_sigma2.sum(1)
    )

    # Total loss
    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
