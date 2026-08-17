import torch.nn.functional as F
import torch


def vae_loss(target, reconstruction, mu, log_var, beta=1.0):

    # Loss = MSE + Beta * D_KL
    reconstruction_loss = F.binary_cross_entropy(reconstruction, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    batch_size = reconstruction_loss.size(0)

    reconstruction_loss /= batch_size
    kl_loss /= batch_size

    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss
