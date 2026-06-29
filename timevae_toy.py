# run_conv_vae_clean.py
# Single-file runnable Conv-VAE for multivariate time series
# Requirements: torch, numpy, matplotlib, scikit-learn, joblib
# Run: python run_conv_vae_clean.py

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# -----------------------------
# Sampling and Base VAE
# -----------------------------
class Sampling(nn.Module):
    def forward(self, mu_logvar_tuple):
        mu, logvar = mu_logvar_tuple
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class BaseVAE(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar, rec_weight=1.0, kl_weight=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec_weight * recon_loss + kl_weight * kld, recon_loss.item(), kld.item()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, map_location=DEVICE):
        self.load_state_dict(torch.load(path, map_location=map_location))


# -----------------------------
# Conv Encoder / Decoder
# -----------------------------
class ConvEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_sizes, latent_dim):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_ch = feat_dim
        for h in hidden_sizes:
            self.conv_layers.append(nn.Conv1d(in_ch, h, 3, stride=2, padding=1))
            in_ch = h
        self.encoder_last_dense_dim = self._get_dense_dim(seq_len, feat_dim)
        self.z_mean = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.z_logvar = nn.Linear(self.encoder_last_dense_dim, latent_dim)
        self.sampling = Sampling()

    def _get_dense_dim(self, seq_len, feat_dim):
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            for conv in self.conv_layers:
                x = conv(x)
            return x.numel()

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, D, T)
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.flatten(1)
        mu, logvar = self.z_mean(x), self.z_logvar(x)
        z = self.sampling((mu, logvar))
        return mu, logvar, z


class ConvDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_sizes, latent_dim, encoder_dense_dim):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_sizes = hidden_sizes
        self.dense = nn.Linear(latent_dim, encoder_dense_dim)
        hidden_last = hidden_sizes[-1]
        L_encoded = encoder_dense_dim // hidden_last
        self.encoded_L = L_encoded

        # deconv layers
        self.deconv_layers = nn.ModuleList()
        rev = list(reversed(hidden_sizes))
        in_ch = rev[0]
        for out_ch in rev[1:]:
            self.deconv_layers.append(nn.ConvTranspose1d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1))
            in_ch = out_ch
        self.deconv_layers.append(nn.ConvTranspose1d(in_ch, feat_dim, 3, stride=2, padding=1, output_padding=1))

        # compute final flattened dimension
        with torch.no_grad():
            tmp = torch.randn(1, hidden_last, L_encoded)
            for d in self.deconv_layers:
                tmp = d(tmp)
            final_flat = feat_dim * tmp.shape[-1]
        self.final_dense = nn.Linear(final_flat, seq_len * feat_dim)

    def forward(self, z):
        B = z.size(0)
        x = F.relu(self.dense(z)).view(B, self.hidden_sizes[-1], self.encoded_L)
        for deconv in self.deconv_layers[:-1]:
            x = F.relu(deconv(x))
        x = F.relu(self.deconv_layers[-1](x))
        x = x.flatten(1)
        x = self.final_dense(x)
        return x.view(B, self.seq_len, self.feat_dim)


class VariationalAutoencoderConv(BaseVAE):
    def __init__(self, seq_len, feat_dim, latent_dim=16, hidden_sizes=None):
        super().__init__(seq_len, feat_dim, latent_dim)
        if hidden_sizes is None:
            hidden_sizes = [32, 64, 128]
        self.hidden_sizes = hidden_sizes
        self.encoder = ConvEncoder(seq_len, feat_dim, hidden_sizes, latent_dim)
        self.decoder = ConvDecoder(seq_len, feat_dim, hidden_sizes, latent_dim, self.encoder.encoder_last_dense_dim)

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    @staticmethod
    def save_params(params, path): joblib.dump(params, path)


# -----------------------------
# Data & Utils
# -----------------------------
def generate_sin_dataset(N=2048, T=64, D=3, noise=0.05):
    t = np.linspace(0, 8, T)
    data = []
    for _ in range(N):
        freqs = np.random.uniform(0.5, 2.5, D)
        phases = np.random.uniform(0, 2*np.pi, D)
        amps = np.random.uniform(0.5, 1.5, D)
        series = np.stack([amps[d]*np.sin(freqs[d]*t + phases[d]) for d in range(D)], axis=1)
        series += noise * np.random.randn(*series.shape)
        data.append(series)
    return np.stack(data).astype(np.float32)


def train(model, loader, epochs=20, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        model.train()
        total_loss = 0
        for xb in loader:
            x = xb[0].to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar, z = model(x)
            loss, _, _ = model.loss_function(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {e+1:03d} loss={total_loss/len(loader.dataset):.6f}")
    return model


def plot_recon(model, loader, n=4, feat=0, path=None):
    model.eval()
    x = next(iter(loader))[0].to(DEVICE)
    with torch.no_grad():
        recon, _, _, _ = model(x)
    x, recon = x.cpu().numpy(), recon.cpu().numpy()
    plt.figure(figsize=(10, 2.5*n))
    idxs = np.random.choice(x.shape[0], n, replace=False)
    t = np.arange(x.shape[1])
    for i, idx in enumerate(idxs):
        plt.subplot(n,1,i+1)
        plt.plot(t, x[idx,:,feat], label="original")
        plt.plot(t, recon[idx,:,feat], linestyle="--", label="recon")
        plt.legend()
    plt.tight_layout()
    if path: plt.savefig(path, dpi=200)
    else: plt.show()


def plot_latent_tsne(mus, path=None):
    z2 = TSNE(n_components=2, random_state=42).fit_transform(mus)
    plt.figure(figsize=(7,6))
    plt.scatter(z2[:,0], z2[:,1], s=8, alpha=0.7)
    plt.title("Latent (mu) t-SNE")
    if path: plt.savefig(path, dpi=200)
    else: plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    seq_len, feat_dim, latent_dim = 64, 3, 16
    hidden_sizes, batch_size, epochs, lr = [32,64,128], 128, 20, 1e-3
    out_dir = "outputs_conv_vae"
    os.makedirs(out_dir, exist_ok=True)

    # data
    data = generate_sin_dataset(N=2048, T=seq_len, D=feat_dim)
    n_train = int(len(data)*0.8)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(data[:n_train])), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(data[n_train:])), batch_size=batch_size, shuffle=False)

    # model
    model = VariationalAutoencoderConv(seq_len, feat_dim, latent_dim, hidden_sizes)
    VariationalAutoencoderConv.save_params(
        dict(seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dim, hidden_sizes=hidden_sizes),
        os.path.join(out_dir, "vae_params.pkl")
    )

    # train & save
    model = train(model, train_loader, epochs, lr)
    model.save_weights(os.path.join(out_dir, "vae_state.pth"))

    # visualize
    mus = []
    model.eval()
    with torch.no_grad():
        for xb in test_loader:
            mu, _, _ = model.encode(xb[0].to(DEVICE))
            mus.append(mu.cpu().numpy())
    mus = np.concatenate(mus)
    plot_recon(model, test_loader, path=os.path.join(out_dir,"recon.png"))
    plot_latent_tsne(mus, path=os.path.join(out_dir,"latent_tsne.png"))
    print("Done! Outputs saved to", out_dir)


if __name__ == "__main__":
    main()
