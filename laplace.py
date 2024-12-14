import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class LaplaceVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super(LaplaceVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_loc = nn.Linear(hidden_dim, input_dim)  
        self.fc_log_scale = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        loc = self.fc_loc(h)
        log_scale = self.fc_log_scale(h)
        return loc, log_scale

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        loc, log_scale = self.decode(z)
        return loc, log_scale, mu, logvar
    
def nll_loss_laplace(x, loc, log_scale):
    scale = torch.exp(log_scale)
    nll = torch.abs(x - loc) / scale + log_scale + math.log(2)  
    return nll.sum(dim=1)

def kl_divergence(mu, log_b):
    b = torch.exp(log_b)  
    kl = -log_b - math.log(2) - 1 + 0.5 * (mu.pow(2) + 2 * b.pow(2)) + 0.5 * math.log(2 * math.pi)
    return torch.sum(kl, dim=1)  


def compute_log_likelihood(model, dataloader, device):
    model.eval()
    total_nll = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            loc, log_scale, mu, logvar = model(x)

            nll = nll_loss_laplace(x, loc, log_scale)
            kl = kl_divergence(mu, logvar)
            loss = nll + kl

            total_nll += loss.sum().item()
            total_samples += x.size(0)

    avg_log_likelihood = -total_nll / total_samples  
    return avg_log_likelihood

def train_vae(model, train_loader, valid_loader, optimizer, num_epochs=20, device="cpu"):
    model.to(device)
    model.train()

    train_losses = []
    valid_log_likelihoods = []
    kl_divergences = []
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            loc, log_scale, mu, logvar = model(x)
            nll = nll_loss_laplace(x, loc, log_scale).mean()
            kl = kl_divergence(mu, logvar).mean()
            loss = nll + kl
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            kl_divergences.append(kl.item()) 

        avg_train_loss = total_loss / len(train_loader)
        avg_valid_log_likelihood = compute_log_likelihood(model, valid_loader, device)

        train_losses.append(avg_train_loss)
        valid_log_likelihoods.append(avg_valid_log_likelihood)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Log-Likelihood: {avg_valid_log_likelihood:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(kl_divergences, label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence over Epochs")
    plt.legend()
    plt.show()
    return train_losses, valid_log_likelihoods

def plot_metrics(train_losses, valid_log_likelihoods):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_log_likelihoods, label='Validation Log-Likelihood')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training and Validation Metrics')

def load_smtp_dataset(seed=42):
    dataset = fetch_kddcup99(subset="smtp", percent10=False, random_state=42)
    data = dataset.data
    split_index = int(data.shape[0] * 0.1)
    X_train = data[:split_index]
    X_test = data[split_index:]

    valid_index = int(X_test.shape[0] * 0.1)
    X_valid = X_test[:valid_index]
    X_test = X_test[valid_index:]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return X_train, X_valid, X_test

if __name__ == "__main__":
    X_train, X_valid, X_test = load_smtp_dataset()
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=64, shuffle=True)
    valid_loader = DataLoader(TensorDataset(torch.tensor(X_valid, dtype=torch.float32)), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=64, shuffle=False)
    vae = LaplaceVAE(input_dim=X_train.shape[1], latent_dim=2)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    train_losses, valid_log_likelihoods = train_vae(vae, train_loader, valid_loader, optimizer, num_epochs=10, device="cuda")
    plot_metrics(train_losses, valid_log_likelihoods)
    test_log_likelihood = compute_log_likelihood(vae, test_loader, device="cuda")
    print(f"Test Log-Likelihood: {test_log_likelihood:.4f}")
    plt.show()
