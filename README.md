# Extending Robust Density Estimation in Variational Autoencoders with Hybrid Heavy-Tailed and Non-Gaussian Priors

This project explores the use of **Laplace** and **Cauchy distributions** as priors in Variational Autoencoders (VAEs), extending the conventional Gaussian framework to better handle **heavy-tailed data** and **non-Gaussian distributions**. By incorporating these robust priors, the model aims to improve density estimation and representation learning, particularly in scenarios with outliers or heavy-tailed noise.

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Methodology](#methodology)
  - [Laplace Distribution in VAEs](#laplace-distribution-in-vaes)
  - [Cauchy Distribution in VAEs](#cauchy-distribution-in-vaes)
- [Implementation](#implementation)
  - [KL Divergence Formulations](#kl-divergence-formulations)
  - [Loss Functions](#loss-functions)
- [Experiments](#experiments)
  - [Datasets](#datasets)
  - [Results](#results)

---

## Introduction
Variational Autoencoders (VAEs) traditionally rely on Gaussian priors for latent space modeling. However, Gaussian distributions can struggle with data that exhibits heavy tails, outliers, or non-Gaussian characteristics. To address these limitations, this project integrates **Laplace** and **Cauchy distributions** as priors in VAEs, offering greater robustness and flexibility in density estimation.

## Motivation
Heavy-tailed and non-Gaussian data are prevalent in real-world applications such as:
- Financial time series analysis
- Medical imaging data with noise
- Anomaly detection

By extending VAEs with non-Gaussian priors, the model becomes better suited for these challenging datasets.

## Methodology

### Laplace Distribution in VAEs
The **Laplace distribution** is characterized by its sharper peak and heavier tails compared to a Gaussian distribution. Its usage as a posterior \( q(Z|X) \sim \text{Laplace}(\mu, b) \) introduces robustness to outliers.

#### KL Divergence for Laplace Prior
The KL divergence between the Laplace posterior and the Gaussian prior is derived as:
\[
\text{KL}(q(Z|X) || p(Z)) = -\log(2b) - 1 + \frac{\mu^2 + 2b^2}{2} + \frac{1}{2}\log(2\pi).
\]

### Cauchy Distribution in VAEs
The **Cauchy distribution** has even heavier tails than the Laplace distribution, making it particularly useful for extreme outlier scenarios. The posterior is defined as \( q(Z|X) \sim \text{Cauchy}(\mu, \gamma) \).

#### KL Divergence for Cauchy Prior
The KL divergence between the Cauchy posterior and the Gaussian prior is numerically approximated due to the intractability of exact computation:
\[
\text{KL}(q(Z|X) || p(Z)) \approx \mathbb{E}_{q(Z|X)} \left[ \frac{Z^2}{2} + \log(\sqrt{2\pi}) - \log(1 + \frac{(Z - \mu)^2}{\gamma^2}) \right].
\]

## Implementation

### KL Divergence Formulations
Custom PyTorch functions are implemented to compute KL divergence for Laplace and Cauchy distributions:
- **Laplace KL Divergence**:
```python
# Laplace KL Divergence
def kl_divergence(mu, log_b):
    b = torch.exp(log_b)
    kl = -log_b - math.log(2) - 1 + 0.5 * (mu.pow(2) + 2 * b.pow(2)) + 0.5 * math.log(2 * math.pi)
    return torch.sum(kl, dim=1)
```

- **Cauchy KL Divergence**:
```python
# Cauchy KL Divergence
def kl_divergence(mu, log_gamma):
    gamma = torch.exp(log_gamma)
    log_cauchy_term = -log_gamma - math.log(math.pi) - torch.log1p(((mu / gamma)**2))
    log_gaussian_term = 0.5 * (mu**2 + gamma**2) + 0.5 * math.log(2 * math.pi)
    kl = log_gaussian_term - log_cauchy_term
    return torch.sum(kl, dim=1)
```

### Loss Functions
The VAE loss combines the reconstruction loss and KL divergence. For Laplace and Cauchy distributions:
\[
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{reconstruction}} + \text{KL}(q(Z|X) || p(Z)).
\]

## Experiments

### Datasets
The model is evaluated on:
- SMTP dataset

### Results
- **Laplace Prior**: Enhanced robustness to moderate outliers.
- **Cauchy Prior**: Superior performance on extreme outlier data but requires intensive training for the stabilisation.
