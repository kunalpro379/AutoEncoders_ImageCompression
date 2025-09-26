# Image Compression using Autoencoders & Variational Autoencoders

A comprehensive project exploring neural network-based image compression techniques using Autoencoders (AE) and Variational Autoencoders (VAE).

## 📌 Theory

### Autoencoders (AE)
- Neural networks trained to reconstruct input data
- **Encoder** compresses the input into a latent representation
- **Decoder** reconstructs the data from this compressed form
- **Deterministic**: same input → same latent code → same reconstruction
- Good for direct compression but latent space may not be continuous or structured

### Variational Autoencoders (VAE)
- A probabilistic extension of autoencoders
- Encoder outputs a **distribution** (mean & variance), not a fixed latent code
- Latent space is smooth and continuous — good for interpolation and sampling
- Reconstructions may be blurrier due to random sampling, but representation is more robust
- Useful for compression + generative modeling

## 🔍 Key Differences in Compression


| Aspect | Autoencoder (AE) | Variational Autoencoder (VAE) |
|--------|------------------|-------------------------------|
| **Latent Code** | Deterministic vector z | Distribution with μ, σ |
| **Loss** | MSE: \|\|x - x̂\|\|² | ELBO: Reconstruction + KL divergence |
| **Compression Quality** | Sharper, high-fidelity | Slightly blurrier |
| **Latent Space** | May be irregular | Smooth, continuous, structured |
| **Extra Benefit** | Direct compression | Compression + generation |




### Autoencoder (AE)

An encoder f_θ(x) maps input x to a latent code z.

A decoder g_φ(z) reconstructs the input.

```
z = f_θ(x),  x̂ = g_φ(z)
```

**Loss function (Reconstruction Loss):**

```
L_AE(x, x̂) = ||x - x̂||²
```

This enforces that the reconstructed image x̂ is close to the original x.

### Variational Autoencoder (VAE)

Instead of mapping to a fixed latent code, the encoder outputs a distribution:

```
q_φ(z|x) = N(z; μ(x), σ²(x)I)
```

A latent sample is drawn using the reparameterization trick:

```
z = μ(x) + σ(x) ⊙ ε,  ε ~ N(0, I)
```

The decoder reconstructs from z:

```
x̂ = g_θ(z)
```

**Loss function (Evidence Lower Bound, ELBO):**

```
L_VAE(x, x̂) = E_q_φ(z|x)[-log p_θ(x|z)] + D_KL(q_φ(z|x) || p(z))
```

where:
- First term: **Reconstruction Loss**
- Second term: **Regularization** (KL divergence)
- p(z) = N(0, I) is the prior


## 📊 Notebooks & Insights

### `ImageCompression_AutoEncoderModel.ipynb`
- Implements a standard Autoencoder for image compression
- Reconstructions are sharp and closely match input images

### `VAE_ImageCompression.ipynb`
- Implements a Variational Autoencoder
- Reconstructions are slightly blurrier, but the model learns a smooth latent space

### `ImageCompression.ipynb`
- Supporting notebook for preprocessing and baseline compression experiments

## 🔥 Key Takeaways

- **AE** = better for sharp, high-fidelity compression
- **VAE** = better for structured latent space & generative tasks, but with some quality trade-off


## 📈 Expected Results

### Autoencoder Results
- High-fidelity reconstructions
- Sharp image quality
- Direct compression ratio control

### VAE Results
- Smooth latent space representation
- Generative capabilities
- Slightly blurrier but more robust reconstructions
