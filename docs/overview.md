## Overview

HE-Diffusion integrates approximate homomorphic encryption (CKKS via TenSEAL) into parts of the Stable Diffusion sampling loop to protect intermediate latent representations during inference. The entrypoint is `scripts/enc_txt2img.py`, which loads a Stable Diffusion model, chooses an HE-enabled sampler when `--plms` is used, and runs sampling while selectively encrypting portions of the latent state.

### What is protected
- **Encrypted latent updates (sparse by default)**: During PLMS sampling, a subset of latent elements are encrypted and updated homomorphically using CKKS-friendly operations (addition and scalar multiplication). The encrypted subset is represented by `ldm/coo_sparse.py:COOSparseTensor` whose `values` are a TenSEAL `CKKSVector`.

### What is not protected (current state)
- **Text conditioning**: The prompt, tokenizer output, and text-conditioning vectors `c` and `uc` are computed in plaintext. `scripts/enc_txt2img.py` contains commented lines hinting at how to encrypt `c`/`uc`, but they are not active.
- **Model forward**: The U-Net forward used to predict the noise `e_t` remains plaintext. Encryption focuses on the latent state update formula, not the full neural network.

### Key components
- `scripts/enc_txt2img.py`: argument parsing, model loading, sampler selection, sampling loop, decoding, and safety check.
- `ldm/models/diffusion/enc_plms.py`: HE-enabled PLMS sampler. Creates a TenSEAL CKKS context and runs encrypted/sparse latent updates per timestep.
- `ldm/coo_sparse.py`: `COOSparseTensor` with encrypted values (`CKKSVector`) and overloaded arithmetic to mix plaintext tensors/scalars with encrypted vectors.
- `ldm/distortion.py`: heuristics to sparsify the latent by zeroing low-impact elements (reduces ciphertext size and runtime).

### Threat model (informal)
- **Goal**: Hide parts of the latent state (and therefore partial information about the final image) from an untrusted compute provider during sampling.
- **Non-goals (today)**: Hiding the prompt or text encoder internals; fully HE U-Net forward; full-image HE decoding. These would require substantially more HE support and are out of scope for this implementation.
