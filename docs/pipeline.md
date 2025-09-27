## End-to-End Pipeline (enc_txt2img.py → ENC_PLMSSampler)

This document walks through the runtime path and pinpoints where homomorphic encryption (TenSEAL CKKS) is used.

### 1) Entry script and setup
- Parse args, seed, load config and checkpoint, default device `cpu` for the model.
- Select sampler. When `--plms` is set, the HE-enabled sampler `ENC_PLMSSampler` is used; otherwise, `DDIMSampler` or `DPMSolverSampler`.
- A TenSEAL context is created in the script (not used directly in sampling), while the sampler constructs its own CKKS context for per-step updates.

### 2) Conditioning computation
- The model briefly moves to GPU to compute text conditioning `c` (and optionally `uc` for guidance), then returns results to CPU.
- The commented lines `#enc_c = ts.ckks_tensor(context, c)` and `#enc_uc = ts.ckks_tensor(context, uc)` show a possible (inactive) path to encrypt conditionings.

### 3) Sampling call
- The script calls `sampler.sample(...)` with the number of steps, latent shape, conditioning, and guidance scale.
- The sampler prepares the PLMS schedule and begins iterating over timesteps.

### 4) HE-enabled PLMS sampler internals
- The sampler (`ldm/models/diffusion/enc_plms.py`) creates a CKKS context:
  - `poly_modulus_degree = 8192`
  - `bits_scale = 26` → `global_scale = 2**26`
  - `coeff_mod_bit_sizes = [31, 26, 26, 26, 26, 26, 26, 31]`
  - Generates Galois keys (useful for rotations; not heavily used by the current sparse vector path).

- Two pathways exist:
  1. **Sparse path (default in code)**
     - Build a sparsified latent `new_image = remove_points(img_cpu, threshold=0.01)` using `ldm/distortion.py`.
     - Split into `remain_img = img_cpu - new_image` and a COO sparse `coo_img = convert_dense_to_coo(new_image)`.
     - Encrypt the sparse values: `coo_img.encrypt(context)`; values become a TenSEAL `CKKSVector`.
     - Call `p_sample_plms_sp(...)` each step to compute encrypted updates for the sparse portion and plaintext updates for the remainder.
     - Decrypt and merge: `x_prev = coo_x_prev.decrypt().merge_tensor(remain_x_prev)` to produce the next latent.

  2. **Non-sparse path (disabled by an in-function flag)**
     - Encrypt the full latent `img` as a CKKS tensor/vector.
     - Compute the PLMS update under encryption, then decrypt back to plaintext for the next step.

- The PLMS update uses the standard pseudo linear multistep formula on the noise prediction `e_t` (computed by the plaintext U-Net):
  - Predict `e_t = model.apply_model(x_t, t, cond)` (or guided version with `unconditional_guidance_scale`).
  - Compute a multistep estimate `e_t_prime` from the current and previous `e_t` values.
  - Update latent with
    - Plaintext form: `x_{t-1} = sqrt(a_prev)*x0 + sqrt(1-a_prev-σ^2)*e_t_prime + σ*ε`
    - Encrypted sparse form: operate on encrypted values with scalar multiply and addition, carrying the remainder in plaintext.

### 5) Encrypted COO arithmetic
- `COOSparseTensor.values` are stored as a `CKKSVector` after `encrypt(context)`.
- Overloaded ops allow mixing with plaintext:
  - `factor * coo_x` multiplies encrypted values by a plaintext scalar.
  - `coo_x + add_part` adds a plaintext tensor’s selected entries to encrypted values.
- These are CKKS-friendly operations and avoid costly ciphertext-ciphertext multiplications where possible.

### 6) Decoding, safety check, and saving
- After sampling finishes, `model.decode_first_stage(latents)` decodes to images.
- Images are clamped to [0,1], optionally checked by the safety checker, watermarked, and saved to `outputs/...`.

### Notes on data movement
- Conditionings `c`/`uc` are computed on GPU then moved to CPU.
- The sampling loop maintains the latent on CPU for HE operations, with only model forward calls invoking the U-Net in plaintext.
