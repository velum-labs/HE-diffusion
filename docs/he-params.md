## HE Parameters and Security Considerations

This project uses the CKKS approximate homomorphic encryption scheme via TenSEAL. Two contexts appear:

- In `scripts/enc_txt2img.py` (not directly used by the sampler):
  - `poly_modulus_degree = 16384`
  - `coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 60]`
  - `global_scale = 2**40`

- In `ldm/models/diffusion/enc_plms.py` (used during sampling each step):
  - `poly_modulus_degree = 8192`
  - `bits_scale = 26` → `global_scale = 2**26`
  - `coeff_mod_bit_sizes = [31, 26, 26, 26, 26, 26, 26, 31]`
  - Galois keys are generated (useful for rotations; the current sparse vector path does not make heavy use of them).

### Why two contexts?
The script-level context illustrates a high-precision, larger modulus configuration. The sampler uses a lighter configuration to keep per-step encryption/decryption and arithmetic feasible in a tight loop.

### CKKS precision vs performance
- Larger `poly_modulus_degree` and coefficient moduli increase ciphertext size and cost but allow deeper circuits and higher precision.
- `global_scale` sets the fixed-point precision of fractional values. `2**26` is a practical compromise for the linear updates performed here.

### Supported operations
- The encrypted path primarily uses:
  - Ciphertext + Plaintext addition
  - Plaintext scalar × Ciphertext multiplication
- These operations are fast in CKKS and sufficient for the PLMS update on the sparse subset. Avoiding ciphertext×ciphertext multiplication reduces noise growth and runtime.

### Security notes
- CKKS is approximate: ciphertexts decrypt to values with small numerical error. This is expected.
- Parameter selection impacts both security and precision. Standard HE guidance suggests `poly_modulus_degree >= 8192` for ~128-bit security depending on the modulus chain. For production, validate with an HE parameter selection tool (e.g., SEAL’s) based on the exact operation depth.
- The current pipeline encrypts parts of the latent, not the entire model forward or conditioning. Leakage through plaintext components is outside HE’s protection boundary here.

### Tuning guidance
- If you increase the encrypted coverage (lower sparsification threshold), consider increasing `poly_modulus_degree` or the modulus chain and `global_scale` accordingly to maintain precision across more operations.
- Keep an eye on runtime: encryption time is printed in logs per step; it scales with ciphertext size and number of encrypted elements.
