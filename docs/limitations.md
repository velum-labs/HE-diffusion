## Limitations and Next Steps

### Current limitations
- **Partial protection**: Text conditioning (`c`, `uc`), U-Net forward passes (noise prediction `e_t`), and decoding are plaintext. Only parts of the latent update are homomorphically encrypted.
- **Sparse encryption heuristic**: Coverage depends on `remove_points` threshold. This is a heuristic that balances visual impact and runtime but is not a formal privacy guarantee.
- **Decryption per step**: Each step merges encrypted sparse updates back into a plaintext latent. Frequent decrypt/merge cycles may be a side channel if the compute provider observes timing or memory access patterns.
- **No end-to-end HE**: Fully HE U-Net inference, attention, and VAE decoding would require significant algorithmic changes and HE-friendly network designs.

### Potential improvements
- **Encrypt conditioning**: Activate encrypted conditioning vectors and propagate through HE-friendly guidance computation.
- **Reduce decrypt frequency**: Accumulate multiple steps in encrypted space (where possible) before merging back, or design a pipeline that minimizes decrypt cycles.
- **Better sparsification**: Learnable or adaptive schemes that optimize the trade-off between utility and HE cost.
- **Broader HE ops**: Explore limited ciphertext×ciphertext multiplications or rotations where they substantially improve coverage, guided by parameter budgets.
- **Parameter tuning**: Auto-tune CKKS parameters based on image size, step count, and sparsity level to maintain precision with minimal overhead.
