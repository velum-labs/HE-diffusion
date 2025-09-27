## Running the HE Diffusion Pipeline

### Requirements
- All Stable Diffusion dependencies as per the repository `README.md`.
- TenSEAL: `pip install tenseal`
- Local SD safety checker weights (the script uses `local_files_only=True`); ensure your HF cache has
  `CompVis/stable-diffusion-safety-checker` available or remove/adjust the safety checker in the script.

### Models and Configs
- Default config: `configs/stable-diffusion/enc-v1-inference.yaml`
- Default checkpoint: `models/ldm/stable-diffusion-v1/model.ckpt`
Ensure these paths exist on your machine.

### Basic command
```bash
python scripts/enc_txt2img.py \
  --prompt "a photograph of an astronaut riding a horse" \
  --plms \
  --outdir outputs/txt2img-samples
```

### Key arguments
- `--prompt`: Text prompt. Alternatively use `--from-file path.txt` for batched prompts.
- `--ddim_steps`: Number of sampling steps (PLMS also uses this count for its schedule).
- `--scale`: Guidance scale (e.g., 7.5).
- `--H`, `--W`: Image size in pixels (multiples of 64 recommended).
- `--n_samples`: Batch size per prompt.
- `--n_iter`: Number of batches to generate per prompt set.
- `--precision`: `autocast` (default) or `full`.
- `--plms`: Use the HE-enabled PLMS sampler.
- `--dpm_solver`: Use DPM-Solver (plaintext) instead of PLMS.

### Reproducibility
- `--seed`: Fix the global seed.
- `--fixed_code`: Use a fixed starting noise `x_T` across runs (same seed + fixed code → identical latents).

### Performance and HE runtime tips
- The HE path in `ENC_PLMSSampler` uses a sparse encrypted representation by default with `threshold=0.01` for sparsification. Lower thresholds encrypt more elements (more secure, slower); higher thresholds encrypt fewer (faster, less coverage).
- The TenSEAL CKKS parameters in the sampler (`poly_modulus_degree=8192`, `bits_scale=26`) are chosen to balance precision and speed. See `he-params.md` for tuning notes.

### Outputs
- Images are saved under `--outdir` (individual samples and optional grids). Watermarking and the safety checker are enabled by default.
