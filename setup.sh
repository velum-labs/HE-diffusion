uv sync
uv pip install git+https://github.com/openai/CLIP.git
uv pip install tenseal
uv pip install torch torchvision . --index-url https://download.pytorch.org/whl/cu118

apt-get install -y libsm6 libxext6 libxrender1 libglib2.0-0 libgl1

python scripts/enc_txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms