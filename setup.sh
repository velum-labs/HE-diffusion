apt-get update
apt-get install -y libsm6 libxext6 libxrender1 libglib2.0-0 libgl1

git config --global user.email "lclc.alen@gmail.com"
git config --global user.name "Alen Rubilar"

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync
uv pip install git+https://github.com/openai/CLIP.git
uv pip install tenseal
uv pip install torch torchvision . --index-url https://download.pytorch.org/whl/cu118

source .venv/bin/activate
# python scripts/enc_txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms
