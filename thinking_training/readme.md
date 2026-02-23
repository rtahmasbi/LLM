conda create -n env_thinking python=3.11
conda activate env_thinking

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets peft bitsandbytes sentencepiece

