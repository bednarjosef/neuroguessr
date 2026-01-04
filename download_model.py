import os
from huggingface_hub import hf_hub_download

os.makedirs('models', exist_ok=True)
hf_hub_download(repo_id="josefbednar/neuroguessr-861-large-acw-streetview-h3-unfrozen", repo_type='model', filename="neuroguessr-861-large-acw-streetview-h3-unfrozen-2-best.pth", local_dir='models')
