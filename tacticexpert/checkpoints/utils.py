from huggingface_hub import hf_hub_download, list_repo_files
import os

repo_id = "Jiabin99/GraphGPT-7B-mix-all"
files = list_repo_files(repo_id=repo_id, repo_type="model")

local_dir = "./tacticexpert/checkpoints/GraphGPT-7B-mix-all"
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

for file in files:
    file_path = os.path.join(local_dir, file)
    if not os.path.exists(file_path):
        file_path = hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)
