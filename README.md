# <img src='assets/TacticExpert_logo.webp' style="width: 6%"> TacticExpert: Spatial-Temporal Graph Language Model for Basketball Tactics

A PyTorch implementation for the paper: 

>**TacticExpert: Spatial-Temporal Graph Language Model for Basketball Tactics**<br />
>Anonymous Author<br />

<p align="center">
<iframe src="assets/framework.pdf" width="800px" height="600px"></iframe>
</p>

In this paper, we propose **TacticExpert**, a spatial-temporal graph language model for basketball tactics. This model explicitly captures delay effects in the spatial space to enhance player node representations across discrete time slices, employing symmetry-invariant priors to guide the attention mechanism. We also introduce an efficient contrastive learning strategy to train a Mixture of Tactics Experts module, facilitating differentiated modeling of offensive tactics. By integrating dense training with sparse inference, we achieve a 2.4x improvement in model efficiency. Moreover, the incorporation of Lightweight Graph Grounding for Large Language Models enables robust performance in open-ended downstream tasks and zero-shot scenarios, including novel teams or players. 


## Environment
You can run the following command to download the codes faster:

```bash
git clone --depth 1 https://github.com/TacticExpert/TacticExpert-main.git && cd TacticExpert-main
```

Then run the following commands to create a conda environment:

```bash
conda create -y -n tacticexpert python=3.9
conda activate tacticexpert

# Torch with CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu117

# Install vicuna base model
pip3 install "fschat[model_worker,webui]"

# Install pyg and pyg-relevant packages
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# Install the required libraries
pip install -r requirements.txt
```

## Code Structure




## Examples to run the codes








## Acknowledgements
You may refer to the related work that serves as foundations for our framework and code repository, 
[Vicuna](https://github.com/lm-sys/FastChat). We also partially draw inspirations from [GraphGPT](https://github.com/HKUDS/GraphGPT). The design of our system deployment was inspired by [gradio](https://www.gradio.app) and [Baize](https://huggingface.co/spaces/project-baize/chat-with-baize). Thanks for their wonderful works.
