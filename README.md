# <img src='assets/logo.webp' style="width: 6%;"> TacticExpert: Spatial-Temporal Graph Language Model for Basketball Tactics

A PyTorch implementation for the paper: 

>**TacticExpert: Spatial-Temporal Graph Language Model for Basketball Tactics**<br />
>Anonymous Author<br />

<p align="center">
<img src="assets/framework.svg" style="width: 80%;"/>
</p>

In this paper, we propose **TacticExpert**, a spatial-temporal graph language model for basketball tactics. This model explicitly captures delay effects in the spatial space to enhance player node representations across discrete time slices, employing symmetry-invariant priors to guide the attention mechanism. We also introduce an efficient contrastive learning strategy to train a Mixture of Tactics Experts module, facilitating differentiated modeling of offensive tactics. By integrating dense training with sparse inference, we achieve a 2.4x improvement in model efficiency. Moreover, the incorporation of Lightweight Graph Grounding for Large Language Models enables robust performance in open-ended downstream tasks and zero-shot scenarios, including novel teams or players. 


## Environments
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

# Install pyg and pyg-relevant packages
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# Install the required libraries
pip install -r requirements.txt
```

## Code structure
```
TacticExpert/
├── src/
│   ├── main.py                            # two-stage training pipeline
│   ├── params.py                          # parameters
│   ├── data/
│   │   ├── data_loader.py                 # data loader
│   │   └── basketball-instants-dataset/   # original data
│   ├── model/
│   │   ├── moe.py                         # Mixture of Experts
│   │   ├── st_encoder.py                  # spatial-temporal graph encoder
│   │   ├── st_graph_transformer.py        # graph transformer module
│   │   └── text_graph_grounding.py        # text-graph contrastive learning module
│   └── utils/
│       └── TimeLogger.py     
├── assets/                  
└── requirements.txt                       # env
```

## Reproduct training pipeline
### Step 1: Prepare the pre-trained base model and meta data
TacticExpert is trained based on `GraphGPT`. Please download its weights [here](https://huggingface.co/Jiabin99/GraphGPT-7B-mix-all/tree/main) and place it in the `./src/checkpoints` folder.

Then you can run the following command to download the meta data from [Kaggle](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset) and place it in the `./src/data` folder:

```bash
cd ./src/data
kaggle datasets download deepsportradar/basketball-instants-dataset
unzip ./basketball-instants-dataset.zip -d .
```

### Step 2: Preprocess graph data and perform multi-modal data augmentation
```python
from data.data_loader import BasketballDataset
dataset = BasketballDataset(data_dir="./src/data/basketball-instants-dataset")
multi_modal_data_augmentation(dataset)
```

### Step 3: Train mixture of tactic experts and spatial-temporal graph encoder

```python
python main.py --stage 1 \
    --num_experts 5 \
    --hidden_dim 256 \
    --lr 2e-4 \
    --epochs 100
```

### Step 4: Train mixture of tactic experts and text-graph grounding

```python
python main.py --stage 2 \
    --load_model "path/to/checkpoint.pt" \ # path to the checkpoint of stage 1
    --lr 1e-4 \
    --epochs 50
```

## Main results
| Supervised Downstream Tasks | Evaluation | TacticExpert | -MoE          | -Delay | -Group | -PE    | -Lap          | -CLIP  | 
| --------------------------- | ---------- | ------------ | ------------- | ------ | ------ | ------ | ------------- | ------ |
| Node Classification         | Macro-F1   | **0.8333**   | 0.7303        | 0.6379 | 0.6667 | 0.7588 | <u>0.7903</u> | 0.5304 |
| Link Prediction             | AUC        | **0.7264**   | <u>0.6788</u> | 0.5758 | 0.5408 | 0.6677 | 0.5301        | 0.6650 |
| Graph Classification        | Macro-F1   | **0.6750**   | 0.4760        | 0.5214 | 0.5786 | 0.6208 | <u>0.6400</u> | 0.5363 |


## Acknowledgements
You may refer to the related works that serves as foundations for TacticExpert, 
[Vicuna](https://github.com/lm-sys/FastChat) and [GraphGPT](https://github.com/HKUDS/GraphGPT). Thanks for their great works.
