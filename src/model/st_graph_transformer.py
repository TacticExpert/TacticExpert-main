import torch as t
from torch import nn
import torch.nn.functional as F
from params import args
import numpy as np
from torch.nn import MultiheadAttention

class GTLayer(nn.Module):
    def __init__(self, latent_dim, num_heads, dropout=0.1):
        super(GTLayer, self).__init__()
        self.multi_head_attention = MultiheadAttention(latent_dim, num_heads, dropout=dropout, bias=False)
        self.dense_layers = nn.Sequential(
            FeedForwardLayer(latent_dim, latent_dim),
            FeedForwardLayer(latent_dim, latent_dim)
        )
        self.layer_norm1 = nn.LayerNorm(latent_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.latent_dim = latent_dim
        self.num_anchors = 32
    
    def _pick_anchors(self, embeds):
        perm = t.randperm(embeds.shape[0])
        anchors = perm[:self.num_anchors]
        return embeds[anchors]
   
    def forward(self, embeds):
        anchor_embeds = self._pick_anchors(embeds)
        _anchor_embeds, _ = self.multi_head_attention(anchor_embeds, embeds, embeds)
        anchor_embeds = _anchor_embeds + anchor_embeds
        
        _embeds, _ = self.multi_head_attention(embeds, anchor_embeds, anchor_embeds)
        embeds = self.layer_norm1(_embeds + embeds)

        _embeds = self.dropout(self.dense_layers(embeds))
        embeds = self.layer_norm2(_embeds + embeds)
        return embeds

class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=True):
        super(FeedForwardLayer, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.act = nn.ReLU()
    
    def forward(self, embeds):
        return self.act(self.linear(embeds))

class raphTransformerG(nn.Module):
    def __init__(self, latent_dim, num_layers=3, num_heads=8, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.gt_layers = nn.ModuleList([
            GTLayer(latent_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        self.scale = np.sqrt(num_layers) 

    def forward(self, embeds):
        for layer in self.gt_layers:
            embeds = layer(embeds) / self.scale
        return embeds