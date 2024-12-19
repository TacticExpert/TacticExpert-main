import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from tslearn.clustering import KShape
from proj_g1.TacticExpert.src.model.st_graph_transformer import TopoEncoder, GTLayer


class STEncoder(nn.Module):
    def __init__(self, hidden_dim=256, num_players=10, dropout=0.1):
        super().__init__()
        
        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.topo_encoder = nn.ModuleList([
            TopoEncoder() for _ in range(2)
        ])
        self.temporal_transformer = nn.ModuleList([
            GTLayer() for _ in range(3)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, data):
        """
        Args:
            data: Data instance:
                - x: [batch_size, num_players, time_steps, 15] 
                - edge_index: [2, num_edges]
                - edge_attr: [num_edges, 1]
        Returns:
            output: [batch_size, time_steps, num_players, hidden_dim]
        """
        batch_size = data.x.size(0)
        num_players = data.x.size(1)
        time_steps = data.x.size(2)
        
        x = self.feature_proj(data.x.view(-1, 15)).view(batch_size, num_players, time_steps, -1)
        x = x.permute(0, 2, 1, 3)  # [batch, time, players, hidden]
        
        adj = self._build_adj_matrix(data.edge_index, data.edge_attr, num_players)
        for topo_layer in self.topo_encoder:
            x = topo_layer(adj, x)
        
        for transformer_layer in self.temporal_transformer:
            x = transformer_layer(x)
            
        return x

    def _build_adj_matrix(self, edge_index, edge_attr, num_nodes):
        adj = torch.sparse.FloatTensor(
            edge_index, 
            edge_attr.squeeze(),
            torch.Size([num_nodes, num_nodes])
        )

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = torch.sparse.mm(
            torch.sparse.mm(
                torch.diag(deg_inv_sqrt),
                adj
            ),
            torch.diag(deg_inv_sqrt)
        )
        return adj

class PositionalEncoding(nn.Module):
    """Transformer positional encoding with normalization
    
    Implements sinusoidal position encoding with:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    
    Adds normalization:
    PE_norm = (PE - mean(PE)) / (std(PE) * 10)
    """
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        pe = self._normalize(pe)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def _normalize(self, pe):
        mean = pe.mean()
        std = pe.std()
        return (pe - mean) / (std * 10)

    def forward(self, x):
        """
        Args:
            x: input tensor [batch_size, seq_len, d_model]
        Returns:
            position encoded tensor with same shape
        """
        seq_len = x.size(1)
        assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
        
        # Add positional encoding
        pos_encoding = self.pe[:, :seq_len]
        x = x + pos_encoding
        
        return self.dropout(x)

class LaplacianPE(nn.Module):
    """Laplacian eigenvector positional encoding with spectral filtering"""
    def __init__(self, hidden_dim, num_nodes, k=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        # Number of smallest non-zero eigenvalues to use (default: num_nodes//2)
        self.k = k if k is not None else num_nodes // 2
        # Learnable projection matrix
        self.weight = nn.Parameter(torch.Tensor(self.k, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform"""
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, adj):
        """
        Args:
            adj: [batch_size, num_nodes, num_nodes] adjacency matrix
        Returns:
            pos_enc: [batch_size, num_nodes, hidden_dim] positional encoding
        """
        batch_size = adj.size(0)
        device = adj.device
        
        # Compute degree matrix
        degree = torch.sum(adj, dim=-1)  # [batch_size, num_nodes]
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
        
        # Compute normalized Laplacian: L = I - D^(-1/2)AD^(-1/2)
        identity = torch.eye(self.num_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        normalized_adj = torch.matmul(torch.matmul(degree_inv_sqrt, adj), degree_inv_sqrt)
        laplacian = identity - normalized_adj
        
        # Eigendecomposition
        eigenvals, eigenvecs = torch.linalg.eigh(laplacian)
        
        # Sort eigenvalues and eigenvectors
        sorted_indices = torch.argsort(eigenvals, dim=-1)
        eigenvals = torch.gather(eigenvals, -1, sorted_indices)
        eigenvecs = torch.gather(eigenvecs, -1, sorted_indices.unsqueeze(-2).expand(-1, self.num_nodes, -1))
        
        # Select k smallest non-zero eigenvalues/vectors (skip first zero eigenvalue)
        selected_eigenvecs = eigenvecs[:, :, 1:self.k+1]  # [batch_size, num_nodes, k]
        
        # Generate positional encoding
        pos_enc = torch.matmul(selected_eigenvecs, self.weight)  # [batch_size, num_nodes, hidden_dim]
        
        return pos_enc

class DelayAttention(nn.Module):
    def __init__(self, d_model=256, d_k=128, window_size=10, n_patterns=8):
        """
        params:
            d_model: input feature dimension
            d_k: query/key/value dimension
            window_size: time window size S
            n_patterns: number of delay effect patterns Np
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_k
        self.window_size = window_size
        self.n_patterns = n_patterns

        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        
        self.W_u = nn.Linear(d_model, d_k)
        self.W_m = nn.Linear(d_model, d_k)
        self.W_c = nn.Linear(d_model, d_k)

        self.patterns = nn.Parameter(torch.randn(n_patterns, window_size, d_model))
        
    def extract_patterns(self, x):
        B, T, N, D = x.shape
        windows = []
        for t in range(self.window_size, T):
            window = x[:, t-self.window_size:t, :, :]
            windows.append(window)
        windows = torch.stack(windows, dim=1)
        
        windows = windows.view(-1, self.window_size, D)

        kshape = KShape(n_clusters=self.n_patterns)
        cluster_centers = kshape.fit_predict(windows.cpu().numpy())
        
        patterns = torch.from_numpy(cluster_centers).to(x.device)
        return patterns
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_nodes, d_model] 输入序列
        Returns:
            attn_output: [batch_size, seq_len, num_nodes, d_model]
        """
        B, T, N, D = x.shape
        
        Q = self.W_Q(x)  # [B, T, N, d_k]
        K = self.W_K(x)  # [B, T, N, d_k]
        V = self.W_V(x)  # [B, T, N, d_k]
        
        for t in range(self.window_size, T):
            hist_window = x[:, t-self.window_size:t, :, :]  # [B, S, N, D]
            
            u_t = self.W_u(hist_window)  # [B, S, N, d_k]
            m_i = self.W_m(self.patterns)  # [n_patterns, S, d_k]
            
            similarity = torch.einsum('bsnd,psd->bpn', u_t, m_i)  # [B, n_patterns, N]
            w_i = F.softmax(similarity, dim=1)  # [B, n_patterns, N]

            pattern_info = torch.einsum('bpn,psd->bnsd', w_i, self.W_c(self.patterns))  # [B, N, S, d_k]
            r_t = pattern_info.sum(dim=2)  # [B, N, d_k]
            
            K[:, t] = K[:, t] + r_t
            
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output

class GroupEquivariantAttention(nn.Module):   
    def __init__(self, d_model=256, d_k=128, n_heads=8):
        """
        params:
            d_model: input feature dimension
            d_k: query/key/value dimension
            n_heads: number of attention heads
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        
        self.W_O = nn.Linear(d_k * n_heads, d_model)
        
        self.group_transforms = [
            self._identity,
            self._flip_x,
            self._flip_y, 
            self._flip_xy
        ]
        
    def _identity(self, x):
        return x
        
    def _flip_x(self, x):
        x = x.clone()
        x[..., 1::3] *= -1
        x[..., 2::3] *= -1
        return x
        
    def _flip_y(self, x):
        x = x.clone()
        x[..., 0::3] *= -1
        x[..., 2::3] *= -1
        return x
        
    def _flip_xy(self, x):
        x = x.clone()
        x[..., 0::3] *= -1
        x[..., 1::3] *= -1
        return x
        
    def message_passing(self, x_g, x_h):
        B, T, N, _ = x_g.shape
        
        Q = self.W_Q(x_g).view(B, T, N, self.n_heads, self.d_k)
        K = self.W_K(x_h).view(B, T, N, self.n_heads, self.d_k)
        V = self.W_V(x_h).view(B, T, N, self.n_heads, self.d_k)
        
        Q = Q.transpose(2, 3)  # [B, T, n_heads, N, d_k]
        K = K.transpose(2, 3)  # [B, T, n_heads, N, d_k]
        V = V.transpose(2, 3)  # [B, T, n_heads, N, d_k]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_weights, V)  # [B, T, n_heads, N, d_k]
        
        output = output.transpose(2, 3).contiguous()  # [B, T, N, n_heads, d_k]
        output = output.view(B, T, N, -1)  # [B, T, N, n_heads*d_k]
        
        return self.W_O(output)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_nodes, d_model] 输入序列
        Returns:
            output: [batch_size, seq_len, num_nodes, d_model]
        """
        B, T, N, D = x.shape
        outputs = []
        
        for g in self.group_transforms:
            x_g = g(x)
            view_outputs = []
            for h in self.group_transforms:
                x_h = h(x)
                g_inv_h = lambda x: g(h(x))
                x_gh = g_inv_h(x)
                
                x_combined = torch.cat([x_h, x_gh], dim=-1)
                
                view_output = self.message_passing(x_g, x_combined)
                view_outputs.append(view_output)
            
            outputs.append(torch.stack(view_outputs).mean(0))
            
        return torch.stack(outputs).mean(0)

class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model=256, d_k=128, n_heads=8, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        
        self.W_O = nn.Linear(d_k * n_heads, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def temporal_attention(self, node_sequence):
        B, T, _ = node_sequence.shape
        
        Q = self.W_Q(node_sequence).view(B, T, self.n_heads, self.d_k)
        K = self.W_K(node_sequence).view(B, T, self.n_heads, self.d_k)
        V = self.W_V(node_sequence).view(B, T, self.n_heads, self.d_k)
        
        Q = Q.transpose(1, 2)  # [B, n_heads, T, d_k]
        K = K.transpose(1, 2)  # [B, n_heads, T, d_k]
        V = V.transpose(1, 2)  # [B, n_heads, T, d_k]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)  # [B, n_heads, T, d_k]
        
        output = output.transpose(1, 2).contiguous()  # [B, T, n_heads, d_k]
        output = output.view(B, T, -1)  # [B, T, n_heads*d_k]

        output = self.W_O(output)
        
        return output
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, num_nodes, d_model] 输入序列
        Returns:
            output: [batch_size, seq_len, num_nodes, d_model]
        """
        B, T, N, D = x.shape
        residual = x
        
        outputs = []
        for n in range(N):
            node_sequence = x[:, :, n, :]  # [B, T, D]

            node_output = self.temporal_attention(node_sequence)  # [B, T, D]
            outputs.append(node_output)

        output = torch.stack(outputs, dim=2)  # [B, T, N, D]

        output = self.layer_norm(output + residual)
        
        return output


model = TemporalSelfAttention(
    d_model=256,
    d_k=128,
    n_heads=8,
    dropout=0.1
)

# input shape: [batch_size, seq_len, num_nodes, d_model]
x = torch.randn(32, 100, 10, 256)
output = model(x)

# print(output.shape)