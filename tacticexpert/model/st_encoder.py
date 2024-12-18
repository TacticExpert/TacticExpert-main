import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from tslearn.clustering import KShape

class STEncoder(nn.Module):
    """Spatio-Temporal Encoder with D2 symmetry views and positional encodings"""
    
    def __init__(self, hidden_dim=256, num_players=10, dropout=0.1):
        super().__init__()
        
        # Static feature embeddings
        self.team_embedding = nn.Embedding(2, hidden_dim)  # home/away team
        self.position_embedding = nn.Embedding(5, hidden_dim)  # 5 positions
        
        # Dynamic feature projection
        self.dynamic_proj = nn.Linear(15, hidden_dim)  # 15 dynamic features
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)  # Transformer PE
        self.lap_pe = LaplacianPE(hidden_dim, num_players)  # Laplacian PE
        
    def forward(self, static_feats, dynamic_feats, adj_matrix=None):
        """
        Args:
            static_feats: [batch_size, num_players, 2] - team, position
            dynamic_feats: [batch_size, num_players, 15] - dynamic features
            adj_matrix: Optional[batch_size, num_players, num_players]
        Returns:
            final_emb: [batch_size, 4, num_players, hidden_dim]
        """
        batch_size = static_feats.size(0)
        
        # 1. Data embeddings (Xdata)
        # Static features
        team_emb = self.team_embedding(static_feats[:,:,0])
        pos_emb = self.position_embedding(static_feats[:,:,1])
        static_emb = team_emb + pos_emb  # [batch, num_players, hidden]
        
        # Dynamic features with D2 symmetry views
        dynamic_views = []
        for transform in [self._identity, self._flip_x, self._flip_y, self._flip_z]:
            transformed_feats = transform(dynamic_feats)
            dynamic_views.append(self.dynamic_proj(transformed_feats))
        dynamic_emb = torch.stack(dynamic_views, dim=1)  # [batch, 4, num_players, hidden]
        
        # 2. Laplacian positional encoding (Xlap)
        if adj_matrix is None:
            adj_matrix = self.compute_adjacency(static_feats, dynamic_feats)
        lap_pe = self.lap_pe(adj_matrix)  # [batch, num_players, hidden]
        
        # 3. Transformer positional encoding (Xpe)
        pos_enc = self.pos_encoder(torch.zeros_like(static_emb))  # [batch, num_players, hidden]
        
        # 4. Combine all embeddings: Xemb = Xdata + Xlap + Xpe
        # Expand static and positional encodings to match dynamic views
        static_expanded = static_emb.unsqueeze(1)  # [batch, 1, num_players, hidden]
        lap_pe_expanded = lap_pe.unsqueeze(1)      # [batch, 1, num_players, hidden]
        pos_enc_expanded = pos_enc.unsqueeze(1)    # [batch, 1, num_players, hidden]
        
        # Broadcast to all 4 views
        static_expanded = static_expanded.expand(-1, 4, -1, -1)
        lap_pe_expanded = lap_pe_expanded.expand(-1, 4, -1, -1)
        pos_enc_expanded = pos_enc_expanded.expand(-1, 4, -1, -1)
        
        # Final combination
        final_emb = (static_expanded +    # Static features
                    dynamic_emb +         # Dynamic features with D2 symmetry
                    lap_pe_expanded +     # Graph structure information
                    pos_enc_expanded)     # Sequential position information
        
        return final_emb

    def _identity(self, x):
        """恒等变换 I = [1 0 0]
                       [0 1 0]
                       [0 0 1]
        Args:
            x: [batch_size, num_players, 15] 动态特征
            动态特征顺序: [head(3), headOrientation(3), hips(3), foot1(3), foot2(3)]
        """
        return x
        
    def _flip_x(self, x):
        """绕x轴旋转180度 Rx = [1  0   0]
                              [0 -1   0]
                              [0  0  -1]
        """
        x = x.clone()
        # 每个3D坐标的y,z分量需要取反
        coord_groups = [
            (1, 2),    # head y,z
            (4, 5),    # headOrientation y,z
            (7, 8),    # hips y,z
            (10, 11),  # foot1 y,z
            (13, 14),  # foot2 y,z
        ]
        
        for y_idx, z_idx in coord_groups:
            x[:, :, [y_idx, z_idx]] *= -1
        
        return x
        
    def _flip_y(self, x):
        """绕y轴旋转180度 Ry = [-1 0  0]
                              [0  1  0]
                              [0  0 -1]
        """
        x = x.clone()
        # 每个3D坐标的x,z分量需要取反
        coord_groups = [
            (0, 2),    # head x,z
            (3, 5),    # headOrientation x,z
            (6, 8),    # hips x,z
            (9, 11),   # foot1 x,z
            (12, 14),  # foot2 x,z
        ]
        
        for x_idx, z_idx in coord_groups:
            x[:, :, [x_idx, z_idx]] *= -1
        
        return x
        
    def _flip_z(self, x):
        """绕z轴旋转180度 Rz = [-1  0  0]
                              [0  -1  0]
                              [0   0  1]
        """
        x = x.clone()
        # 每个3D坐标的x,y分量需要取反
        coord_groups = [
            (0, 1),    # head x,y
            (3, 4),    # headOrientation x,y
            (6, 7),    # hips x,y
            (9, 10),   # foot1 x,y
            (12, 13),  # foot2 x,y
        ]
        
        for x_idx, y_idx in coord_groups:
            x[:, :, [x_idx, y_idx]] *= -1
        
        return x

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

        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Compute sinusoidal position encoding
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        # Normalize position encoding
        pe = self._normalize(pe)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def _normalize(self, pe):
        """Normalize position encoding for better training stability
        
        Args:
            pe: position encoding [max_len, d_model]
        Returns:
            normalized position encoding
        """
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
    """时空延迟效应注意力模块"""
    
    def __init__(self, d_model=256, d_k=128, window_size=10, n_patterns=8):
        """
        参数:
            d_model: 输入特征维度
            d_k: query/key/value的维度
            window_size: 时间窗口大小S
            n_patterns: 延迟效应模式数量Np
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_k
        self.window_size = window_size
        self.n_patterns = n_patterns
        
        # 空域注意力的Q/K/V投影矩阵
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        
        # 延迟效应模式的参数矩阵
        self.W_u = nn.Linear(d_model, d_k)
        self.W_m = nn.Linear(d_model, d_k)
        self.W_c = nn.Linear(d_model, d_k)
        
        # 初始化延迟效应模式
        self.patterns = nn.Parameter(torch.randn(n_patterns, window_size, d_model))
        
    def extract_patterns(self, x):
        """使用k-Shape聚类提取延迟效应模式
        
        Args:
            x: [batch_size, seq_len, num_nodes, d_model]
        Returns:
            patterns: [n_patterns, window_size, d_model]
        """
        # 准备数据
        B, T, N, D = x.shape
        windows = []
        for t in range(self.window_size, T):
            window = x[:, t-self.window_size:t, :, :]
            windows.append(window)
        windows = torch.stack(windows, dim=1)
        
        # 重塑为k-Shape可处理的形状
        windows = windows.view(-1, self.window_size, D)
        
        # 使用k-Shape聚类
        kshape = KShape(n_clusters=self.n_patterns)
        cluster_centers = kshape.fit_predict(windows.cpu().numpy())
        
        # 将聚类中心转换为tensor
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
        
        # 1. 计算Q/K/V矩阵
        Q = self.W_Q(x)  # [B, T, N, d_k]
        K = self.W_K(x)  # [B, T, N, d_k]
        V = self.W_V(x)  # [B, T, N, d_k]
        
        # 2. 对每个时间步处理延迟效应
        for t in range(self.window_size, T):
            # 提取历史窗口
            hist_window = x[:, t-self.window_size:t, :, :]  # [B, S, N, D]
            
            # 计算与模式的相似度
            u_t = self.W_u(hist_window)  # [B, S, N, d_k]
            m_i = self.W_m(self.patterns)  # [n_patterns, S, d_k]
            
            # 计算注意力分数
            similarity = torch.einsum('bsnd,psd->bpn', u_t, m_i)  # [B, n_patterns, N]
            w_i = F.softmax(similarity, dim=1)  # [B, n_patterns, N]
            
            # 加权融合模式信息
            pattern_info = torch.einsum('bpn,psd->bnsd', w_i, self.W_c(self.patterns))  # [B, N, S, d_k]
            r_t = pattern_info.sum(dim=2)  # [B, N, d_k]
            
            # 更新key值
            K[:, t] = K[:, t] + r_t
            
        # 3. 计算注意力输出
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output

class GroupEquivariantAttention(nn.Module):
    """空域群等变注意力机制，处理D2群的四种对称视图"""
    
    def __init__(self, d_model=256, d_k=128, n_heads=8):
        """
        参数:
            d_model: 输入特征维度
            d_k: query/key/value的维度
            n_heads: 注意力头数量
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        
        # 每个注意力头的Q/K/V投影矩阵
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        
        # 输出投影
        self.W_O = nn.Linear(d_k * n_heads, d_model)
        
        # D2群的四种变换: [id, flip_x, flip_y, flip_xy]
        self.group_transforms = [
            self._identity,
            self._flip_x,
            self._flip_y, 
            self._flip_xy
        ]
        
    def _identity(self, x):
        """恒等变换"""
        return x
        
    def _flip_x(self, x):
        """绕x轴翻转"""
        x = x.clone()
        x[..., 1::3] *= -1  # y坐标取反
        x[..., 2::3] *= -1  # z坐标取反
        return x
        
    def _flip_y(self, x):
        """绕y轴翻转"""
        x = x.clone()
        x[..., 0::3] *= -1  # x坐标取反
        x[..., 2::3] *= -1  # z坐标取反
        return x
        
    def _flip_xy(self, x):
        """绕x和y轴翻转"""
        x = x.clone()
        x[..., 0::3] *= -1  # x坐标取反
        x[..., 1::3] *= -1  # y坐标取反
        return x
        
    def message_passing(self, x_g, x_h):
        """实现群等变注意力的消息传递
        
        Args:
            x_g: [batch_size, seq_len, num_nodes, d_model] g变换后的输入
            x_h: [batch_size, seq_len, num_nodes, d_model] h变换后的输入
        Returns:
            output: [batch_size, seq_len, num_nodes, d_model]
        """
        B, T, N, _ = x_g.shape
        
        # 1. 计算Q/K/V
        Q = self.W_Q(x_g).view(B, T, N, self.n_heads, self.d_k)
        K = self.W_K(x_h).view(B, T, N, self.n_heads, self.d_k)
        V = self.W_V(x_h).view(B, T, N, self.n_heads, self.d_k)
        
        # 2. 转置以进行注意力计算
        Q = Q.transpose(2, 3)  # [B, T, n_heads, N, d_k]
        K = K.transpose(2, 3)  # [B, T, n_heads, N, d_k]
        V = V.transpose(2, 3)  # [B, T, n_heads, N, d_k]
        
        # 3. 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 4. 加权求和得到输出
        output = torch.matmul(attn_weights, V)  # [B, T, n_heads, N, d_k]
        
        # 5. 转置回原始维度并合并多头
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
        
        # 对每个g变换
        for g in self.group_transforms:
            x_g = g(x)
            view_outputs = []
            
            # 与其他h变换交互
            for h in self.group_transforms:
                x_h = h(x)
                # g^(-1)h变换
                g_inv_h = lambda x: g(h(x))  # 在D2群中g^(-1) = g
                x_gh = g_inv_h(x)
                
                # 拼接h变换和g^(-1)h变换的特征
                x_combined = torch.cat([x_h, x_gh], dim=-1)
                
                # 消息传递
                view_output = self.message_passing(x_g, x_combined)
                view_outputs.append(view_output)
            
            # 平均所有h变换的输出
            outputs.append(torch.stack(view_outputs).mean(0))
            
        # 平均所有g变换的输出
        return torch.stack(outputs).mean(0)

class TemporalSelfAttention(nn.Module):
    """时域自注意力模块，处理球员节点的时间依赖关系"""
    
    def __init__(self, d_model=256, d_k=128, n_heads=8, dropout=0.1):
        """
        参数:
            d_model: 输入特征维度
            d_k: query/key/value的维度
            n_heads: 注意力头数量
            dropout: dropout比率
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        
        # 时域注意力的Q/K/V投影矩阵
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        
        # 输出投影
        self.W_O = nn.Linear(d_k * n_heads, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def temporal_attention(self, node_sequence):
        """对单个球员节点的时间序列计算自注意力
        
        Args:
            node_sequence: [batch_size, seq_len, d_model] 单个节点的时间序列
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        B, T, _ = node_sequence.shape
        
        # 1. 计算Q/K/V
        Q = self.W_Q(node_sequence).view(B, T, self.n_heads, self.d_k)
        K = self.W_K(node_sequence).view(B, T, self.n_heads, self.d_k)
        V = self.W_V(node_sequence).view(B, T, self.n_heads, self.d_k)
        
        # 2. 转置以进行注意力计算
        Q = Q.transpose(1, 2)  # [B, n_heads, T, d_k]
        K = K.transpose(1, 2)  # [B, n_heads, T, d_k]
        V = V.transpose(1, 2)  # [B, n_heads, T, d_k]
        
        # 3. 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 4. 加权求和得到输出
        output = torch.matmul(attn_weights, V)  # [B, n_heads, T, d_k]
        
        # 5. 转置回原始维度并合并多头
        output = output.transpose(1, 2).contiguous()  # [B, T, n_heads, d_k]
        output = output.view(B, T, -1)  # [B, T, n_heads*d_k]
        
        # 6. 输出投影
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
        
        # 对每个球员��点分别计算时域自注意力
        outputs = []
        for n in range(N):
            # 提取单个节点的时间序列
            node_sequence = x[:, :, n, :]  # [B, T, D]
            
            # 计算时域自注意力
            node_output = self.temporal_attention(node_sequence)  # [B, T, D]
            outputs.append(node_output)
            
        # 拼接所有节点的输出
        output = torch.stack(outputs, dim=2)  # [B, T, N, D]
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output
model = TemporalSelfAttention(
    d_model=256,
    d_k=128,
    n_heads=8,
    dropout=0.1
)

# 输入形状: [batch_size, seq_len, num_nodes, d_model]
x = torch.randn(32, 100, 10, 256)
output = model(x)

