import torch
from PIL import Image
import os
import json
import prompt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils.TimeLogger import log
from model.st_encoder import STEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

class MoERouter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=384, num_experts=5, temperature=0.07, dropout=0.1):
        super(MoERouter, self).__init__()
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        self.num_experts = num_experts
        self.temperature = temperature
        
    def forward(self, x):
        logits = self.router(x)
        routing_weights = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        expert_index = torch.argmax(routing_weights, dim=-1)
        return routing_weights, expert_index

class ExpertModule(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256, dropout=0.1):
        super(ExpertModule, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.self_attention = nn.MultiheadAttention(output_dim, num_heads=4, dropout=dropout)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.expert(x)
        attn_output, _ = self.self_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = x + attn_output.squeeze(0)
        x = self.norm(x)
        return x

class TacticExpert(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=384, output_dim=256, num_experts=5):
        super(TacticExpert, self).__init__()
        
        self.router = MoERouter(
            input_dim=15 * 10,
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
        
        self.experts = nn.ModuleList([
            STEncoder(
                hidden_dim=hidden_dim,
                num_players=10
            ) for _ in range(num_experts)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, batch_data, return_all_experts=False):
        routing_features = batch_data.x[:, :, 0, :].reshape(batch_data.x.size(0), -1)
        routing_weights, expert_index = self.router(routing_features)
        
        if return_all_experts:
            all_expert_outputs = []
            for expert in self.experts:
                expert_output = expert(batch_data)  # [batch, time, players, hidden]
                expert_output = expert_output.mean(dim=[1,2])  # [batch, hidden]
                expert_output = self.output_proj(expert_output)  # [batch, output]
                all_expert_outputs.append(expert_output)
            outputs = torch.stack(all_expert_outputs, dim=1)  # [batch, num_experts, output]
        else:
            outputs = torch.zeros(
                batch_data.x.size(0), 
                self.output_proj[-1].out_features
            ).to(batch_data.x.device)
            
            for i in range(batch_data.x.size(0)):
                expert_idx = expert_index[i]
                expert_output = self.experts[expert_idx](batch_data[i:i+1])
                expert_output = expert_output.mean(dim=[1,2])
                outputs[i] = self.output_proj(expert_output.squeeze(0))
                
        return outputs, routing_weights, expert_index

def info_nce_loss(expert_outputs, similarity_matrix, k_negatives=50, temperature=0.07):
    """ InfoNCE contrastive loss
    
    Args:
        expert_outputs: shape [batch_size, num_experts, output_dim]
        similarity_matrix: pre-computed similarity matrix
        k_negatives: number of negative samples
        temperature: temperature parameter
    """
    batch_size, num_experts, _ = expert_outputs.shape
    device = expert_outputs.device
    
    expert_outputs = F.normalize(expert_outputs, dim=-1)

    sim_matrix = torch.matmul(
        expert_outputs.view(-1, expert_outputs.shape[-1]),
        expert_outputs.view(-1, expert_outputs.shape[-1]).t()
    )
    

    pos_mask = (similarity_matrix > 0.8).float()
    neg_mask = (similarity_matrix < 0.3).float()

    neg_indices = torch.multinomial(
        neg_mask.view(-1), 
        k_negatives * batch_size * num_experts, 
        replacement=True
    )
    
    total_loss = 0
    n_valid_pairs = 0
    
    for i in range(batch_size * num_experts):
        pos_indices = torch.where(pos_mask[i])[0]
        if len(pos_indices) == 0:
            continue
            
        anchor = expert_outputs.view(-1, expert_outputs.shape[-1])[i]
        positives = expert_outputs.view(-1, expert_outputs.shape[-1])[pos_indices]
        negatives = expert_outputs.view(-1, expert_outputs.shape[-1])[
            neg_indices[i*k_negatives:(i+1)*k_negatives]
        ]
        
        logits = torch.cat([
            torch.matmul(anchor.unsqueeze(0), positives.T),
            torch.matmul(anchor.unsqueeze(0), negatives.T)
        ], dim=1) / temperature
        
        labels = torch.zeros(1, device=device, dtype=torch.long)
        total_loss += F.cross_entropy(logits, labels)
        n_valid_pairs += 1
    
    return total_loss / max(n_valid_pairs, 1)



def train_moe(model, optimizer, batch_data, batch_labels, similarity_matrix, alpha=0.5):
    model.train()
    optimizer.zero_grad()
    
    expert_outputs, routing_weights, expert_index = model(batch_data, return_all_experts=True)
    
    contra_loss = info_nce_loss(expert_outputs, similarity_matrix)
    
    selected_outputs = torch.gather(
        expert_outputs, 
        1, 
        expert_index.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, expert_outputs.size(-1))
    ).squeeze(1)
    main_loss = info_nce_loss(selected_outputs.unsqueeze(1), batch_labels)
    
    routing_loss = compute_routing_balance_loss(routing_weights)
    
    total_loss = main_loss + alpha * contra_loss + 0.1 * routing_loss
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'main_loss': main_loss.item(),
        'contra_loss': contra_loss.item(),
        'routing_loss': routing_loss.item()
    }

def compute_routing_balance_loss(routing_weights):
    expert_usage = routing_weights.mean(dim=0)
    target_usage = torch.ones_like(expert_usage) / expert_usage.size(0)
    return F.kl_div(expert_usage.log(), target_usage, reduction='batchmean')

def evaluate_model(model, val_data, val_labels, batch_size=32):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch_data = val_data[i:i+batch_size].cuda()
            batch_labels = val_labels[i:i+batch_size].cuda()
            
            outputs, _, _ = model(batch_data)
            loss = info_nce_loss(outputs.unsqueeze(1), batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
    
    return {
        'total_loss': total_loss / len(val_data),
        'accuracy': correct / total
    }

def train_model(model, train_loader, val_loader, train_similarity, val_similarity, num_epochs=200):
    initial_lr = 2e-5
    main_lr = 2e-3
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    
    for epoch in range(5):
        train_losses = []
        for batch_data, batch_labels in train_loader:
            loss_dict = train_moe(
                model, 
                optimizer, 
                batch_data, 
                batch_labels, 
                train_similarity[batch_labels][:, batch_labels]
            )
            train_losses.append(loss_dict['total_loss'])
        log(f"Epoch {epoch}: loss={np.mean(train_losses):.4f}")
            
    for param_group in optimizer.param_groups:
        param_group['lr'] = main_lr
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for batch_data, batch_labels in train_loader:
            loss_dict = train_moe(
                model, 
                optimizer, 
                batch_data, 
                batch_labels, 
                train_similarity[batch_labels][:, batch_labels]
            )
            train_losses.append(loss_dict['total_loss'])
            
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.cuda()
                expert_outputs, _, _ = model(batch_data, return_all_experts=True)
                val_loss = info_nce_loss(
                    expert_outputs, 
                    val_similarity[batch_labels][:, batch_labels]
                )
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        avg_train_loss = np.mean(train_losses)
        
        log(f"Epoch {epoch}:")
        log(f"- train_loss: {avg_train_loss:.4f}")
        log(f"- val_loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            log(f"early stop at epoch {epoch}")
            break
            
        scheduler.step()

def main():
    embeddings = np.load("./src/data/offensive_tactics_embeddings.npy")
    similarity_matrix = np.load("./src/data/offensive_tactics_similarity.npy")
    
    embeddings = torch.FloatTensor(embeddings)
    similarity_matrix = torch.FloatTensor(similarity_matrix)

    labels = torch.arange(len(embeddings))
    
    train_size = int(0.8 * len(embeddings))
    indices = torch.randperm(len(embeddings))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data = embeddings[train_indices]
    train_labels = labels[train_indices]
    train_similarity = similarity_matrix[train_indices][:, train_indices]
    
    val_data = embeddings[val_indices]
    val_labels = labels[val_indices]
    val_similarity = similarity_matrix[val_indices][:, val_indices]
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = TacticExpert(
        input_dim=768,
        hidden_dim=384,
        output_dim=256,
        num_experts=5
    ).cuda()
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_similarity=train_similarity,
        val_similarity=val_similarity,
        num_epochs=100
    )

if __name__ == "__main__":
    main()
