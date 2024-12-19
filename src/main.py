import sys
sys.path.append('./src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from params import args
from model.moe import TacticExpert
from model.text_graph_grounding import CLIP, contrastive_loss
from utils.TimeLogger import log
from data.data_loader import create_dataloader

class Trainer:
    def __init__(self):
        self.setup_logging()
        self.setup_device()
        self.setup_model()
        self.setup_optimizer()
        
    def setup_logging(self):
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(f'../Logs/stage{args.stage}.log'),
                logging.StreamHandler()
            ]
        )
        
    def setup_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Using device: {self.device}")
        
    def setup_model(self):
        if args.stage == 1:
            self.model = TacticExpert(
                input_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.embed_dim,
                num_experts=args.num_experts
            ).to(self.device)
            
        else:
            self.model = CLIP(args).to(self.device)
            
            if args.load_model:
                checkpoint = torch.load(f'../Models/{args.load_model}.pt')
                self.model.st_encoder.load_state_dict(checkpoint['st_encoder'])
                log(f"Loaded pretrained ST Encoder from {args.load_model}")
                for param in self.model.st_encoder.parameters():
                    param.requires_grad = False
                    
    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=args.epochs,
            eta_min=args.min_lr
        )

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }
        
        if args.stage == 1:
            checkpoint['st_encoder'] = self.model.experts[0].state_dict()
            
        save_path = f'../Models/{args.save_path}_stage{args.stage}'
        if is_best:
            save_path += '_best'
        save_path += '.pt'
        
        torch.save(checkpoint, save_path)
        log(f"Checkpoint saved: {save_path}")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            if args.stage == 1:
                # Stage 1: Train MoE classification
                data, labels = batch
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs, routing_weights, expert_index = self.model(data)
                loss = F.cross_entropy(outputs, labels)
                
            else:
                # Stage 2: Train contrastive learning
                graphs, texts = batch
                graphs = graphs.to(self.device)
                texts = texts.to(self.device)
                
                graph_features, text_features, logit_scale = self.model(graphs, texts)
                loss = contrastive_loss(
                    graph_features, 
                    text_features,
                    logit_scale,
                    self.device
                )
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
        
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if args.stage == 1:
                    data, labels = batch
                    data = data.to(self.device) 
                    labels = labels.to(self.device)
                    
                    outputs, _, _ = self.model(data)
                    loss = F.cross_entropy(outputs, labels)
                    
                else:
                    graphs, texts = batch
                    graphs = graphs.to(self.device)
                    texts = texts.to(self.device)
                    
                    graph_features, text_features, logit_scale = self.model(graphs, texts)
                    loss = contrastive_loss(
                        graph_features,
                        text_features, 
                        logit_scale,
                        self.device
                    )
                    
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader):
        best_loss = float('inf')
        patience_counter = 0
        
        log(f"Starting training stage {args.stage}")
        log(f"Total epochs: {args.epochs}")
        log(f"Patience: {args.patience}")
        
        for epoch in range(args.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            self.scheduler.step()
            
            log(f'Epoch {epoch}:')
            log(f'Train Loss: {train_loss:.4f}')
            log(f'Val Loss: {val_loss:.4f}')
            log(f'Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}')
            
            # Save regular checkpoint
            if (epoch + 1) % args.save_freq == 0:
                self.save_checkpoint(epoch, val_loss)
            
            # Save best checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                log(f'New best validation loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                log(f"Early stopping at epoch {epoch}")
                break

def setup_data_loaders():
    train_loader = create_dataloader(
        data_dir=os.path.join(args.data_dir, 'train'),
        window_size=args.window_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = create_dataloader(
        data_dir=os.path.join(args.data_dir, 'val'),
        window_size=args.window_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    return train_loader, val_loader

def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    os.makedirs('../Models', exist_ok=True)
    os.makedirs('../Logs', exist_ok=True)
    
    train_loader, val_loader = setup_data_loaders()
    
    trainer = Trainer()
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()