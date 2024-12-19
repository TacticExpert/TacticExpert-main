import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='TacticExpert Parameters')
    
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_path', default='tactic_expert', type=str)
    parser.add_argument('--load_model', default=None, type=str)
    parser.add_argument('--data_dir', default='../data', type=str)
    
    parser.add_argument('--stage', default=1, type=int, help='training stage (1 or 2)')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_experts', default=5, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    
    parser.add_argument('--context_length', default=77, type=int)
    parser.add_argument('--vocab_size', default=49408, type=int)
    parser.add_argument('--transformer_width', default=512, type=int)
    parser.add_argument('--transformer_heads', default=8, type=int)
    parser.add_argument('--transformer_layers', default=12, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)

    return parser.parse_args()

args = parse_args()