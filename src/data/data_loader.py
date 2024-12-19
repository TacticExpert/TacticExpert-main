import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch_geometric.data as g_data
from typing import List, Dict, Tuple
import os
from utils.TimeLogger import log

class BasketballDataset(Dataset):
    def __init__(self, data_dir: str = "./src/data/basketball-instants-dataset"):
        """Initialize the dataset.
        Args:
            data_dir: The path to the original data directory.
        """
        self.data_dir = data_dir
    
        self.frame_data = self._load_and_preprocess()
        self.frame_ids = sorted(list(self.frame_data.keys()))
        
        self.sequence_lengths = self._get_sequence_lengths()
        self.window_size = self._determine_window_size()
        
        self.player_to_idx = {}
        self._build_player_mapping()
        
        log(f"Dataset loaded with:")
        log(f"- Total sequences: {len(self.sequence_lengths)}")
        log(f"- Window size: {self.window_size} frames")
        log(f"- Total players: {len(self.player_to_idx)}")

    def _load_and_preprocess(self) -> Dict:
        """load and preprocess the original data into a dict with frame time as the key
        Returns:
            Dict: {
                relative_time: [{player1_info}, {player2_info}, ...],
                ...
            }
        """
        processed_data = {}

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found. Please follow the instructions in README.md to download the data.")
            
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError(f"No json files found in {self.data_dir}")
            
        for file in json_files:
            file_path = os.path.join(self.data_dir, file)
            try:
                with open(file_path, 'r') as f:
                    raw_data = json.load(f)
                    
                for frame in raw_data:
                    relative_time = frame['relative_time']
                    frame_id = f"{file}_{relative_time}"
                    
                    if frame_id not in processed_data:
                        processed_data[frame_id] = []

                    player_info = {
                        'player_id': frame['player_id'],
                        'team': frame['team'],
                        'position': frame['position'],
                        'relative_time': relative_time,
                        'head': frame['head'],
                        'head_orientation': frame['headOrientation'],
                        'hips': frame['hips'],
                        'foot1': frame['foot1'],
                        'foot2': frame['foot2'],
                        'foot1_grounded': frame['foot1_at_the_ground'],
                        'foot2_grounded': frame['foot2_at_the_ground'],
                        'speed': frame['speed_mps'],
                        'has_ball': frame['has_ball']
                    }
                    
                    player_exists = False
                    for existing_player in processed_data[frame_id]:
                        if existing_player['player_id'] == player_info['player_id']:
                            player_exists = True
                            break
                            
                    if not player_exists:
                        processed_data[frame_id].append(player_info)
                        
            except Exception as e:
                log(f"Error loading file {file}: {str(e)}")
                continue
                
        return processed_data

    def _get_sequence_lengths(self) -> List[int]:
        """get the length of each sequence by the prefix of frame_id(file name)"""
        sequences = {}
        for frame_id in self.frame_ids:
            seq_name = frame_id.split('_')[0]
            if seq_name not in sequences:
                sequences[seq_name] = 0
            sequences[seq_name] += 1
            
        return list(sequences.values())

    def _determine_window_size(self) -> int:
        """determine the appropriate window size
        
        rules:
        1. use the length of the shortest sequence by default
        2. if the shortest sequence is too short(< 5 frames), use the second shortest sequence
        3. if all sequences are short, use the minimum value 5
        """
        MIN_WINDOW_SIZE = 5
        sorted_lengths = sorted(self.sequence_lengths)
        
        if len(sorted_lengths) == 0:
            return MIN_WINDOW_SIZE
            
        if sorted_lengths[0] < MIN_WINDOW_SIZE:
            if len(sorted_lengths) > 1:
                return max(sorted_lengths[1], MIN_WINDOW_SIZE)
            return MIN_WINDOW_SIZE
            
        return sorted_lengths[0]

    def _build_player_mapping(self):
        """build the mapping from player_id to index"""
        unique_players = set()
        for frame_data in self.frame_data.values():
            for player in frame_data:
                unique_players.add(player['player_id'])
        
        for idx, player_id in enumerate(sorted(unique_players)):
            self.player_to_idx[player_id] = idx
            
        self.num_players = len(self.player_to_idx)

    def _create_node_features(self, frame_data: List[Dict]) -> torch.Tensor:
        """create the node feature matrix
        
        features include: [x coordinate, y coordinate, speed, whether holding the ball, team number(one-hot), position number(one-hot)]
        """
        features = torch.zeros(self.num_players, 15)
        
        for player in frame_data:
            idx = self.player_to_idx[player['player_id']]
            
            pos = np.mean([player['head'], player['hips']], axis=0)
            features[idx, 0:2] = torch.tensor(pos)

            features[idx, 2] = player['speed']
            
            features[idx, 3] = float(player['has_ball'])
            
            team_idx = 0 if player['team'] == 'team_A' else 1
            features[idx, 4+team_idx] = 1.0
            
            pos_map = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4}
            pos_idx = pos_map[player['position']]
            features[idx, 6+pos_idx] = 1.0
            
            features[idx, 11:14] = torch.tensor(player['head_orientation'])
            
            features[idx, 14] = float(player['foot1_grounded'] or player['foot2_grounded'])
            
        return features

    def _create_edge_index(self, frame_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """build the edge index matrix based on the team, position relation and head orientation
        
        rules:
        1. players in the same team are more likely to connect
        2. players who are closer to each other are more likely to connect
        3. players who are in adjacent positions are more likely to connect
        4. players who have similar head orientations are more likely to connect
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (edge_index, edge_attr)
        """
        edges = []
        edge_weights = []
        
        pos_adjacent = {
            'PG': ['SG', 'SF'],
            'SG': ['PG', 'SF', 'PF'],
            'SF': ['SG', 'PF', 'C'],
            'PF': ['SG', 'SF', 'C'],
            'C': ['SF', 'PF']
        }
        
        MAX_DISTANCE = 15.0
        
        for i in range(len(frame_data)):
            for j in range(i + 1, len(frame_data)):
                player_i = frame_data[i]
                player_j = frame_data[j]

                weight = 0.0

                if player_i['team'] == player_j['team']:
                    weight += 0.3

                pos_i = np.mean([player_i['head'], player_i['hips']], axis=0)
                pos_j = np.mean([player_j['head'], player_j['hips']], axis=0)
                distance = np.linalg.norm(pos_i - pos_j)
                distance_weight = max(0, 1 - distance / MAX_DISTANCE)
                weight += 0.3 * distance_weight
                
                if player_j['position'] in pos_adjacent[player_i['position']]:
                    weight += 0.2
                    
                cos_sim = torch.nn.functional.cosine_similarity(
                    torch.tensor(player_i['head_orientation']),
                    torch.tensor(player_j['head_orientation']),
                    dim=0
                )
                weight += 0.2 * max(0, cos_sim.item())
                
                if weight > 0.6:
                    i_idx = self.player_to_idx[player_i['player_id']]
                    j_idx = self.player_to_idx[player_j['player_id']]
                    edges.extend([[i_idx, j_idx], [j_idx, i_idx]])
                    edge_weights.extend([weight, weight])
        
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
            
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        
        return edge_index, edge_attr

    def __len__(self) -> int:
        """return the number of available time windows"""
        return len(self.frame_ids) - self.window_size + 1

    def __getitem__(self, idx: int) -> g_data.Data:
        """get the graph data for the time window
        
        Returns:
            g_data.Data: the graph data
        """
        window_frames = self.frame_ids[idx:idx + self.window_size]
        
        node_features = []
        edge_indices = []
        edge_attrs = []
        
        for frame_id in window_frames:
            frame_data = self.frame_data[frame_id]
            frame_features = self._create_node_features(frame_data)
            edge_index, edge_attr = self._create_edge_index(frame_data)
            
            node_features.append(frame_features)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
        
        node_features = torch.stack(node_features, dim=-1)
        
        data = g_data.Data(
            x=node_features,
            edge_index=edge_indices[0],
            edge_attr=edge_attrs[0]
        )
        
        return data

def create_dataloader(
    data_dir: str,
    window_size: int = 10,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """create a dataloader for the dataset  
    
    Args:
        data_dir: the path to the dataset
        window_size: the size of the time window
        batch_size: the batch size
        shuffle: whether to shuffle the data
        num_workers: the number of processes to load the data
    """
    dataset = BasketballDataset(data_dir, window_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader