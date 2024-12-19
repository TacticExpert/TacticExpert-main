from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer, AutoModel
import torch
from PIL import Image
import os
import json
import prompt
import numpy as np
import datetime
from data_loader import BasketballDataset
from utils.TimeLogger import logger


class TacticDescriptionGenerator:
    def __init__(self):
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mixtral-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mixtral-7b-hf", 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
    def _get_initial_frames_and_images(self, data_dir: str, image_dir: str) -> tuple:
        """get all frames with time 0 and corresponding images
        Args:
            data_dir: json data directory path
            image_dir: image directory path
        Returns:
            Tuple[Dict, Dict]: (frame_data, frame_images)
        """
        dataset = BasketballDataset(data_dir)
        initial_frames = {}
        initial_images = {}
        
        for frame_id, frame_data in dataset.frame_data.items():
            sequence_name = frame_id.split('_')[0]
            time = float(frame_id.split('_')[1])
            
            if time == 0:
                initial_frames[sequence_name] = frame_data
                image_path = os.path.join(image_dir, f"{time}.jpg")
                if os.path.exists(image_path):
                    initial_images[sequence_name] = Image.open(image_path)
                else:
                    logger.warning(f"Image not found for frame: {frame_id}")
                
        logger.info(f"Found {len(initial_frames)} initial frames with {len(initial_images)} images")
        return initial_frames, initial_images
        
    def _load_official_frames_and_images(self, official_data_dir: str, official_image_dir: str) -> tuple:
        """load frames and images from official dataset
        Args:
            official_data_dir: official data json directory path
            official_image_dir: official data image directory path
        Returns:
            Tuple[Dict, Dict]: (frame_data, frame_images)
        """
        official_frames = {}
        official_images = {}
        
        if not os.path.exists(official_data_dir):
            raise FileNotFoundError(f"Official data directory {official_data_dir} not found")
            
        for file in os.listdir(official_data_dir):
            if file.endswith('.json'):
                with open(os.path.join(official_data_dir, file), 'r') as f:
                    frame_data = json.load(f)
                    frame_name = file.replace('.json', '')
                    official_frames[frame_name] = frame_data
                    
                    image_path = os.path.join(official_image_dir, f"{frame_name}.jpg")
                    if os.path.exists(image_path):
                        official_images[frame_name] = Image.open(image_path)
                    else:
                        logger.warning(f"Image not found for official frame: {frame_name}")
                    
        logger.info(f"Loaded {len(official_frames)} official frames with {len(official_images)} images")
        return official_frames, official_images
        
    def generate_descriptions(self, 
                            raw_data_dir: str, 
                            raw_image_dir: str,
                            official_data_dir: str, 
                            official_image_dir: str, 
                            save_dir: str):
        """generate descriptions for all frames and calculate similarity
        Args:
            raw_data_dir: raw data json directory path
            raw_image_dir: raw data image directory path
            official_data_dir: official data json directory path
            official_image_dir: official data image directory path
            save_dir: save result directory path
        """
        os.makedirs(save_dir, exist_ok=True)
        
        initial_frames, initial_images = self._get_initial_frames_and_images(raw_data_dir, raw_image_dir)
        official_frames, official_images = self._load_official_frames_and_images(official_data_dir, official_image_dir)
        
        descriptions = {}
        
        for seq_name in initial_frames.keys():
            if seq_name not in initial_images:
                logger.warning(f"Skipping initial frame {seq_name} due to missing image")
                continue
                
            prompt_text = "[INST] <image> " + prompt.PROMPTS["tactical_description"] + " [/INST]"
            inputs = self.processor(prompt_text, initial_images[seq_name], return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=150)
            descriptions[f"initial_{seq_name}"] = self.processor.decode(outputs[0], skip_special_tokens=True)
            
        for frame_name in official_frames.keys():
            if frame_name not in official_images:
                logger.warning(f"Skipping official frame {frame_name} due to missing image")
                continue
                
            prompt_text = "[INST] <image> " + prompt.PROMPTS["tactical_description"] + " [/INST]"
            inputs = self.processor(prompt_text, official_images[frame_name], return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=150)
            descriptions[f"official_{frame_name}"] = self.processor.decode(outputs[0], skip_special_tokens=True)
            
        all_descriptions = list(descriptions.values())
        embeddings = self.compute_embeddings(all_descriptions)
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        result = {
            "descriptions": descriptions,
            "embeddings": embeddings.tolist(),
            "similarity_matrix": similarity_matrix.tolist(),
            "frame_names": list(descriptions.keys()),
            "timestamp": str(datetime.datetime.now())
        }
        
        result_file = os.path.join(save_dir, "tactical_descriptions_and_similarities.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Results saved to {result_file}")
        return result
    
    def compute_embeddings(self, descriptions):
        encoded_input = self.tokenizer(
            descriptions, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
            
        return torch.mean(model_output.last_hidden_state, dim=1).cpu().numpy()
    
    def compute_similarity_matrix(self, embeddings):
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i][j] = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
        return similarity_matrix

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    
    generator = TacticDescriptionGenerator()
    
    raw_data_dir = "./src/data/basketball-instants-dataset"
    raw_image_dir = "./src/data/basketball-instants-dataset/images"
    official_data_dir = "./src/data/official_data"
    official_image_dir = "./src/data/official_data/images"
    save_dir = "./src/data/processed_data"
    
    result = generator.generate_descriptions(
        raw_data_dir=raw_data_dir,
        raw_image_dir=raw_image_dir,
        official_data_dir=official_data_dir,
        official_image_dir=official_image_dir,
        save_dir=save_dir
    )
    
    return result

if __name__ == "__main__":
    main()