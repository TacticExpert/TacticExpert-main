import sys
sys.path.append("./tacticexpert")

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer, AutoModel
import torch
from PIL import Image
import requests
import os
import json
import prompt
import numpy as np
import datetime

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
        
    def generate_descriptions(self, image_dir, save_path):
        des_dict = {}
        for file in os.listdir(image_dir):
            url = os.path.join(image_dir, file)
            image = Image.open(requests.get(url, stream=True).raw)
            
            prompt_text = "[INST] <image> " + prompt.PROMPTS["offensive_tactics_description"] + " [/INST]"
            # prompt = "[INST] <image> Please generate a more detailed and professional description based on the input of the standard initial position diagram of the \"One inside and four outside\" basketball tactic. This tactic is usually used by teams with excellent low-post offensive players to provide single-player space for inside players and break the zone defense. [/INST]"
            inputs = self.processor(prompt_text, image, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=150)
            des_dict[file] = self.processor.decode(outputs[0], skip_special_tokens=True)
            
        with open(save_path, "w") as f:
            json.dump(des_dict, f)
        return des_dict
    
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

    url_dir = "./tacticexpert/raw_data/offensive_tactics_image/"
    output_dir = "./tacticexpert/raw_data/"
    os.makedirs(output_dir, exist_ok=True)

    des_file = os.path.join(output_dir, "offensive_tactics_description.json")
    descriptions = generator.generate_descriptions(url_dir, des_file)

    embeddings = generator.compute_embeddings(list(descriptions.values()))
    embeddings_file = os.path.join(output_dir, "offensive_tactics_embeddings.npy")
    np.save(embeddings_file, embeddings)

    similarity_matrix = generator.compute_similarity_matrix(embeddings)
    similarity_file = os.path.join(output_dir, "offensive_tactics_similarity.npy")
    np.save(similarity_file, similarity_matrix)

    result = {
        "descriptions": descriptions,
        "embeddings_file": embeddings_file,
        "similarity_file": similarity_file,
        "files": list(descriptions.keys()),
        "timestamp": str(datetime.datetime.now())
    }
    
    result_file = os.path.join(output_dir, "offensive_tactics_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

if __name__ == "__main__":
    main()