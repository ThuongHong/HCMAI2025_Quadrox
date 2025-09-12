import torch
import numpy as np
from PIL import Image
from transformers import SiglipModel, SiglipProcessor
from typing import Union
import os


class SigLIPModelService:
    def __init__(self, model_path: str):
        """
        Initialize SigLIP model service
        Args:
            model_path: Path to the SigLIP model directory
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        # Load SigLIP model and processor
        self.processor = SiglipProcessor.from_pretrained(model_path)
        self.model = SiglipModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        print(f"SigLIP model loaded from {model_path} on {self.device}")
    
    def embedding(self, query_text: str) -> np.ndarray:
        """
        Generate text embedding using SigLIP
        Args:
            query_text: Input text query
        Returns:
            numpy array with shape (1, embedding_dim)
        """
        with torch.no_grad():
            inputs = self.processor(text=[query_text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_features = self.model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            return text_features.cpu().detach().numpy().astype(np.float32)
    
    def image_embedding(self, image: Union[Image.Image, str]) -> np.ndarray:
        """
        Generate image embedding using SigLIP
        Args:
            image: PIL Image or path to image
        Returns:
            numpy array with shape (1, embedding_dim)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            image_features = self.model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            return image_features.cpu().detach().numpy().astype(np.float32)
    
    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension of the SigLIP model
        """
        # For SigLIP, the embedding dimension is in the text/vision config
        return self.model.config.text_config.hidden_size