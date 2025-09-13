import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
from typing import Union


class SigLIPModelService:
    def __init__(self, model_path: str):
        """
        Initialize SigLIP2 model service
        Args:
            model_path: Path to the SigLIP2 model directory
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        # Load SigLIP2 model and processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        print(f"SigLIP2 model loaded from {model_path} on {self.device}")
    
    def embedding(self, query_text: str) -> np.ndarray:
        """
        Generate text embedding using SigLIP2
        Args:
            query_text: Input text query
        Returns:
            numpy array with shape (1, embedding_dim)
        """
        try:
            # Clear any cached state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                # Process with clean state - force no padding and truncation
                print("🔍 Query right before embedding:", query_text)
                q = query_text.lower()  # SigLIP2 trained on lowercased text
                inputs = self.processor(
                    text=[q],
                    return_tensors="pt",
                    padding="max_length",  
                    truncation=True,
                    max_length=64
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}      
                
                # Debug log
                seq_len = inputs['input_ids'].shape[1]
                print(f"🔍 DEBUG - Query: '{query_text[:50]}...' -> {seq_len} tokens")
                
                text_features = self.model.get_text_features(**inputs)
                # Normalize embeddings
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                # Clean up inputs to prevent memory leaks
                del inputs
                
                return text_features.cpu().detach().numpy().astype(np.float32)
                
        except Exception as e:
            print(f"❌ SigLIP2 embedding error: {e}")
            # Clear any partial state on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def image_embedding(self, image: Union[Image.Image, str]) -> np.ndarray:
        """
        Generate image embedding using SigLIP2
        Args:
            image: PIL Image or path to image
        Returns:
            numpy array with shape (1, embedding_dim)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        try:
            # Clear any cached state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            with torch.no_grad():
                inputs = self.processor(images=[image], return_tensors="pt", padding=False)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Clean up inputs
                del inputs
                
                return image_features.cpu().detach().numpy().astype(np.float32)
                
        except Exception as e:
            print(f"❌ SigLIP2 image embedding error: {e}")
            # Clear any partial state on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension of the SigLIP2 model
        """
        # For SigLIP2, the embedding dimension is in the text/vision config
        return getattr(self.model.config.text_config, "projection_size",
                        self.model.config.text_config.hidden_size)