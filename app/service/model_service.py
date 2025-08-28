import torch
import numpy as np
from PIL import Image


class ModelService:
    def __init__(
        self,
        model ,
        preprocess ,
        tokenizer
        ):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.model.eval()
    
    def embedding(self, query_text: str) -> np.ndarray:
        """
        Return (1, ndim 1024) torch.Tensor
        """
        with torch.no_grad():
            text_tokens = self.tokenizer([query_text]).to(self.device)
            query_embedding = self.model.encode_text(text_tokens).cpu().detach().numpy().astype(np.float32) # (1, 1024)
        return query_embedding

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Return (1, ndim 1024) torch.Tensor from image
        """
        with torch.no_grad():
            image_preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
            image_embedding = self.model.encode_image(image_preprocessed).cpu().detach().numpy().astype(np.float32) # (1, 1024)
        return image_embedding

            