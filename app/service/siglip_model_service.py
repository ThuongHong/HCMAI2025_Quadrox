import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
from typing import Union

# ----- Text preprocessing for SigLIP2 -----
import re
import unicodedata

# Common control/instruction words (English)
_EN_INSTR = r"(?:please|kindly|find|search|retrieve|return|show|select|get|take|identify|detect|locate)"
# Temporal/order scaffolding (English)
_EN_TIME  = r"(?:first|second|third|next|then|finally|beginning|ending|start(?:ing)?|ends?|earliest|latest|initial|final)"
# Enumeration prefixes like E1:, Scene 2:, Step-3:, (1), 1), 1.
_ENUM_PAT = r"(?:(?:e|scene|step|shot|frame|moment|q|question)\s*[:#-]?\s*\d+|\(\d+\)|\d+\)|^\s*\d+\.)"

def preprocess_query_siglip2_en(q: str) -> str:
    """
    Normalize an already-refined English query for SigLIP2 retrieval.
    - Unicode NFC, collapse whitespace
    - Drop enumeration prefixes and control/temporal scaffolding
    - Convert dashes/semicolons to commas (CLIP-style descriptors)
    - Preserve quoted text; add 'text "<...>"' hint (max 2 items)
    - Lowercase at the end
    - Keep it concise; true truncation handled by tokenizer (max_length=64)
    """
    if not q or not isinstance(q, str):
        return ""

    # 1) Unicode normalize + collapse newlines/whitespace
    q = unicodedata.normalize("NFC", q)
    q = re.sub(r"[\r\n]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    # 2) Remove enumeration prefixes anywhere (E1:, Scene 2:, (1), 1) …)
    q = re.sub(_ENUM_PAT, "", q, flags=re.IGNORECASE)

    # 3) Convert separators -> comma
    #   em-dash/en-dash/hyphen/bullets/semicolon -> comma
    q = re.sub(r"\s*[–—\-•;]+\s*", ", ", q)

    # 4) Remove leading instruction/temporal scaffolding terms (but keep nouns/verbs around them)
    #   We only remove the "control word" tokens themselves; content words remain.
    q = re.sub(rf"\b{_EN_INSTR}\b", "", q, flags=re.IGNORECASE)
    q = re.sub(rf"\b{_EN_TIME}\b",   "", q, flags=re.IGNORECASE)

    # Clean double commas/spaces from previous steps
    q = re.sub(r"\s*,\s*,\s*", ", ", q)
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"^[, ]+|[, ]+$", "", q)

    # 5) Preserve quoted text and add a light OCR hint (max 2)
    #   Covers “smart quotes”, ASCII quotes, single quotes
    quoted = re.findall(r"“([^”]+)”|\"([^\"]+)\"|‘([^’]+)’|'([^']+)'", q)
    texts = [t for grp in quoted for t in grp if t]
    for t in texts[:2]:
        t_clean = re.sub(r"\s+", " ", t.strip())
        if t_clean:
            # If not already present as text "...", append one hint
            if f'text "{t_clean}"' not in q.lower():
                q += f', text "{t_clean}"'

    # 6) Lowercase last (SigLIP2 trained with lowercase text)
    q = q.lower()

    # 7) De-duplicate comma-separated descriptors while preserving order
    parts = [p.strip() for p in q.split(",") if p.strip()]
    seen = set()
    dedup = []
    for p in parts:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    q = ", ".join(dedup)

    return q

# ----- SigLIP2 Model Service -----
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
                print("🔍 Query before preprocessing for SigLIP2:", query_text)
                q = preprocess_query_siglip2_en(query_text)
                print("🔍 Preprocessed query for SigLIP2:", q)
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