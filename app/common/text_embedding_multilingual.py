"""Multilingual text embedding for Vietnamese-English text similarity."""

import logging
import time
from pathlib import Path
from typing import List, Union, Optional

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class MultilingualTextEmbedder:
    """
    Multilingual text embedding using CLIP-ViT-B-32-multilingual-v1.
    Optimized for Vietnamese-English cross-lingual similarity.
    """

    def __init__(
        self,
        model_path: str = "./models/clip-multilingual/clip-ViT-B-32-multilingual-v1",
        device: str = "cpu"
    ):
        """
        Initialize multilingual text embedder.

        Args:
            model_path: Path to sentence-transformers multilingual model
            device: Device to use (cpu/cuda)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model: Optional[SentenceTransformer] = None

        logger.info(
            f"MultilingualTextEmbedder initialized: model_path={model_path}, device={device}")

    def _load_model(self) -> None:
        """Load sentence-transformers model if not already loaded."""
        if self.model is not None:
            return

        try:
            logger.info(
                f"Loading multilingual text embedding model from {self.model_path}")
            start_time = time.time()

            # Load model from local path
            self.model = SentenceTransformer(
                str(self.model_path), device=self.device)

            # Set model to evaluation mode
            self.model.eval()

            elapsed = time.time() - start_time
            logger.info(
                f"Multilingual text embedding model loaded successfully in {elapsed:.2f}s")

        except Exception as e:
            logger.error(
                f"Failed to load multilingual text embedding model: {e}")
            raise

    def encode_text_vi(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode Vietnamese/multilingual texts to embeddings.

        Args:
            texts: List of text strings to encode
            normalize_embeddings: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        try:
            # Load model if needed
            self._load_model()

            # Filter out empty texts
            valid_texts = [text.strip() if text.strip()
                           else "no text" for text in texts]

            logger.debug(
                f"Encoding {len(valid_texts)} texts with batch_size={batch_size}")
            start_time = time.time()

            # Encode texts using sentence-transformers
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False
            )

            elapsed = time.time() - start_time
            logger.debug(
                f"Text encoding completed in {elapsed:.2f}s for {len(texts)} texts")

            return embeddings

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            # Return zero embeddings as fallback
            if hasattr(self, 'model') and self.model:
                # Get embedding dimension from model
                embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                # Default CLIP embedding dimension
                embedding_dim = 512

            return np.zeros((len(texts), embedding_dim), dtype=np.float32)

    def compute_similarity(
        self,
        text1: Union[str, List[str], np.ndarray],
        text2: Union[str, List[str], np.ndarray]
    ) -> np.ndarray:
        """
        Compute cosine similarity between texts.

        Args:
            text1: First text(s) or embeddings
            text2: Second text(s) or embeddings

        Returns:
            Similarity matrix (n_text1, n_text2)
        """
        try:
            # Handle different input types
            if isinstance(text1, str):
                emb1 = self.encode_text_vi([text1])
            elif isinstance(text1, list):
                emb1 = self.encode_text_vi(text1)
            else:
                emb1 = text1

            if isinstance(text2, str):
                emb2 = self.encode_text_vi([text2])
            elif isinstance(text2, list):
                emb2 = self.encode_text_vi(text2)
            else:
                emb2 = text2

            # Ensure embeddings are normalized for cosine similarity
            emb1_norm = normalize(emb1, axis=1)
            emb2_norm = normalize(emb2, axis=1)

            # Compute cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm.T)

            return similarity

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            # Return zero similarity as fallback
            if isinstance(text1, str):
                n1 = 1
            elif isinstance(text1, list):
                n1 = len(text1)
            else:
                n1 = text1.shape[0]

            if isinstance(text2, str):
                n2 = 1
            elif isinstance(text2, list):
                n2 = len(text2)
            else:
                n2 = text2.shape[0]

            return np.zeros((n1, n2), dtype=np.float32)

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'model') and self.model:
                del self.model
        except:
            pass


# Global instance for reuse
_global_embedder: Optional[MultilingualTextEmbedder] = None


def get_multilingual_embedder(
    model_path: str = "./models/clip-multilingual/clip-ViT-B-32-multilingual-v1",
    device: str = "cpu"
) -> MultilingualTextEmbedder:
    """
    Get global multilingual text embedder instance.

    Args:
        model_path: Path to model (used only on first call)
        device: Device to use (used only on first call)

    Returns:
        MultilingualTextEmbedder instance
    """
    global _global_embedder

    if _global_embedder is None:
        _global_embedder = MultilingualTextEmbedder(
            model_path=model_path, device=device)

    return _global_embedder
