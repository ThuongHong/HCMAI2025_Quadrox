# Vietnamese Captioning & Multilingual Reranking

This system provides Vietnamese image captioning using Vintern-1B-v3.5 and multilingual text similarity for enhanced reranking.

## Requirements

- Python ≥3.10
- CPU-only (no GPU/CUDA required)
- Dependencies: `torch`, `torchvision`, `transformers>=4.43`, `sentence-transformers`, `accelerate`, `safetensors`

## Model Setup

Download required models offline:

```bash
# Download Vintern-1B-v3.5 for Vietnamese captioning
huggingface-cli download 5CD-AI/Vintern-1B-v3_5 --local-dir ./models/Vintern-1B-v3_5

# Download multilingual CLIP for Vietnamese-English text similarity
huggingface-cli download sentence-transformers/clip-ViT-B-32-multilingual-v1 --local-dir ./models/clip-multilingual/clip-ViT-B-32-multilingual-v1
```

## Configuration

Enable Vietnamese captioning in your config:

```yaml
# config/retrieval.yaml or equivalent
captioner:
  enabled: true
  model: vintern_cpu # Use VinternCaptionerCPU
  model_path: ./models/Vintern-1B-v3_5 # Path to Vintern model
  style: dense # dense | short | tags | ocr
  max_new_tokens: 64 # Max caption length
  allow_on_demand: false # false = use cache only (faster UI)
  workers: 2 # Parallel workers (2-4 optimal for CPU)
  alpha: 1.0 # CLIP score weight
  beta: 0.25 # Caption score weight

text_embedding:
  multilingual:
    enabled: true
    model_path: ./models/clip-multilingual/clip-ViT-B-32-multilingual-v1
```

## Precompute Captions (Recommended)

For faster online performance, precompute captions offline:

```bash
# Precompute captions for all keyframes
python tools/precompute_captions.py --images_glob "resources/keyframes/**/*.jpg"

# Custom options
python tools/precompute_captions.py \
  --images_glob "resources/keyframes/L21/**/*.jpg" \
  --style dense \
  --max_new_tokens 64 \
  --max_workers 4 \
  --batch_size 10
```

## Usage Examples

### Basic Captioning

```python
from app.retrieval.rerank.vintern_captioner import VinternCaptionerCPU

# Initialize captioner
captioner = VinternCaptionerCPU(model_path="./models/Vintern-1B-v3_5")

# Single image
result = captioner.caption_image("resources/keyframes/L01/L01_V001/001.jpg")
print(result["caption"])  # Vietnamese caption

# Batch processing (2-4 images in parallel)
results = captioner.batch_caption([
    "resources/keyframes/L01/L01_V001/001.jpg",
    "resources/keyframes/L01/L01_V001/002.jpg"
])
```

### Multilingual Text Similarity

```python
from app.common.text_embedding_multilingual import get_multilingual_embedder

# Initialize embedder
embedder = get_multilingual_embedder()

# Vietnamese-English similarity
query = "người đàn ông đi bộ"  # Vietnamese query
captions = [
    "một người đàn ông đang đi bộ trong công viên",  # Vietnamese caption
    "man walking in park",                           # English caption
    "xe hơi màu đỏ"                                 # Unrelated Vietnamese
]

similarity = embedder.compute_similarity(query, captions)
print(similarity[0])  # [0.85, 0.82, 0.15] - high similarity for both languages
```

### Reranking Pipeline

```python
from app.retrieval.rerank.captioning import CaptionRanker

# Enable Vietnamese captioning
ranker = CaptionRanker(
    model_name="vintern_cpu",
    vintern_model_path="./models/Vintern-1B-v3_5",
    allow_on_demand=False  # Use cache only for speed
)

# Rerank with Vietnamese captions
results = await ranker.rerank_with_captions(
    query="người đàn ông đi bộ",
    candidates=search_candidates,
    top_t=20
)
```

## Performance Guidelines

### CPU Optimization

- Use 2-4 workers for parallel caption generation
- Set `allow_on_demand=false` in production for consistent response times
- Precompute captions offline when possible

### Typical Performance (CPU)

- Single caption: 3-8 seconds (first time), <0.1s (cached)
- Batch 10 images: 15-30 seconds (2 workers)
- Top-100 reranking: 1-5 minutes total (with precomputed captions: <30s)

### Memory Usage

- Vintern model: ~2.5GB RAM
- Multilingual CLIP: ~400MB RAM
- Combined: ~3GB RAM total

## Testing

Run smoke tests to verify installation:

```bash
# Quick test
python tests/test_vintern_smoke.py

# Manual test
python - << 'PY'
from app.retrieval.rerank.vintern_captioner import VinternCaptionerCPU
c = VinternCaptionerCPU(model_path="./models/Vintern-1B-v3_5")
print(c.caption_image("resources/keyframes/L01/L01_V001/001.jpg")["caption"])
PY
```

## Feature Flags

The system uses feature flags for safe deployment:

- `captioner.enabled`: Master switch for Vietnamese captioning
- `captioner.allow_on_demand`: Allow real-time caption generation (may be slow)
- `text_embedding.multilingual.enabled`: Enable multilingual similarity

When disabled, the system falls back to the original synthetic caption pipeline.

## Troubleshooting

### Models not found

```bash
# Check model directories exist
ls ./models/Vintern-1B-v3_5/
ls ./models/clip-multilingual/clip-ViT-B-32-multilingual-v1/

# Re-download if missing
huggingface-cli download 5CD-AI/Vintern-1B-v3_5 --local-dir ./models/Vintern-1B-v3_5
```

### Memory issues

- Reduce `max_workers` to 1-2
- Use `torch.float32` instead of `torch.bfloat16` if needed
- Process smaller batches

### Slow performance

- Precompute captions offline using `tools/precompute_captions.py`
- Set `allow_on_demand=false` to use cache only
- Verify CPU has sufficient cores (4+ recommended)

## Architecture

```
Query (Vietnamese/English)
    ↓
Search Pipeline → Base Results
    ↓
Reranking Pipeline:
  1. SuperGlobal (visual similarity)
  2. Caption (Vietnamese captions via Vintern)
  3. LLM (advanced reasoning)
    ↓
Ensemble Scoring:
  - α × CLIP_score + β × Caption_score
    ↓
Final Ranked Results
```

The Vietnamese captioning integrates seamlessly with the existing pipeline while maintaining backward compatibility.
